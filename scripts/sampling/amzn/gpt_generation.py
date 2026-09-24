import os
import io
import sys
import json
import time
import base64
import shutil
import argparse
from tqdm import tqdm
from PIL import Image, ImageOps
from openai import OpenAI

# Paste the key here, or leave it empty and export OPENAI_API_KEY / pass --api_key.
API_KEY = ""

def save_auxiliary_files(ref_image_path, gt_var_src_path, var_file, prompt, asin_out_dir):
    """Save reference image, GT variation image, and prompt txt file if missing."""
    os.makedirs(asin_out_dir, exist_ok=True)

    # 1. Save reference image
    ref_out_path = os.path.join(asin_out_dir, "reference.jpg")
    if os.path.exists(ref_image_path) and not os.path.exists(ref_out_path):
        shutil.copy(ref_image_path, ref_out_path)

    # Extract variation suffix and extension
    var_basename = os.path.basename(var_file)
    var_stem, ext = os.path.splitext(var_basename)
    if var_stem.lower().startswith("variation_"):
        suffix = var_stem[len("variation_"):]
    else:
        suffix = var_stem

    # 2. Save GT variation image
    gt_out_file_1 = f"gt_variation_{suffix}{ext}"
    gt_out_file_2 = f"GT_variation_{suffix}{ext}"
    gt_out_path_1 = os.path.join(asin_out_dir, gt_out_file_1)
    gt_out_path_2 = os.path.join(asin_out_dir, gt_out_file_2)
    if os.path.exists(gt_var_src_path) and not os.path.exists(gt_out_path_1) and not os.path.exists(gt_out_path_2):
        shutil.copy(gt_var_src_path, gt_out_path_1)

    # 3. Save SDG prompt text file
    txt_filename = f"variation_{suffix}.txt"
    txt_out_path = os.path.join(asin_out_dir, txt_filename)
    if not os.path.exists(txt_out_path):
        with open(txt_out_path, "w", encoding="utf-8") as f:
            f.write(prompt.strip() + "\n")


def as_png_upload(path):
    """Re-encode the reference as a plain PNG the API will accept.

    Amazon product shots include iPhone MPO (multi-frame JPEG) and CMYK files,
    which the API rejects with `invalid_image_file`. EXIF rotation is baked in.
    """
    im = Image.open(path)
    im.seek(0)                      # MPO/GIF: keep only the first frame
    im = ImageOps.exif_transpose(im).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    buf.name = os.path.splitext(os.path.basename(path))[0] + ".png"
    return buf


def main():
    parser = argparse.ArgumentParser(description="OpenAI image-edit generation benchmark script for AMZN dataset")
    parser.add_argument('--data_dir', type=str, default='/home/daewon/Dreambench-Meticulous/assets/data/amzn')
    parser.add_argument('--metadata_filename', type=str, default='prompt-updated_metadata.json', help="Metadata filename to read (default: prompt-updated_metadata.json)")
    parser.add_argument('--samples_dir', type=str, default='/home/daewon/Dreambench-Meticulous/samples/amzn/GPT')
    parser.add_argument('--category', type=str, default=None, help="Process only a specific category folder (e.g. raw_meta_Appliances)")
    parser.add_argument('--product', type=str, default=None, help="Process only a specific product ASIN")
    parser.add_argument('--model', type=str, default="gpt-image-2", help="Image model id available to your account")
    parser.add_argument('--size', type=str, default="1024x1024")
    parser.add_argument('--quality', type=str, default="high", choices=["low", "medium", "high", "auto"])
    parser.add_argument('--max_retries', type=int, default=3, help="Retries per image on API errors")
    parser.add_argument('--api_key', type=str, default=None, help="Overrides API_KEY / OPENAI_API_KEY")
    parser.add_argument('--skip_existing', type=str, default="True", help="Skip existing files ('True' or 'False')")

    args = parser.parse_args()

    skip_existing = args.skip_existing.lower() == 'true'

    api_key = args.api_key or API_KEY or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("no API key: set API_KEY at the top of this file, pass --api_key, or export OPENAI_API_KEY")

    # Determine categories to process
    if args.category:
        categories = [args.category]
    else:
        # Scan data_dir for raw_meta_* subdirectories
        categories = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d)) and d.startswith('raw_meta_')]
        categories.sort()

    to_generate = []
    for category in categories:
        metadata_path = os.path.join(args.data_dir, category, args.metadata_filename)
        if not os.path.exists(metadata_path):
            fallback_meta = os.path.join(args.data_dir, category, 'metadata.json')
            if os.path.exists(fallback_meta):
                metadata_path = fallback_meta
            else:
                print(f"Warning: Neither {args.metadata_filename} nor metadata.json found in {category}. Skipping category.")
                continue

        print(f"Reading metadata for {category} from: {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        for item in metadata:
            asin = item['asin']

            if args.product and asin != args.product:
                continue

            ref_file = item['reference_file']
            ref_image_path = os.path.join(args.data_dir, category, ref_file)

            for variation in item['variation_files']:
                var_file = variation['file']
                var_prompt = variation['prompt']
                out_path = os.path.join(args.samples_dir, category, var_file)
                if skip_existing and os.path.exists(out_path):
                    continue

                to_generate.append({
                    'category': category,
                    'asin': asin,
                    'ref_image_path': ref_image_path,
                    'gt_var_src_path': os.path.join(args.data_dir, category, var_file),
                    'var_file': var_file,
                    'out_path': out_path,
                    'prompt': var_prompt
                })

    if not to_generate:
        print("All targets already generated or no targets found. Exiting.")
        return

    print(f"Found {len(to_generate)} targets to generate.")

    client = OpenAI(api_key=api_key)
    print(f"Using model: {args.model} ({args.size}, quality={args.quality})")

    for task in tqdm(to_generate, desc="Generating images"):
        category = task['category']
        asin = task['asin']
        ref_image_path = task['ref_image_path']
        gt_var_src_path = task['gt_var_src_path']
        var_file = task['var_file']
        out_path = task['out_path']
        target_prompt = task['prompt']

        print(f"\nProcessing {category} / {asin} -> {os.path.basename(out_path)}")
        print(f"Prompt: {target_prompt}")

        if not os.path.exists(ref_image_path):
            print(f"Error: Reference image {ref_image_path} does not exist. Skipping.")
            continue

        # Create target folder and copy reference / GT / prompt alongside the output.
        # Done here rather than during the scan, so targets we never generate do not
        # leave behind folders holding auxiliary files and no sample.
        save_auxiliary_files(ref_image_path, gt_var_src_path, var_file,
                             target_prompt, os.path.dirname(out_path))

        # The reference image is the conditioning image, the variation prompt the text prompt
        for attempt in range(1, args.max_retries + 1):
            try:
                result = client.images.edit(
                    model=args.model,
                    image=as_png_upload(ref_image_path),
                    prompt=target_prompt,
                    n=1,
                    size=args.size,
                    quality=args.quality,
                )
                
                # The API returns base64 PNG; re-encode to whatever out_path asks for
                image = Image.open(io.BytesIO(base64.b64decode(result.data[0].b64_json)))
                image.convert("RGB").save(out_path)
                print(f"Saved generated image to {out_path}")
                break
            except Exception as e:
                print(f"Attempt {attempt}/{args.max_retries} failed for {category}/{asin} to {out_path}: {e}")
                if attempt < args.max_retries:
                    time.sleep(5 * attempt)   # back off on rate limits


if __name__ == "__main__":
    main()
