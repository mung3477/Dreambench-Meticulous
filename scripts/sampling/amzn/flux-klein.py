import os
import sys
import json
import torch
import shutil
import argparse
from tqdm import tqdm
from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image

def save_auxiliary_files(ref_image_path, gt_var_src_path, var_file, prompt, asin_out_dir):
    """Save reference image, GT variation image, and prompt txt file if missing."""
    os.makedirs(asin_out_dir, exist_ok=True)

    # 1. Save reference image if available
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

def main():
    parser = argparse.ArgumentParser(description="FLUX.2-klein-9B T2I / TI2I sampling benchmark script for AMZN dataset")
    parser.add_argument('--data_dir', type=str, default='/root/Desktop/workspace/woosung/commercial-dreambench/assets/data/amzn')
    parser.add_argument('--metadata_filename', type=str, default='prompt-updated_metadata.json', help="Metadata filename to read (default: prompt-updated_metadata.json)")
    parser.add_argument('--mode', type=str, choices=['t2i', 'ti2i'], default='ti2i', help="Sampling mode: 't2i' (text-to-image) or 'ti2i' (text+image-to-image)")
    parser.add_argument('--samples_dir', type=str, default=None, help="Output directory for generated samples (default: amzn/flux-klein for ti2i, amzn-T2I/flux-klein for t2i)")
    parser.add_argument('--category', type=str, default=None, help="Process only a specific category folder (e.g. raw_meta_Appliances)")
    parser.add_argument('--product', type=str, default=None, help="Process only a specific product ASIN")
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--num_inference_steps', type=int, default=4)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--device', type=str, default="cuda:0", help="CUDA device identifier (default: cuda:0)")
    parser.add_argument('--skip_existing', type=str, default="True", help="Skip existing files ('True' or 'False')")

    args = parser.parse_args()

    # Determine default samples_dir if not specified
    if args.samples_dir is None:
        if args.mode == 'ti2i':
            args.samples_dir = '/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn/flux-klein'
        else:
            args.samples_dir = '/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn-T2I/flux-klein'

    skip_existing = args.skip_existing.lower() == 'true'

    # Determine categories to process
    if args.category:
        categories = [args.category]
    else:
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
            categorical_name = item['categorical_name']

            if args.product and asin != args.product:
                continue

            ref_file = item['reference_file']
            ref_image_path = os.path.join(args.data_dir, category, ref_file)

            for variation in item['variation_files']:
                var_file = variation['file']
                var_prompt = variation['prompt']
                out_path = os.path.join(args.samples_dir, category, var_file)
                gt_var_src_path = os.path.join(args.data_dir, category, var_file)
                asin_out_dir = os.path.dirname(out_path)

                # Save auxiliary files in output directory if missing
                save_auxiliary_files(ref_image_path, gt_var_src_path, var_file, var_prompt, asin_out_dir)

                if skip_existing and os.path.exists(out_path):
                    continue

                to_generate.append({
                    'category': category,
                    'asin': asin,
                    'ref_image_path': ref_image_path,
                    'gt_var_src_path': gt_var_src_path,
                    'var_file': var_file,
                    'out_path': out_path,
                    'prompt': var_prompt
                })

    if not to_generate:
        print("All targets already generated or no targets found. Exiting.")
        return

    print(f"Found {len(to_generate)} targets to generate in {args.mode.upper()} mode.")

    # Explicitly enforce GPU / target device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading FLUX.2-klein-9B model on {device}...")
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9B",
        torch_dtype=torch.bfloat16
    ).to(device)

    for task in tqdm(to_generate, desc=f"Generating {args.mode.upper()} images"):
        category = task['category']
        asin = task['asin']
        ref_image_path = task['ref_image_path']
        gt_var_src_path = task['gt_var_src_path']
        var_file = task['var_file']
        out_path = task['out_path']
        target_prompt = task['prompt']

        print(f"\nProcessing {category} / {asin} -> {os.path.basename(out_path)}")
        print(f"Prompt: {target_prompt}")

        try:
            # Create target folder and save auxiliary files
            asin_out_dir = os.path.dirname(out_path)
            save_auxiliary_files(ref_image_path, gt_var_src_path, var_file, target_prompt, asin_out_dir)

            generator = torch.Generator(device=device.type).manual_seed(args.seed)
            pipeline_kwargs = {
                "prompt": target_prompt,
                "guidance_scale": args.guidance_scale,
                "num_inference_steps": args.num_inference_steps,
                "generator": generator,
                "height": args.height if args.height else None,
                "width": args.width if args.width else None
            }

            # In TI2I mode, pass reference image as input_images / image parameter
            if args.mode == 'ti2i':
                if not os.path.exists(ref_image_path):
                    print(f"Error: Reference image {ref_image_path} does not exist. Skipping.")
                    continue
                reference_image = load_image(ref_image_path).convert("RGB")
                if args.width and args.height:
                    reference_image = reference_image.resize((args.width, args.height))
                pipeline_kwargs["image"] = reference_image

            result = pipe(**pipeline_kwargs).images[0]

            # Save final generated image
            result.save(out_path)
            print(f"Saved generated image to {out_path}")
        except Exception as e:
            print(f"Error generating {category}/{asin} to {out_path}: {e}")

if __name__ == "__main__":
    main()
