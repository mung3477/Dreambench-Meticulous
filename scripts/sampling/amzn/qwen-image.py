import os
import sys
import json
import torch
import shutil
import argparse
from tqdm import tqdm
from diffusers import QwenImagePipeline

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

def main():
    parser = argparse.ArgumentParser(description="Qwen-Image-2512 T2I benchmark generation script for AMZN dataset")
    parser.add_argument('--data_dir', type=str, default='/root/Desktop/workspace/woosung/AMZN-review-2023/detail_benchmark/greedy')
    parser.add_argument('--samples_dir', type=str, default='/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn-T2I/Qwen-Image-2512')
    parser.add_argument('--category', type=str, default=None, help="Process only a specific category folder (e.g. raw_meta_Appliances)")
    parser.add_argument('--product', type=str, default=None, help="Process only a specific product ASIN")
    parser.add_argument('--true_cfg_scale', type=float, default=4.0)
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--skip_existing', type=str, default="True", help="Skip existing files ('True' or 'False')")

    args = parser.parse_args()

    skip_existing = args.skip_existing.lower() == 'true'

    # Determine categories to process
    if args.category:
        categories = [args.category]
    else:
        categories = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d)) and d.startswith('raw_meta_')]
        categories.sort()

    to_generate = []
    for category in categories:
        metadata_path = os.path.join(args.data_dir, category, 'metadata.json')
        if not os.path.exists(metadata_path):
            print(f"Warning: metadata.json not found in {category}. Skipping category.")
            continue

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

                # Always ensure auxiliary files exist in output directory
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

    print(f"Found {len(to_generate)} targets to generate.")

    # Load Qwen-Image-2512 model
    print("Loading Qwen-Image-2512 model...")
    pipeline = QwenImagePipeline.from_pretrained(
        "Qwen/Qwen-Image-2512",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    pipeline.set_progress_bar_config(disable=None)

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

        try:
            # Create target folder and save auxiliary files
            asin_out_dir = os.path.dirname(out_path)
            save_auxiliary_files(ref_image_path, gt_var_src_path, var_file, target_prompt, asin_out_dir)

            # Set up inputs for Text-to-Image pipeline
            generator = torch.Generator(device="cuda").manual_seed(args.seed)
            inputs = {
                "prompt": target_prompt,
                "generator": generator,
                "true_cfg_scale": args.true_cfg_scale,
                "negative_prompt": " ",
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "num_images_per_prompt": 1,
            }
            if args.width and args.height:
                inputs["width"] = args.width
                inputs["height"] = args.height

            # Run generation
            with torch.inference_mode():
                output = pipeline(**inputs)
                result = output.images[0]

            # Save final image
            result.save(out_path)
            print(f"Saved generated image to {out_path}")
        except Exception as e:
            print(f"Error generating {category}/{asin} to {out_path}: {e}")

if __name__ == "__main__":
    main()
