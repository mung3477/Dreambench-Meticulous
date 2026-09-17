import os
import sys
import json
import torch
import shutil
import argparse
from tqdm import tqdm
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image
from PIL import Image

DEFAULT_VARIANT_FOLDERS = [
    "original",
    "noised-0.875x_masked-dino-adaptive_height-ratio-1.0_timestep-10",
    "noised-0.625x_masked-dino-adaptive_height-ratio-1.0_timestep-10",
    "noised-0.375x_masked-dino-adaptive_height-ratio-1.0_timestep-10",
    "noised-0.25x_masked-dino-adaptive_height-ratio-1.0_timestep-10",
]

def save_auxiliary_files(ref_image_path, gt_var_src_path, var_file, prompt, asin_out_dir):
    os.makedirs(asin_out_dir, exist_ok=True)
    ref_out_path = os.path.join(asin_out_dir, "reference.jpg")
    if os.path.exists(ref_image_path) and not os.path.exists(ref_out_path):
        shutil.copy(ref_image_path, ref_out_path)

    var_basename = os.path.basename(var_file)
    var_stem, ext = os.path.splitext(var_basename)
    if var_stem.lower().startswith("variation_"):
        suffix = var_stem[len("variation_"):]
    else:
        suffix = var_stem

    gt_out_file_1 = f"gt_variation_{suffix}{ext}"
    gt_out_file_2 = f"GT_variation_{suffix}{ext}"
    gt_out_path_1 = os.path.join(asin_out_dir, gt_out_file_1)
    gt_out_path_2 = os.path.join(asin_out_dir, gt_out_file_2)
    if os.path.exists(gt_var_src_path) and not os.path.exists(gt_out_path_1) and not os.path.exists(gt_out_path_2):
        shutil.copy(gt_var_src_path, gt_out_path_1)

    txt_filename = f"variation_{suffix}.txt"
    txt_out_path = os.path.join(asin_out_dir, txt_filename)
    with open(txt_out_path, "w", encoding="utf-8") as f:
        f.write(prompt.strip() + "\n")

def main():
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit sampling with noised reference images")
    parser.add_argument("--data_dir", type=str, default="/root/Desktop/workspace/woosung/commercial-dreambench/assets/data/amzn")
    parser.add_argument("--ref_only_dir", type=str, default="/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn_ref-only")
    parser.add_argument("--output_base_dir", type=str, default="/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn_distorted/qwen-image-edit")
    parser.add_argument("--category", type=str, default="")
    parser.add_argument("--product", type=str, default="")
    parser.add_argument("--variant_folders", nargs="+", type=str, default=None)
    parser.add_argument("--target_variations", nargs="+", type=str, default=None)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_existing", type=str, default="True")

    args = parser.parse_args()
    skip_existing = args.skip_existing.lower() == "true"

    if args.variant_folders:
        variant_folders = []
        for vf in args.variant_folders:
            variant_folders.extend([v.strip() for v in vf.split(",") if v.strip()])
    else:
        variant_folders = DEFAULT_VARIANT_FOLDERS

    target_set = set()
    if args.target_variations:
        for tv in args.target_variations:
            if tv.endswith(".json") and os.path.exists(tv):
                with open(tv) as f:
                    target_set.update(json.load(f))
            else:
                target_set.add(tv)

    if args.category:
        categories = [args.category]
    else:
        categories = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d)) and d.startswith("raw_meta_")]
        categories.sort()

    tasks = []
    for category in categories:
        metadata_path = os.path.join(args.data_dir, category, "metadata.json")
        if not os.path.exists(metadata_path):
            continue

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        for item in metadata:
            asin = item["asin"]
            if args.product and asin != args.product:
                continue

            for var_folder in variant_folders:
                ref_image_path = os.path.join(args.ref_only_dir, var_folder, category, asin, "noised.jpg")
                if not os.path.exists(ref_image_path):
                    continue

                for variation in item["variation_files"]:
                    orig_var_file = variation["file"]
                    var_file = orig_var_file[:-4] + ".jpg"
                    var_prompt = variation["prompt"]
                    gt_var_src_path = os.path.join(args.data_dir, category, orig_var_file)
                    asin_out_dir = os.path.join(args.output_base_dir, var_folder, category, asin)
                    out_path = os.path.join(args.output_base_dir, var_folder, category, var_file)

                    task_key_1 = f"{category}/{orig_var_file}"
                    task_key_2 = f"{category}_{asin}_{os.path.splitext(os.path.basename(orig_var_file))[0]}"
                    if target_set and (task_key_1 not in target_set and task_key_2 not in target_set and orig_var_file not in target_set):
                        continue

                    save_auxiliary_files(ref_image_path, gt_var_src_path, var_file, var_prompt, asin_out_dir)

                    if skip_existing and os.path.exists(out_path):
                        continue

                    tasks.append({
                        "category": category,
                        "asin": asin,
                        "var_folder": var_folder,
                        "ref_image_path": ref_image_path,
                        "prompt": var_prompt,
                        "out_path": out_path
                    })

    if not tasks:
        print("No tasks to process for Qwen-Image-Edit.")
        return

    print(f"Total images to generate for Qwen-Image-Edit: {len(tasks)}")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Qwen-Image-Edit-2511 model on {device_str}...")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        torch_dtype=torch.bfloat16
    ).to(device_str)
    pipeline.set_progress_bar_config(disable=None)

    for task in tqdm(tasks, desc="Sampling Qwen-Image-Edit"):
        os.makedirs(os.path.dirname(task["out_path"]), exist_ok=True)
        ref_image = load_image(task["ref_image_path"]).convert("RGB")
        if args.width and args.height:
            ref_image = ref_image.resize((args.width, args.height))

        generator = torch.Generator(device=device_str).manual_seed(args.seed)
        inputs = {
            "image": [ref_image],
            "prompt": task["prompt"],
            "generator": generator,
            "true_cfg_scale": args.true_cfg_scale,
            "negative_prompt": " ",
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "num_images_per_prompt": 1,
        }

        with torch.inference_mode():
            output = pipeline(**inputs)
            result = output.images[0]

        result.save(task["out_path"])

    print("Qwen-Image-Edit Sampling complete.")

if __name__ == "__main__":
    main()
