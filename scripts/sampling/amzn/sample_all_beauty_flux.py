#!/usr/bin/env python3
import os
import sys
import json
import torch
from tqdm import tqdm
from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image

def main():
    metadata_path = "/root/Desktop/workspace/woosung/commercial-dreambench/assets/data/amzn/raw_meta_All_Beauty/prompt-updated_metadata.json"
    out_dir = "/root/.gemini/antigravity-ide/brain/e25ea296-2822-42bb-a7be-747ecedf4d3f/scratch/all_beauty_samples"
    cat_dir = "/root/Desktop/workspace/woosung/commercial-dreambench/assets/data/amzn/raw_meta_All_Beauty"

    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found.")
        return

    os.makedirs(out_dir, exist_ok=True)

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading FLUX.2-klein-9B model on {device}...")
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9B",
        torch_dtype=torch.bfloat16
    ).to(device)

    print("\nGenerating FLUX.2-Klein-9B verification samples...")
    for item in tqdm(metadata, desc="Items in All_Beauty"):
        asin = item['asin']
        ref_file = os.path.join(cat_dir, item['reference_file'])
        if not os.path.exists(ref_file):
            print(f"Warning: Ref image {ref_file} missing. Skipping {asin}.")
            continue

        ref_img = load_image(ref_file).convert("RGB").resize((1024, 1024))

        asin_out_dir = os.path.join(out_dir, asin)
        os.makedirs(asin_out_dir, exist_ok=True)

        for var_idx, var in enumerate(item.get('variation_files', []), start=1):
            prompt = var['prompt']
            out_filename = f"variation_{var_idx}_sample_seed-602.png"
            out_path = os.path.join(asin_out_dir, out_filename)

            # Save prompt txt for reference
            txt_path = os.path.join(asin_out_dir, f"variation_{var_idx}_prompt.txt")
            with open(txt_path, 'w', encoding='utf-8') as txt_f:
                txt_f.write(prompt + "\n")

            generator = torch.Generator(device=device.type).manual_seed(602)
            try:
                res = pipe(prompt=prompt, image=ref_img, guidance_scale=1.0, num_inference_steps=4, generator=generator).images[0]
                res.save(out_path)
                print(f"Saved: {out_path}")
            except Exception as e:
                print(f"Error sampling for {asin} var {var_idx}: {e}")

    print("\nAll verification sampling completed! Output directory:", out_dir)

if __name__ == '__main__':
    main()
