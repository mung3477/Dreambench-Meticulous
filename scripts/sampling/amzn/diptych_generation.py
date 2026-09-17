import os
import sys
import json
import torch
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import re

# Append DiptychPrompting directory to sys.path to resolve imports
sys.path.append('/root/Desktop/workspace/woosung/DiptychPrompting')

from controlnet_flux import FluxControlNetModel
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
from diptych_prompting_inference import CustomFluxAttnProcessor2_0, grounded_segmentation
from diffusers.utils import load_image
from transformers import AutoProcessor, pipeline, AutoModelForMaskGeneration

def main():
    parser = argparse.ArgumentParser(description="Diptych generation benchmark script for AMZN dataset")
    parser.add_argument('--data_dir', type=str, default='/root/Desktop/workspace/woosung/AMZN-review-2023/detail_benchmark/greedy')
    parser.add_argument('--samples_dir', type=str, default='/root/Desktop/workspace/woosung/commercial-dreambench/samples/diptych-amzn')
    parser.add_argument('--category', type=str, default=None, help="Process only a specific category folder (e.g. raw_meta_Appliances)")
    parser.add_argument('--product', type=str, default=None, help="Process only a specific product ASIN")
    parser.add_argument('--attn_enforce', type=float, default=1.3)
    parser.add_argument('--ctrl_scale', type=float, default=0.95)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--height', type=int, default=768)
    parser.add_argument('--pixel_offset', type=int, default=8)
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
            if args.product and asin != args.product:
                continue

            ref_file = item['reference_file']
            ref_image_path = os.path.join(args.data_dir, category, ref_file)
            categorical_name = item.get('categorical_name', '')
            if not categorical_name:
                categorical_name = 'product'

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
                    'subject_name': categorical_name,
                    'out_path': out_path,
                    'prompt': var_prompt
                })

    if not to_generate:
        print("All targets already generated or no targets found. Exiting.")
        return

    print(f"Found {len(to_generate)} targets to generate.")

    # Load models
    print("Loading models...")
    controlnet = FluxControlNetModel.from_pretrained(
        "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta",
        torch_dtype=torch.bfloat16
    )
    pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        controlnet=controlnet,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.transformer.to(torch.bfloat16)
    pipe.controlnet.to(torch.bfloat16)
    base_attn_procs = pipe.transformer.attn_processors.copy()

    detector_id = "IDEA-Research/grounding-dino-tiny"
    segmenter_id = "facebook/sam-vit-base"

    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).cuda()
    segment_processor = AutoProcessor.from_pretrained(segmenter_id)
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=torch.device("cuda"))

    def segment_image(image, object_name):
        try:
            image_array, detections = grounded_segmentation(
                object_detector,
                segmentator,
                segment_processor,
                image=image,
                labels=[object_name],
                threshold=0.3,
                polygon_refinement=True,
            )
            if not detections or len(detections) == 0:
                print(f"Warning: No detections for {object_name}, using original image for reference.")
                return image

            mask = detections[0].mask
            if mask is None:
                print(f"Warning: No mask for {object_name}, using original image for reference.")
                return image

            segment_result = image_array * np.expand_dims(mask / 255, axis=-1) + np.ones_like(image_array) * (
                    1 - np.expand_dims(mask / 255, axis=-1)) * 255
            segmented_image = Image.fromarray(segment_result.astype(np.uint8))
            return segmented_image
        except Exception as e:
            print(f"Error segmenting {object_name}: {e}. Falling back to original image.")
            return image

    def make_diptych(image):
        ref_image = np.array(image)
        ref_image = np.concatenate([ref_image, np.zeros_like(ref_image)], axis=1)
        ref_image = Image.fromarray(ref_image)
        return ref_image

    # Dimension calculation
    width = args.width + args.pixel_offset * 2
    height = args.height + args.pixel_offset * 2
    size = (width * 2, height)

    # Setup custom attention processors
    new_attn_procs = base_attn_procs.copy()
    for k in new_attn_procs.keys():
        new_attn_procs[k] = CustomFluxAttnProcessor2_0(
            height=height // 16,
            width=width // 16 * 2,
            attn_enforce=args.attn_enforce
        )
    pipe.transformer.set_attn_processor(new_attn_procs)

    for task in tqdm(to_generate, desc="Generating images"):
        category = task['category']
        asin = task['asin']
        ref_image_path = task['ref_image_path']
        subject_name = task['subject_name']
        out_path = task['out_path']
        target_prompt = task['prompt']

        print(f"\nProcessing {category} / {asin} -> {os.path.basename(out_path)} (Subject: {subject_name})")
        print(f"Prompt: {target_prompt}")

        if not os.path.exists(ref_image_path):
            print(f"Error: Reference image {ref_image_path} does not exist. Skipping.")
            continue

        try:
            # Create target folder if it doesn't exist
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # Load and resize reference image
            reference_image = load_image(ref_image_path).resize((width, height)).convert("RGB")

            # Segment the image using SAM and DINO
            segmented_image = segment_image(reference_image, subject_name)

            # Create diptych mask (left 0s, right 255s)
            mask_image = np.concatenate([np.zeros((height, width, 3)), np.ones((height, width, 3)) * 255], axis=1)
            mask_image = Image.fromarray(mask_image.astype(np.uint8))

            # Create diptych control image
            diptych_image_prompt = make_diptych(segmented_image)

            # Construct final prompt
            base_prompt = f"a photo of {subject_name}"
            diptych_text_prompt = f"A diptych with two side-by-side images of same {subject_name}. On the left, {base_prompt}. On the right, replicate this {subject_name} exactly but as {target_prompt}"

            # Run generation
            generator = torch.Generator(device="cuda").manual_seed(args.seed)
            result = pipe(
                prompt=diptych_text_prompt,
                height=size[1],
                width=size[0],
                control_image=diptych_image_prompt,
                control_mask=mask_image,
                num_inference_steps=30,
                generator=generator,
                controlnet_conditioning_scale=args.ctrl_scale,
                guidance_scale=3.5,
                negative_prompt="",
                true_guidance_scale=3.5
            ).images[0]

            # Crop the generated image on the right
            result = result.crop((width, 0, width * 2, height))
            result = result.crop((args.pixel_offset, args.pixel_offset, width - args.pixel_offset, height - args.pixel_offset))

            # Save final image
            result.save(out_path)
            print(f"Saved generated image to {out_path}")
        except Exception as e:
            print(f"Error generating {category}/{asin} to {out_path}: {e}")

if __name__ == "__main__":
    main()
