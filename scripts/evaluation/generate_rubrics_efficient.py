#!/usr/bin/env python3
import os
import sys
import re
import time
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# Set VLM generation environment variables
os.environ["greedy"] = "true"

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageOps

# Try importing vLLM engine
HAS_VLLM = False
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None


def local_load_image(img_path):
    image = Image.open(img_path)
    image = ImageOps.exif_transpose(image)
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, "white")
        image = Image.alpha_composite(background, image).convert("RGB")
    return image.convert("RGB")


def parse_json_from_vlm(response_text: str):
    cleaned = response_text.strip()
    match = re.search(r'(\[.*\]|\{.*\})', cleaned, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        json_str = cleaned

    parsed = json.loads(json_str)
    if isinstance(parsed, dict) and "rubrics" in parsed:
        return parsed["rubrics"]
    if isinstance(parsed, dict) and "evaluations" in parsed:
        return parsed["evaluations"]
    return parsed


def batch_generate_hf(batch_items, model, processor, max_new_tokens=2048):
    """
    Batched inference using standard HuggingFace PyTorch pipeline.
    """
    batch_messages = [item["messages"] for item in batch_items]
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in batch_messages
    ]

    images_batch = []
    for msg in batch_messages:
        imgs = []
        for turn in msg:
            if isinstance(turn.get("content"), list):
                for elem in turn["content"]:
                    if elem.get("type") == "image":
                        img_val = elem.get("image")
                        if isinstance(img_val, str):
                            imgs.append(local_load_image(img_val))
                        elif isinstance(img_val, Image.Image):
                            imgs.append(img_val)
        images_batch.append(imgs if imgs else None)

    inputs = processor(text=texts, images=images_batch, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    output_texts = []
    input_ids = inputs["input_ids"]
    for in_ids, out_ids in zip(input_ids, generated_ids):
        trimmed = out_ids[len(in_ids):]
        decoded = processor.decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        output_texts.append(decoded)

    return output_texts


def batch_generate_vllm(batch_items, vllm_engine, processor, max_new_tokens=2048):
    """
    High-throughput batched inference using vLLM engine.
    """
    vllm_inputs = []
    for item in batch_items:
        messages = item["messages"]
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        images = []
        for turn in messages:
            if isinstance(turn.get("content"), list):
                for elem in turn["content"]:
                    if elem.get("type") == "image":
                        img_val = elem.get("image")
                        if isinstance(img_val, str):
                            images.append(local_load_image(img_val))
                        elif isinstance(img_val, Image.Image):
                            images.append(img_val)

        vllm_input = {
            "prompt": prompt_text,
            "multi_modal_data": {
                "image": images
            }
        }
        vllm_inputs.append(vllm_input)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_tokens=max_new_tokens
    )

    outputs = vllm_engine.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=True)
    results = [out.outputs[0].text for out in outputs]
    return results


def main():
    parser = argparse.ArgumentParser(description="Generates subject identity rubrics efficiently using vLLM batched inference.")
    parser.add_argument("-m", "--model_name", type=str, default="Qwen/Qwen3-VL-32B-Instruct", help="Name of local VLM model.")
    parser.add_argument("--data_dir", type=str, default="/root/Desktop/workspace/woosung/commercial-dreambench/assets/data/amzn", help="Root wordy data directory containing category subdirectories.")
    parser.add_argument("-c", "--category", "--categories", nargs="+", default=None, help="Optional specific category/categories to filter.")
    parser.add_argument("-a", "--asin", "--asins", nargs="+", default=None, help="Optional specific ASIN(s) to filter (space-separated or comma-separated).")
    parser.add_argument("-p", "--prompt_file", type=str, default="user_prompt_generate_rubric_comprehensive_bbox-titles-coarseNfine.txt", help="Name of prompt file in prompts/ folder.")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="Device specification.")
    parser.add_argument("--root_dir", type=str, default="/root/Desktop/workspace/woosung/commercial-dreambench", help="Root workspace path.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for parallel rubric generation.")
    parser.add_argument("--use_vllm", action="store_true", help="Force use of vLLM backend engine for inference.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization factor for vLLM.")
    parser.add_argument("--limit_mm_per_prompt", type=int, default=1, help="Max image count per prompt for vLLM limit.")

    args = parser.parse_args()

    rubrics_root = os.path.join(args.root_dir, "assets", "rubrics", "amzn")
    os.makedirs(rubrics_root, exist_ok=True)

    # 1. Gather all categories
    if args.category is not None:
        target_cats = set()
        for entry in args.category:
            for item in entry.split(","):
                val = item.strip()
                if val:
                    target_cats.add(val)
        categories = sorted(list(target_cats))
    else:
        categories = sorted([
            d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d)) and d.startswith("raw_meta_")
        ])

    print(f"Discovered {len(categories)} categories to process.")

    # Parse target ASINs if provided
    target_asins = set()
    if args.asin is not None:
        for entry in args.asin:
            for item in entry.split(","):
                val = item.strip()
                if val:
                    target_asins.add(val)
        print(f"Filtering for {len(target_asins)} specific ASIN(s).")

    # 2. Collect pending items (skip rubrics that already exist)
    pending_items = []
    skipped_count = 0
    matched_asins = set()

    for cat in categories:
        meta_json_path = os.path.join(args.data_dir, cat, "metadata.json")
        if not os.path.exists(meta_json_path):
            print(f"[WARNING] Skipping category '{cat}' as metadata.json was not found.")
            continue

        with open(meta_json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        output_cat_dir = os.path.join(rubrics_root, cat)

        for item in metadata:
            curr_asin = item.get("asin")
            if target_asins and curr_asin not in target_asins:
                continue

            matched_asins.add(curr_asin)

            ref_file = item.get("reference_file", f"{curr_asin}/reference.jpg")
            ref_image_path = os.path.abspath(os.path.join(args.data_dir, cat, ref_file))
            if not os.path.exists(ref_image_path):
                print(f"[WARNING] Reference image not found at {ref_image_path}. Skipping.")
                continue

            output_file = os.path.join(output_cat_dir, f"{curr_asin}.json")
            if os.path.exists(output_file):
                skipped_count += 1
                continue

            pending_items.append({
                "cat": cat,
                "asin": curr_asin,
                "ref_image_path": ref_image_path,
                "output_file": output_file,
                "output_cat_dir": output_cat_dir
            })

    if target_asins:
        unfound_asins = target_asins - matched_asins
        if unfound_asins:
            print(f"[WARNING] {len(unfound_asins)} target ASIN(s) were not found in metadata: {sorted(unfound_asins)}")

    print(f"Found {skipped_count} existing rubrics (skipped).")
    print(f"Total pending rubrics to generate: {len(pending_items)}")

    if not pending_items:
        print("All target rubrics already exist. Nothing to generate.")
        return

    # 3. Load rubric prompt template
    prompt_template_path = os.path.join(args.root_dir, "prompts", args.prompt_file)
    print(f"Loading rubric generation prompt from: {prompt_template_path}")
    with open(prompt_template_path, "r", encoding="utf-8") as f:
        user_prompt_rubric = f.read().strip()

    # 4. Decide Backend Engine (vLLM vs HuggingFace PyTorch Batching)
    use_vllm_engine = HAS_VLLM and (args.use_vllm or HAS_VLLM)
    print(f"Loading local VLM processor for model: {args.model_name}...")
    processor = AutoProcessor.from_pretrained(args.model_name)

    vllm_engine = None
    hf_model_obj = None

    if use_vllm_engine:
        print(f"Initializing vLLM Engine backend for high-throughput batching...")
        try:
            vllm_engine = LLM(
                model=args.model_name,
                trust_remote_code=True,
                max_model_len=8192,
                limit_mm_per_prompt={"image": args.limit_mm_per_prompt},
                gpu_memory_utilization=args.gpu_memory_utilization
            )
            print("vLLM Engine initialized successfully!")
        except Exception as e:
            print(f"[WARNING] Failed to initialize vLLM engine ({e}). Falling back to PyTorch batched inference.")
            use_vllm_engine = False

    if not use_vllm_engine:
        print(f"Loading PyTorch model on {args.device} for batched inference (batch_size={args.batch_size})...")
        if "Qwen3" in args.model_name and Qwen3VLForConditionalGeneration is not None:
            hf_model_obj = Qwen3VLForConditionalGeneration.from_pretrained(
                args.model_name, torch_dtype=torch.bfloat16, device_map=args.device
            )
        else:
            try:
                hf_model_obj = AutoModelForCausalLM.from_pretrained(
                    args.model_name, torch_dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True
                )
            except Exception:
                from transformers import AutoModelImageTextToText
                hf_model_obj = AutoModelImageTextToText.from_pretrained(
                    args.model_name, torch_dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True
                )
        print("PyTorch model loaded successfully!")

    # 5. Build prompt messages for pending items
    for item in pending_items:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt_rubric}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item["ref_image_path"]},
                ]
            }
        ]
        item["messages"] = messages

    # 6. Execute batched rubric generation
    print(f"Running batched rubric generation for {len(pending_items)} pending ASINs...")

    if use_vllm_engine:
        raw_outputs = batch_generate_vllm(pending_items, vllm_engine, processor, max_new_tokens=2048)
    else:
        raw_outputs = []
        for i in tqdm(range(0, len(pending_items), args.batch_size), desc="PyTorch Batched Rubrics"):
            chunk = pending_items[i : i + args.batch_size]
            chunk_raw = batch_generate_hf(chunk, hf_model_obj, processor, max_new_tokens=2048)
            raw_outputs.extend(chunk_raw)

    # 7. Parse and save generated rubrics
    saved_count = 0
    for item, raw_res in zip(pending_items, raw_outputs):
        cat = item["cat"]
        curr_asin = item["asin"]
        output_file = item["output_file"]
        output_cat_dir = item["output_cat_dir"]

        os.makedirs(output_cat_dir, exist_ok=True)
        try:
            rubrics = parse_json_from_vlm(raw_res)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(rubrics, f, indent=4, ensure_ascii=False)
            saved_count += 1
            print(f"Successfully generated & saved rubric for {cat} / {curr_asin} -> {output_file}")
        except Exception as e:
            print(f"[ERROR] Failed to parse/save rubric for {curr_asin}: {e}")

    print(f"\nCompleted! Saved {saved_count} / {len(pending_items)} pending rubrics.")


if __name__ == "__main__":
    main()
