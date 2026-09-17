import os
import json
import re
import time
import torch
import argparse
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

def generate_with_local_vlm(messages, model, processor, max_new_tokens=128):
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def parse_json_from_vlm(response_text: str) -> list:
    cleaned = response_text.strip()
    match = re.search(r'(\[.*\]|\{.*\})', cleaned, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        json_str = cleaned
    
    parsed = json.loads(json_str)
    # Ensure it returns a list of rubrics
    if isinstance(parsed, dict) and "rubrics" in parsed:
        return parsed["rubrics"]
    if isinstance(parsed, dict) and "evaluations" in parsed:
        return parsed["evaluations"]
    return parsed

def main():
    parser = argparse.ArgumentParser(description="Generates subject identity rubrics for the amzn dataset and caches them.")
    parser.add_argument("-m", "--model_name", type=str, default="Qwen/Qwen3-VL-32B-Instruct", help="Name of local Qwen3-VL model.")
    parser.add_argument("--data_dir", type=str, default="/root/Desktop/workspace/woosung/commercial-dreambench/assets/data/wordy", help="Root wordy data directory containing category subdirectories.")
    parser.add_argument("-c", "--category", type=str, default=None, help="Optional specific category subdirectory to filter (e.g. raw_meta_All_Beauty).")
    parser.add_argument("-a", "--asin", type=str, default=None, help="Optional specific ASIN to filter.")
    parser.add_argument("-p", "--prompt_file", type=str, default="user_prompt_generate_rubric.txt", help="Name of prompt file in prompts/ folder to use.")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="Device map specification (e.g. cuda:0, cuda:1).")
    parser.add_argument("--root_dir", type=str, default="/root/Desktop/workspace/woosung/commercial-dreambench", help="Root workspace path.")
    
    args = parser.parse_args()

    rubrics_root = os.path.join(args.root_dir, "assets", "rubrics", "amzn")
    os.makedirs(rubrics_root, exist_ok=True)

    # 1. Gather all categories
    if args.category is not None:
        categories = [args.category]
    else:
        categories = sorted([
            d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d)) and d.startswith("raw_meta_")
        ])
    
    print(f"Discovered {len(categories)} categories to process.")

    # 2. Collect pending items (skip rubrics that already exist)
    pending_items = []
    skipped_count = 0

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
            if args.asin is not None and curr_asin != args.asin:
                continue

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

    # 4. Load the local VLM model
    print(f"Loading local VLM model: {args.model_name} on {args.device}...")
    processor = AutoProcessor.from_pretrained(args.model_name)
    if "Qwen3" in args.model_name and Qwen3VLForConditionalGeneration is not None:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16, device_map=args.device
        )
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name, torch_dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True
            )
        except Exception:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                args.model_name, torch_dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True
            )
    print(f"Model loaded successfully on {args.device}!")

    # 5. Generate rubrics for pending items
    for item in pending_items:
        cat = item["cat"]
        curr_asin = item["asin"]
        ref_image_path = item["ref_image_path"]
        output_file = item["output_file"]
        output_cat_dir = item["output_cat_dir"]

        os.makedirs(output_cat_dir, exist_ok=True)

        if os.path.exists(output_file):
            print(f"Rubric already exists for {cat} / {curr_asin}. Skipping.")
            continue

        print(f"\nGenerating rubric for {cat} / {curr_asin}...")
        
        messages_rubric = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt_rubric}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": ref_image_path},
                ]
            }
        ]

        try:
            res_rubric = generate_with_local_vlm(messages_rubric, model, processor, max_new_tokens=2048)
            rubrics = parse_json_from_vlm(res_rubric)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(rubrics, f, indent=4, ensure_ascii=False)
            print(f"Successfully saved rubric to {output_file}")
        except Exception as e:
            print(f"[ERROR] Failed to generate rubric for {curr_asin}: {e}")

if __name__ == "__main__":
    # Standard generation configurations as environment variables
    os.environ["greedy"] = "false"
    os.environ["top_p"] = "0.8"
    os.environ["top_k"] = "20"
    os.environ["temperature"] = "0.7"
    os.environ["repetition_penalty"] = "1.0"
    os.environ["presence_penalty"] = "1.5"
    os.environ["out_seq_length"] = "16384"
    main()
