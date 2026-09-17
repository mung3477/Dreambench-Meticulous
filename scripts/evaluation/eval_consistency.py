#!/usr/bin/env python3
import os
import sys
import re
import time
import json
import argparse
import math
from pathlib import Path
from tqdm import tqdm

# Set VLM generation environment variables
os.environ["greedy"] = "false"
os.environ["top_p"] = "0.8"
os.environ["top_k"] = "20"
os.environ["temperature"] = "0.7"
os.environ["repetition_penalty"] = "1.0"
os.environ["presence_penalty"] = "1.5"
os.environ["out_seq_length"] = "16384"

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageOps

def local_load_image(img_path):
    image = Image.open(img_path)
    image = ImageOps.exif_transpose(image)
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, "white")
        image = Image.alpha_composite(background, image).convert("RGB")
    return image.convert("RGB")

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

def load_local_vlm(model_name="Qwen/Qwen3-VL-32B-Instruct"):
    print(f"Loading local VLM model: {model_name} on cuda:0...")
    if not torch.cuda.is_available():
        print("WARNING: TORCH DOES NOT DETECT ANY GPUS! torch.cuda.is_available() is False.")

    processor = AutoProcessor.from_pretrained(model_name)

    if "Qwen3" in model_name and Qwen3VLForConditionalGeneration is not None:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="cuda:0"
        )
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
            )
        except Exception:
            try:
                from transformers import AutoModelForImageTextToText
                model = AutoModelForImageTextToText.from_pretrained(
                    model_name, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
                )
            except Exception:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(
                    model_name, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
                )

    print("Model loaded successfully on cuda:0.")
    return processor, model

def generate_with_local_vlm(messages, model, processor, max_new_tokens=128):
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def resize_and_save(img_path, out_dir, filename, size=512):
    image = local_load_image(img_path).resize((size, size))
    tmp_path = os.path.abspath(os.path.join(out_dir, filename))
    image.save(tmp_path)
    return tmp_path

def evaluate_pair(src_path, tgt_path, model, processor, prompts_dir, out_dir="/tmp", max_retry=3):
    os.makedirs(out_dir, exist_ok=True)
    tmp_src_path = resize_and_save(src_path, out_dir, "tmp_src.jpg", size=512)
    tmp_tgt_path = resize_and_save(tgt_path, out_dir, "tmp_tgt.jpg", size=512)

    with open(os.path.join(prompts_dir, "user_prompt_subject_full.txt"), "r") as f:
        user_prompt = f.read().strip()
    with open(os.path.join(prompts_dir, "gpt_prompt_subject_full.txt"), "r") as f:
        gpt_prompt = f.read().strip()

    messages = [
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": gpt_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": tmp_src_path},
                {"type": "image", "image": tmp_tgt_path},
            ],
        },
    ]

    cur_retry = 0
    score = None
    content = ""

    while True:
        try:
            content = generate_with_local_vlm(messages, model, processor, max_new_tokens=256)
            pattern = r"(score|Score):\s*[a-zA-Z]*\s*(\d+)"
            matches = re.findall(pattern, content)
            scores = [int(s) for _, s in matches]

            assert len(scores) == 1, f"Expected exactly one score pattern match, got {scores}"
            score = scores[0]
            break
        except Exception as e:
            cur_retry += 1
            print(f"Retry {cur_retry}/{max_retry} failed due to: {e}")
            if cur_retry >= max_retry:
                print("Reached maximum retry limit. Could not parse score.")
                break
            time.sleep(0.1)

    # Clean up resized temp images
    for p in [tmp_src_path, tmp_tgt_path]:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    return score, content

def discover_subjects(samples_dir, model_filter=None, subject_filter=None, category_filter=None):
    if model_filter:
        models = [model_filter]
    else:
        models = [d for d in os.listdir(samples_dir) if os.path.isdir(os.path.join(samples_dir, d)) and d not in ['deprecated']]

    subjects = []
    for m in models:
        m_dir = os.path.join(samples_dir, m)
        if not os.path.exists(m_dir):
            continue

        for name in sorted(os.listdir(m_dir)):
            p1 = os.path.join(m_dir, name)
            if not os.path.isdir(p1) or name == 'deprecated':
                continue

            subdirs = [d for d in os.listdir(p1) if os.path.isdir(os.path.join(p1, d)) and d != '.ipynb_checkpoints']
            if subdirs:
                # Category-nested structure (e.g. category/asin)
                for subname in sorted(subdirs):
                    p2 = os.path.join(p1, subname)
                    if category_filter and name != category_filter:
                        continue
                    if subject_filter and subname != subject_filter:
                        continue
                    subjects.append({
                        'model': m,
                        'category': name,
                        'subject': subname,
                        'full_path': p2
                    })
            else:
                # Flat structure (e.g. subject)
                if category_filter:
                    continue
                if subject_filter and name != subject_filter:
                    continue
                subjects.append({
                    'model': m,
                    'category': None,
                    'subject': name,
                    'full_path': p1
                })
    return subjects

def get_reference_path(subj_dir, category=None, subject=None, data_dir=None, remove_bg=False):
    if remove_bg:
        for name in ['reference_bg_removed.jpg', 'reference_bg_removed.png', 'ref_bg_removed.jpg', 'ref_bg_removed.png']:
            path = os.path.join(subj_dir, name)
            if os.path.exists(path):
                return path

    for name in ['reference.jpg', 'reference.png', 'ref.jpg', 'ref.png']:
        path = os.path.join(subj_dir, name)
        if os.path.exists(path):
            return path

    if data_dir:
        dirs_to_try = []
        if category:
            dirs_to_try.append(os.path.join(data_dir, category, subject))
        dirs_to_try.append(os.path.join(data_dir, subject))

        for d in dirs_to_try:
            if remove_bg:
                for name in ['reference_bg_removed.jpg', 'reference_bg_removed.png', 'ref_bg_removed.jpg', 'ref_bg_removed.png']:
                    path = os.path.join(d, name)
                    if os.path.exists(path):
                        return path
            for name in ['reference.jpg', 'reference.png', 'ref.jpg', 'ref.png']:
                path = os.path.join(d, name)
                if os.path.exists(path):
                    return path
    return None

def get_generated_images(subj_dir):
    images = []
    for f in os.listdir(subj_dir):
        path = os.path.join(subj_dir, f)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(f)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.webp']:
            continue
        fname = f.lower()
        if 'ref' in fname or 'gt' in fname:
            continue
        images.append(f)
    images.sort()
    return images

def calculate_std(scores):
    if len(scores) <= 1:
        return 0.0
    mean = sum(scores) / len(scores)
    variance = sum((x - mean) ** 2 for x in scores) / (len(scores) - 1)
    return math.sqrt(variance)

def main():
    parser = argparse.ArgumentParser(description="DreamBench++ Score Consistency Evaluation")
    parser.add_argument('--samples_dir', type=str, required=True, help="Directory containing target samples")
    parser.add_argument('--data_dir', type=str, default=None, help="Directory containing original reference images")
    parser.add_argument('--category', type=str, default=None, help="Process only a specific category folder")
    parser.add_argument('--subject', type=str, default=None, help="Process only a specific subject")
    parser.add_argument('--model', type=str, default=None, help="Process only a specific model name")
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-VL-32B-Instruct', help="Pretrained VLM name/path")
    parser.add_argument('--prompts_dir', type=str, default='/root/Desktop/workspace/woosung/dreambench_plus/dreambench_plus/prompts', help="Directory containing user and gpt prompt template files")
    parser.add_argument('--out_dir', type=str, default='/tmp', help="Temp output folder for resizing")
    parser.add_argument('--max_retry', type=int, default=3, help="Max retry limit for VLM parsing")
    parser.add_argument('--remove_bg', type=str, default="False", help="Evaluate against background-removed reference image ('True' or 'False')")
    parser.add_argument('--num_runs', type=int, default=3, help="Number of evaluation runs to perform for each item")
    args = parser.parse_args()
    remove_bg = args.remove_bg.lower() == 'true'
    num_runs = args.num_runs

    # Load local VLM weights only once
    processor, model = load_local_vlm(args.model_name)

    # Determine models to evaluate
    if args.model:
        models = [args.model]
    else:
        models = [d for d in os.listdir(args.samples_dir) if os.path.isdir(os.path.join(args.samples_dir, d)) and d not in ['deprecated']]

    models.sort()

    for m in models:
        model_dir = os.path.join(args.samples_dir, m)
        if not os.path.exists(model_dir):
            continue

        print(f"\nEvaluating DreamBench++ Consistency for model: {m} ({num_runs} runs)")

        model_subjects = discover_subjects(args.samples_dir, model_filter=m, subject_filter=args.subject, category_filter=args.category)
        if not model_subjects:
            print(f"No subjects found for model {m}.")
            continue

        consistency_file = os.path.join(model_dir, 'dreambench_plus_consistency.json')
        runs_file = os.path.join(model_dir, 'dreambench_plus_consistency_runs.json')

        # Load existing run logs if they exist
        raw_runs = {}
        if os.path.exists(runs_file):
            try:
                with open(runs_file, 'r') as f:
                    raw_runs = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load existing runs file {runs_file}: {e}")

        for s in tqdm(model_subjects, desc="Subjects"):
            subj_dir = s['full_path']
            ref_path = get_reference_path(subj_dir, category=s['category'], subject=s['subject'], data_dir=args.data_dir, remove_bg=remove_bg)
            if not ref_path:
                continue

            gen_images = get_generated_images(subj_dir)
            if not gen_images:
                continue

            print(f"Evaluating subject '{s['category']}/{s['subject']}' or '{s['subject']}' ({len(gen_images)} images)...")
            for img_name in gen_images:
                img_path = os.path.join(subj_dir, img_name)
                img_base = os.path.splitext(img_name)[0]

                if s['category']:
                    key = f"{s['category']}_{s['subject']}_{img_base}"
                else:
                    key = f"{s['subject']}_{img_base}"

                # We will collect N scores for this key
                item_scores = []
                for run_idx in range(num_runs):
                    print(f"  Evaluating {key} (Run {run_idx + 1}/{num_runs})...")
                    score, _ = evaluate_pair(
                        src_path=ref_path,
                        tgt_path=img_path,
                        model=model,
                        processor=processor,
                        prompts_dir=args.prompts_dir,
                        out_dir=args.out_dir,
                        max_retry=args.max_retry
                    )
                    if score is not None:
                        item_scores.append(score)
                        print(f"    Run {run_idx + 1} Score: {score}")
                    else:
                        print(f"    Run {run_idx + 1} Score: Failed to parse.")

                if len(item_scores) > 0:
                    raw_runs[key] = item_scores

        # Now compute per-run means and per-item consistency statistics
        # Find all keys that have valid scores
        valid_keys = [k for k, v in raw_runs.items() if len(v) > 0]
        if not valid_keys:
            print(f"No DreamBench++ scores computed for model '{m}'.")
            continue

        # We want to support case where some keys might have different number of runs, 
        # but ideally we align to num_runs. Let's find the max number of runs actually completed.
        actual_runs = max(len(raw_runs[k]) for k in valid_keys)
        
        # Calculate per-run mean score
        per_run_means = []
        for run_idx in range(actual_runs):
            run_scores = []
            for k in valid_keys:
                if run_idx < len(raw_runs[k]):
                    run_scores.append(raw_runs[k][run_idx])
            if run_scores:
                per_run_means.append(sum(run_scores) / len(run_scores))
            else:
                per_run_means.append(0.0)

        # Calculate per-item statistics
        item_stats = {}
        item_stds = []
        item_ranges = []
        perfect_agreements = []
        modal_agreements = []

        for k in valid_keys:
            scores = raw_runs[k]
            mean_val = sum(scores) / len(scores)
            std_val = calculate_std(scores)
            range_val = max(scores) - min(scores)
            
            # Modal agreement
            score_counts = {}
            for s_val in scores:
                score_counts[s_val] = score_counts.get(s_val, 0) + 1
            max_count = max(score_counts.values())
            modal_agreement_val = max_count / len(scores)
            
            perfect_agreement_val = 1.0 if len(set(scores)) == 1 else 0.0

            item_stats[k] = {
                "scores": scores,
                "mean": mean_val,
                "std": std_val,
                "range": range_val,
                "perfect_agreement": perfect_agreement_val == 1.0,
                "modal_agreement": modal_agreement_val
            }

            item_stds.append(std_val)
            item_ranges.append(range_val)
            perfect_agreements.append(perfect_agreement_val)
            modal_agreements.append(modal_agreement_val)

        avg_std = sum(item_stds) / len(item_stds) if item_stds else 0.0
        avg_range = sum(item_ranges) / len(item_ranges) if item_ranges else 0.0
        perfect_agreement_rate = sum(perfect_agreements) / len(perfect_agreements) if perfect_agreements else 0.0
        avg_modal_agreement_rate = sum(modal_agreements) / len(modal_agreements) if modal_agreements else 0.0

        overall_stats = {
            "num_runs": num_runs,
            "actual_max_runs": actual_runs,
            "per_run_means": per_run_means,
            "avg_std": avg_std,
            "avg_range": avg_range,
            "perfect_agreement_rate": perfect_agreement_rate,
            "avg_modal_agreement_rate": avg_modal_agreement_rate
        }

        output_data = {
            "overall_statistics": overall_stats,
            "items": item_stats
        }

        # Save files
        try:
            with open(runs_file, 'w') as f:
                json.dump(raw_runs, f, indent=4)
            print(f"Saved raw run scores to {runs_file}")
            
            with open(consistency_file, 'w') as f:
                json.dump(output_data, f, indent=4)
            print(f"Saved aggregated consistency results to {consistency_file}")
        except Exception as e:
            print(f"Error saving results: {e}")

        # Print summary table
        print("\n" + "="*50)
        print(f"CONSISTENCY EVALUATION SUMMARY FOR MODEL: {m}")
        print("="*50)
        print(f"Number of items evaluated: {len(valid_keys)}")
        print(f"Number of scheduled runs: {num_runs}")
        print(f"Per-Run Mean Scores: " + ", ".join([f"Run {i+1}: {val:.4f}" for i, val in enumerate(per_run_means)]))
        print(f"Average Standard Deviation: {avg_std:.4f}")
        print(f"Average Score Range (Max - Min): {avg_range:.4f}")
        print(f"Perfect Agreement Rate (All runs same): {perfect_agreement_rate*100:.2f}%")
        print(f"Average Modal Agreement Rate: {avg_modal_agreement_rate*100:.2f}%")
        print("="*50 + "\n")

if __name__ == "__main__":
    main()
