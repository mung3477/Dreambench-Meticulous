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
        if 'ref' in fname or 'gt' in fname or '_mask' in fname or '_noised' in fname or '_error' in fname:
            continue
        images.append(f)
    images.sort()
    return images

def main():
    parser = argparse.ArgumentParser(description="Dreambench++ GPU-efficient batch DreamBench++ evaluation")
    parser.add_argument('--samples_dir', type=str, required=True, help="Directory containing target samples")
    parser.add_argument('--rating_dir', type=str, default=None, help="Directory to save evaluation ratings")
    parser.add_argument('--data_dir', type=str, default=None, help="Directory containing original reference images")
    parser.add_argument('--category', type=str, default=None, help="Process only a specific category folder")
    parser.add_argument('--subject', type=str, default=None, help="Process only a specific subject")
    parser.add_argument('--model', type=str, default=None, help="Process only a specific model name")
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-VL-32B-Instruct', help="Pretrained VLM name/path")
    parser.add_argument('--prompts_dir', type=str, default='/root/Desktop/workspace/woosung/dreambench_plus/dreambench_plus/prompts', help="Directory containing user and gpt prompt template files")
    parser.add_argument('--out_dir', type=str, default='/tmp', help="Temp output folder for resizing")
    parser.add_argument('--max_retry', type=int, default=3, help="Max retry limit for VLM parsing")
    parser.add_argument('--remove_bg', type=str, default="False", help="Evaluate against background-removed reference image ('True' or 'False')")
    args = parser.parse_args()
    remove_bg = args.remove_bg.lower() == 'true'

    if args.rating_dir:
        rating_dir = args.rating_dir
    elif '/samples/' in args.samples_dir:
        rating_dir = args.samples_dir.replace('/samples/', '/rating/')
    else:
        rating_dir = os.path.join(os.path.dirname(args.samples_dir.rstrip('/')), 'rating', os.path.basename(args.samples_dir.rstrip('/')))

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

        print(f"\nEvaluating DreamBench++ for model subfolder: {m}")

        model_subjects = discover_subjects(args.samples_dir, model_filter=m, subject_filter=args.subject, category_filter=args.category)
        if not model_subjects:
            print(f"No subjects found for model {m}.")
            continue

        out_dir = os.path.join(rating_dir, m)
        os.makedirs(out_dir, exist_ok=True)
        results_file = os.path.join(out_dir, 'dreambench_plus_results.json')
        legacy_results_file = os.path.join(model_dir, 'dreambench_plus_results.json')

        # Load existing results if they exist
        results = {}
        target_file = results_file if os.path.exists(results_file) else legacy_results_file
        if os.path.exists(target_file):
            try:
                with open(target_file, 'r') as f:
                    results = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load existing results file {target_file}: {e}")

        # Remove overall_mean if it exists in loaded dict to avoid polluting score lists
        results.pop('overall_mean', None)

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

                if key in results:
                    print(f"  {key}: Already evaluated, skipping.")
                    continue

                score, raw_response = evaluate_pair(
                    src_path=ref_path,
                    tgt_path=img_path,
                    model=model,
                    processor=processor,
                    prompts_dir=args.prompts_dir,
                    out_dir=args.out_dir,
                    max_retry=args.max_retry
                )

                if score is not None:
                    results[key] = score
                    print(f"  {key}: {score}")
                    # Save incrementally
                    try:
                        with open(results_file, 'w') as f:
                            json.dump(results, f, indent=4)
                    except Exception as e:
                        print(f"Error saving incremental results: {e}")
                else:
                    print(f"  {key}: Failed to parse score.")

        # Compute overall mean
        scores = [v for k, v in results.items() if k != 'overall_mean']
        if scores:
            overall_mean = sum(scores) / len(scores)
            results['overall_mean'] = overall_mean
            print(f"Model '{m}' overall mean DreamBench++ score: {overall_mean:.4f}")

            # Save results
            try:
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=4)
                print(f"Saved DreamBench++ results to {results_file}")
            except Exception as e:
                print(f"Error saving results file: {e}")
        else:
            print(f"No DreamBench++ scores computed for model '{m}'.")

if __name__ == "__main__":
    main()
