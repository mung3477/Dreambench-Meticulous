from datetime import datetime, timezone, timedelta

def log_eval_summary(metric_name, model_name, num_images, overall_mean):
    kst = timezone(timedelta(hours=9))
    now_kst = datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S KST')
    log_dir = '/root/Desktop/workspace/woosung/commercial-dreambench/scripts/evaluation/log'
    os.makedirs(log_dir, exist_ok=True)
    
    filename_map = {
        'CLIP': 'clip_eval.log',
        'DINO': 'dino_eval.log',
        'DreamBench++': 'dreambench_plus_eval.log',
        'VIEScore': 'viescore_eval.log'
    }
    filename = filename_map.get(metric_name, f"{metric_name.lower()}_eval.log")
    log_path = os.path.join(log_dir, filename)
    
    log_line = f"[{now_kst}] Metric: {metric_name:<12} | Model: {model_name} | Images: {num_images} | Avg Score: {overall_mean:.4f}\n"
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_line)
        combined_log_path = os.path.join(log_dir, 'evaluation_summary.log')
        with open(combined_log_path, 'a', encoding='utf-8') as f:
            f.write(log_line)
        print(f"Logged evaluation summary to {log_path}")
    except Exception as e:
        print(f"Failed to write log entry: {e}")

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


def local_load_image(img_path, size=512):
    image = Image.open(img_path)
    image = ImageOps.exif_transpose(image)
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, "white")
        image = Image.alpha_composite(background, image).convert("RGB")
    image = image.convert("RGB")
    if size is not None:
        image = image.resize((size, size))
    return image


def parse_viescore(content):
    score = None
    # 1. Try parsing JSON out of content
    try:
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1:
            data = json.loads(content[start:end+1])
            if 'score' in data:
                val = data['score']
                if isinstance(val, list) and len(val) > 0:
                    score = int(val[0])
                elif isinstance(val, (int, float)):
                    score = int(val)
    except Exception:
        pass

    # 2. Try regex fallback
    if score is None:
        pattern = r'"score"\s*:\s*\[?\s*(\d+)\s*\]?'
        match = re.search(pattern, content)
        if match:
            score = int(match.group(1))

    if score is not None and not (0 <= score <= 10):
        score = None
    return score


def build_messages(user_prompt, gpt_prompt, src_img, tgt_img):
    return [
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": gpt_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": src_img},
                {"type": "image", "image": tgt_img},
            ],
        },
    ]


def batch_generate_hf(batch_items, model, processor, max_new_tokens=256):
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
                            imgs.append(local_load_image(img_val, size=512))
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


def batch_generate_vllm(batch_items, vllm_engine, processor, max_new_tokens=256):
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
                            images.append(local_load_image(img_val, size=512))
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

    outputs = vllm_engine.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=False)
    results = [out.outputs[0].text for out in outputs]
    return results


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
    parser = argparse.ArgumentParser(description="Dreambench++ High-Throughput vLLM Batch VIEScore resemblance evaluation")
    parser.add_argument('--samples_dir', type=str, required=True, help="Directory containing target samples")
    parser.add_argument('--rating_dir', type=str, default=None, help="Directory to save evaluation ratings")
    parser.add_argument('--data_dir', type=str, default=None, help="Directory containing original reference images")
    parser.add_argument('--category', type=str, default=None, help="Process only a specific category folder")
    parser.add_argument('--subject', type=str, default=None, help="Process only a specific subject")
    parser.add_argument('--model', type=str, nargs="+", default=None, help="Process specific model name(s)")
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-VL-32B-Instruct', help="Pretrained VLM name/path")
    parser.add_argument('--prompt_file', type=str, default='/root/Desktop/workspace/woosung/commercial-dreambench/prompts/VIEScore_resemblance.txt', help="Path to VIEScore_resemblance.txt")
    parser.add_argument('--out_dir', type=str, default='/tmp', help="Temp output folder for resizing")
    parser.add_argument('--max_retry', type=int, default=3, help="Max retry limit for VLM parsing")
    parser.add_argument('--remove_bg', type=str, default="False", help="Evaluate against background-removed reference image ('True' or 'False')")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for parallel evaluation")
    parser.add_argument('--use_vllm', action='store_true', help="Force use of vLLM backend engine for inference")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help="GPU memory utilization factor for vLLM")
    parser.add_argument('--skip_if_done', action='store_true', help="Skip evaluation if result is already in JSON")

    args = parser.parse_args()
    remove_bg = args.remove_bg.lower() == 'true'

    if args.rating_dir:
        rating_dir = args.rating_dir
    elif '/samples/' in args.samples_dir:
        rating_dir = args.samples_dir.replace('/samples/', '/rating/')
    else:
        rating_dir = os.path.join(os.path.dirname(args.samples_dir.rstrip('/')), 'rating', os.path.basename(args.samples_dir.rstrip('/')))

    with open(args.prompt_file, "r") as f:
        user_prompt = f.read().strip()

    gpt_prompt = (
        "Yes, I understand the task. I will evaluate the resemblance between the token subject "
        "in the first image and the subject in the second generated image based on the provided rules. "
        "I will output my response in the requested JSON format containing the 'score' list (ranging from 0 to 10) "
        "and a concise reasoning. Please provide the images."
    )

    # Determine models to evaluate
    if args.model is not None:
        models = []
        for m in args.model:
            for sub_m in m.replace(',', ' ').split():
                models.append(sub_m)
    else:
        models = [d for d in os.listdir(args.samples_dir) if os.path.isdir(os.path.join(args.samples_dir, d)) and d not in ['deprecated']]

    models = sorted(list(set(models)))
    print(f"Discovered {len(models)} model(s) to evaluate: {models}")

    # Setup Lazy Model Engine Initialization
    use_vllm_engine = HAS_VLLM and (args.use_vllm or HAS_VLLM)
    processor = None
    vllm_engine = None
    hf_model_obj = None

    def ensure_model_loaded():
        nonlocal processor, vllm_engine, hf_model_obj, use_vllm_engine
        if processor is not None or vllm_engine is not None or hf_model_obj is not None:
            return
        print(f"Loading local VLM processor for model: {args.model_name}...")
        processor = AutoProcessor.from_pretrained(args.model_name)

        if use_vllm_engine:
            print(f"Initializing vLLM Engine backend for high-throughput batching...")
            try:
                vllm_engine = LLM(
                    model=args.model_name,
                    trust_remote_code=True,
                    max_model_len=8192,
                    limit_mm_per_prompt={"image": 2},
                    gpu_memory_utilization=args.gpu_memory_utilization
                )
                print("vLLM Engine initialized successfully!")
            except Exception as e:
                print(f"[WARNING] Failed to initialize vLLM engine ({e}). Falling back to PyTorch batched inference.")
                use_vllm_engine = False

        if not use_vllm_engine:
            print(f"Loading PyTorch model on cuda:0 for batched inference (batch_size={args.batch_size})...")
            if "Qwen3" in args.model_name and Qwen3VLForConditionalGeneration is not None:
                hf_model_obj = Qwen3VLForConditionalGeneration.from_pretrained(
                    args.model_name, torch_dtype=torch.bfloat16, device_map="cuda:0"
                )
            else:
                try:
                    hf_model_obj = AutoModelForCausalLM.from_pretrained(
                        args.model_name, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
                    )
                except Exception:
                    from transformers import AutoModelImageTextToText
                    hf_model_obj = AutoModelImageTextToText.from_pretrained(
                        args.model_name, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
                    )
            print("PyTorch model loaded successfully!")

    for m in models:
        model_dir = os.path.join(args.samples_dir, m)
        if not os.path.exists(model_dir):
            continue

        print(f"\nEvaluating VIEScore for model subfolder: {m}")

        model_subjects = discover_subjects(args.samples_dir, model_filter=m, subject_filter=args.subject, category_filter=args.category)
        if not model_subjects:
            print(f"No subjects found for model {m}.")
            continue

        out_dir = os.path.join(rating_dir, m)
        os.makedirs(out_dir, exist_ok=True)
        results_file = os.path.join(out_dir, 'viescore_results.json')
        legacy_results_file = os.path.join(model_dir, 'viescore_results.json')

        # Load existing results if present
        results = {}
        target_file = results_file if os.path.exists(results_file) else legacy_results_file
        if os.path.exists(target_file):
            try:
                with open(target_file, 'r') as f:
                    results = json.load(f)
                if args.skip_if_done and 'overall_mean' in results:
                    overall_mean = results['overall_mean']
                    scores = [v for k, v in results.items() if k != 'overall_mean']
                    print(f"Model '{m}' VIEScore evaluation already completed (mean: {overall_mean:.4f}). Skipping.")
                    log_eval_summary("VIEScore", m, len(scores), overall_mean)
                    continue
            except Exception as e:
                print(f"Warning: Failed to load existing results file {target_file}: {e}")

        results.pop('overall_mean', None)

        # Collect evaluation queue for this model
        eval_queue = []
        for s in model_subjects:
            subj_dir = s['full_path']
            ref_path = get_reference_path(subj_dir, category=s['category'], subject=s['subject'], data_dir=args.data_dir, remove_bg=remove_bg)
            if not ref_path:
                continue

            gen_images = get_generated_images(subj_dir)
            if not gen_images:
                continue

            for img_name in gen_images:
                img_path = os.path.join(subj_dir, img_name)
                img_base = os.path.splitext(img_name)[0]

                if s['category']:
                    key = f"{s['category']}_{s['subject']}_{img_base}"
                else:
                    key = f"{s['subject']}_{img_base}"

                if key in results and args.skip_if_done:
                    continue

                src_img = local_load_image(ref_path, size=512)
                tgt_img = local_load_image(img_path, size=512)
                msgs = build_messages(user_prompt, gpt_prompt, src_img, tgt_img)

                eval_queue.append({
                    "key": key,
                    "ref_path": ref_path,
                    "img_path": img_path,
                    "messages": msgs,
                    "retry_count": 0
                })

        if not eval_queue:
            print(f"No pending image pairs to evaluate for model {m}.")
            continue

        ensure_model_loaded()
        print(f"Executing batched evaluation on {len(eval_queue)} pair prompt tasks for model {m}...")

        # Process in batches with retry tracking
        pending_items = eval_queue
        while pending_items:
            # Batch execution
            all_raw_responses = []
            if use_vllm_engine:
                # Process all pending items using vLLM high-throughput engine
                all_raw_responses = batch_generate_vllm(pending_items, vllm_engine, processor)
            else:
                # Chunk into mini-batches for PyTorch pipeline
                for i in tqdm(range(0, len(pending_items), args.batch_size), desc="PyTorch Batched Generation"):
                    chunk = pending_items[i : i + args.batch_size]
                    chunk_raw = batch_generate_hf(chunk, hf_model_obj, processor, max_new_tokens=256)
                    all_raw_responses.extend(chunk_raw)

            retry_queue = []
            for task_item, raw_res in zip(pending_items, all_raw_responses):
                key = task_item["key"]
                score = parse_viescore(raw_res)

                if score is not None:
                    results[key] = score
                    print(f"  {key}: {score}")
                else:
                    task_item["retry_count"] += 1
                    if task_item["retry_count"] < args.max_retry:
                        print(f"  {key}: Parsing failed (retry {task_item['retry_count']}/{args.max_retry})")
                        retry_queue.append(task_item)
                    else:
                        print(f"  {key}: Failed to parse score after {args.max_retry} retries.")

            # Incrementally save results
            try:
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=4)
            except Exception as e:
                print(f"Error saving incremental results: {e}")

            pending_items = retry_queue

        # Compute overall mean
        scores = [v for k, v in results.items() if k != 'overall_mean']
        if scores:
            overall_mean = sum(scores) / len(scores)
            results['overall_mean'] = overall_mean
            print(f"Model '{m}' overall mean VIEScore score: {overall_mean:.4f}")
            log_eval_summary("VIEScore", m, len(scores), overall_mean)

            try:
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=4)
                print(f"Saved VIEScore results to {results_file}")
            except Exception as e:
                print(f"Error saving results file: {e}")
        else:
            print(f"No VIEScore scores computed for model '{m}'.")


if __name__ == "__main__":
    main()
