#!/usr/bin/env python3
"""
eval_qwen-reranker.py

Evaluates SDG subject consistency using Qwen3-VL-Reranker-2B.
Pairs reference images (query) with candidate SDG variation images (documents)
collected from commercial-dreambench dataset directories.

Saves results to:
  {rating_dir}/{model_name}/qwen_reranker_results.json
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone, timedelta
import torch
from PIL import Image
from tqdm import tqdm


def log_eval_summary(metric_name, model_name, num_images, overall_mean):
    kst = timezone(timedelta(hours=9))
    now_kst = datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S KST')
    log_dir = '/root/Desktop/workspace/woosung/commercial-dreambench/scripts/evaluation/log'
    os.makedirs(log_dir, exist_ok=True)

    filename = "qwen_reranker_eval.log"
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


def discover_subjects(samples_dir, model_filter=None, subject_filter=None, category_filter=None):
    if model_filter:
        if isinstance(model_filter, list):
            models = model_filter
        elif isinstance(model_filter, str):
            models = [sub_m for sub_m in model_filter.replace(',', ' ').split()]
        else:
            models = [model_filter]
    else:
        models = [
            d for d in os.listdir(samples_dir)
            if os.path.isdir(os.path.join(samples_dir, d)) and d not in ['deprecated']
        ]

    subjects = []
    for m in models:
        m_dir = os.path.join(samples_dir, m)
        if not os.path.exists(m_dir):
            continue

        for name in sorted(os.listdir(m_dir)):
            p1 = os.path.join(m_dir, name)
            if not os.path.isdir(p1) or name == 'deprecated':
                continue

            subdirs = [
                d for d in os.listdir(p1)
                if os.path.isdir(os.path.join(p1, d)) and d != '.ipynb_checkpoints'
            ]
            if subdirs:
                # Category-nested structure (e.g., category/asin)
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
                # Flat structure (e.g., subject)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SDG subject consistency using Qwen3-VL-Reranker-2B")
    parser.add_argument('--samples_dir', type=str, default='/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn', help="Directory containing target samples")
    parser.add_argument('--rating_dir', type=str, default=None, help="Directory to save evaluation ratings")
    parser.add_argument('--data_dir', type=str, default='/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn/original', help="Directory containing original reference images")
    parser.add_argument('--category', type=str, default=None, help="Process only a specific category folder")
    parser.add_argument('--subject', type=str, default=None, help="Process only a specific subject")
    parser.add_argument('--model', type=str, nargs='+', default=None, help="Process specific model name(s)")
    parser.add_argument('--model_id', type=str, default='Qwen/Qwen3-VL-Reranker-2B', help="HuggingFace model ID")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for reranker inference")
    parser.add_argument('--remove_bg', type=str, default="False", help="Evaluate against background-removed reference image ('True' or 'False')")
    parser.add_argument('--skip_if_done', action='store_true', help="Skip evaluation if result is already in JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    remove_bg = args.remove_bg.lower() == 'true'

    if args.rating_dir:
        rating_dir = args.rating_dir
    elif '/samples/' in args.samples_dir:
        rating_dir = args.samples_dir.replace('/samples/', '/rating/')
    else:
        rating_dir = os.path.join(os.path.dirname(args.samples_dir.rstrip('/')), 'rating', os.path.basename(args.samples_dir.rstrip('/')))

    # Determine models to evaluate
    if args.model is not None:
        models = []
        model_inputs = args.model if isinstance(args.model, list) else [args.model]
        for m in model_inputs:
            for sub_m in m.replace(',', ' ').split():
                models.append(sub_m)
    else:
        models = [
            d for d in os.listdir(args.samples_dir)
            if os.path.isdir(os.path.join(args.samples_dir, d)) and d not in ['deprecated']
        ]

    models = sorted(list(set(models)))

    model_obj = None

    def get_model():
        nonlocal model_obj
        if model_obj is None:
            from sentence_transformers import CrossEncoder
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading Qwen3-VL-Reranker model '{args.model_id}' on {device}...")
            model_obj = CrossEncoder(
                args.model_id,
                trust_remote_code=True,
                model_kwargs={'torch_dtype': torch.bfloat16} if torch.cuda.is_available() else {}
            )
        return model_obj

    for m in models:
        model_dir = os.path.join(args.samples_dir, m)
        if not os.path.exists(model_dir):
            continue

        model_subjects = discover_subjects(
            args.samples_dir, model_filter=m, subject_filter=args.subject, category_filter=args.category
        )
        if not model_subjects:
            continue

        out_dir = os.path.join(rating_dir, m)
        os.makedirs(out_dir, exist_ok=True)
        results_file = os.path.join(out_dir, 'qwen_reranker_results.json')
        legacy_results_file = os.path.join(model_dir, 'qwen_reranker_results.json')

        results = {}
        target_file = results_file if os.path.exists(results_file) else legacy_results_file
        if os.path.exists(target_file):
            try:
                with open(target_file, 'r') as f:
                    results = json.load(f)
                if args.skip_if_done and 'overall_mean' in results:
                    overall_mean = results['overall_mean']
                    scores = [v for k, v in results.items() if k != 'overall_mean']
                    print(f"Model '{m}' Qwen-Reranker evaluation already completed (mean: {overall_mean:.4f}). Skipping.")
                    log_eval_summary("Qwen-Reranker", m, len(scores), overall_mean)
                    continue
            except Exception as e:
                print(f"Warning: Failed to load existing results file {target_file}: {e}")

        results.pop('overall_mean', None)

        # Collect all (query, doc) pairs for batch inference
        pair_metadata = []
        for s in model_subjects:
            subj_dir = s['full_path']
            ref_path = get_reference_path(
                subj_dir, category=s['category'], subject=s['subject'], data_dir=args.data_dir, remove_bg=remove_bg
            )
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

                pair_metadata.append((key, ref_path, img_path))

        if not pair_metadata:
            print(f"No new image pairs to process for model '{m}'.")
            if results:
                valid_scores = [v for k, v in results.items() if k != 'overall_mean']
                if valid_scores:
                    overall_mean = float(sum(valid_scores) / len(valid_scores))
                    results['overall_mean'] = overall_mean
                    log_eval_summary("Qwen-Reranker", m, len(valid_scores), overall_mean)
            continue

        print(f"\nEvaluating Qwen-Reranker for model subfolder: {m}")
        print(f"Running Qwen-Reranker inference on {len(pair_metadata)} image pairs (batch_size={args.batch_size})...")

        # Lazy load model right before inference
        reranker_model = get_model()

        # Run inference in batches
        pairs_to_predict = [(p[1], p[2]) for p in pair_metadata]
        raw_scores = reranker_model.predict(pairs_to_predict, batch_size=args.batch_size, show_progress_bar=True)

        for (key, _, _), score in zip(pair_metadata, raw_scores):
            results[key] = float(score)

        valid_scores = [v for k, v in results.items() if k != 'overall_mean']
        if valid_scores:
            overall_mean = float(sum(valid_scores) / len(valid_scores))
            results['overall_mean'] = overall_mean

            print(f"Completed evaluation for '{m}'. Total evaluated images: {len(valid_scores)}, Mean Score: {overall_mean:.4f}")
            log_eval_summary("Qwen-Reranker", m, len(valid_scores), overall_mean)

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {results_file}")


if __name__ == '__main__':
    main()
