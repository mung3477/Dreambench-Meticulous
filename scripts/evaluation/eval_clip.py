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

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

def discover_subjects(samples_dir, model_filter=None, subject_filter=None, category_filter=None):
    if model_filter:
        if isinstance(model_filter, list):
            models = model_filter
        elif isinstance(model_filter, str):
            models = [sub_m for sub_m in model_filter.replace(',', ' ').split()]
        else:
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
    parser = argparse.ArgumentParser(description="Dreambench++ GPU-efficient batch CLIP similarity evaluation")
    parser.add_argument('--samples_dir', type=str, required=True, help="Directory containing target samples")
    parser.add_argument('--rating_dir', type=str, default=None, help="Directory to save evaluation ratings")
    parser.add_argument('--data_dir', type=str, default=None, help="Directory containing original reference images")
    parser.add_argument('--category', type=str, default=None, help="Process only a specific category folder")
    parser.add_argument('--subject', type=str, default=None, help="Process only a specific subject")
    parser.add_argument('--model', type=str, nargs='+', default=None, help="Process specific model name(s)")
    parser.add_argument('--remove_bg', type=str, default="False", help="Evaluate against background-removed reference image ('True' or 'False')")
    parser.add_argument('--skip_if_done', action='store_true', help="Skip evaluation if result is already in JSON")
    args = parser.parse_args()
    remove_bg = args.remove_bg.lower() == 'true'

    if args.rating_dir:
        rating_dir = args.rating_dir
    elif '/samples/' in args.samples_dir:
        rating_dir = args.samples_dir.replace('/samples/', '/rating/')
    else:
        rating_dir = os.path.join(os.path.dirname(args.samples_dir.rstrip('/')), 'rating', os.path.basename(args.samples_dir.rstrip('/')))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model on {device}...")
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    # Determine models to evaluate
    if args.model is not None:
        models = []
        model_inputs = args.model if isinstance(args.model, list) else [args.model]
        for m in model_inputs:
            for sub_m in m.replace(',', ' ').split():
                models.append(sub_m)
    else:
        models = [d for d in os.listdir(args.samples_dir) if os.path.isdir(os.path.join(args.samples_dir, d)) and d not in ['deprecated']]
    
    models = sorted(list(set(models)))
    
    for m in models:
        model_dir = os.path.join(args.samples_dir, m)
        if not os.path.exists(model_dir):
            continue
            
        print(f"\nEvaluating CLIP for model subfolder: {m}")
        
        model_subjects = discover_subjects(args.samples_dir, model_filter=m, subject_filter=args.subject, category_filter=args.category)
        if not model_subjects:
            print(f"No subjects found for model {m}.")
            continue
            
        out_dir = os.path.join(rating_dir, m)
        os.makedirs(out_dir, exist_ok=True)
        results_file = os.path.join(out_dir, 'clip_results.json')
        legacy_results_file = os.path.join(model_dir, 'clip_results.json')
        
        # Load existing results if they exist
        results = {}
        target_file = results_file if os.path.exists(results_file) else legacy_results_file
        if os.path.exists(target_file):
            try:
                with open(target_file, 'r') as f:
                    results = json.load(f)
                if args.skip_if_done and 'overall_mean' in results:
                    overall_mean = results['overall_mean']
                    scores = [v for k, v in results.items() if k != 'overall_mean']
                    print(f"Model '{m}' CLIP evaluation already completed (mean: {overall_mean:.4f}). Skipping.")
                    log_eval_summary("CLIP", m, len(scores), overall_mean)
                    continue
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
                
            try:
                ref_img = Image.open(ref_path).convert("RGB")
                ref_inputs = processor(images=ref_img, return_tensors="pt").to(device)
                with torch.no_grad():
                    ref_features = model.get_image_features(**ref_inputs)
                ref_features = F.normalize(ref_features, p=2, dim=-1)
            except Exception as e:
                print(f"Error loading reference image {ref_path}: {e}")
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
                
                try:
                    img = Image.open(img_path).convert("RGB")
                    inputs = processor(images=img, return_tensors="pt").to(device)
                    with torch.no_grad():
                        features = model.get_image_features(**inputs)
                    features = F.normalize(features, p=2, dim=-1)
                    similarity = F.cosine_similarity(ref_features, features).item()
                    results[key] = similarity
                except Exception as e:
                    print(f"Error evaluating CLIP similarity for {img_path}: {e}")
                    
        # Compute overall mean
        scores = [v for k, v in results.items() if k != 'overall_mean']
        if scores:
            overall_mean = sum(scores) / len(scores)
            results['overall_mean'] = overall_mean
            print(f"Model '{m}' overall mean CLIP score: {overall_mean:.4f}")
            log_eval_summary("CLIP", m, len(scores), overall_mean)
            
            # Save results
            try:
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=4)
                print(f"Saved CLIP results to {results_file}")
            except Exception as e:
                print(f"Error saving results file: {e}")
        else:
            print(f"No CLIP scores computed for model '{m}'.")

if __name__ == "__main__":
    main()
