#!/usr/bin/env python3
"""
eval_AKI.py

Evaluates keypoint matches between reference images and generated images
using OmniGlue (https://github.com/google-research/omniglue).

Outputs `keypoints_results.json` per model under `rating_dir/<model>/`,
logging sample-level matched keypoint counts N(M(I_ref, I_gen)).
Downstream pairwise AKI and K_Gain metrics can be computed instantly via:
    scripts/evaluation/utils/calculate_aki.py
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime, timezone, timedelta
from PIL import Image, ImageOps
from tqdm import tqdm


def log_eval_summary(metric_name, model_name, num_images, overall_mean):
    kst = timezone(timedelta(hours=9))
    now_kst = datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S KST')
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log')
    os.makedirs(log_dir, exist_ok=True)

    filename_map = {
        'Keypoints': 'keypoints_eval.log',
        'AKI': 'aki_eval.log',
        'DINO': 'dino_eval.log',
        'CLIP': 'clip_eval.log'
    }
    filename = filename_map.get(metric_name, f"{metric_name.lower()}_eval.log")
    log_path = os.path.join(log_dir, filename)

    log_line = f"[{now_kst}] Metric: {metric_name:<12} | Model: {model_name} | Images: {num_images} | Avg Keypoints: {overall_mean:.2f}\n"
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_line)
        combined_log_path = os.path.join(log_dir, 'evaluation_summary.log')
        with open(combined_log_path, 'a', encoding='utf-8') as f:
            f.write(log_line)
        print(f"Logged evaluation summary to {log_path}")
    except Exception as e:
        print(f"Failed to write log entry: {e}")


def load_image_as_numpy(img):
    """Load an image (or accept existing array) and convert to RGB numpy array (H, W, 3)."""
    if isinstance(img, np.ndarray):
        return img
    pil_img = Image.open(img)
    pil_img = ImageOps.exif_transpose(pil_img)
    if pil_img.mode == "RGBA":
        background = Image.new("RGBA", pil_img.size, "white")
        pil_img = Image.alpha_composite(background, pil_img).convert("RGB")
    elif pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return np.array(pil_img)


class OmniGlueMatcher:
    """Wrapper class for OmniGlue keypoint matching."""

    def __init__(self, og_export=None, sp_export=None, dino_export=None, match_threshold=0.02, mock=False):
        self.match_threshold = match_threshold
        self.mock = mock
        self.matcher = None

        if self.mock:
            print("[Info] Running OmniGlue in mock/test mode.")
            return

        try:
            import omniglue
            self.matcher = omniglue.OmniGlue(
                og_export=og_export or './models/og_export',
                sp_export=sp_export or './models/sp_v6',
                dino_export=dino_export or './models/dinov2_vitb14_pretrain.pth'
            )
            print("Successfully initialized OmniGlue matcher.")
        except ImportError:
            print(
                "\n[Warning] 'omniglue' package is not installed in the current environment.\n"
                "To install OmniGlue, run:\n"
                "    git clone https://github.com/google-research/omniglue.git\n"
                "    cd omniglue && pip install -e .\n"
                "And download model checkpoints for OmniGlue, SuperPoint, and DINOv2.\n"
            )
            self.matcher = None
        except Exception as e:
            print(f"\n[Warning] Failed to initialize OmniGlue: {e}")
            self.matcher = None

    def count_matches(self, img0, img1):
        """
        Calculates N(M(img0, img1)), the number of matched keypoints between two images.
        Accepts file paths or numpy arrays.
        """
        arr0 = load_image_as_numpy(img0)
        arr1 = load_image_as_numpy(img1)

        if self.mock:
            h0, w0 = arr0.shape[:2]
            h1, w1 = arr1.shape[:2]
            return int((h0 + h1 + w0 + w1) % 100 + 10)

        if self.matcher is None:
            raise RuntimeError(
                "OmniGlue matcher is not initialized. Please ensure 'omniglue' is installed "
                "and valid model weights are provided."
            )

        match_kp0, match_kp1, match_confidences = self.matcher.FindMatches(arr0, arr1)

        if self.match_threshold is not None and self.match_threshold > 0:
            if match_confidences is not None and len(match_confidences) > 0:
                valid_mask = match_confidences > self.match_threshold
                return int(np.sum(valid_mask))
            return 0

        return int(len(match_kp0)) if match_kp0 is not None else 0


def discover_subjects(samples_dir, model_filter=None, subject_filter=None, category_filter=None):
    """Discovers subject directories following Dreambench conventions."""
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
    """Finds the reference image path for a given subject."""
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
    """Retrieves generated image filenames within a subject directory."""
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
    parser = argparse.ArgumentParser(
        description="OmniGlue Reference Keypoint Evaluation (N(M(I_ref, I_gen)))"
    )
    parser.add_argument('--samples_dir', type=str, required=True, help="Directory containing target samples")
    parser.add_argument('--rating_dir', type=str, default=None, help="Directory to save evaluation ratings")
    parser.add_argument('--data_dir', type=str, default=None, help="Directory containing original reference images")
    parser.add_argument('--category', type=str, default=None, help="Process only a specific category folder")
    parser.add_argument('--subject', type=str, default=None, help="Process only a specific subject")
    parser.add_argument('--model', type=str, nargs='+', default=None, help="Process specific target model name(s)")
    parser.add_argument('--remove_bg', type=str, default="False", help="Evaluate against background-removed reference image ('True' or 'False')")
    parser.add_argument('--skip_if_done', action='store_true', help="Skip evaluation if result is already in JSON")
    parser.add_argument('--match_threshold', type=float, default=0.02, help="OmniGlue confidence threshold for filtering valid matches")
    parser.add_argument('--og_export', type=str, default='./models/og_export', help="Path to OmniGlue weights export")
    parser.add_argument('--sp_export', type=str, default='./models/sp_v6', help="Path to SuperPoint export")
    parser.add_argument('--dino_export', type=str, default='./models/dinov2_vitb14_pretrain.pth', help="Path to DINOv2 weights")
    parser.add_argument('--mock', action='store_true', help="Use mock matcher for dry-run testing without OmniGlue weights")

    args = parser.parse_args()
    remove_bg = args.remove_bg.lower() == 'true'

    if args.rating_dir:
        rating_dir = args.rating_dir
    elif '/samples/' in args.samples_dir:
        rating_dir = args.samples_dir.replace('/samples/', '/rating/')
    else:
        rating_dir = os.path.join(
            os.path.dirname(args.samples_dir.rstrip('/')),
            'rating',
            os.path.basename(args.samples_dir.rstrip('/'))
        )

    # Initialize keypoint matcher
    matcher = OmniGlueMatcher(
        og_export=args.og_export,
        sp_export=args.sp_export,
        dino_export=args.dino_export,
        match_threshold=args.match_threshold,
        mock=args.mock
    )

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

    for m in models:
        model_dir = os.path.join(args.samples_dir, m)
        if not os.path.exists(model_dir):
            continue

        print(f"\n=======================================================")
        print(f"Evaluating Keypoint Matches for Model: {m}")
        print(f"=======================================================")

        model_subjects = discover_subjects(
            args.samples_dir,
            model_filter=m,
            subject_filter=args.subject,
            category_filter=args.category
        )
        if not model_subjects:
            print(f"No subjects found for model {m}.")
            continue

        out_dir = os.path.join(rating_dir, m)
        os.makedirs(out_dir, exist_ok=True)
        results_file = os.path.join(out_dir, 'keypoints_results.json')
        legacy_results_file = os.path.join(model_dir, 'keypoints_results.json')

        results = {}
        target_file = results_file if os.path.exists(results_file) else legacy_results_file
        if os.path.exists(target_file):
            try:
                with open(target_file, 'r') as f:
                    results = json.load(f)
                if args.skip_if_done and 'overall_mean' in results:
                    overall_mean = results['overall_mean']
                    scores = [v for k, v in results.items() if k != 'overall_mean']
                    print(f"Model '{m}' keypoints evaluation already completed (mean: {overall_mean:.2f}). Skipping.")
                    log_eval_summary("Keypoints", m, len(scores), overall_mean)
                    continue
            except Exception as e:
                print(f"Warning: Failed to load existing results file {target_file}: {e}")

        # Clear previous overall_mean
        results.pop('overall_mean', None)

        for s in tqdm(model_subjects, desc="Subjects"):
            subj_dir = s['full_path']
            ref_path = get_reference_path(
                subj_dir,
                category=s['category'],
                subject=s['subject'],
                data_dir=args.data_dir,
                remove_bg=remove_bg
            )
            if not ref_path:
                continue

            gen_images = get_generated_images(subj_dir)
            if not gen_images:
                continue

            try:
                ref_arr = load_image_as_numpy(ref_path)
            except Exception as e:
                print(f"Error loading reference image {ref_path}: {e}")
                continue

            for img_name in gen_images:
                img_path = os.path.join(subj_dir, img_name)
                img_base = os.path.splitext(img_name)[0]

                key = f"{s['category']}_{s['subject']}_{img_base}" if s['category'] else f"{s['subject']}_{img_base}"

                if key in results and args.skip_if_done:
                    continue

                try:
                    num_matches = matcher.count_matches(ref_arr, img_path)
                    results[key] = num_matches
                except Exception as e:
                    print(f"Error evaluating sample {img_path}: {e}")

        # Compute overall mean
        scores = [v for k, v in results.items() if k != 'overall_mean' and isinstance(v, (int, float))]
        if scores:
            overall_mean = float(np.mean(scores))
            results['overall_mean'] = overall_mean
            print(f"Model '{m}' overall mean keypoints: {overall_mean:.2f}")
            log_eval_summary("Keypoints", m, len(scores), overall_mean)

            try:
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=4)
                print(f"Saved keypoints results to {results_file}")
            except Exception as e:
                print(f"Error saving results file: {e}")
        else:
            print(f"No keypoint evaluations computed for model '{m}'.")


if __name__ == "__main__":
    main()
