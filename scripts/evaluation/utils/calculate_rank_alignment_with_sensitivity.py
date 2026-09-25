#!/usr/bin/env python3
"""
calculate_rank_alignment_with_sensitivity.py

Calculates rank alignment statistics, detail sensitivity, dynamic range, effect sizes,
and continuous Epsilon-Sweep / Z-Score Sweep curves across configurable distortion ladders.

Supported CLI Options:
  --preset [3-step | 5-step | 8-step | area-all | custom]
  --scales original 0.875x 0.625x 0.375x 0.25x
  --dimension ["Noise Strength" | "Noise Area"]
  --plot_curves (generates high-DPI publication comparison plots)

Outputs:
  - alignment_sensitivity_summary.csv (main summary table with AUCC scores)
  - epsilon_sweep_curves.csv (dense curve data points)
  - epsilon_sweep_zscore.png (Z-score sweep comparison plot)
  - discordant_pairs_sensitivity.csv (logged discordant pairs)
"""

import argparse
import json
import os
import glob
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, pearsonr
import matplotlib.pyplot as plt, matplotlib.patches as mpatches

# Full Dictionary of Noise Strength Steps (Ordered Weakest Noise -> Strongest Noise)
NOISE_STRENGTH_DICT = {
    "refined": ("refined", 0),
    "original": ("original", 1),
    "1.0x": ("noised-1.0x_masked-dino-adaptive_height-ratio-1.0_timestep-10", 2),
    "1.0": ("noised-1.0x_masked-dino-adaptive_height-ratio-1.0_timestep-10", 2),
    "0.875x": ("noised-0.875x_masked-dino-adaptive_height-ratio-1.0_timestep-10", 3),
    "0.875": ("noised-0.875x_masked-dino-adaptive_height-ratio-1.0_timestep-10", 3),
    "0.75x": ("noised-0.75x_masked-dino-adaptive_height-ratio-1.0_timestep-10", 4),
    "0.75": ("noised-0.75x_masked-dino-adaptive_height-ratio-1.0_timestep-10", 4),
    "0.625x": ("noised-0.625x_masked-dino-adaptive_height-ratio-1.0_timestep-10", 5),
    "0.625": ("noised-0.625x_masked-dino-adaptive_height-ratio-1.0_timestep-10", 5),
    "0.5x": ("noised-0.5x_masked-dino-adaptive_height-ratio-1.0_timestep-10", 6),
    "0.5": ("noised-0.5x_masked-dino-adaptive_height-ratio-1.0_timestep-10", 6),
    "0.375x": ("noised-0.375x_masked-dino-adaptive_height-ratio-1.0_timestep-10", 7),
    "0.375": ("noised-0.375x_masked-dino-adaptive_height-ratio-1.0_timestep-10", 7),
    "0.25x": ("noised-0.25x_masked-dino-adaptive_height-ratio-1.0_timestep-10", 8),
    "0.25": ("noised-0.25x_masked-dino-adaptive_height-ratio-1.0_timestep-10", 8),
}

# Full Dictionary of Noise Area Steps (Ordered Smallest Area -> Largest Area)
NOISE_AREA_DICT = {
    "original": ("original", 0),
    "0.25": ("noised-0.5x_masked-dino-adaptive_height-ratio-0.25_timestep-10", 1),
    "0.375": ("noised-0.5x_masked-dino-adaptive_height-ratio-0.375_timestep-10", 2),
    "0.5": ("noised-0.5x_masked-dino-adaptive_height-ratio-0.5_timestep-10", 3),
    "0.625": ("noised-0.5x_masked-dino-adaptive_height-ratio-0.625_timestep-10", 4),
    "0.75": ("noised-0.5x_masked-dino-adaptive_height-ratio-0.75_timestep-10", 5),
    "0.875": ("noised-0.5x_masked-dino-adaptive_height-ratio-0.875_timestep-10", 6),
    "1.0": ("noised-0.5x_masked-dino-adaptive_height-ratio-1.0_timestep-10", 7),
}

PRESET_SEQUENCES = {
    "3-step": [
        NOISE_STRENGTH_DICT["original"],
        NOISE_STRENGTH_DICT["0.625x"],
        NOISE_STRENGTH_DICT["0.25x"],
    ],
    "refined": [
            NOISE_STRENGTH_DICT["refined"],
            NOISE_STRENGTH_DICT["original"],
            NOISE_STRENGTH_DICT["0.625x"],
            NOISE_STRENGTH_DICT["0.25x"],
    ],
    "5-step": [
        NOISE_STRENGTH_DICT["original"],
        NOISE_STRENGTH_DICT["0.875x"],
        NOISE_STRENGTH_DICT["0.625x"],
        NOISE_STRENGTH_DICT["0.375x"],
        NOISE_STRENGTH_DICT["0.25x"],
    ],
    "8-step": [
        NOISE_STRENGTH_DICT["original"],
        NOISE_STRENGTH_DICT["1.0x"],
        NOISE_STRENGTH_DICT["0.875x"],
        NOISE_STRENGTH_DICT["0.75x"],
        NOISE_STRENGTH_DICT["0.625x"],
        NOISE_STRENGTH_DICT["0.5x"],
        NOISE_STRENGTH_DICT["0.375x"],
        NOISE_STRENGTH_DICT["0.25x"],
    ],
    "area-all": [
        NOISE_AREA_DICT["original"],
        NOISE_AREA_DICT["0.25"],
        NOISE_AREA_DICT["0.375"],
        NOISE_AREA_DICT["0.5"],
        NOISE_AREA_DICT["0.625"],
        NOISE_AREA_DICT["0.75"],
        NOISE_AREA_DICT["0.875"],
        NOISE_AREA_DICT["1.0"],
    ]
}

metric_files = {
    "CLIP": "clip_results.json",
    "DINO": "dino_results.json",
    "DreamBench++": "dreambench_plus_results.json",
    "VIEScore": "viescore_results.json",
    "Keypoints": "keypoints_results.json",
    # "Qwen-Reranker": "qwen_reranker_results.json",
    # "CLIP_bbox-crop": "clip_bbox-crop_results.json",
    # "DINO_bbox-crop": "dino_bbox-crop_results.json",
    # "Qwen-Reranker_bbox-crop": "qwen_reranker_bbox-crop_results.json",
    "Ours": "ours-Qwen3-VL-8B-Instruct_bbox-crop_results.json",
    # "Qwen3-VL-8B_bbox-crop_reversed": "ours-Qwen3-VL-8B-Instruct_image-reversed_bbox-crop_results.json",
    # "Qwen3-VL-32B_bbox-crop": "ours_bbox-crop_results.json",
    # "GLM-4.6V-Flash_bbox-crop": "ours-GLM-4.6V-Flash_bbox-crop_results.json",
    # "InternVL3-8B_bbox-crop": "ours-InternVL3-8B_bbox-crop_results.json",
    # "Qwen3-VL-8B_bbox-overlay": "ours-Qwen3-VL-8B-Instruct_bbox-overlay_results.json",
    # "Ours_each_item": "ours/each_item/summary.json",
    # "Qwen3-VL-8B_coarse-bbox-crop": "ours-Qwen3-VL-8B-Instruct_coarse_bbox-crop_results.json",
    # "Qwen3-VL-8B_fine-bbox-crop": "ours-Qwen3-VL-8B-Instruct_fine_bbox-crop_results.json",
}

# Dense sweep vectors
# ZSCORE_THRESHOLDS_K = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ZSCORE_THRESHOLDS_K = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate rank alignment, sensitivity, and Epsilon/Z-Score sweeps.")
    parser.add_argument(
        "--rating_dir",
        type=str,
        default="/root/Desktop/workspace/woosung/commercial-dreambench/rating/amzn_distorted/flux-klein",
        help="Path to rating directory containing model subfolders with JSON evaluation results."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/root/Desktop/workspace/woosung/commercial-dreambench/assets/data/amzn",
        help="Root wordy data directory for ordering evaluation items."
    )
    parser.add_argument(
        "--rubrics_dir",
        type=str,
        default="/root/Desktop/workspace/woosung/commercial-dreambench/assets/rubrics/amzn",
        help="Root cached rubrics directory for ordering evaluation items."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/Desktop/workspace/woosung/commercial-dreambench/scripts/evaluation/log",
        help="Directory to save output CSV summary and plots."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="alignment_sensitivity_summary.csv",
        help="Filename for sensitivity summary CSV."
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["3-step", "refined", "5-step", "8-step", "area-all", "custom"],
        default="3-step",
        help="Predefined distortion ladder preset (default: 3-step)."
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        default=None,
        help="Custom list of noise scale keys (e.g. --scales original 0.875x 0.625x 0.25x)."
    )
    parser.add_argument(
        "--dimension",
        type=str,
        default="Noise Strength",
        help="Name of distortion dimension."
    )
    parser.add_argument(
        "--exclude_file",
        type=str,
        default="/root/Desktop/workspace/woosung/commercial-dreambench/scripts/evaluation/filtering/excluded_keys.json",
        help="Path to JSON file containing list of image keys to exclude from evaluation."
    )
    parser.add_argument(
        "--discordant_file",
        type=str,
        default="discordant_pairs_sensitivity.csv",
        help="Filename for logging discordant pairs in output_dir."
    )
    parser.add_argument(
        "--plot_curves",
        action="store_true",
        default=True,
        help="Generate and save PNG plots for Epsilon and Z-Score sweep curves."
    )
    return parser.parse_args()

def resolve_sequence(args):
    if args.scales:
        seq = []
        mapping = NOISE_AREA_DICT if "area" in args.dimension.lower() else NOISE_STRENGTH_DICT
        for s in args.scales:
            cleaned_s = s.strip().lower()
            if cleaned_s in mapping:
                seq.append(mapping[cleaned_s])
            else:
                seq.append((s, len(seq) + 1))
        return seq

    if args.preset in PRESET_SEQUENCES:
        return PRESET_SEQUENCES[args.preset]

    return PRESET_SEQUENCES["3-step"]

@lru_cache(maxsize=4)
def resolve_compositional_eval_order(data_dir=None, rubrics_dir=None):
    """
    Reconstructs the exact sequence of image keys generated by evaluate_compositional_efficient.py.
    """
    if data_dir is None:
        data_dir = "/root/Desktop/workspace/woosung/commercial-dreambench/assets/data/amzn"
    if rubrics_dir is None:
        rubrics_dir = "/root/Desktop/workspace/woosung/commercial-dreambench/assets/rubrics/amzn"

    if not os.path.exists(data_dir):
        return []

    try:
        categories = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("raw_meta_")
        ])
    except Exception:
        return []

    ordered_keys = []
    for cat in categories:
        meta_json_path = os.path.join(data_dir, cat, "metadata.json")
        if not os.path.exists(meta_json_path):
            continue

        try:
            with open(meta_json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            continue

        for item in metadata:
            curr_asin = item.get("asin")
            rubric_path = os.path.join(rubrics_dir, cat, f"{curr_asin}.json")
            if not os.path.exists(rubric_path):
                continue

            for var_file in item.get("variation_files", []):
                var_rel_path = var_file.get("file", "")
                target_base = os.path.splitext(os.path.basename(var_rel_path))[0]
                img_key = f"{cat}_{curr_asin}_{target_base}"
                ordered_keys.append(img_key)

    return ordered_keys

def parse_rating_json(json_path, data_dir=None, rubrics_dir=None):
    """
    Parses a rating file which may be:
      1. Flat mapping: {image_key: score, ...}
      2. Summary file from evaluate_compositional_efficient.py:
         {"model": ..., "mode": "each_item", "total_pairs_evaluated": N, "scores": [...]}
         In this case, image keys are resolved using the per-item files in the directory
         or ordered keys reconstructed according to evaluate_compositional_efficient.py.
    """
    if not os.path.exists(json_path):
        return {}

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: Failed to read {json_path}: {e}")
        return {}

    # 1. Summary JSON format
    if isinstance(data, dict) and "scores" in data and isinstance(data["scores"], list):
        scores_list = data["scores"]
        dir_path = os.path.dirname(json_path)

        # First, try reading per-item evaluation JSON files in sibling/child folders
        per_item_scores = {}
        if os.path.isdir(dir_path):
            for root_dir, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith(".json") and file != "summary.json":
                        item_json_path = os.path.join(root_dir, file)
                        try:
                            with open(item_json_path, "r", encoding="utf-8") as item_f:
                                item_data = json.load(item_f)
                            cat = item_data.get("category", "")
                            asin = item_data.get("asin", "")
                            target_file = item_data.get("target_file", "")
                            target_base = item_data.get("target_base", "")
                            if not target_base and target_file:
                                target_base = os.path.splitext(os.path.basename(target_file))[0]

                            if cat and asin and target_base:
                                img_key = f"{cat}_{asin}_{target_base}"
                            else:
                                img_key = os.path.splitext(file)[0]

                            score = item_data.get("score")
                            if score is None:
                                pct_score = item_data.get("percentage_score")
                                if pct_score is not None:
                                    score = pct_score / 20.0

                            if score is not None:
                                per_item_scores[img_key] = float(score)
                        except Exception:
                            continue

        if per_item_scores and len(per_item_scores) == len(scores_list):
            return per_item_scores

        # Second, resolve ordering using evaluate_compositional_efficient order
        ordered_keys = resolve_compositional_eval_order(data_dir, rubrics_dir)
        if ordered_keys and len(ordered_keys) == len(scores_list):
            return {k: float(v) for k, v in zip(ordered_keys, scores_list)}

        if per_item_scores:
            return per_item_scores

        # Fallback to indexed keys
        return {f"item_{i}": float(v) for i, v in enumerate(scores_list)}

    # 2. Standard flat mapping: {image_key: score}
    if isinstance(data, dict):
        valid_dict = {}
        for k, v in data.items():
            if isinstance(v, (int, float)):
                valid_dict[k] = float(v)
        return valid_dict

    return {}

def load_all_data(rating_dir, data_dir=None, rubrics_dir=None):
    dataset = {}
    for metric_name in metric_files:
        dataset[metric_name] = {}

    if not os.path.exists(rating_dir):
        return dataset

    for model_dir in os.listdir(rating_dir):
        full_dir = os.path.join(rating_dir, model_dir)
        if not os.path.isdir(full_dir):
            continue

        for metric_name, filename in metric_files.items():
            json_path = os.path.join(full_dir, filename)
            if not os.path.exists(json_path):
                # Search recursively for relative path or basename
                target_base = os.path.basename(filename)
                found_paths = glob.glob(os.path.join(full_dir, "**", target_base), recursive=True)
                for candidate in found_paths:
                    norm_candidate = candidate.replace("\\", "/")
                    norm_filename = filename.replace("\\", "/")
                    if norm_candidate.endswith(norm_filename) or os.path.basename(candidate) == target_base:
                        json_path = candidate
                        break

            if os.path.exists(json_path):
                parsed = parse_rating_json(json_path, data_dir=data_dir, rubrics_dir=rubrics_dir)
                if parsed:
                    dataset[metric_name][model_dir] = parsed

    return dataset

def load_exclusion_file(path):
    if not path or not os.path.exists(path):
        return set(), {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return set(data), {}
        elif isinstance(data, dict):
            ex_keys = set(data.get("excluded_keys", []))
            ex_scales = {k: set(v) for k, v in data.get("excluded_scales", {}).items()}
            return ex_keys, ex_scales
    except Exception as e:
        print(f"Warning: Could not load exclude file {path}: {e}")
    return set(), {}

def compute_cohens_d(group1, group2):
    g1 = np.array(group1, dtype=float)
    g2 = np.array(group2, dtype=float)
    if len(g1) < 2 or len(g2) < 2:
        return np.nan
    n1, n2 = len(g1), len(g2)
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0 or np.isnan(pooled_sd):
        return 0.0
    return (np.mean(g1) - np.mean(g2)) / pooled_sd

def compute_aucc(x_vals, y_vals):
    """Computes Area Under the Concordance Curve normalized by x_max."""
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)
    if len(x) < 2 or (x[-1] - x[0]) == 0:
        return np.nan
    area = np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)
    return area / (x[-1] - x[0])

def evaluate_sequence_with_sweeps(dataset, metric_name, sequence, dimension_name="Noise Strength", excluded_keys=None, excluded_scales=None):
    model_names = [m for m, _ in sequence]

    candidate_keys = set()
    for model_name in model_names:
        candidate_keys.update(dataset[metric_name].get(model_name, {}).keys())

    if excluded_keys:
        candidate_keys = candidate_keys - set(excluded_keys)

    sorted_keys = sorted(list(candidate_keys))
    # Pre-collect all valid image score lists to determine metric variance (sigma_m)
    all_scores_flat = []
    valid_sequences = []

    for img in sorted_keys:
        scores = []
        ranks = []
        ladder_steps = []
        per_step_score = {}

        for idx, (model_name, seq_idx) in enumerate(sequence):
            if excluded_scales and img in excluded_scales and model_name in excluded_scales[img]:
                continue
            if model_name in dataset[metric_name] and img in dataset[metric_name][model_name]:
                val = round(dataset[metric_name][model_name][img], 4)
                scores.append(val)
                ranks.append(len(sequence) - 1 - seq_idx)
                ladder_steps.append((model_name, idx))
                per_step_score[idx] = val
                all_scores_flat.append(val)

        if len(scores) >= 2:
            valid_sequences.append({
                "img": img,
                "scores": scores,
                "ranks": ranks,
                "ladder_steps": ladder_steps,
                "per_step_score": per_step_score
            })

    if not valid_sequences:
        return {
            "num_images": 0,
            "total_pairs": 0,
            "metric_sigma": np.nan,
            "strict_concordance": np.nan,
            "conc_2pct": np.nan,
            "conc_5pct": np.nan,
            "aucc_relative": np.nan,
            "aucc_zscore": np.nan,
            "first_mild_drop_pct_mean": np.nan,
            "total_drop_pct_mean": np.nan,
            "mean_adjacent_drop_pct": np.nan,
            "cohens_d_subtle": np.nan,
            "conc_margin_pct": np.nan,
            "disc_margin_pct": np.nan,
            "margin_ratio": np.nan,
            "tie_rate_mean": np.nan,
            "discordant_rate_mean": np.nan,
            "pearson_r_mean": np.nan,
            "somers_d_mean": np.nan,
            "epsilon_curve_rel": [],
            "zscore_curve_k": [],
            "discordant_records": []
        }

    # Per-metric standard deviation for Z-score normalization
    metric_sigma = np.std(all_scores_flat, ddof=1) if len(all_scores_flat) > 1 else 1.0
    if metric_sigma == 0 or np.isnan(metric_sigma):
        metric_sigma = 1.0

    evaluated_images = len(valid_sequences)
    total_pairs = sum(len(item["scores"]) * (len(item["scores"]) - 1) // 2 for item in valid_sequences)

    # Accumulators for sweep evaluations
    zscore_margin_counts = {k: 0 for k in ZSCORE_THRESHOLDS_K}

    # --- [DIAGNOSTIC] Pairwise decomposition accumulators ---
    pair_stats = {}
    for i_idx in range(len(sequence)):
        for j_idx in range(i_idx + 1, len(sequence)):
            pair_stats[(i_idx, j_idx)] = {
                "total": 0,
                "conc": 0,
                "zscore_counts": {k: 0 for k in ZSCORE_THRESHOLDS_K}
            }

    P_total = 0
    Q_total = 0
    T_total = 0

    step0_scores = []
    step1_scores = []

    first_mild_drops = []
    total_ladder_drops = []
    adjacent_step_drops = []

    kendall_tau_bs = []
    spearman_rhos = []
    pearson_rs = []
    discordant_records = []

    for item in valid_sequences:
        scores = item["scores"]
        ranks = item["ranks"]
        ladder_steps = item["ladder_steps"]
        per_step_score = item["per_step_score"]
        n_pts = len(scores)

        seq_max = max(scores) if scores else 1.0
        denom_drop = seq_max if seq_max > 0 else 1.0

        if 0 in per_step_score and 1 in per_step_score:
            s0, s1 = per_step_score[0], per_step_score[1]
            step0_scores.append(s0)
            step1_scores.append(s1)
            first_mild_drops.append(((s0 - s1) / denom_drop) * 100.0)

        if 0 in per_step_score and (len(sequence) - 1) in per_step_score:
            s0, s_last = per_step_score[0], per_step_score[len(sequence) - 1]
            total_ladder_drops.append(((s0 - s_last) / denom_drop) * 100.0)

        adj_drops = [((scores[i] - scores[i+1]) / denom_drop) * 100.0 for i in range(len(scores) - 1)]
        if adj_drops:
            adjacent_step_drops.append(np.mean(adj_drops))

        # Pairwise iterations
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                s_i, s_j = scores[i], scores[j]
                abs_diff = s_i - s_j

                # --- [DIAGNOSTIC] Pairwise key tracking ---
                pair_key = (ladder_steps[i][1], ladder_steps[j][1])
                if pair_key in pair_stats:
                    pair_stats[pair_key]["total"] += 1

                if s_i > s_j:
                    P_total += 1
                    if pair_key in pair_stats:
                        pair_stats[pair_key]["conc"] += 1

                    # Z-Score sweep check
                    for k in ZSCORE_THRESHOLDS_K:
                        if abs_diff >= (k * metric_sigma):
                            zscore_margin_counts[k] += 1
                            if pair_key in pair_stats:
                                pair_stats[pair_key]["zscore_counts"][k] += 1

                elif s_i < s_j:
                    Q_total += 1
                    step_i_name, idx_i = ladder_steps[i]
                    step_j_name, idx_j = ladder_steps[j]
                    discordant_records.append({
                        "Metric": metric_name,
                        "Dimension": dimension_name,
                        "Image Key": item["img"],
                        "Step 1 (Weaker)": step_i_name,
                        "Rank 1": ranks[i],
                        "Score 1": scores[i],
                        "Step 2 (Stronger)": step_j_name,
                        "Rank 2": ranks[j],
                        "Score 2": scores[j],
                        "Score Diff": scores[j] - scores[i]
                    })
                else:
                    T_total += 1

        tau_b, _ = kendalltau(ranks, scores)
        if not np.isnan(tau_b):
            kendall_tau_bs.append(tau_b)

        rho, _ = spearmanr(ranks, scores)
        if not np.isnan(rho):
            spearman_rhos.append(rho)

        r, _ = pearsonr(ranks, scores)
        if not np.isnan(r):
            pearson_rs.append(r)

    # Compute continuous Z-score sweep curve
    zscore_curve_k = []
    for k in ZSCORE_THRESHOLDS_K:
        conc_pct = (zscore_margin_counts[k] / total_pairs) * 100.0 if total_pairs > 0 else 0.0
        zscore_curve_k.append((k, conc_pct))

    # Compute AUCC (Area Under Concordance Curve)
    x_z = [pt[0] for pt in zscore_curve_k]
    y_z = [pt[1] for pt in zscore_curve_k]
    aucc_z = compute_aucc(x_z, y_z)

    # --- [DIAGNOSTIC] Compute per-pair AUCC and Concordance ---
    pair_decomposition = {}
    for pair_key, pdata in pair_stats.items():
        p_tot = pdata["total"]
        p_conc = (pdata["conc"] / p_tot * 100.0) if p_tot > 0 else 0.0
        p_z_curve = []
        for k in ZSCORE_THRESHOLDS_K:
            pct = (pdata["zscore_counts"][k] / p_tot * 100.0) if p_tot > 0 else 0.0
            p_z_curve.append((k, pct))
        p_aucc = compute_aucc([p[0] for p in p_z_curve], [p[1] for p in p_z_curve])
        pair_decomposition[pair_key] = {
            "conc": float(p_conc),
            "aucc": float(p_aucc)
        }

    strict_conc = (P_total / total_pairs) * 100.0 if total_pairs > 0 else 0.0
    disc_rate = (Q_total / total_pairs) * 100.0 if total_pairs > 0 else 0.0
    tie_rate = (T_total / total_pairs) * 100.0 if total_pairs > 0 else 0.0
    somers_d = (P_total - Q_total) / (P_total + Q_total + T_total) if total_pairs > 0 else 0.0

    cohens_d = compute_cohens_d(step0_scores, step1_scores)

    return {
        "num_images": evaluated_images,
        "total_pairs": total_pairs,
        "metric_sigma": float(metric_sigma),
        "strict_concordance": float(strict_conc),
        "aucc_zscore": float(aucc_z),
        "first_mild_drop_pct_mean": float(np.mean(first_mild_drops)) if first_mild_drops else np.nan,
        "total_drop_pct_mean": float(np.mean(total_ladder_drops)) if total_ladder_drops else np.nan,
        "mean_adjacent_drop_pct": float(np.mean(adjacent_step_drops)) if adjacent_step_drops else np.nan,
        "cohens_d_subtle": float(cohens_d),
        "tie_rate_mean": float(tie_rate),
        "discordant_rate_mean": float(disc_rate),
        "pearson_r_mean": float(np.mean(pearson_rs)) if pearson_rs else np.nan,
        "somers_d_mean": float(somers_d),
        "zscore_curve_k": zscore_curve_k,
        "pair_decomposition": pair_decomposition,
        "discordant_records": discordant_records
    }

def generate_publication_plots(curve_records_z, output_dir, dimension_name):
    """Generates clean publication-ready matplotlib comparison plots for Z-score sweep."""
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    color_map = {
        "CLIP": "#7f7f7f",
        "CLIP_bbox-crop": "#1f77b4",
        "DINO": "#bcbd22",
        "DINO_bbox-crop": "#2ca02c",
        "DreamBench++": "#8c564b",
        "VIEScore": "#e377c2",
        "keypoints": "#ff7f0e",
        "Ours-GLM-4.6V-Flash_bbox-crop": "#9467bd",
        "Qwen3-VL-8B_bbox-crop": "#d62728",
        "Ours-Qwen3-VL-8B-Instruct_bbox-crop": "#d62728",
        "Ours-Qwen3-VL-32B-Instruct_image-reversed_bbox-crop": "#8c2d04",
        "Ours_each_item": "#9467bd",
    }

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral", "Liberation Serif"],
        "mathtext.fontset": "stix",       # STIX provides Times New Roman-compatible math styling on Linux
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
    })

    fig, ax = plt.subplots(figsize=(4, 5), dpi=300)
    for metric_name, pts in curve_records_z.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        color = color_map.get(metric_name, None)
        lw = 2.5 if "Ours" in metric_name or "Qwen3" in metric_name else 1.8
        ls = "-" if "crop" in metric_name.lower() or "Ours" in metric_name else "--"
        # label = metric_name.replace("_bbox-crop", " (Crop)").replace("Ours-", "Ours: ")
        label = "MetMag(Ours)" if "Ours" in metric_name else metric_name.replace("_bbox-crop", " (Crop)").replace("Ours-", "Ours: ")
        ax.plot(xs, ys, label=label, color=color, linewidth=lw, linestyle=ls, marker="s" if "Ours" in metric_name else None, markersize=4)

    # ax.set_title(f"Scale-Normalized Stability Curve ({dimension_name})\nConcordance vs. Z-Score Margin Threshold ($k\\cdot\\sigma$)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Threshold $k$", fontsize=14, fontweight="normal")
    ax.set_ylabel("Pairwise Accuracy (%)", fontsize=14, fontweight="normal")
    ax.set_xlim(0, 0.52)
    ax.set_ylim(0, 100)
    ax.axhline(50, color="gray", linestyle=":", linewidth=1.2)
    handles, labels = ax.get_legend_handles_labels()
    ncols = (len(labels) + 1) // 3
    remainder = len(labels) % ncols
    if remainder != 0:
        n_missing = ncols - remainder
        blank_handle = mpatches.Rectangle((0, 0), 0, 0, fill=False, edgecolor="none", visible=False)
        handles = handles[:-remainder] + [blank_handle] * n_missing + handles[-remainder:]
        labels = labels[:-remainder] + [""] * n_missing + labels[-remainder:]
    ax.legend(handles, labels, fontsize=10, loc="upper center", framealpha=0.5, ncols=ncols)

    plot_path_z = os.path.join(output_dir, "fig_controlled_distortion_AUC.pdf")
    fig.savefig(plot_path_z, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Z-Score Sweep Plot: {plot_path_z}")

def main():
    args = parse_args()
    sequence = resolve_sequence(args)

    print(f"Loading data from: {args.rating_dir}")
    print(f"Active Sequence ({len(sequence)} steps): {[m for m, _ in sequence]}")

    dataset = load_all_data(args.rating_dir, data_dir=args.data_dir, rubrics_dir=args.rubrics_dir)
    excluded_keys, excluded_scales = load_exclusion_file(args.exclude_file)
    print(f"Loaded {len(excluded_keys)} excluded keys, {len(excluded_scales)} items with dropped scales.")

    results = []
    all_discordant_records = []
    curve_records_z = {}
    dense_curve_rows = []
    pairwise_records = {}

    # Strictly evaluate only metrics specified in metric_files
    metrics = [m for m in metric_files.keys() if m in dataset and len(dataset[m]) > 0]

    for m in metrics:
        res = evaluate_sequence_with_sweeps(
            dataset, m, sequence,
            dimension_name=args.dimension,
            excluded_keys=excluded_keys,
            excluded_scales=excluded_scales
        )

        all_discordant_records.extend(res.get("discordant_records", []))
        curve_records_z[m] = res["zscore_curve_k"]
        pairwise_records[m] = res.get("pair_decomposition", {})

        for k, conc in res["zscore_curve_k"]:
            dense_curve_rows.append({
                "Metric": m,
                "Threshold_Type": "ZScore_k_sigma",
                "Threshold": k,
                "Concordance %": conc
            })

        results.append({
            "Metric": m,
            # "Images": res["num_images"],
            # "Pairs": res["total_pairs"],
            "Pairwise Accuracy %": res["strict_concordance"],
            "Tie Rate %": res["tie_rate_mean"],
            "AUC_z": res["aucc_zscore"],
        })

    df = pd.DataFrame(results)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print("\n" + "="*120)
    print(f"RANK ALIGNMENT & PERTURBATION STABILITY (AUCC SUMMARY - {len(sequence)} steps)")
    print("="*120)
    print(df.to_string(index=False))

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, args.output_csv)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved Sensitivity Summary CSV: {csv_path}")

    # --- [DIAGNOSTIC] Output Pairwise Decomposition Table ---
    if len(sequence) == 3:
        step0_name = sequence[0][0].split("_")[0]
        step1_name = sequence[1][0].split("_")[0]
        step2_name = sequence[2][0].split("_")[0]

        col_om_conc = f"Acc. ({step0_name}->{step1_name}) %"
        col_om_aucc = f"AUC_z ({step0_name}->{step1_name})"
        col_ms_conc = f"Acc. ({step1_name}->{step2_name}) %"
        col_ms_aucc = f"AUC_z ({step1_name}->{step2_name})"
        col_os_conc = f"Acc. ({step0_name}->{step2_name}) %"
        col_os_aucc = f"AUC_z ({step0_name}->{step2_name})"

        decomp_rows = []
        for m in metrics:
            pdecomp = pairwise_records.get(m, {})
            om = pdecomp.get((0, 1), {"conc": np.nan, "aucc": np.nan})
            ms = pdecomp.get((1, 2), {"conc": np.nan, "aucc": np.nan})
            os_pair = pdecomp.get((0, 2), {"conc": np.nan, "aucc": np.nan})

            decomp_rows.append({
                "Metric": m,
                col_om_conc: om["conc"],
                col_om_aucc: om["aucc"],
                col_ms_conc: ms["conc"],
                col_ms_aucc: ms["aucc"],
                col_os_conc: os_pair["conc"],
                col_os_aucc: os_pair["aucc"],
            })

        df_decomp = pd.DataFrame(decomp_rows)
        print("\n" + "="*120)
        print("DIAGNOSTIC: PAIRWISE DECOMPOSITION SUMMARY (Subtle vs. Moderate vs. Severe)")
        print("="*120)
        print(df_decomp.to_string(index=False))

        decomp_csv_path = os.path.join(args.output_dir, "pairwise_decomposition_summary.csv")
        df_decomp.to_csv(decomp_csv_path, index=False)
        print(f"Saved Pairwise Decomposition CSV: {decomp_csv_path}")

    # Save dense curve CSV
    curve_csv_path = os.path.join(args.output_dir, "epsilon_sweep_curves.csv")
    pd.DataFrame(dense_curve_rows).to_csv(curve_csv_path, index=False)
    print(f"Saved Dense Sweep Curves CSV: {curve_csv_path}")

    if args.plot_curves:
        generate_publication_plots(curve_records_z, args.output_dir, args.dimension)

    if args.discordant_file:
        discordant_csv_path = os.path.join(args.output_dir, args.discordant_file)
        if all_discordant_records:
            discordant_df = pd.DataFrame(all_discordant_records)
            discordant_df.to_csv(discordant_csv_path, index=False)
            print(f"Saved Discordant Pairs CSV ({len(discordant_df)} pairs): {discordant_csv_path}")

if __name__ == "__main__":
    main()
