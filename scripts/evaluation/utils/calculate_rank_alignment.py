#!/usr/bin/env python3
"""
calculate_rank_alignment.py

Calculates Kendall's tau-b, Kendall's tau-a, Spearman's rho, Pearson's r,
Standard Concordance Rate (%), Strict Concordance Rate (%), and Somers' D
for evaluation metrics across reference image rating datasets.

Usage:
    python3 calculate_rank_alignment.py \
        --rating_dir /path/to/rating/directory \
        --output_dir /path/to/output_directory
"""

import argparse
import json
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, pearsonr
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate rank alignment statistics and concordance rates.")
    parser.add_argument(
        "--rating_dir",
        type=str,
        default="/root/Desktop/workspace/woosung/commercial-dreambench/rating/amzn_distorted/qwen-image-edit",
        help="Path to rating directory containing model subfolders with JSON evaluation results."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/Desktop/workspace/woosung/commercial-dreambench/scripts/evaluation/log",
        help="Directory to save output CSV summary and correlation bar plots."
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
        default="discordant_pairs.csv",
        help="Filename for logging discordant pairs in output_dir (default: discordant_pairs.csv)."
    )
    return parser.parse_args()

metric_files = {
    "CLIP": "clip_results.json",
    "DINO": "dino_results.json",
    "DreamBench++": "dreambench_plus_results.json",
    "VIEScore": "viescore_results.json",
    "Qwen-Reranker": "qwen_reranker_results.json",
    "CLIP_bbox-crop": "clip_bbox-crop_results.json",
    "DINO_bbox-crop": "dino_bbox-crop_results.json",
    "Qwen-Reranker_bbox-crop": "qwen_reranker_bbox-crop_results.json",
    "Ours_bbox-crop": "ours_bbox-crop_results.json",
    # "Visual-Likert-Scale_Qwen-Reranker_bbox-crop": "visual-likert-scale_qwen_reranker_bbox-crop_results.json",
    # "Visual-Likert-Scale_Ref-Distorted_Qwen-Reranker_bbox-crop": "visual-likert-scale_ref-distorted_qwen_reranker_bbox-crop_results.json",
    # "Visual-Likert-Scale_Crop-Distorted_Qwen-Reranker_bbox-crop": "visual-likert-scale_crop-distorted_qwen_reranker_bbox-crop_results.json"
}

# Sequences ordered from WEAKEST noise (expected score high) to STRONGEST noise (expected score low)
noise_strength_seq = [
    ("original", 0),
    # ("noised-1.0x_masked-dino-adaptive_height-ratio-1.0_timestep-10", 1),
    # ("noised-0.875x_masked-dino-adaptive_height-ratio-1.0_timestep-10", 2),
    # ("noised-0.75x_masked-dino-adaptive_height-ratio-1.0_timestep-10", 3),
    ("noised-0.625x_masked-dino-adaptive_height-ratio-1.0_timestep-10", 4),
    # ("noised-0.5x_masked-dino-adaptive_height-ratio-1.0_timestep-10", 5),
    # ("noised-0.375x_masked-dino-adaptive_height-ratio-1.0_timestep-10", 6),
    ("noised-0.25x_masked-dino-adaptive_height-ratio-1.0_timestep-10", 7),
]

noise_area_seq = [
    # ("original", 0),
    # ("noised-0.5x_masked-dino-adaptive_height-ratio-0.25_timestep-10", 1),
    # ("noised-0.5x_masked-dino-adaptive_height-ratio-0.375_timestep-10", 2),
    # ("noised-0.5x_masked-dino-adaptive_height-ratio-0.5_timestep-10", 3),
    # ("noised-0.5x_masked-dino-adaptive_height-ratio-0.625_timestep-10", 4),
    # ("noised-0.5x_masked-dino-adaptive_height-ratio-0.75_timestep-10", 5),
    # ("noised-0.5x_masked-dino-adaptive_height-ratio-0.875_timestep-10", 6),
    # ("noised-0.5x_masked-dino-adaptive_height-ratio-1.0_timestep-10", 7),
]

def load_all_data(rating_dir):
    dataset = {}
    for metric_name in metric_files:
        dataset[metric_name] = {}

    for model_dir in os.listdir(rating_dir):
        full_dir = os.path.join(rating_dir, model_dir)
        if not os.path.isdir(full_dir):
            continue

        # 1. Load standard metrics (clip_results.json, dino_results.json, etc.)
        for metric_name, filename in metric_files.items():
            json_path = os.path.join(full_dir, filename)
            if not os.path.exists(json_path):
                # Search under subfolders (e.g. ours/{mode}/{filename})
                found_paths = glob.glob(os.path.join(full_dir, "**", filename), recursive=True)
                if found_paths:
                    json_path = found_paths[0]

            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    dataset[metric_name][model_dir] = json.load(f)

        # 1b. Load model-specific 'ours' results (ours-{judge_model}_bbox-crop_results.json)
        for fname in os.listdir(full_dir):
            if fname.startswith("ours-") and fname.endswith("_bbox-crop_results.json"):
                base_tag = fname[:-len("_results.json")]
                metric_key = base_tag[0].upper() + base_tag[1:]
                if metric_key not in dataset:
                    dataset[metric_key] = {}
                try:
                    with open(os.path.join(full_dir, fname), "r", encoding="utf-8") as f:
                        dataset[metric_key][model_dir] = json.load(f)
                except Exception as e:
                    print(f"[Warning] Failed to load {fname} in {full_dir}: {e}")

        # 2. Load nested 'ours' structure: rating/{model}/ours/{mode}/{category}/*.json
        for sub_name in os.listdir(full_dir):
            if not sub_name.startswith("ours"):
                continue
            ours_path = os.path.join(full_dir, sub_name)
            if not os.path.isdir(ours_path):
                continue

            for mode in os.listdir(ours_path):
                mode_path = os.path.join(ours_path, mode)
                if not os.path.isdir(mode_path):
                    continue

                metric_name = f"{sub_name} ({mode})"
                if metric_name not in dataset:
                    dataset[metric_name] = {}
                if model_dir not in dataset[metric_name]:
                    dataset[metric_name][model_dir] = {}

                # Walk through per-item evaluation JSON files
                for root_dir, _, files in os.walk(mode_path):
                    for file in files:
                        if file.endswith(".json") and file != "summary.json":
                            item_json_path = os.path.join(root_dir, file)
                            try:
                                with open(item_json_path, "r", encoding="utf-8") as f:
                                    item_data = json.load(f)

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

                                score = item_data.get("percentage_score")
                                if score is None:
                                    score = item_data.get("score")

                                if score is not None:
                                    dataset[metric_name][model_dir][img_key] = score
                            except Exception:
                                continue

    return dataset

def load_exclusion_file(path):
    """Loads exclusions supporting whole-sequence keys and granular scale drops."""
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

def calc_pair_stats(scores):
    n = len(scores)
    total_pairs = n * (n - 1) // 2
    P = 0
    Q = 0
    T_Y = 0

    for i in range(n):
        for j in range(i + 1, n):
            if scores[i] > scores[j]:
                P += 1
            elif scores[i] < scores[j]:
                Q += 1
            else:
                T_Y += 1

    tau_a = (P - Q) / total_pairs if total_pairs > 0 else 0
    standard_concordance = ((P + 0.5 * T_Y) / total_pairs) * 100.0 if total_pairs > 0 else 0
    strict_concordance = (P / total_pairs) * 100.0 if total_pairs > 0 else 0
    discordant_rate = (Q / total_pairs) * 100.0 if total_pairs > 0 else 0
    tie_rate = (T_Y / total_pairs) * 100.0 if total_pairs > 0 else 0
    somers_d = (P - Q) / (P + Q + T_Y) if (P + Q + T_Y) > 0 else 0

    return tau_a, standard_concordance, strict_concordance, discordant_rate, tie_rate, somers_d

def calc_strict_monotonicity(scores):
    if len(scores) < 2:
        return False
    return all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

def evaluate_sequence(dataset, metric_name, sequence, dimension_name="Noise Strength", excluded_keys=None, excluded_scales=None):
    model_names = [m for m, _ in sequence]

    # Collect candidate image keys across sequence models
    candidate_keys = set()
    for model_name in model_names:
        candidate_keys.update(dataset[metric_name].get(model_name, {}).keys())

    if excluded_keys:
        candidate_keys = candidate_keys - set(excluded_keys)

    sorted_keys = sorted(list(candidate_keys))

    kendall_tau_bs = []
    kendall_tau_as = []
    spearman_rhos = []
    pearson_rs = []
    std_concordances = []
    strict_concordances = []
    discordant_rates = []
    tie_rates = []
    somers_ds = []
    strict_monotonic_flags = []
    discordant_records = []
    evaluated_images = 0
    total_pairs = 0

    for img in sorted_keys:
        scores = []
        ranks = []
        ladder_steps = []
        for model_name, idx in sequence:
            # Skip if specific scale is marked as dropped for this image
            if excluded_scales and img in excluded_scales and model_name in excluded_scales[img]:
                continue
            if model_name in dataset[metric_name] and img in dataset[metric_name][model_name]:
                scores.append(round(dataset[metric_name][model_name][img], 4))
                # Ground truth rank: higher index in noise_strength_seq means stronger noise -> lower expected score rank
                ranks.append(len(sequence) - 1 - idx)
                ladder_steps.append((model_name, idx))

        # Require at least 2 valid points in the ladder to evaluate alignment
        if len(scores) < 2:
            continue

        evaluated_images += 1
        n_pts = len(scores)
        total_pairs += n_pts * (n_pts - 1) // 2

        # Extract discordant pairs (where weaker noise scored lower than stronger noise)
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                if scores[i] < scores[j]:
                    step_i_name, idx_i = ladder_steps[i]
                    step_j_name, idx_j = ladder_steps[j]
                    discordant_records.append({
                        "Metric": metric_name,
                        "Dimension": dimension_name,
                        "Image Key": img,
                        "Step 1 (Weaker)": step_i_name,
                        "Rank 1": ranks[i],
                        "Score 1": scores[i],
                        "Step 2 (Stronger)": step_j_name,
                        "Rank 2": ranks[j],
                        "Score 2": scores[j],
                        "Score Diff (Stronger - Weaker)": scores[j] - scores[i]
                    })

        tau_b, _ = kendalltau(ranks, scores)
        if not np.isnan(tau_b):
            kendall_tau_bs.append(tau_b)

        rho, _ = spearmanr(ranks, scores)
        if not np.isnan(rho):
            spearman_rhos.append(rho)

        r, _ = pearsonr(ranks, scores)
        if not np.isnan(r):
            pearson_rs.append(r)

        tau_a, std_conc, strict_conc, disc_rate, t_rate, somers_d = calc_pair_stats(scores)
        kendall_tau_as.append(tau_a)
        std_concordances.append(std_conc)
        strict_concordances.append(strict_conc)
        discordant_rates.append(disc_rate)
        tie_rates.append(t_rate)
        somers_ds.append(somers_d)
        strict_monotonic_flags.append(calc_strict_monotonicity(scores))

    if evaluated_images == 0:
        return {
            "num_images": 0,
            "total_pairs": 0,
            "kendall_tau_b_mean": np.nan,
            "kendall_tau_a_mean": np.nan,
            "spearman_rho_mean": np.nan,
            "pearson_r_mean": np.nan,
            "std_concordance_mean": np.nan,
            "strict_concordance_mean": np.nan,
            "discordant_rate_mean": np.nan,
            "tie_rate_mean": np.nan,
            "somers_d_mean": np.nan,
            "strict_monotonic_pct": np.nan
        }

    return {
        "num_images": evaluated_images,
        "total_pairs": total_pairs,
        "kendall_tau_b_mean": np.mean(kendall_tau_bs) if kendall_tau_bs else np.nan,
        "kendall_tau_a_mean": np.mean(kendall_tau_as) if kendall_tau_as else np.nan,
        "spearman_rho_mean": np.mean(spearman_rhos) if spearman_rhos else np.nan,
        "pearson_r_mean": np.mean(pearson_rs) if pearson_rs else np.nan,
        "std_concordance_mean": np.mean(std_concordances) if std_concordances else np.nan,
        "strict_concordance_mean": np.mean(strict_concordances) if strict_concordances else np.nan,
        "discordant_rate_mean": np.mean(discordant_rates) if discordant_rates else np.nan,
        "tie_rate_mean": np.mean(tie_rates) if tie_rates else np.nan,
        "somers_d_mean": np.mean(somers_ds) if somers_ds else np.nan,
        "strict_monotonic_pct": (np.mean(strict_monotonic_flags) * 100.0) if strict_monotonic_flags else np.nan,
        "discordant_records": discordant_records
    }

def main():
    args = parse_args()
    if not os.path.exists(args.rating_dir):
        raise FileNotFoundError(f"Rating directory not found: {args.rating_dir}")

    dataset = load_all_data(args.rating_dir)
    metrics = [m for m, model_dict in dataset.items() if any(len(scores) > 0 for scores in model_dict.values())]

    print("=" * 110)
    print(" COMPREHENSIVE EVALUATION ALIGNMENT REPORT (TIE-ADJUSTED VS TIE-PENALIZED METRICS)")
    print("=" * 110)

    results = []

    exclude_keys = set()
    if args.exclude_file and os.path.exists(args.exclude_file):
        try:
            with open(args.exclude_file, 'r', encoding='utf-8') as f:
                exclude_keys = set(json.load(f))
            print(f'Loaded {len(exclude_keys)} excluded keys from {args.exclude_file}')
        except Exception as e:
            print(f'Warning: Could not load exclude file {args.exclude_file}: {e}')

    excluded_keys, excluded_scales = load_exclusion_file(args.exclude_file)
    if excluded_keys or excluded_scales:
        dropped_img_cnt = sum(len(v) for v in excluded_scales.values())
        print(f'Loaded exclusions: {len(excluded_keys)} whole keys, {dropped_img_cnt} dropped images.')

    all_discordant_records = []

    for m in metrics:
        res_strength = evaluate_sequence(dataset, m, noise_strength_seq, dimension_name="Noise Strength", excluded_keys=excluded_keys, excluded_scales=excluded_scales)
        # res_area = evaluate_sequence(dataset, m, noise_area_seq, dimension_name="Noise Area", excluded_keys=excluded_keys, excluded_scales=excluded_scales)

        all_discordant_records.extend(res_strength.get("discordant_records", []))
        # all_discordant_records.extend(res_area.get("discordant_records", []))

        results.append({
            "Metric": m,
            "Dimension": "Noise Strength",
            "Images": res_strength["num_images"],
            "Pairs": res_strength.get("total_pairs", 0),
            "Strict Concordance %": res_strength["strict_concordance_mean"],
            "Discordant Rate %": res_strength["discordant_rate_mean"],
            "Tie Rate %": res_strength["tie_rate_mean"],
            # # "Kendall Tau-a (Penalized)": res_strength["kendall_tau_a_mean"],
            # # "Kendall Tau-b (Standard)": res_strength["kendall_tau_b_mean"],
            # # "Spearman Rho": res_strength["spearman_rho_mean"],
            # # "Pearson R": res_strength["pearson_r_mean"],
            # # "Somers D": res_strength["somers_d_mean"],
            # "Std Concordance %": res_strength["std_concordance_mean"],
            # "Strict Monotonic %": res_strength["strict_monotonic_pct"]
        })

        # results.append({
        #     "Metric": m,
        #     "Dimension": "Noise Area",
        #     "Images": res_area["num_images"],
        #     "Pairs": res_area.get("total_pairs", 0),
        #     "Strict Concordance %": res_area["strict_concordance_mean"],
        #     "Discordant Rate %": res_area["discordant_rate_mean"],
        #     "Tie Rate %": res_area["tie_rate_mean"],
        #     "Kendall Tau-a (Penalized)": res_area["kendall_tau_a_mean"],
        #     "Kendall Tau-b (Standard)": res_area["kendall_tau_b_mean"],
        #     "Spearman Rho": res_area["spearman_rho_mean"],
        #     "Pearson R": res_area["pearson_r_mean"],
        #     "Somers D": res_area["somers_d_mean"],
        #     "Std Concordance %": res_area["std_concordance_mean"],
        #     "Strict Monotonic %": res_area["strict_monotonic_pct"]
        # })

    df = pd.DataFrame(results)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df.to_string(index=False))

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "alignment_analysis_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    if args.discordant_file:
        discordant_csv_path = os.path.join(args.output_dir, args.discordant_file)
        if all_discordant_records:
            discordant_df = pd.DataFrame(all_discordant_records)
            discordant_df.to_csv(discordant_csv_path, index=False)
            print(f"Saved Discordant Pairs CSV ({len(discordant_df)} pairs): {discordant_csv_path}")
        else:
            print(f"No discordant pairs found to log to {discordant_csv_path}.")

    # # Generate Comparison Plot
    # fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    # dimensions = ["Noise Strength"]
    # x = np.arange(len(metrics))
    # width = 0.15

    # for idx, dim in enumerate(dimensions):
    #     ax = axes[idx]
    #     sub_df = df[df["Dimension"] == dim]

    #     # tau_bs = sub_df["Kendall Tau-b (Standard)"].values
    #     # tau_as = sub_df["Kendall Tau-a (Penalized)"].values
    #     # std_concs = sub_df["Std Concordance %"].values / 100.0
    #     # strict_concs = sub_df["Strict Concordance %"].values / 100.0

    #     # rects1 = ax.bar(x - 1.5*width, tau_bs, width, label="Kendall Tau-b (Standard)", color="#1f77b4")
    #     # rects2 = ax.bar(x - 0.5*width, tau_as, width, label="Kendall Tau-a (Tie-Penalized)", color="#ff7f0e")
    #     # rects3 = ax.bar(x + 0.5*width, std_concs, width, label="Std Concordance (0-1)", color="#2ca02c")
    #     # rects4 = ax.bar(x + 1.5*width, strict_concs, width, label="Strict Concordance (0-1)", color="#d62728")

    #     ax.set_title(f"Tie-Penalization Impact ({dim})", fontsize=14, fontweight="bold", pad=12)
    #     ax.set_ylabel("Metric Value", fontsize=11, fontweight="bold")
    #     ax.set_xticks(x)
    #     ax.set_xticklabels(metrics, fontsize=9, fontweight="bold", rotation=15)
    #     ax.set_ylim(-0.1, 1.15)
    #     ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    #     ax.grid(True, linestyle="--", alpha=0.5)
    #     ax.legend(fontsize=9, loc="upper left")

    #     for rect in rects1 + rects2 + rects3 + rects4:
    #         h = rect.get_height()
    #         if not np.isnan(h):
    #             ax.annotate(f"{h:.2f}",
    #                         xy=(rect.get_x() + rect.get_width() / 2, h),
    #                         xytext=(0, 3 if h >= 0 else -10),
    #                         textcoords="offset points",
    #                         ha="center", va="bottom", fontsize=7, fontweight="bold")

    # plt.tight_layout()
    # plot_path = os.path.join(args.output_dir, "tie_penalized_metrics_comparison.png")
    # fig.savefig(plot_path, bbox_inches="tight")
    # plt.close(fig)
    # print(f"Saved Plot: {plot_path}")

if __name__ == "__main__":
    main()
