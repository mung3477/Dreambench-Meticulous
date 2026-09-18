#!/usr/bin/env python3
"""
calculate_model_rank.py

Evaluates whether evaluation metrics (e.g., CLIP, DINO, Qwen-Reranker, Ours)
align with model enhancement history (e.g., Flux.1 -> Flux.2, or SD1.5 -> SDXL -> Flux.1)
both at the macro-average level and the paired per-sample (image-by-image and crop-by-crop) level.

Usage Examples:
    # 1. Pairwise head-to-head comparison (e.g. Flux.1 vs Flux.2):
    python3 calculate_model_rank.py \
        --rating_dir /path/to/rating \
        --baseline_model Flux.1 \
        --target_model Flux.2 \
        --eval_mode all \
        --save_pairwise_details

    # 2. Ordered sequence evaluation (chronological / quality progression):
    python3 calculate_model_rank.py \
        --rating_dir /path/to/rating \
        --models SD-1.5 SDXL Flux.1 Flux.2 \
        --metrics CLIP DINO Qwen-Reranker

    # 3. Crop-level metrics only (e.g. bbox-crop results):
    python3 calculate_model_rank.py \
        --rating_dir /path/to/rating \
        --baseline_model Flux.1 \
        --target_model Flux.2 \
        --eval_mode crop

    # 4. Multi-family sequence configuration via JSON:
    python3 calculate_model_rank.py \
        --rating_dir /path/to/rating \
        --rank_config progression_families.json \
        --output_dir /path/to/output
"""

import os
import sys
import glob
import json
import argparse
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_METRIC_FILES = {
    # Full image level
    "CLIP": "clip_results.json",
    "DINO": "dino_results.json",
    "DreamBench++": "dreambench_plus_results.json",
    "VIEScore": "viescore_results.json",
    "Qwen-Reranker": "qwen_reranker_results.json",
    # Crop / BBox level
    "CLIP_bbox-crop": "clip_bbox-crop_results.json",
    "DINO_bbox-crop": "dino_bbox-crop_results.json",
    "Qwen-Reranker_bbox-crop": "qwen_reranker_bbox-crop_results.json",
    "Ours_bbox-crop": "ours_bbox-crop_results.json",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check if metric scores align with model enhancement history at average and sample/crop level."
    )
    parser.add_argument(
        "--rating_dir",
        type=str,
        required=True,
        help="Path to rating directory containing model subfolders with JSON evaluation results."
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Ordered list of models from lowest to highest expected quality (e.g., --models Flux.1 Flux.2)."
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        default=None,
        help="Baseline model for pairwise head-to-head comparison."
    )
    parser.add_argument(
        "--target_model",
        type=str,
        default=None,
        help="Target (improved) model for pairwise head-to-head comparison."
    )
    parser.add_argument(
        "--rank_config",
        type=str,
        default=None,
        help="Path to JSON file containing model progression sequences, e.g. {'flux': ['Flux.1', 'Flux.2']}."
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Subset of metrics to evaluate. If not specified, evaluates all found metrics."
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        choices=["all", "image", "crop"],
        default="all",
        help="Evaluation scope: 'image' for full-image metrics, 'crop' for bbox-crop metrics, 'all' for both."
    )
    parser.add_argument(
        "--exclude_file",
        type=str,
        default=None,
        help="Path to JSON file containing list of sample keys to exclude."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save summary CSV and JSON reports (defaults to rating_dir/model_rank_summary)."
    )
    parser.add_argument(
        "--save_pairwise_details",
        action="store_true",
        help="Save detailed per-sample comparison deltas to CSV in output_dir."
    )
    return parser.parse_args()


# ==============================================================================
# 1. Data Ingestion (Image-level and Crop-level)
# ==============================================================================

def load_exclusion_file(path: Optional[str]) -> set:
    if not path or not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return set(data)
        elif isinstance(data, dict):
            return set(data.get("excluded_keys", []))
    except Exception as e:
        print(f"[Warning] Could not load exclude file {path}: {e}")
    return set()


def get_available_metric_map(eval_mode: str = "all") -> Dict[str, str]:
    if eval_mode == "image":
        return {k: v for k, v in DEFAULT_METRIC_FILES.items() if not k.endswith("_bbox-crop")}
    elif eval_mode == "crop":
        return {k: v for k, v in DEFAULT_METRIC_FILES.items() if k.endswith("_bbox-crop")}
    return dict(DEFAULT_METRIC_FILES)


def load_model_scores(rating_dir: str, model_name: str, metric_map: Dict[str, str], excluded_keys: set) -> Dict[str, Dict[str, float]]:
    """
    Loads all metric scores for a given model directory.
    Returns: {metric_name: {sample_key: score}}
    """
    model_dir = os.path.join(rating_dir, model_name)
    if not os.path.isdir(model_dir):
        return {}

    scores_by_metric: Dict[str, Dict[str, float]] = {}

    # 1. Load standard flat JSON results
    for metric_name, filename in metric_map.items():
        json_path = os.path.join(model_dir, filename)
        if not os.path.exists(json_path):
            found = glob.glob(os.path.join(model_dir, "**", filename), recursive=True)
            if found:
                json_path = found[0]

        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)

                scores = {}
                for k, v in raw_data.items():
                    if k in ["overall_mean", "mean", "std", "count"]:
                        continue
                    if k in excluded_keys:
                        continue
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        scores[k] = float(v)
                    elif isinstance(v, dict):
                        # Some nested ratings have {'score': float} or {'percentage_score': float}
                        val = v.get("percentage_score", v.get("score"))
                        if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
                            scores[k] = float(val)

                if scores:
                    scores_by_metric[metric_name] = scores
            except Exception as e:
                print(f"[Warning] Failed to load {json_path}: {e}")

    # 2. Check for model-specific ours-judge files: ours-{judge}_bbox-crop_results.json
    if eval_mode_allows_crop(metric_map):
        for fname in os.listdir(model_dir):
            if fname.startswith("ours-") and fname.endswith("_bbox-crop_results.json"):
                base_tag = fname[:-len("_results.json")]
                metric_key = base_tag[0].upper() + base_tag[1:]
                fpath = os.path.join(model_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        raw_data = json.load(f)
                    scores = {
                        k: float(v) for k, v in raw_data.items()
                        if k not in ["overall_mean", "mean"] and k not in excluded_keys and isinstance(v, (int, float))
                    }
                    if scores:
                        scores_by_metric[metric_key] = scores
                except Exception as e:
                    print(f"[Warning] Failed to load {fname}: {e}")

    # 3. Check for nested ours structure: rating/{model}/ours/{mode}/{category}/*.json
    if eval_mode_allows_crop(metric_map):
        for sub_name in os.listdir(model_dir):
            if not sub_name.startswith("ours"):
                continue
            ours_path = os.path.join(model_dir, sub_name)
            if not os.path.isdir(ours_path):
                continue

            for mode in os.listdir(ours_path):
                mode_path = os.path.join(ours_path, mode)
                if not os.path.isdir(mode_path):
                    continue

                metric_name = f"{sub_name} ({mode})"
                scores = {}
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

                                if img_key in excluded_keys:
                                    continue

                                score = item_data.get("percentage_score", item_data.get("score"))
                                if score is not None and isinstance(score, (int, float)):
                                    scores[img_key] = float(score)
                            except Exception:
                                continue
                if scores:
                    scores_by_metric[metric_name] = scores

    return scores_by_metric


def eval_mode_allows_crop(metric_map: Dict[str, str]) -> bool:
    return any(k.endswith("_bbox-crop") or "crop" in k.lower() or "ours" in k.lower() for k in metric_map)


# ==============================================================================
# 2. Rank Alignment Statistics
# ==============================================================================

def compute_pairwise_stats(scores_base: List[float], scores_target: List[float]) -> Dict[str, Any]:
    """
    Computes strict concordant, tie, and discordant rates assuming target > base in GT ranking.
    - Strict Concordant: target > base
    - Tie: target == base
    - Discordant: target < base
    """
    n = len(scores_base)
    if n == 0:
        return {}

    strict_concordant = 0  # Target > Base
    discordant = 0         # Target < Base
    ties = 0               # Target == Base
    deltas = []

    for b, t in zip(scores_base, scores_target):
        deltas.append(t - b)
        if t > b:
            strict_concordant += 1
        elif t < b:
            discordant += 1
        else:
            ties += 1

    strict_concordant_pct = (strict_concordant / n) * 100.0
    tie_pct = (ties / n) * 100.0
    discordant_pct = (discordant / n) * 100.0
    mean_delta = float(np.mean(deltas))

    return {
        "num_pairs": n,
        "strict_concordant_pct": strict_concordant_pct,
        "tie_pct": tie_pct,
        "discordant_pct": discordant_pct,
        "mean_delta": mean_delta,
    }


def compute_sequence_rank_alignment(model_means: List[float]) -> Dict[str, Any]:
    """
    Evaluates sequence monotonicity and rank correlation against model enhancement rank (0, 1, 2, ...).
    """
    num_models = len(model_means)
    if num_models < 2:
        return {}

    gt_ranks = list(range(num_models))

    # Strict monotonicity: each subsequent model has strictly higher mean
    is_strictly_monotonic = all(model_means[i] < model_means[i+1] for i in range(num_models - 1))
    # Weak monotonicity: each subsequent model has greater or equal mean
    is_weakly_monotonic = all(model_means[i] <= model_means[i+1] for i in range(num_models - 1))

    # Rank correlations
    try:
        spearman_rho, spearman_p = stats.spearmanr(gt_ranks, model_means)
    except Exception:
        spearman_rho, spearman_p = np.nan, np.nan

    try:
        kendall_tau, kendall_p = stats.kendalltau(gt_ranks, model_means)
    except Exception:
        kendall_tau, kendall_p = np.nan, np.nan

    # Pairwise mean comparisons across all ordered pairs
    p_count = 0
    q_count = 0
    t_count = 0
    total_pairs = num_models * (num_models - 1) // 2

    for i in range(num_models):
        for j in range(i + 1, num_models):
            if model_means[j] > model_means[i]:
                p_count += 1
            elif model_means[j] < model_means[i]:
                q_count += 1
            else:
                t_count += 1

    macro_concordance = ((p_count + 0.5 * t_count) / total_pairs) * 100.0 if total_pairs > 0 else 0.0

    return {
        "num_models": num_models,
        "strictly_monotonic": is_strictly_monotonic,
        "weakly_monotonic": is_weakly_monotonic,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "kendall_tau": kendall_tau,
        "kendall_p": kendall_p,
        "macro_concordance": macro_concordance,
    }


# ==============================================================================
# 3. Main Evaluation Runner
# ==============================================================================

def run_evaluation(
    rating_dir: str,
    sequences: Dict[str, List[str]],
    metric_filter: Optional[List[str]] = None,
    eval_mode: str = "all",
    exclude_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    save_pairwise_details: bool = False
):
    excluded_keys = load_exclusion_file(exclude_file)
    if excluded_keys:
        print(f"Loaded {len(excluded_keys)} excluded keys from {exclude_file}")

    metric_map = get_available_metric_map(eval_mode)
    if metric_filter:
        metric_map = {k: v for k, v in metric_map.items() if k in metric_filter or any(f.lower() in k.lower() for f in metric_filter)}

    if not metric_map:
        print("No matching metrics selected. Exiting.")
        return

    if not output_dir:
        output_dir = os.path.join(rating_dir, "model_rank_summary")
    os.makedirs(output_dir, exist_ok=True)

    # Collect all unique models needed
    all_needed_models = set()
    for seq in sequences.values():
        all_needed_models.update(seq)

    print(f"Loading evaluation scores from: {rating_dir}")
    print(f"Models to evaluate: {sorted(all_needed_models)}")
    print(f"Active metrics: {list(metric_map.keys())}\n")

    # Cache loaded scores: {model_name: {metric_name: {key: score}}}
    model_scores_cache: Dict[str, Dict[str, Dict[str, float]]] = {}
    for m in all_needed_models:
        scores = load_model_scores(rating_dir, m, metric_map, excluded_keys)
        if not scores:
            print(f"[Warning] No scores found for model '{m}'. Check directory name.")
        model_scores_cache[m] = scores

    # Records for output tables
    avg_summary_records = []
    sample_summary_records = []
    pairwise_delta_records = []

    for seq_name, model_seq in sequences.items():
        print(f"================================================================================")
        print(f"Evaluation Sequence: {seq_name} (Order: {' -> '.join(model_seq)})")
        print(f"================================================================================")

        # Collect metrics that are present in at least 2 models of the sequence
        common_metrics = set()
        for m in model_seq:
            common_metrics.update(model_scores_cache.get(m, {}).keys())

        for metric_name in sorted(common_metrics):
            # Check availability
            valid_models = [m for m in model_seq if metric_name in model_scores_cache.get(m, {}) and len(model_scores_cache[m][metric_name]) > 0]
            if len(valid_models) < 2:
                continue

            # ------------------------------------------------------------------
            # A. Macro-Average Level Evaluation
            # ------------------------------------------------------------------
            means = []
            stds = []
            counts = []
            for m in valid_models:
                m_scores = list(model_scores_cache[m][metric_name].values())
                means.append(float(np.mean(m_scores)))
                stds.append(float(np.std(m_scores, ddof=1)) if len(m_scores) > 1 else 0.0)
                counts.append(len(m_scores))

            avg_stats = compute_sequence_rank_alignment(means)
            avg_record = {
                "sequence": seq_name,
                "metric": metric_name,
                "num_models": len(valid_models),
                "models": " -> ".join(valid_models),
                "means": " | ".join([f"{m}:{val:.4f}" for m, val in zip(valid_models, means)]),
                "strictly_monotonic": avg_stats.get("strictly_monotonic", False),
                "weakly_monotonic": avg_stats.get("weakly_monotonic", False),
                "spearman_rho": avg_stats.get("spearman_rho", np.nan),
                "kendall_tau": avg_stats.get("kendall_tau", np.nan),
                "macro_concordance_pct": avg_stats.get("macro_concordance", np.nan),
            }
            avg_summary_records.append(avg_record)

            # ------------------------------------------------------------------
            # B. Paired Sample-by-Sample / Crop-by-Crop Level Evaluation
            # ------------------------------------------------------------------
            # Evaluate all adjacent steps (M_i -> M_i+1) as well as full span (M_0 -> M_last)
            pairs_to_eval = []
            for i in range(len(valid_models) - 1):
                pairs_to_eval.append((valid_models[i], valid_models[i+1], "adjacent"))
            if len(valid_models) > 2:
                pairs_to_eval.append((valid_models[0], valid_models[-1], "full_span"))

            for base_m, tgt_m, pair_type in pairs_to_eval:
                base_dict = model_scores_cache[base_m][metric_name]
                tgt_dict = model_scores_cache[tgt_m][metric_name]

                # Find shared sample / crop keys
                shared_keys = sorted(set(base_dict.keys()) & set(tgt_dict.keys()))
                if not shared_keys:
                    continue

                b_scores = [base_dict[k] for k in shared_keys]
                t_scores = [tgt_dict[k] for k in shared_keys]

                pair_stats = compute_pairwise_stats(b_scores, t_scores)
                if not pair_stats:
                    continue

                sample_record = {
                    "sequence": seq_name,
                    "metric": metric_name,
                    "pair_type": pair_type,
                    "baseline_model": base_m,
                    "target_model": tgt_m,
                    "num_shared_samples": pair_stats["num_pairs"],
                    "strict_concordant_pct": pair_stats["strict_concordant_pct"],
                    "tie_pct": pair_stats["tie_pct"],
                    "discordant_pct": pair_stats["discordant_pct"],
                    "mean_delta": pair_stats["mean_delta"],
                }
                sample_summary_records.append(sample_record)

                if save_pairwise_details:
                    for k, b, t in zip(shared_keys, b_scores, t_scores):
                        pairwise_delta_records.append({
                            "sequence": seq_name,
                            "metric": metric_name,
                            "baseline_model": base_m,
                            "target_model": tgt_m,
                            "sample_key": k,
                            "baseline_score": b,
                            "target_score": t,
                            "delta": t - b,
                            "relation": "strict_concordant" if t > b else ("discordant" if t < b else "tie")
                        })

    # --------------------------------------------------------------------------
    # 4. Print & Save Summaries
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(" 1. MACRO-AVERAGE LEVEL RANK ALIGNMENT SUMMARY")
    print("=" * 90)
    if avg_summary_records:
        df_avg = pd.DataFrame(avg_summary_records)
        display_cols = ["sequence", "metric", "num_models", "strictly_monotonic", "spearman_rho", "kendall_tau", "macro_concordance_pct"]
        print(df_avg[display_cols].to_string(index=False))

        avg_csv_path = os.path.join(output_dir, "macro_average_alignment.csv")
        df_avg.to_csv(avg_csv_path, index=False)
        print(f"\n[Saved] Macro average alignment summary -> {avg_csv_path}")
    else:
        print("No valid multi-model sequences found to evaluate average alignment.")

    print("\n" + "=" * 90)
    print(" 2. PAIRED SAMPLE-BY-SAMPLE / CROP-BY-CROP LEVEL ALIGNMENT SUMMARY")
    print("=" * 90)
    if sample_summary_records:
        df_sample = pd.DataFrame(sample_summary_records)
        display_cols = [
            "sequence", "metric", "baseline_model", "target_model",
            "num_shared_samples", "strict_concordant_pct", "tie_pct", "discordant_pct", "mean_delta"
        ]
        print(df_sample[display_cols].to_string(index=False))

        sample_csv_path = os.path.join(output_dir, "sample_level_alignment.csv")
        df_sample.to_csv(sample_csv_path, index=False)
        print(f"\n[Saved] Paired sample alignment summary -> {sample_csv_path}")
    else:
        print("No shared sample pairs found across models for pairwise evaluation.")

    if save_pairwise_details and pairwise_delta_records:
        df_details = pd.DataFrame(pairwise_delta_records)
        details_path = os.path.join(output_dir, "pairwise_sample_deltas.csv")
        df_details.to_csv(details_path, index=False)
        print(f"[Saved] Per-sample deltas -> {details_path}")

    # Save complete JSON summary
    json_summary_path = os.path.join(output_dir, "model_rank_summary.json")
    with open(json_summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "rating_dir": rating_dir,
            "macro_average_results": avg_summary_records,
            "sample_level_results": sample_summary_records
        }, f, indent=2)
    print(f"[Saved] Full JSON summary -> {json_summary_path}")


def main():
    args = parse_args()

    # Build sequence dictionary: {seq_name: [model_0, model_1, ...]}
    sequences: Dict[str, List[str]] = {}

    if args.rank_config and os.path.exists(args.rank_config):
        with open(args.rank_config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            if isinstance(cfg, dict):
                sequences = cfg
            elif isinstance(cfg, list):
                sequences["default_sequence"] = cfg
    elif args.models:
        sequences["cli_sequence"] = args.models
    elif args.baseline_model and args.target_model:
        sequences[f"{args.baseline_model}_vs_{args.target_model}"] = [args.baseline_model, args.target_model]
    else:
        print("Error: You must provide either --models, (--baseline_model and --target_model), or --rank_config.")
        sys.exit(1)

    run_evaluation(
        rating_dir=args.rating_dir,
        sequences=sequences,
        metric_filter=args.metrics,
        eval_mode=args.eval_mode,
        exclude_file=args.exclude_file,
        output_dir=args.output_dir,
        save_pairwise_details=args.save_pairwise_details
    )


if __name__ == "__main__":
    main()

