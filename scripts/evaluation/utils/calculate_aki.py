#!/usr/bin/env python3
"""
calculate_aki.py

Calculates Absolute Keypoint Increase (AKI) and Keypoint Matching Gain (K_Gain)
across model combinations using precomputed keypoint matching results.

Formulation from the paper:
    AKI = N(M(I_ref, I_hat_gen)) - N(M(I_ref, I_gen))
    K_Gain = 1/N * sum_{i=1}^N delta(AKI_i, tau) * 100% (default tau = 0)

Usage:
    # 1. Compare a specific target model against a baseline:
    python3 calculate_aki.py \
        --rating_dir /path/to/rating \
        --target_model proposed_model \
        --baseline_model baseline_model

    # 2. Compare all models in rating_dir against a baseline:
    python3 calculate_aki.py \
        --rating_dir /path/to/rating \
        --baseline_model baseline_model

    # 3. All-pairs comparison across all models in rating_dir:
    python3 calculate_aki.py \
        --rating_dir /path/to/rating \
        --all_pairs
"""

import os
import sys
import json
import argparse
import itertools
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate AKI and K_Gain across models from precomputed keypoints_results.json."
    )
    parser.add_argument(
        "--rating_dir",
        type=str,
        required=True,
        help="Directory containing model folders with keypoints_results.json."
    )
    parser.add_argument(
        "--target_model",
        type=str,
        default=None,
        help="Target model name to evaluate (if not specified, evaluates all non-baseline models)."
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        default=None,
        help="Baseline model name to compare against."
    )
    parser.add_argument(
        "--all_pairs",
        action="store_true",
        help="Calculate AKI and K_Gain for all ordered pairs (A vs B) of models."
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.0,
        help="Threshold tau for K_Gain indicator delta(AKI, tau) (default: 0.0)."
    )
    parser.add_argument(
        "--keypoint_file",
        type=str,
        default="keypoints_results.json",
        help="Keypoint results filename within each model rating directory (default: keypoints_results.json)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save summary CSV and logs (defaults to rating_dir/aki_summary)."
    )
    parser.add_argument(
        "--save_pairwise_json",
        action="store_true",
        help="Save detailed per-sample AKI JSON under rating_dir/<target_model>/aki_vs_<baseline_model>.json."
    )
    return parser.parse_args()


def load_model_keypoints(model_dir, filename="keypoints_results.json"):
    """Loads keypoint results from a model's rating folder, returning {sample_key: count}."""
    filepath = os.path.join(model_dir, filename)
    if not os.path.isfile(filepath):
        # Also check fallback filename
        fallback = os.path.join(model_dir, "aki_results.json")
        if os.path.isfile(fallback):
            filepath = fallback
        else:
            return None

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        scores = {}
        for k, v in data.items():
            if k in ['overall_mean', 'mean_keypoints', 'mean_aki', 'k_gain', 'tau']:
                continue
            if isinstance(v, (int, float)):
                scores[k] = float(v)
            elif isinstance(v, dict) and 'keypoints' in v:
                scores[k] = float(v['keypoints'])
            elif isinstance(v, dict) and 'target_keypoints' in v:
                scores[k] = float(v['target_keypoints'])
        return scores
    except Exception as e:
        print(f"[Warning] Failed to read {filepath}: {e}")
        return None


def calculate_pair_metrics(target_scores, baseline_scores, tau=0.0):
    """
    Computes AKI and K_Gain between target and baseline for shared sample keys.
    Returns:
        mean_aki (float), k_gain (float %), detailed_samples (dict), num_common (int)
    """
    common_keys = sorted(list(set(target_scores.keys()) & set(baseline_scores.keys())))
    if not common_keys:
        return None

    aki_values = []
    k_gain_hits = []
    details = {}

    for k in common_keys:
        t_kp = target_scores[k]
        b_kp = baseline_scores[k]
        aki = t_kp - b_kp
        hit = 1 if aki > tau else 0

        aki_values.append(aki)
        k_gain_hits.append(hit)
        details[k] = {
            "target_keypoints": t_kp,
            "baseline_keypoints": b_kp,
            "aki": aki,
            "gain_hit": hit
        }

    mean_aki = float(np.mean(aki_values))
    k_gain = float(np.mean(k_gain_hits) * 100.0)

    return {
        "mean_aki": mean_aki,
        "k_gain": k_gain,
        "mean_target_kp": float(np.mean([target_scores[k] for k in common_keys])),
        "mean_baseline_kp": float(np.mean([baseline_scores[k] for k in common_keys])),
        "num_samples": len(common_keys),
        "details": details
    }


def log_summary_entry(output_dir, target_name, baseline_name, num_samples, mean_aki, k_gain, tau):
    kst = timezone(timedelta(hours=9))
    now_kst = datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S KST')
    log_dir = os.path.join(output_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'aki_gain_summary.log')

    line = (
        f"[{now_kst}] Target: {target_name:<20} | Baseline: {baseline_name:<20} | "
        f"Samples: {num_samples:<5} | Mean AKI: {mean_aki:+6.2f} | K_Gain (tau={tau}): {k_gain:6.2f}%\n"
    )
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(line)


def main():
    args = parse_args()
    rating_dir = os.path.abspath(args.rating_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(rating_dir, 'aki_summary')
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(rating_dir):
        print(f"Error: Rating directory '{rating_dir}' does not exist.")
        sys.exit(1)

    # Discover available model folders that contain keypoint files
    available_models = {}
    for d in sorted(os.listdir(rating_dir)):
        m_dir = os.path.join(rating_dir, d)
        if not os.path.isdir(m_dir) or d in ['aki_summary', 'log']:
            continue
        scores = load_model_keypoints(m_dir, filename=args.keypoint_file)
        if scores:
            available_models[d] = scores

    if not available_models:
        print(f"No valid keypoint result files found in {rating_dir}.")
        sys.exit(1)

    print(f"Found {len(available_models)} models with keypoints: {', '.join(available_models.keys())}")

    pairs_to_evaluate = []

    if args.all_pairs:
        # All ordered permutations (A vs B)
        for t, b in itertools.permutations(available_models.keys(), 2):
            pairs_to_evaluate.append((t, b))
    elif args.baseline_model:
        if args.baseline_model not in available_models:
            print(f"Error: Baseline model '{args.baseline_model}' keypoint results not found in {rating_dir}.")
            sys.exit(1)
        if args.target_model:
            if args.target_model not in available_models:
                print(f"Error: Target model '{args.target_model}' keypoint results not found in {rating_dir}.")
                sys.exit(1)
            pairs_to_evaluate.append((args.target_model, args.baseline_model))
        else:
            for m in available_models.keys():
                if m != args.baseline_model:
                    pairs_to_evaluate.append((m, args.baseline_model))
    elif args.target_model and not args.baseline_model:
        print("Error: Please specify --baseline_model when evaluating a target model.")
        sys.exit(1)
    else:
        print("Error: Specify either (--target_model and --baseline_model), --baseline_model, or --all_pairs.")
        sys.exit(1)

    print(f"\nEvaluating {len(pairs_to_evaluate)} model combination(s) (tau = {args.tau}):")
    print(f"{'-'*75}")
    print(f"{'Target Model':<22} {'Baseline Model':<22} {'Samples':<8} {'Mean AKI':<12} {'K_Gain (%)':<10}")
    print(f"{'-'*75}")

    summary_rows = []

    for target_m, baseline_m in pairs_to_evaluate:
        res = calculate_pair_metrics(
            available_models[target_m],
            available_models[baseline_m],
            tau=args.tau
        )
        if res is None:
            print(f"{target_m:<22} {baseline_m:<22} {'0 (no overlap)':<30}")
            continue

        print(
            f"{target_m:<22} {baseline_m:<22} {res['num_samples']:<8} "
            f"{res['mean_aki']:+7.2f}      {res['k_gain']:6.2f}%"
        )

        log_summary_entry(
            output_dir,
            target_m,
            baseline_m,
            res['num_samples'],
            res['mean_aki'],
            res['k_gain'],
            args.tau
        )

        summary_rows.append({
            "target_model": target_m,
            "baseline_model": baseline_m,
            "num_samples": res["num_samples"],
            "target_mean_keypoints": res["mean_target_kp"],
            "baseline_mean_keypoints": res["mean_baseline_kp"],
            "mean_aki": res["mean_aki"],
            "k_gain": res["k_gain"],
            "tau": args.tau
        })

        if args.save_pairwise_json:
            target_dir = os.path.join(rating_dir, target_m)
            pair_json_path = os.path.join(target_dir, f"aki_vs_{baseline_m}.json")
            save_payload = {
                "target_model": target_m,
                "baseline_model": baseline_m,
                "mean_aki": res["mean_aki"],
                "k_gain": res["k_gain"],
                "tau": args.tau,
                "samples": res["details"]
            }
            try:
                with open(pair_json_path, "w") as f:
                    json.dump(save_payload, f, indent=4)
            except Exception as e:
                print(f"Warning: Failed to save pairwise JSON {pair_json_path}: {e}")

    print(f"{'-'*75}")

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(output_dir, "aki_gain_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved summary CSV to: {csv_path}")
        print(f"Saved summary log to: {os.path.join(output_dir, 'log', 'aki_gain_summary.log')}")


if __name__ == "__main__":
    main()

