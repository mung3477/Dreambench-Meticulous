#!/usr/bin/env python3
"""
calculate_elo.py

Calculates Elo / Bradley-Terry ratings of models based on pairwise comparisons
across identical prompts and references.

Usage:
    python3 calculate_elo.py
    python3 calculate_elo.py --models flux-kontext flux-klein qwen-image-edit diptych gpt-image-2
    python3 calculate_elo.py --common_samples --n_boot 1000 --output_path elo_ratings.csv
"""

import os
import sys
import json
import argparse
from itertools import combinations
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DEFAULT_MODELS = [
    "flux-kontext",
    "flux-klein",
    "qwen-image-edit",
    "diptych",
    "gpt-image-2-high",
    "ominicontrol"
]
DEFAULT_RATING_DIR = "/root/Desktop/workspace/woosung/commercial-dreambench/rating/amzn"
DEFAULT_RESULTS_FILENAME = "ours-Qwen3-VL-8B-Instruct_bbox-crop_results.json"
SCALE = 400.0  # Elo scale constant
ANCHOR = 1000.0  # mean rating anchor (gauge fix)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate Elo / Bradley-Terry ratings for models based on pairwise score comparisons."
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"List of model names to evaluate (default: {' '.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--rating_dir",
        type=str,
        default=DEFAULT_RATING_DIR,
        help=f"Base directory containing model evaluation subfolders (default: {DEFAULT_RATING_DIR})",
    )
    parser.add_argument(
        "--results_filename",
        type=str,
        default=DEFAULT_RESULTS_FILENAME,
        help=f"Evaluation result filename in each model folder (default: {DEFAULT_RESULTS_FILENAME})",
    )
    parser.add_argument(
        "--common_samples",
        action="store_true",
        help="Only evaluate on common samples evaluated across all specified models.",
    )
    parser.add_argument(
        "--n_boot",
        type=int,
        default=1000,
        help="Number of bootstrap iterations for computing confidence intervals (default: 1000). Set to 0 to disable.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for confidence intervals (default: 0.05 for 95%% CI).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrapping (default: 42).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional path to save Elo ratings (.csv, .json, or .md).",
    )
    parser.add_argument(
        "--plot_path",
        type=str,
        default=None,
        help="Optional path to save Elo rating bar plot (e.g. elo_leaderboard.png, elo_leaderboard.pdf).",
    )
    parser.add_argument(
        "--plot_title",
        type=str,
        default="Model Elo Leaderboard (Bradley-Terry)",
        help="Title for the Elo rating bar plot.",
    )
    parser.add_argument(
        "--plot_dpi",
        type=int,
        default=300,
        help="DPI resolution for the exported plot (default: 300).",
    )
    return parser.parse_args()


def load_model_scores(rating_dir: str, models: List[str], filename: str) -> Dict[str, Dict[str, float]]:
    """Loads evaluation results for each model from JSON files."""
    model_data = {}
    for model in models:
        file_path = os.path.join(rating_dir, model, filename)
        if not os.path.exists(file_path):
            print(f"[Warning] File not found for model '{model}': {file_path}", file=sys.stderr)
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            scores = {k: float(v) for k, v in data.items() if isinstance(v, (int, float)) and not np.isnan(v)}
            model_data[model] = scores
        except Exception as e:
            print(f"[Error] Failed to load {file_path}: {e}", file=sys.stderr)
    return model_data


def build_comparisons_by_prompt(
    model_scores: Dict[str, Dict[str, float]],
    models: List[str],
    common_samples: bool = False,
) -> Tuple[List[str], np.ndarray, np.ndarray, int]:
    """
    Constructs pairwise comparisons indexed by prompt.

    Returns:
        models: sorted list of participating model names
        comps: ndarray of shape (N, 3) with columns [i, j, s_i] where s_i in {1.0, 0.5, 0.0}
        prompt_indices: ndarray of shape (N,) associating each comparison to its unique prompt index
        num_prompts: total number of distinct prompts
    """
    if common_samples:
        common_keys = sorted(set.intersection(*(set(scores.keys()) for scores in model_scores.values())))
        print(f"[Info] Evaluating on {len(common_keys)} common samples across {len(model_scores)} models.")
        prompts = common_keys
    else:
        all_keys = set()
        for scores in model_scores.values():
            all_keys.update(scores.keys())
        prompts = sorted(all_keys)
        print(f"[Info] Evaluating across {len(prompts)} total unique samples (any overlap).")

    idx = {m: k for k, m in enumerate(models)}
    rows = []
    prompt_ids = []

    for p_idx, p in enumerate(prompts):
        available = [(m, model_scores[m][p]) for m in models if p in model_scores[m]]
        if len(available) < 2:
            continue
        for (ma, ra), (mb, rb) in combinations(available, 2):
            s = 1.0 if ra > rb else (0.0 if ra < rb else 0.5)
            rows.append((idx[ma], idx[mb], s))
            prompt_ids.append(p_idx)

    if not rows:
        return models, np.empty((0, 3)), np.empty((0,), dtype=int), len(prompts)

    return models, np.array(rows, dtype=float), np.array(prompt_ids, dtype=int), len(prompts)


def fit_bradley_terry(n_models: int, comps: np.ndarray) -> np.ndarray:
    """MLE fit of Bradley-Terry model. Returns ratings on Elo scale, mean-anchored to ANCHOR."""
    if len(comps) == 0 or n_models == 0:
        return np.full(n_models, ANCHOR)

    i = comps[:, 0].astype(int)
    j = comps[:, 1].astype(int)
    s = comps[:, 2]
    c = np.log(10) / SCALE  # convert Elo-scale diffs to logits

    def neg_ll(theta):
        d = theta[i] - theta[j]
        # log-sigmoid, numerically stable
        logp = -np.logaddexp(0.0, -c * d)  # log P(i beats j)
        logq = -np.logaddexp(0.0, c * d)   # log P(j beats i)
        return -np.sum(s * logp + (1 - s) * logq)

    theta0 = np.zeros(n_models)
    res = minimize(neg_ll, theta0, method="L-BFGS-B")
    r = res.x
    return r - r.mean() + ANCHOR


def bootstrap_ci(
    n_models: int,
    comps: np.ndarray,
    prompt_indices: np.ndarray,
    num_prompts: int,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes bootstrap confidence intervals by resampling prompt clusters.
    Uses comparison-to-prompt indexing for fast aggregation without pandas grouping.
    """
    if n_boot <= 0 or len(comps) == 0:
        return np.full(n_models, np.nan), np.full(n_models, np.nan)

    rng = np.random.default_rng(seed)
    samples = np.empty((n_boot, n_models))

    # Pre-index comparisons by prompt ID for speed
    prompt_to_comp_indices = [[] for _ in range(num_prompts)]
    for comp_idx, p_id in enumerate(prompt_indices):
        prompt_to_comp_indices[p_id].append(comp_idx)

    # Filter out empty prompts
    active_prompt_ids = np.array([pid for pid, c_idxs in enumerate(prompt_to_comp_indices) if len(c_idxs) > 0])
    num_active = len(active_prompt_ids)

    if num_active == 0:
        return np.full(n_models, np.nan), np.full(n_models, np.nan)

    for b in range(n_boot):
        # Sample prompt clusters with replacement
        sampled_prompts = rng.choice(active_prompt_ids, size=num_active, replace=True)
        # Gather all comparisons for the sampled prompts
        sampled_comp_indices = []
        for pid in sampled_prompts:
            sampled_comp_indices.extend(prompt_to_comp_indices[pid])

        boot_comps = comps[sampled_comp_indices]
        samples[b] = fit_bradley_terry(n_models, boot_comps)

    lo = np.percentile(samples, 100 * alpha / 2, axis=0)
    hi = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


def main():
    args = parse_args()

    model_scores = load_model_scores(args.rating_dir, args.models, args.results_filename)
    loaded_models = [m for m in args.models if m in model_scores]

    if not loaded_models:
        print("[Error] No model scores could be loaded. Please check directory and model names.", file=sys.stderr)
        sys.exit(1)

    print(f"[Info] Successfully loaded scores for {len(loaded_models)} models: {loaded_models}")

    models, comps, prompt_indices, num_prompts = build_comparisons_by_prompt(
        model_scores=model_scores,
        models=loaded_models,
        common_samples=args.common_samples,
    )

    if len(comps) == 0:
        print("[Error] No pairwise comparisons could be formed between the loaded models.", file=sys.stderr)
        sys.exit(1)

    print(f"[Info] Running Bradley-Terry MLE estimation on {len(comps)} pairwise comparisons...")
    ratings = fit_bradley_terry(len(models), comps)

    if args.n_boot > 0:
        print(f"[Info] Computing 95% bootstrap confidence intervals ({args.n_boot} iterations)...")
        lo, hi = bootstrap_ci(
            n_models=len(models),
            comps=comps,
            prompt_indices=prompt_indices,
            num_prompts=num_prompts,
            n_boot=args.n_boot,
            alpha=args.alpha,
            seed=args.seed,
        )
    else:
        lo = np.full(len(models), np.nan)
        hi = np.full(len(models), np.nan)

    # Calculate match stats per model
    total_matches = np.zeros(len(models), dtype=int)
    wins = np.zeros(len(models), dtype=float)
    idx = {m: k for k, m in enumerate(models)}

    for m_idx in range(len(models)):
        as_i = comps[:, 0] == m_idx
        as_j = comps[:, 1] == m_idx
        total_matches[m_idx] = np.sum(as_i) + np.sum(as_j)
        wins[m_idx] = np.sum(comps[as_i, 2]) + np.sum(1.0 - comps[as_j, 2])

    win_rate = np.where(total_matches > 0, wins / total_matches * 100.0, 0.0)

    out_df = pd.DataFrame({
        "Model": models,
        "Elo": np.round(ratings, 1),
        "95% CI": [f"[{l:.1f}, {h:.1f}]" if not np.isnan(l) else "N/A" for l, h in zip(lo, hi)],
        "Win Rate (%)": np.round(win_rate, 2),
        "Total Matches": total_matches,
        "Sample Count": [len(model_scores[m]) for m in models],
    }).sort_values("Elo", ascending=False).reset_index(drop=True)
    out_df.index = out_df.index + 1
    out_df.index.name = "Rank"

    print("\n" + "=" * 80)
    print("                      MODEL ELO LEADERBOARD (Bradley-Terry)")
    print("=" * 80)
    print(out_df.to_string())
    print("=" * 80 + "\n")

    if args.output_path:
        ext = os.path.splitext(args.output_path)[-1].lower()
        if ext == ".csv":
            out_df.to_csv(args.output_path)
        elif ext == ".json":
            out_df.to_json(args.output_path, orient="records", indent=2)
        elif ext in [".md", ".markdown"]:
            out_df.to_markdown(args.output_path)
        else:
            out_df.to_csv(args.output_path)
        print(f"[Info] Saved Elo ratings to {args.output_path}")

    if args.plot_path:
        plot_elo_bars(
            models=models,
            ratings=ratings,
            ci_low=lo,
            ci_high=hi,
            output_path=args.plot_path,
            title=args.plot_title,
            dpi=args.plot_dpi,
        )


def plot_elo_bars(
    models: List[str],
    ratings: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    output_path: str,
    title: str = "Model Elo Leaderboard (Bradley-Terry)",
    dpi: int = 300,
):
    """
    Plots a tech-report quality horizontal bar chart of Elo ratings with confidence intervals.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("[Error] matplotlib is required for plotting. Please run 'pip install matplotlib'.", file=sys.stderr)
        return

    # Sort models by rating ascending for horizontal bar chart (top ranked on top)
    sorted_order = np.argsort(ratings)
    sorted_models = [models[idx] for idx in sorted_order]
    sorted_ratings = ratings[sorted_order]
    sorted_ci_low = ci_low[sorted_order]
    sorted_ci_high = ci_high[sorted_order]

    # Calculate asymmetric error bars: [rating - ci_low, ci_high - rating]
    has_valid_ci = not np.isnan(sorted_ci_low).any() and not np.isnan(sorted_ci_high).any()
    if has_valid_ci:
        xerr_low = np.maximum(0, sorted_ratings - sorted_ci_low)
        xerr_high = np.maximum(0, sorted_ci_high - sorted_ratings)
        xerr = np.vstack([xerr_low, xerr_high])
    else:
        xerr = None

    # Styling settings reminiscent of OpenAI / LMSYS / Anthropic technical reports
    fig_height = max(4.0, len(models) * 0.75 + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height), dpi=dpi)

    y_pos = np.arange(len(sorted_models))

    # Color palette: top model in primary highlight (deep modern blue), rest in slate gray
    colors = ["#e3ecf0"] * len(sorted_models)
    colors[-1] = "#1e475e"  # Highlight rank 1

    bars = ax.barh(
        y_pos,
        sorted_ratings,
        xerr=xerr,
        align="center",
        color=colors,
        alpha=0.88,
        edgecolor="none",
        height=0.42,
        capsize=4.5 if has_valid_ci else 0,
        error_kw={"elinewidth": 1.2, "ecolor": "#1F2937", "capthick": 1.4},
        zorder=3,
    )

    # Spacing and limits
    min_x = np.nanmin(sorted_ci_low) if has_valid_ci else np.min(sorted_ratings)
    max_x = np.nanmax(sorted_ci_high) if has_valid_ci else np.max(sorted_ratings)
    padding = (max_x - min_x) * 0.18 if (max_x > min_x) else 50.0
    x_lim_left = max(0, min_x - padding)
    x_lim_right = max_x + padding * 1.5
    ax.set_xlim(x_lim_left, x_lim_right)
    ax.set_ylim(-0.6, len(sorted_models) - 0.4)

    # Format ticks and labels
    # ax.set_yticks(y_pos)
    # ax.set_yticklabels(sorted_models, fontsize=14, fontweight="medium", color="#1F2937")
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=14, colors="#4B5563")
    ax.tick_params(axis="y", length=0)

    # Clean borders (spines)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.spines["bottom"].set_linewidth(1.0)

    # Subtle vertical grid behind bars
    ax.xaxis.grid(True, linestyle="--", linewidth=0.7, color="#E2E8F0", zorder=0)
    ax.set_axisbelow(True)

    # Value annotations at end of bars
    for idx, (model, val, y) in enumerate(zip(sorted_models, sorted_ratings, y_pos)):
        # Put model name directly above the bar, aligned to the left of the plot
        ax.text(
            x_lim_left,
            y + 0.22,
            model,
            fontsize=12,
            fontweight="bold",
            color="#1F2937",
            va="bottom",
            ha="left"
        )

        if has_valid_ci:
            err_plus = sorted_ci_high[idx] - val
            err_minus = val - sorted_ci_low[idx]
            label_text = f" {val:.1f}\n(-{err_minus:.1f}, +{err_plus:.1f})"
            annot_x = sorted_ci_high[idx]
        else:
            label_text = f" {val:.1f}"
            annot_x = val

        font_weight = "bold" if idx == len(sorted_models) - 1 else "normal"
        text_color = "#1E40AF" if idx == len(sorted_models) - 1 else "#1F2937"
        ax.text(
            annot_x,
            y,
            label_text,
            va="center",
            ha="left",
            fontsize=9.5,
            color=text_color,
            fontweight=font_weight,
            zorder=4,
        )

    # Titles and labels
    ax.set_xlabel("Bradley-Terry Elo Rating (Mean Anchor = 1000)", fontsize=11, color="#374151", labelpad=10)
    ax.set_title(title, fontsize=16, fontweight="bold", color="#111827", pad=16, loc="left")

    # Add reference line at ANCHOR
    ax.axvline(1000.0, color="#94A3B8", linestyle=":", linewidth=1.1, zorder=1)

    plt.tight_layout()

    # Ensure parent dir exists
    plot_dir = os.path.dirname(output_path)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Info] Saved Elo rating bar plot to {output_path}")


if __name__ == "__main__":
    main()


