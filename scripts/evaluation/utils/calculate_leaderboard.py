#!/usr/bin/env python3
"""
calculate_leaderboard.py

Calculates model leaderboard rankings based on average rating scores.
Supports customizable model lists, rating directories, evaluation files,
common sample intersection filtering, per-category breakdown, and exporting to CSV/JSON/Markdown.

Usage:
    python3 calculate_leaderboard.py
    python3 calculate_leaderboard.py --models flux-klein flux-kontext qwen-image-edit
    python3 calculate_leaderboard.py --common_samples --category_breakdown --output_path leaderboard.csv
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd


DEFAULT_MODELS = [
    "diptych",
    "flux-klein",
    "flux-kontext",
    "qwen-image-edit",
    "gpt-image-2-high",
    "ominicontrol"
]

DEFAULT_RATING_DIR = "/root/Desktop/workspace/woosung/commercial-dreambench/rating/amzn"
DEFAULT_RESULTS_FILENAME = "ours-Qwen3-VL-8B-Instruct_bbox-crop_results.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate model leaderboard rankings based on evaluation rating scores."
    )
    parser.add_argument(
        "--rating_dir",
        type=str,
        default=DEFAULT_RATING_DIR,
        help=f"Base directory containing model evaluation subfolders (default: {DEFAULT_RATING_DIR})",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"List of model names to evaluate (default: {' '.join(DEFAULT_MODELS)})",
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
        help="Only evaluate on the common intersection of samples evaluated across all models.",
    )
    parser.add_argument(
        "--category_breakdown",
        action="store_true",
        help="Include per-category breakdown in the leaderboard.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional path to save leaderboard (supports .csv, .json, .md).",
    )
    return parser.parse_args()


def extract_category(sample_id: str) -> str:
    """Extract category name from standard sample_id e.g. 'raw_meta_All_Beauty_B000BBPSFA_variation_2'."""
    parts = sample_id.split("_")
    if len(parts) >= 3 and parts[0] == "raw" and parts[1] == "meta":
        # Usually ends with item id + variation + index (e.g., _B00..._variation_1)
        # Find index of item ID starting with B0 or asin-like token, or join middle parts
        cat_tokens = []
        for token in parts[2:]:
            if token.startswith("B0") or token == "variation":
                break
            cat_tokens.append(token)
        if cat_tokens:
            return "_".join(cat_tokens)
    return "Overall"


def load_model_scores(rating_dir: str, models: List[str], filename: str) -> Dict[str, Dict[str, float]]:
    model_data = {}
    for model in models:
        file_path = os.path.join(rating_dir, model, filename)
        if not os.path.exists(file_path):
            print(f"[Warning] File not found for model '{model}': {file_path}", file=sys.stderr)
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Filter valid numerical entries
            scores = {k: float(v) for k, v in data.items() if isinstance(v, (int, float)) and not np.isnan(v)}
            model_data[model] = scores
        except Exception as e:
            print(f"[Error] Failed to load {file_path}: {e}", file=sys.stderr)
    return model_data


def compute_leaderboard(
    model_scores: Dict[str, Dict[str, float]],
    common_samples: bool = False,
    category_breakdown: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if not model_scores:
        return pd.DataFrame(), None

    # Handle common sample filtering if requested
    if common_samples:
        common_keys = set.intersection(*(set(scores.keys()) for scores in model_scores.values()))
        print(f"[Info] Evaluating on {len(common_keys)} common samples across {len(model_scores)} models.")
        filtered_scores = {
            m: {k: v for k, v in scores.items() if k in common_keys}
            for m, scores in model_scores.items()
        }
    else:
        filtered_scores = model_scores

    summary_rows = []
    category_rows = []

    for model, scores in filtered_scores.items():
        if not scores:
            continue
        vals = list(scores.values())
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        median_val = float(np.median(vals))
        min_val = float(np.min(vals))
        max_val = float(np.max(vals))
        count_val = len(vals)

        summary_rows.append({
            "Model": model,
            "Average Score": mean_val,
            "Std": std_val,
            "Median": median_val,
            "Min": min_val,
            "Max": max_val,
            "Sample Count": count_val,
        })

        if category_breakdown:
            cat_grouped: Dict[str, List[float]] = {}
            for k, v in scores.items():
                cat = extract_category(k)
                cat_grouped.setdefault(cat, []).append(v)
            for cat, c_vals in cat_grouped.items():
                category_rows.append({
                    "Model": model,
                    "Category": cat,
                    "Average Score": float(np.mean(c_vals)),
                    "Sample Count": len(c_vals),
                })

    df_leaderboard = pd.DataFrame(summary_rows)
    if not df_leaderboard.empty:
        df_leaderboard = df_leaderboard.sort_values(by="Average Score", ascending=False).reset_index(drop=True)
        df_leaderboard.index = df_leaderboard.index + 1
        df_leaderboard.index.name = "Rank"

    df_cat = None
    if category_breakdown and category_rows:
        df_cat = pd.DataFrame(category_rows)
        # Pivot table: Categories as columns, models as rows sorted by overall rank
        df_cat_pivot = df_cat.pivot(index="Model", columns="Category", values="Average Score")
        # Re-index models by overall leaderboard rank
        ordered_models = df_leaderboard["Model"].tolist()
        df_cat_pivot = df_cat_pivot.reindex([m for m in ordered_models if m in df_cat_pivot.index])
        df_cat = df_cat_pivot

    return df_leaderboard, df_cat


def save_leaderboard(df_leaderboard: pd.DataFrame, df_cat: Optional[pd.DataFrame], output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower()

    if ext == ".csv":
        df_leaderboard.to_csv(output_path)
        if df_cat is not None:
            cat_path = output_path.replace(".csv", "_category.csv")
            df_cat.to_csv(cat_path)
            print(f"[Info] Category breakdown saved to: {cat_path}")
        print(f"[Info] Leaderboard saved to: {output_path}")

    elif ext == ".json":
        out_dict = {
            "leaderboard": df_leaderboard.reset_index().to_dict(orient="records"),
        }
        if df_cat is not None:
            out_dict["category_breakdown"] = df_cat.reset_index().to_dict(orient="records")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, indent=4)
        print(f"[Info] Leaderboard saved to: {output_path}")

    elif ext in [".md", ".markdown"]:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Model Evaluation Leaderboard\n\n")
            f.write(df_leaderboard.to_markdown())
            f.write("\n\n")
            if df_cat is not None:
                f.write("## Category Breakdown\n\n")
                f.write(df_cat.to_markdown())
                f.write("\n")
        print(f"[Info] Leaderboard saved to: {output_path}")
    else:
        print(f"[Warning] Unsupported extension '{ext}'. Saving as CSV.", file=sys.stderr)
        df_leaderboard.to_csv(output_path)
        print(f"[Info] Leaderboard saved to: {output_path}")


def main():
    args = parse_args()
    print("=" * 70)
    print("Commercial-DreamBench Model Leaderboard")
    print(f"Rating Directory : {args.rating_dir}")
    print(f"Results File     : {args.results_filename}")
    print(f"Models ({len(args.models)})  : {', '.join(args.models)}")
    print(f"Common Samples   : {args.common_samples}")
    print("=" * 70)

    model_scores = load_model_scores(args.rating_dir, args.models, args.results_filename)
    if not model_scores:
        print("[Error] No valid model score data could be loaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    df_leaderboard, df_cat = compute_leaderboard(
        model_scores,
        common_samples=args.common_samples,
        category_breakdown=args.category_breakdown,
    )

    if df_leaderboard.empty:
        print("[Error] Failed to compute leaderboard.", file=sys.stderr)
        sys.exit(1)

    # Format numeric columns for display
    display_df = df_leaderboard.copy()
    display_df["Average Score"] = display_df["Average Score"].map(lambda x: f"{x:.4f}")
    display_df["Std"] = display_df["Std"].map(lambda x: f"{x:.4f}")
    display_df["Median"] = display_df["Median"].map(lambda x: f"{x:.4f}")
    display_df["Min"] = display_df["Min"].map(lambda x: f"{x:.4f}")
    display_df["Max"] = display_df["Max"].map(lambda x: f"{x:.4f}")

    print("\n--- Overall Leaderboard ---")
    print(display_df.to_string())

    if df_cat is not None and not df_cat.empty:
        print("\n--- Category Breakdown (Average Score) ---")
        cat_display = df_cat.map(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
        print(cat_display.to_string())

    if args.output_path:
        save_leaderboard(df_leaderboard, df_cat, args.output_path)


if __name__ == "__main__":
    main()
