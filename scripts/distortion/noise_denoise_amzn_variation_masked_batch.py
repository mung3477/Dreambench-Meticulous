import os
import argparse
import json
import shutil
from PIL import Image
from tqdm import tqdm

# Import the workflow function from noise_denoise_one_step
from noise_denoise_one_step import single_step_denoise_workflow

# Import masking libraries
from lib.mask_ocr import generate_ocr_mask
from lib.mask_opencv import generate_contour_mask
from lib.mask_dino_adaptive import generate_dino_adaptive_mask

try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS

def parse_args():
    parser = argparse.ArgumentParser(description="Batch process AMJN variations using masked one-step noising & denoising.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/root/Desktop/workspace/woosung/commercial-dreambench/assets/data/amzn",
        help="Path to the dataset directory containing category subdirectories."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn",
        help="Base path to the directory where results will be saved."
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=0.5,
        help="Single scale factor for degradation downscaling (default: 0.5)."
    )
    parser.add_argument(
        "--scale_factors",
        type=float,
        nargs="+",
        default=None,
        help="List of scale factors to iterate over in a single run (e.g. 1.0 0.75 0.5 0.25)."
    )
    parser.add_argument(
        "--noise_timestep",
        type=int,
        default=0,
        help="Timestep at which to add noise (0 to 1000, default: 0)."
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Name of a specific category folder to process (e.g. 'raw_meta_Appliances')."
    )
    parser.add_argument(
        "--asin",
        type=str,
        default=None,
        help="Name of a specific ASIN folder to process (e.g. 'B005H47EJ4')."
    )
    parser.add_argument(
        "--target_mode",
        type=str,
        choices=["variation", "reference"],
        default="variation",
        help="Distortion target mode: 'variation' selects images starting with 'variation_', 'reference' selects reference images (default: 'variation')."
    )
    # Masking options
    parser.add_argument(
        "--mask_method",
        type=str,
        choices=["ocr", "opencv", "dino_adaptive"],
        default="opencv",
        help="Masking method to identify regions (choices: ocr, opencv, dino_adaptive. default: opencv)."
    )
    parser.add_argument(
        "--ocr_padding",
        type=int,
        default=15,
        help="Padding size around OCR text bounding boxes (default: 15)."
    )
    parser.add_argument(
        "--ocr_sample_ratio",
        type=float,
        default=1.0,
        help="Fraction of OCR text areas to randomly keep (e.g. 0.33 for 1/3, default: 1.0)."
    )
    parser.add_argument(
        "--cv_min_area_ratio",
        type=float,
        default=0.00001,
        help="Minimum area ratio for OpenCV contour detection (default: 0.00001)."
    )
    parser.add_argument(
        "--cv_max_area_ratio",
        type=float,
        default=0.10,
        help="Maximum area ratio for OpenCV contour detection (default: 0.1)."
    )
    parser.add_argument(
        "--cv_threshold",
        type=int,
        default=120,
        help="Gradient threshold for OpenCV contour detection (default: 30)."
    )
    parser.add_argument(
        "--cv_dilation_size",
        type=int,
        default=3,
        help="Dilation kernel size for OpenCV contour detection (default: 3)."
    )
    parser.add_argument(
        "--dino_bbox_height_ratio",
        type=float,
        default=1.0,
        help="Fraction of the detected contour bounding box height to use for masking from top (e.g. 0.1 for top 10%, default: 1.0)."
    )
    parser.add_argument(
        "--dino_bbox_height_ratios",
        type=float,
        nargs="+",
        default=None,
        help="List of bounding box height ratios to iterate over (e.g. 1.0 0.75 0.5 0.25)."
    )
    parser.add_argument(
        "--gpu_device",
        type=int,
        default=0,
        help="GPU device ID to run detection/distortion on (default: 0)."
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.17,
        help="Box threshold for GroundingDINO detection (default: 0.17)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, logs detailed masking steps and saves intermediate mask and noised variation files to the output directory."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, lists the files that would be processed without executing the noise-denoise workflow"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, forces re-processing and overwriting of existing output files instead of skipping them."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Verify input directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Dataset directory {args.data_dir} doesZnot exist.")
        return

    # Determine scale factors and height ratios to iterate over
    scale_factors = args.scale_factors if args.scale_factors is not None else [args.scale_factor]
    if args.mask_method == "dino_adaptive":
        dino_height_ratios = args.dino_bbox_height_ratios if args.dino_bbox_height_ratios is not None else [args.dino_bbox_height_ratio]
    else:
        dino_height_ratios = [args.dino_bbox_height_ratio]

    print(f"==============================================================")
    print(f"Batch Processor Running in Single Session")
    print(f"Scale Factors:        {scale_factors}")
    print(f"Height Ratios (DINO): {dino_height_ratios if args.mask_method == 'dino_adaptive' else 'N/A'}")
    print(f"Mask Method:          {args.mask_method}")
    print(f"Target Mode:          {args.target_mode}")
    print(f"Noise Timestep:       {args.noise_timestep}")
    print(f"=============================================================")

    # Cache for category metadata to look up categorical_names
    metadata_cache = {}
    def get_categorical_name(cat, asin):
        if cat not in metadata_cache:
            metadata_path = os.path.join(args.data_dir, cat, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        metadata_cache[cat] = json.load(f)
                except Exception as e:
                    print(f"Error loading metadata for category {cat}: {e}")
                    metadata_cache[cat] = []
            else:
                metadata_cache[cat] = []

        for item in metadata_cache[cat]:
            if item.get("asin") == asin:
                return item.get("categorical_name", "product")
        return "product"

    # Determine categories to process
    if args.category:
        categories = [args.category]
    else:
        categories = [
            d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d)) and not d.startswith(".")
        ]
        categories.sort()

    print(f"Found {len(categories)} categories to scan.")

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    # Pre-scan image input paths
    discovered_items = []  # list of (cat, asin, item, img_in_path)
    for cat in categories:
        cat_in_path = os.path.join(args.data_dir, cat)
        if not os.path.isdir(cat_in_path):
            continue

        if args.asin:
            asins = [args.asin]
        else:
            asins = [
                d for d in os.listdir(cat_in_path)
                if os.path.isdir(os.path.join(cat_in_path, d)) and not d.startswith(".")
            ]
            asins.sort()

        for asin in asins:
            asin_in_path = os.path.join(cat_in_path, asin)
            if not os.path.isdir(os.path.join(cat_in_path, asin)):
                continue
            for item in os.listdir(asin_in_path):
                ext = os.path.splitext(item.lower())[1]
                if ext in valid_extensions:
                    stem = os.path.splitext(item)[0]
                    is_target = False
                    if args.target_mode == "variation" and item.startswith("variation_"):
                        is_target = True
                    elif args.target_mode == "reference" and stem == "reference":
                        is_target = True

                    if is_target:
                        img_in_path = os.path.join(asin_in_path, item)
                        discovered_items.append((cat, asin, item, img_in_path))

    print(f"Discovered {len(discovered_items)} {args.target_mode} source images.")

    if not discovered_items:
        print(f"No {args.target_mode} images found matching the criteria.")
        return

    # Global DINO mask pre-generation (single pass for all images across all scales & ratios)
    image_masks_cache = {}
    image_bbox_stats = {}
    image_bbox_stats = {}
    if args.mask_method == "dino_adaptive":
        print(f"\n=============================================================")
        print(f"Pre-generating DINO adaptive masks for all {len(discovered_items)} images (1 global pass across all scales & ratios)...")
        print(f"=============================================================")
        for cat, asin, item, img_in_path in tqdm(discovered_items, desc="DINO Global Single-Pass Masking"):
            categorical_name = get_categorical_name(cat, asin)
            try:
                masks_dict, bboxes_info = generate_dino_adaptive_mask(
                    image_path=img_in_path,
                    categorical_name=categorical_name,
                    block_size=25,
                    min_area_ratio=args.cv_min_area_ratio,
                    max_area_ratio=args.cv_max_area_ratio,
                    dilation_size=args.cv_dilation_size,
                    verbose=args.verbose,
                    box_threshold=args.box_threshold,
                    bbox_height_ratios=dino_height_ratios,
                    device=args.gpu_device,
                    return_bbox_info=True
                )
                image_masks_cache[img_in_path] = masks_dict
                image_bbox_stats[img_in_path] = bboxes_info
            except Exception as e:
                print(f"Error generating DINO mask for {img_in_path}: {e}")

        # Compile and log contour bbox statistics and plot histogram
        if image_bbox_stats:
            os.makedirs(args.output_dir, exist_ok=True)
            json_log_path = os.path.join(args.output_dir, "dino_contour_bbox_stats.json")
            hist_plot_path = os.path.join(args.output_dir, "dino_contour_bbox_histogram.png")

            all_raw_ratios = []
            per_ratio_ratios = {r: [] for r in dino_height_ratios}
            per_image_logs = {}

            for img_path, bboxes in image_bbox_stats.items():
                raw_ratios = [b["raw_area_ratio"] for b in bboxes]
                all_raw_ratios.extend(raw_ratios)

                eff_logs = {}
                for r in dino_height_ratios:
                    r_ratios = [b["effective_area_ratios"].get(r, b["raw_area_ratio"] * r) for b in bboxes]
                    per_ratio_ratios[r].extend(r_ratios)
                    eff_logs[str(r)] = {
                        "ratios": r_ratios,
                        "avg_ratio": float(sum(r_ratios) / len(r_ratios)) if r_ratios else 0.0
                    }

                per_image_logs[img_path] = {
                    "num_contour_bboxes": len(bboxes),
                    "raw_bbox_ratios": raw_ratios,
                    "avg_raw_bbox_ratio": float(sum(raw_ratios) / len(raw_ratios)) if raw_ratios else 0.0,
                    "effective_ratios_by_height_ratio": eff_logs
                }

            import numpy as np
            dataset_summary = {
                "total_images": len(discovered_items),
                "images_processed": len(image_bbox_stats),
                "total_contour_bboxes": len(all_raw_ratios),
                "avg_raw_bbox_ratio": float(np.mean(all_raw_ratios)) if all_raw_ratios else 0.0,
                "median_raw_bbox_ratio": float(np.median(all_raw_ratios)) if all_raw_ratios else 0.0,
                "std_raw_bbox_ratio": float(np.std(all_raw_ratios)) if all_raw_ratios else 0.0,
                "min_raw_bbox_ratio": float(np.min(all_raw_ratios)) if all_raw_ratios else 0.0,
                "max_raw_bbox_ratio": float(np.max(all_raw_ratios)) if all_raw_ratios else 0.0,
                "height_ratio_summaries": {}
            }

            for r in dino_height_ratios:
                r_list = per_ratio_ratios[r]
                dataset_summary["height_ratio_summaries"][str(r)] = {
                    "avg_effective_bbox_ratio": float(np.mean(r_list)) if r_list else 0.0,
                    "median_effective_bbox_ratio": float(np.median(r_list)) if r_list else 0.0,
                    "std_effective_bbox_ratio": float(np.std(r_list)) if r_list else 0.0
                }

            full_report = {
                "dataset_summary": dataset_summary,
                "per_image_logs": per_image_logs
            }

            with open(json_log_path, "w") as f:
                json.dump(full_report, f, indent=2)

            print(f"\n=============================================================")
            print(f"DINO Contour BBox Mask Statistics Summary")
            print(f"=============================================================")
            print(f"Total Source Images Scanned:     {len(discovered_items)}")
            print(f"Total Contour BBoxes Detected:  {len(all_raw_ratios)}")
            if len(all_raw_ratios) > 0:
                print(f"Avg Raw Contour BBox Area Fraction:  {dataset_summary['avg_raw_bbox_ratio']:.6f} ({dataset_summary['avg_raw_bbox_ratio']*100:.4f}%)")
                print(f"Median Contour BBox Area Fraction:   {dataset_summary['median_raw_bbox_ratio']:.6f} ({dataset_summary['median_raw_bbox_ratio']*100:.4f}%)")
                print(f"Min / Max Contour BBox Area Fraction:{dataset_summary['min_raw_bbox_ratio']:.6f} / {dataset_summary['max_raw_bbox_ratio']:.6f}")
                for r in dino_height_ratios:
                    r_avg = dataset_summary['height_ratio_summaries'][str(r)]['avg_effective_bbox_ratio']
                    print(f"  Height Ratio {r} Avg BBox Fraction: {r_avg:.6f} ({r_avg*100:.4f}%)")
            print(f"Saved JSON bbox log to: {json_log_path}")

            if all_raw_ratios:
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 6))
                    for r in dino_height_ratios:
                        r_list = per_ratio_ratios[r]
                        mean_val = float(np.mean(r_list))
                        plt.hist(r_list, bins=50, alpha=0.6, label=f"Height Ratio {r} (mean={mean_val:.5f})")

                    plt.title("Distribution of Contour BBox Area relative to Image Size (DINO Adaptive Mode)", fontsize=13)
                    plt.xlabel("BBox Area Fraction (BBox Area / Image Area)", fontsize=11)
                    plt.ylabel("Frequency (Count of BBoxes)", fontsize=11)
                    plt.grid(True, linestyle="--", alpha=0.5)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(hist_plot_path, dpi=300)
                    plt.close()
                    print(f"Saved BBox size histogram plot to: {hist_plot_path}")
                except Exception as e:
                    print(f"Failed to generate bbox histogram plot: {e}")
            print(f"=============================================================\n")

        # Compile and log contour bbox statistics and plot histogram
        if image_bbox_stats:
            os.makedirs(args.output_dir, exist_ok=True)
            json_log_path = os.path.join(args.output_dir, "dino_contour_bbox_stats.json")
            hist_plot_path = os.path.join(args.output_dir, "dino_contour_bbox_histogram.png")

            all_raw_ratios = []
            per_ratio_ratios = {r: [] for r in dino_height_ratios}
            per_image_logs = {}

            for img_path, bboxes in image_bbox_stats.items():
                raw_ratios = [b["raw_area_ratio"] for b in bboxes]
                all_raw_ratios.extend(raw_ratios)

                eff_logs = {}
                for r in dino_height_ratios:
                    r_ratios = [b["effective_area_ratios"].get(r, b["raw_area_ratio"] * r) for b in bboxes]
                    per_ratio_ratios[r].extend(r_ratios)
                    eff_logs[str(r)] = {
                        "ratios": r_ratios,
                        "avg_ratio": float(sum(r_ratios) / len(r_ratios)) if r_ratios else 0.0
                    }

                per_image_logs[img_path] = {
                    "num_contour_bboxes": len(bboxes),
                    "raw_bbox_ratios": raw_ratios,
                    "avg_raw_bbox_ratio": float(sum(raw_ratios) / len(raw_ratios)) if raw_ratios else 0.0,
                    "effective_ratios_by_height_ratio": eff_logs
                }

            import numpy as np
            dataset_summary = {
                "total_images": len(discovered_items),
                "images_processed": len(image_bbox_stats),
                "total_contour_bboxes": len(all_raw_ratios),
                "avg_raw_bbox_ratio": float(np.mean(all_raw_ratios)) if all_raw_ratios else 0.0,
                "median_raw_bbox_ratio": float(np.median(all_raw_ratios)) if all_raw_ratios else 0.0,
                "std_raw_bbox_ratio": float(np.std(all_raw_ratios)) if all_raw_ratios else 0.0,
                "min_raw_bbox_ratio": float(np.min(all_raw_ratios)) if all_raw_ratios else 0.0,
                "max_raw_bbox_ratio": float(np.max(all_raw_ratios)) if all_raw_ratios else 0.0,
                "height_ratio_summaries": {}
            }

            for r in dino_height_ratios:
                r_list = per_ratio_ratios[r]
                dataset_summary["height_ratio_summaries"][str(r)] = {
                    "avg_effective_bbox_ratio": float(np.mean(r_list)) if r_list else 0.0,
                    "median_effective_bbox_ratio": float(np.median(r_list)) if r_list else 0.0,
                    "std_effective_bbox_ratio": float(np.std(r_list)) if r_list else 0.0
                }

            full_report = {
                "dataset_summary": dataset_summary,
                "per_image_logs": per_image_logs
            }

            with open(json_log_path, "w") as f:
                json.dump(full_report, f, indent=2)

            print(f"\n=============================================================")
            print(f"DINO Contour BBox Mask Statistics Summary")
            print(f"=============================================================")
            print(f"Total Source Images Scanned:     {len(discovered_items)}")
            print(f"Total Contour BBoxes Detected:  {len(all_raw_ratios)}")
            if len(all_raw_ratios) > 0:
                print(f"Avg Raw Contour BBox Area Fraction:  {dataset_summary['avg_raw_bbox_ratio']:.6f} ({dataset_summary['avg_raw_bbox_ratio']*100:.4f}%)")
                print(f"Median Contour BBox Area Fraction:   {dataset_summary['median_raw_bbox_ratio']:.6f} ({dataset_summary['median_raw_bbox_ratio']*100:.4f}%)")
                print(f"Min / Max Contour BBox Area Fraction:{dataset_summary['min_raw_bbox_ratio']:.6f} / {dataset_summary['max_raw_bbox_ratio']:.6f}")
                for r in dino_height_ratios:
                    r_avg = dataset_summary['height_ratio_summaries'][str(r)]['avg_effective_bbox_ratio']
                    print(f"  Height Ratio {r} Avg BBox Fraction: {r_avg:.6f} ({r_avg*100:.4f}%)")
            print(f"Saved JSON bbox log to: {json_log_path}")

            if all_raw_ratios:
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 6))
                    for r in dino_height_ratios:
                        r_list = per_ratio_ratios[r]
                        mean_val = float(np.mean(r_list))
                        plt.hist(r_list, bins=50, alpha=0.6, label=f"Height Ratio {r} (mean={mean_val:.5f})")

                    plt.title("Distribution of Contour BBox Area relative to Image Size (DINO Adaptive Mode)", fontsize=13)
                    plt.xlabel("BBox Area Fraction (BBox Area / Image Area)", fontsize=11)
                    plt.ylabel("Frequency (Count of BBoxes)", fontsize=11)
                    plt.grid(True, linestyle="--", alpha=0.5)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(hist_plot_path, dpi=300)
                    plt.close()
                    print(f"Saved BBox size histogram plot to: {hist_plot_path}")
                except Exception as e:
                    print(f"Failed to generate bbox histogram plot: {e}")
            print(f"=============================================================\n")

    # Outer loop across scale factors
    for scale in scale_factors:
        print(f"\n=============================================================")
        print(f"[SCALE SESSION] Processing Scale Factor: {scale}x")
        print(f"=============================================================")

        # Pre-generate single-step SDXL denoised images (single pass per scale factor per image)
        denoised_cache = {}
        if not args.dry_run:
            print(f"Generating SDXL single-step noise & denoise images for scale {scale}x...")
            for cat, asin, item, img_in_path in tqdm(discovered_items, desc=f"SDXL Denoise {scale}x"):
                try:
                    degraded, denoised = single_step_denoise_workflow(
                        image_path=img_in_path,
                        scale_factor=scale,
                        noise_timestep=args.noise_timestep,
                        debug=False
                    )
                    orig_img = Image.open(img_in_path).convert("RGB")
                    denoised_resized = denoised.resize(orig_img.size, LANCZOS)
                    degraded_resized = degraded.resize(orig_img.size, LANCZOS)
                    denoised_cache[img_in_path] = (orig_img, degraded_resized, denoised_resized)
                except Exception as e:
                    print(f"Error in single_step_denoise_workflow for {img_in_path}: {e}")

        # Apply masks and save outputs for each height ratio
        for height_ratio in dino_height_ratios:
            if args.mask_method == "ocr":
                mask_suffix = f"ocr-{args.ocr_sample_ratio}"
            elif args.mask_method == "dino_adaptive":
                mask_suffix = f"dino-adaptive_height-ratio-{height_ratio}"
            else:
                mask_suffix = "opencv"

            dynamic_output_dir = os.path.join(
                args.output_dir,
                f"noised-{scale}x_masked-{mask_suffix}_timestep-{args.noise_timestep}"
            )

            tasks = []
            for cat, asin, item, img_in_path in discovered_items:
                if args.target_mode == "reference":
                    img_out_path = os.path.join(dynamic_output_dir, cat, asin, "noised.jpg")
                else:
                    img_out_path = os.path.join(dynamic_output_dir, cat, asin, item)
                tasks.append((img_in_path, img_out_path))

            print(f"\n-------------------------------------------------------------")
            print(f"[RUNNING] Scale: {scale}x | Height Ratio: {height_ratio} | Output: {dynamic_output_dir}")
            print(f"--------------------------------------------------------------")

            pending_tasks = [(in_p, out_p) for in_p, out_p in tasks if args.overwrite or not os.path.exists(out_p)]
            skipped_count = len(tasks) - len(pending_tasks)

            if args.dry_run:
                print(f"[Dry Run] Total tasks: {len(tasks)}, Existing (to skip): {skipped_count}, Pending: {len(pending_tasks)}")
                for in_p, out_p in pending_tasks[:10]:
                    print(f"  Input:  {in_p}")
                    print(f"  Output: {out_p}")
                if len(pending_tasks) > 10:
                    print(f"  ... and {len(pending_tasks) - 10} more pending tasks.")
                continue

            if not pending_tasks:
                print(f"All {len(tasks)} tasks already completed. Skipping.")
                continue

            processed_count = 0
            for img_in_path, img_out_path in tqdm(tasks, desc=f"Compositing Scale {scale}x | Ratio {height_ratio}"):
                if not args.overwrite and os.path.exists(img_out_path):
                    continue

                os.makedirs(os.path.dirname(img_out_path), exist_ok=True)

                try:
                    # Retrieve mask
                    if args.mask_method == "ocr":
                        mask = generate_ocr_mask(
                            image_path=img_in_path,
                            padding=args.ocr_padding,
                            sample_ratio=args.ocr_sample_ratio
                        )
                    elif args.mask_method == "dino_adaptive":
                        mask = image_masks_cache[img_in_path][height_ratio]
                    else:
                        mask = generate_contour_mask(
                            image_path=img_in_path,
                            min_area_ratio=args.cv_min_area_ratio,
                            max_area_ratio=args.cv_max_area_ratio,
                            threshold_value=args.cv_threshold,
                            dilation_size=args.cv_dilation_size
                        )

                    # Retrieve cached denoised components
                    orig_img, degraded_resized, denoised_resized = denoised_cache[img_in_path]

                    # Composite and save
                    result = Image.composite(denoised_resized, orig_img, mask)
                    result.save(img_out_path)

                    out_dir_path = os.path.dirname(img_out_path)
                    in_dir_path = os.path.dirname(img_in_path)
                    img_basename = os.path.basename(img_in_path)

                    if args.verbose:
                        img_name_no_ext, _ = os.path.splitext(img_basename)
                        mask_save_path = os.path.join(out_dir_path, f"{img_name_no_ext}_mask.png")
                        mask.save(mask_save_path)

                        noised_save_path = os.path.join(out_dir_path, f"{img_name_no_ext}_noised.jpg")
                        degraded_resized.save(noised_save_path)

                    if args.target_mode == "variation":
                        ref_in_path = os.path.join(in_dir_path, "reference.jpg")
                        if os.path.exists(ref_in_path):
                            ref_out_path = os.path.join(out_dir_path, "reference.jpg")
                            if not os.path.exists(ref_out_path):
                                shutil.copy(ref_in_path, ref_out_path)
                        gt_out_path = os.path.join(out_dir_path, f"GT_{img_basename}")
                        shutil.copy(img_in_path, gt_out_path)
                    elif args.target_mode == "reference":
                        ref_out_path = os.path.join(out_dir_path, "reference.jpg")
                        shutil.copy(img_in_path, ref_out_path)

                    processed_count += 1
                except Exception as e:
                    print(f"Error processing {img_in_path}: {e}")

            print(f"Finished Scale {scale}x | Ratio {height_ratio}: Processed {processed_count}, Skipped {skipped_count}.")

    print(f"\n=============================================================")
    print(f"All requested scale factors and ratios completed successfully!")
    print(f"=============================================================")

if __name__ == "__main__":
    main()
