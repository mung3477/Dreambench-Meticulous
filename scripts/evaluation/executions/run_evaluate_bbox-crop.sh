#!/bin/bash
set -e

# ==============================================================================
# Base Configuration & Paths
# ==============================================================================
REF_DIR="/root/Desktop/workspace/woosung/commercial-dreambench/assets/data/amzn"
RUBRICS_DIR="/root/Desktop/workspace/woosung/commercial-dreambench/assets/rubrics/amzn"

SAMPLING_MODEL_DIR="amzn_distorted/flux-klein"
SAMPLES_BASE_DIR="/root/Desktop/workspace/woosung/commercial-dreambench/samples/${SAMPLING_MODEL_DIR}"
OUTPUT_BASE_DIR="/root/Desktop/workspace/woosung/commercial-dreambench/outputs/bbox_crop_eval/${SAMPLING_MODEL_DIR}"
RATING_BASE_DIR="/root/Desktop/workspace/woosung/commercial-dreambench/rating/${SAMPLING_MODEL_DIR}"

# Shared cache file across all models (reuses Reference image detections)
SHARED_CACHE_FILE="${OUTPUT_BASE_DIR}/bbox_detections_cache.json"
# ANCHOR_CACHE_DIR_REF="/root/Desktop/workspace/woosung/commercial-dreambench/outputs/visual_likert_anchors_from_distorted_ref"
# ANCHOR_CACHE_DIR_CROP="/root/Desktop/workspace/woosung/commercial-dreambench/outputs/visual_likert_anchors"
# ANCHOR_CACHE_DIR="${ANCHOR_CACHE_DIR_REF}"

VENV_VLLM="/root/Desktop/workspace/woosung/commercial-dreambench/.venv-vllm"
VENV_RERANKER="/root/Desktop/workspace/woosung/commercial-dreambench/.venv-qwen-reranker"
VENV_FLOWFIXER="/root/Desktop/workspace/woosung/commercial-dreambench/.venv-flowfixer"

# Models/Distortions to evaluate
SAMPLING_MODELS=(noised-0.375x_masked-dino-adaptive_height-ratio-1.0_timestep-10)

BBOX_MARGIN=0.15
MIN_CROP_SIZE=224
METRICS=${METRICS:-"ours"}
CUDA_DEVICE="${CUDA_DEVICE:-1}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"

# Parse CLI arguments
EXTRA_ARGS=()
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --judge_model|--judge-model)
            JUDGE_MODEL="$2"
            shift 2
            ;;
        --cuda|--cuda_device)
            CUDA_DEVICE="$2"
            shift 2
            ;;
        --metrics)
            METRICS="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# mkdir -p "${OUTPUT_BASE_DIR}" "${RATING_BASE_DIR}" "${ANCHOR_CACHE_DIR}" "${ANCHOR_CACHE_DIR_CROP}"
mkdir -p "${OUTPUT_BASE_DIR}" "${RATING_BASE_DIR}"

# ==============================================================================
# Metric Partitioning across Virtual Environments
# ==============================================================================
VLLM_METRICS=()
EMBED_METRICS=()
NEED_ANCHOR_GEN=false

for m in $METRICS; do
    if [[ "$m" =~ ^(ours|visual-likert-scale_vlm)$ ]]; then
        VLLM_METRICS+=("$m")
    elif [[ "$m" =~ ^(clip|dino|qwen_reranker|visual-likert-scale_.*)$ ]]; then
        EMBED_METRICS+=("$m")
    fi

    if [[ "$m" =~ ^visual-likert-scale_ ]]; then
        NEED_ANCHOR_GEN=true
    fi
done

echo "======================================================================"
echo " BBox-Crop Unified Evaluation Orchestrator"
echo " CUDA Device:             ${CUDA_DEVICE}"
echo " Judge Model:             ${JUDGE_MODEL}"
echo " Models to evaluate:      ${SAMPLING_MODELS[*]}"
echo " Anchor Generation:       ${NEED_ANCHOR_GEN}"
echo " VLM Metrics (.venv-vllm): ${VLLM_METRICS[*]:-None}"
echo " Embedding Metrics:       ${EMBED_METRICS[*]:-None}"
echo "======================================================================"

# ==============================================================================
# Multi-Model Evaluation Loop
# ==============================================================================
for MODEL in "${SAMPLING_MODELS[@]}"; do
    SDG_DIR="${SAMPLES_BASE_DIR}/${MODEL}"
    MODEL_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL}"
    MODEL_RATING_DIR="${RATING_BASE_DIR}/${MODEL}"

    if [ ! -d "${SDG_DIR}" ]; then
        echo "[Warning] SDG directory not found: ${SDG_DIR}. Skipping..."
        continue
    fi

    mkdir -p "${MODEL_OUTPUT_DIR}" "${MODEL_RATING_DIR}"

    echo ""
    echo "======================================================================"
    echo " Evaluating Model/Distortion: ${MODEL}"
    echo " SDG Directory:    ${SDG_DIR}"
    echo " Output Directory: ${MODEL_OUTPUT_DIR}"
    echo " Rating Directory: ${MODEL_RATING_DIR}"
    echo "======================================================================"

    # --------------------------------------------------------------------------
    # Step 1: High-Throughput BBox Detection (vLLM) -> Populates Shared Cache
    # --------------------------------------------------------------------------
    echo ">>> [Step 1/4] Running BBox Detection via vLLM for ${MODEL}..."
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${VENV_VLLM}/bin/python" /root/Desktop/workspace/woosung/commercial-dreambench/scripts/evaluation/evaluate_bbox-crop_efficient.py \
        --ref_dir "${REF_DIR}" \
        --sdg_dir "${SDG_DIR}" \
        --rubrics_dir "${RUBRICS_DIR}" \
        --output_dir "${MODEL_OUTPUT_DIR}" \
        --rating_dir "${MODEL_RATING_DIR}" \
        --cache_file "${SHARED_CACHE_FILE}" \
        --anchor_cache_dir "${ANCHOR_CACHE_DIR}" \
        --backend vllm \
        --metrics none \
        "${EXTRA_ARGS[@]}"

    # --------------------------------------------------------------------------
    # Step 2: Visual Likert Anchor Precomputation / Extraction
    # --------------------------------------------------------------------------
    # if [ "$NEED_ANCHOR_GEN" = true ]; then
    #     if [[ "$METRICS" =~ visual-likert-scale_crop-distorted ]]; then
    #         echo ">>> [Step 2/4] Precomputing Visual Likert Reference Anchors via Live SDXL Diffusion (.venv-flowfixer)..."
    #         CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${VENV_FLOWFIXER}/bin/python" /root/Desktop/workspace/woosung/commercial-dreambench/scripts/evaluation/generate_visual_likert_anchors.py \
    #             --ref_dir "${REF_DIR}" \
    #             --rubrics_dir "${RUBRICS_DIR}" \
    #             --cache_file "${SHARED_CACHE_FILE}" \
    #             --anchor_cache_dir "${ANCHOR_CACHE_DIR_CROP}" \
    #             --bbox_margin "${BBOX_MARGIN}" \
    #             --min_crop_size "${MIN_CROP_SIZE}" \
    #             "${EXTRA_ARGS[@]}"
    #     else
    #         echo ">>> [Step 2/4] Extracting Visual Likert Reference Anchors from Pre-Distorted Full Images..."
    #         python3 /root/Desktop/workspace/woosung/commercial-dreambench/scripts/evaluation/generate_visual_likert_anchors_from_distorted_ref.py \
    #             --ref_dir "${REF_DIR}" \
    #             --rubrics_dir "${RUBRICS_DIR}" \
    #             --cache_file "${SHARED_CACHE_FILE}" \
    #             --anchor_cache_dir "${ANCHOR_CACHE_DIR_REF}" \
    #             --bbox_margin "${BBOX_MARGIN}" \
    #             --min_crop_size "${MIN_CROP_SIZE}" \
    #             "${EXTRA_ARGS[@]}"
    #     fi
    # fi

    # --------------------------------------------------------------------------
    # Step 3: VLM-based Evaluation (Ours, Visual-Likert-VLM via .venv-vllm)
    # --------------------------------------------------------------------------
    if [ ${#VLLM_METRICS[@]} -gt 0 ]; then
        echo ">>> [Step 3/4] Running VLM Metrics (${VLLM_METRICS[*]}) with judge ${JUDGE_MODEL} via vLLM for ${MODEL}..."
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${VENV_VLLM}/bin/python" /root/Desktop/workspace/woosung/commercial-dreambench/scripts/evaluation/evaluate_bbox-crop_efficient.py \
            --ref_dir "${REF_DIR}" \
            --sdg_dir "${SDG_DIR}" \
            --rubrics_dir "${RUBRICS_DIR}" \
            --output_dir "${MODEL_OUTPUT_DIR}" \
            --rating_dir "${MODEL_RATING_DIR}" \
            --cache_file "${SHARED_CACHE_FILE}" \
            --judge_model "${JUDGE_MODEL}" \
            --backend vllm \
            --metrics "${VLLM_METRICS[@]}" \
            --bbox_margin "${BBOX_MARGIN}" \
            --min_crop_size "${MIN_CROP_SIZE}" \
            --save_visualizations \
            --skip_if_done \
            "${EXTRA_ARGS[@]}"
            # --skip_if_done \
            #        --anchor_cache_dir "${ANCHOR_CACHE_DIR}" \
    fi

    # --------------------------------------------------------------------------
    # Step 4: Embedding & CrossEncoder Evaluation (CLIP / DINO / Qwen-Reranker / Visual-Likert)
    # --------------------------------------------------------------------------
    if [ ${#EMBED_METRICS[@]} -gt 0 ]; then
        CURRENT_ANCHOR_DIR="${ANCHOR_CACHE_DIR_REF}"
        if [[ "${EMBED_METRICS[*]}" =~ visual-likert-scale_crop-distorted ]]; then
            CURRENT_ANCHOR_DIR="${ANCHOR_CACHE_DIR_CROP}"
        fi

        echo ">>> [Step 4/4] Running Embedding/Reranker Metrics (${EMBED_METRICS[*]}) for ${MODEL}..."
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${VENV_RERANKER}/bin/python" /root/Desktop/workspace/woosung/commercial-dreambench/scripts/evaluation/evaluate_bbox-crop_efficient.py \
            --ref_dir "${REF_DIR}" \
            --sdg_dir "${SDG_DIR}" \
            --rubrics_dir "${RUBRICS_DIR}" \
            --output_dir "${MODEL_OUTPUT_DIR}" \
            --rating_dir "${MODEL_RATING_DIR}" \
            --cache_file "${SHARED_CACHE_FILE}" \
            --anchor_cache_dir "${CURRENT_ANCHOR_DIR}" \
            --metrics "${EMBED_METRICS[@]}" \
            --bbox_margin "${BBOX_MARGIN}" \
            --min_crop_size "${MIN_CROP_SIZE}" \
            --save_visualizations \
            --skip_if_done \
            "${EXTRA_ARGS[@]}"
            # --skip_if_done \
    fi

    echo ">>> Finished evaluation for ${MODEL}. Ratings saved to ${MODEL_RATING_DIR}/*_bbox-crop_results.json"
done

echo ""
echo "======================================================================"
echo " All evaluations completed across models: ${SAMPLING_MODELS[*]}"
echo " Rating outputs stored under: ${RATING_BASE_DIR}"
echo "======================================================================"
