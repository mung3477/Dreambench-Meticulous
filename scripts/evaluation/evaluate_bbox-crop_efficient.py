#!/usr/bin/env python3
"""
Efficient Bounding Box Crop Multi-Metric Evaluation Pipeline
============================================================
Evaluates visual consistency between Reference and SDG (Synthetic Data Generation) images
by detecting rubric-specified object categories, extracting context-aware high-resolution crops,
and comparing crop pairs across multiple metrics (CLIP, DINOv2, Qwen-reranker, Ours-VLM, Visual-Likert-Scale) in a single pass.

Features:
- Memory-Safe Execution:
  - Chunked mini-batching for vLLM detection (prevents holding thousands of PIL images in RAM).
  - Lazy on-the-fly crop extraction during metric evaluation (prevents RAM ballooning).
  - Incremental cache saving per chunk.
- Strict Variation Filtering: Evaluates ONLY variation_N.jpg images against clean reference.jpg from REF_DIR.
- Primary Backbones:
  - BBox Detector: Qwen/Qwen3-VL-32B-Instruct (vLLM / Transformers)
  - Reranker Evaluator: Qwen/Qwen3-VL-Reranker-2B (sentence_transformers.CrossEncoder)
  - Ours VLM Judge: Qwen/Qwen3-VL-32B-Instruct with 5-level uniform scale (Very Poor=0 to Excellent=4).
  - Visual-Likert-Scale Evaluator: Qwen-Reranker & VLM judging against 8 flowfixer-distorted reference anchor levels.
- Full Post-Mortem Qualitative Logging: Records `status` and `visual_evidence` alongside numeric scores in `evaluation_results.json`.
- Exact prompt matching with concrete category prefix example.
- BBox coordinate mapping matching notebook Cell 22 (bbox[0]=x1, bbox[1]=y1, bbox[2]=x2, bbox[3]=y2)
- Unified metric evaluation supporting both legacy and modern Transformers model outputs (CLIP/DINO).
- Category & ASIN filtering for focused dry-runs and subsets.
- Rating files saved in standard benchmark flat dictionary format: `{ "key": score, ... }`
  as `{metric}_bbox-crop_results.json` under `--rating_dir`.
- Strict `--skip_if_done`: Verifies every target item key is present with a valid numeric score in all requested rating files.
- Deterministic BBox caching (Greedy decoding, temperature=0.0) to eliminate redundant VLM passes.
- Visual BBox overlay logging with colored category bounding boxes and labels.
- Rubric matching based on `rubric['title']` and `description.split(":")[0]`.
- Spatial context margin expansion (`--bbox_margin`) with image boundary clamping.
- High-quality Lanczos upscaling (`--min_crop_size` / `--crop_upscale_factor`) to prevent ViT patch collapse.
- Vectorized batch metric evaluations for maximum GPU throughput.
- Robust JSON parsing, inverted bbox recovery, and missing-detection penalty handling.
- Uniform score aggregation: (Sum of matched crop scores / Total maximum possible score).
"""

import os
import sys
import json
import re
import math
import gc
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ==============================================================================
# 1. Rubric Loading & Formatting
# ==============================================================================

def load_rubric(rubric_path: Union[str, Path]) -> List[Dict[str, str]]:
    rubric_path = Path(rubric_path)
    if not rubric_path.exists():
        raise FileNotFoundError(f"Rubric file not found at: {rubric_path}")

    with open(rubric_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, dict) and "rubrics" in data:
                return data["rubrics"]
            elif isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [{"title": k, "description": v} for k, v in data.items()]
        except json.JSONDecodeError:
            f.seek(0)
            items = []
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
            return items
    return []


def load_all_rubrics(rubrics_dir: Union[str, Path]) -> Dict[str, List[Dict[str, str]]]:
    rubrics_dir = Path(rubrics_dir)
    rubrics_map = {}
    if not rubrics_dir.exists():
        print(f"[Warning] Rubrics directory {rubrics_dir} does not exist.")
        return rubrics_map

    for r_file in rubrics_dir.rglob("*.json"):
        try:
            r_data = load_rubric(r_file)
            rubrics_map[r_file.stem] = r_data
            rubrics_map[f"{r_file.parent.name}/{r_file.stem}"] = r_data
        except Exception as e:
            print(f"[Warning] Failed to load rubric {r_file}: {e}")
    return rubrics_map


def build_detection_prompt(rubrics: List[Dict[str, str]]) -> str:
    titles = [rubric['title'] for rubric in rubrics if 'title' in rubric]
    if not titles:
        titles = [str(r) for r in rubrics]

    sample_category = titles[0] if titles else "category_name"

    user_prompt = """
	Locate every instance that belongs to the following categories: "{categories}".
	Report bbox coordinates in JSON format.

	<output_format>
	```json
	[
 		{{"bbox": [y1 x1 y2 x2], "description": "{category_example}: detailed description of visual appearance..."}},
		...
	]
	```
	</output_format>
""".format(categories=titles, category_example=sample_category)
    return user_prompt


# ==============================================================================
# 2. BBox Parsing, Normalization & Visual Overlay Logging
# ==============================================================================

def extract_label_from_description(description: str) -> str:
    if not description:
        return ""
    cleaned = description.strip().strip('"\'')
    label = cleaned.split(":")[0].strip()
    return label


def parse_vlm_bbox_json(raw_text: str, image_width: int, image_height: int) -> List[Dict[str, Any]]:
    if not raw_text or not isinstance(raw_text, str):
        return []

    text = raw_text.strip()
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if json_match:
        text = json_match.group(1).strip()

    start_idx = text.find("[")
    end_idx = text.rfind("]")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx:end_idx + 1]

    parsed_boxes = []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            parsed_boxes = data
    except Exception:
        pattern = re.compile(r'\{[^{}]*?"bbox"\s*:\s*\[\s*([0-9.,\s]+)\s*\][^{}]*?\}')
        for match in pattern.finditer(raw_text):
            try:
                block = match.group(0)
                if not block.endswith("}"):
                    block += "}"
                obj = json.loads(block)
                parsed_boxes.append(obj)
            except Exception:
                continue

    valid_boxes = []
    for item in parsed_boxes:
        if not isinstance(item, dict) or "bbox" not in item:
            continue
        bbox = item["bbox"]
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue

        try:
            c0, c1, c2, c3 = [float(v) for v in bbox]
        except (ValueError, TypeError):
            continue

        # In Qwen3-VL convention matching notebook Cell 22:
        # c0 = x1, c1 = y1, c2 = x2, c3 = y2 in 0-1000 scale
        if max(c0, c1, c2, c3) > 1.0 and max(c0, c1, c2, c3) <= 1000.0:
            abs_x1 = (c0 / 1000.0) * image_width
            abs_y1 = (c1 / 1000.0) * image_height
            abs_x2 = (c2 / 1000.0) * image_width
            abs_y2 = (c3 / 1000.0) * image_height
        elif max(c0, c1, c2, c3) <= 1.0:
            abs_x1 = c0 * image_width
            abs_y1 = c1 * image_height
            abs_x2 = c2 * image_width
            abs_y2 = c3 * image_height
        else:
            abs_x1, abs_y1, abs_x2, abs_y2 = c0, c1, c2, c3

        # Inverted box handling
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        # Boundary clamping
        abs_y1 = max(0.0, min(float(image_height), abs_y1))
        abs_y2 = max(0.0, min(float(image_height), abs_y2))
        abs_x1 = max(0.0, min(float(image_width), abs_x1))
        abs_x2 = max(0.0, min(float(image_width), abs_x2))

        if (abs_x2 - abs_x1) < 1.0 or (abs_y2 - abs_y1) < 1.0:
            continue

        desc = str(item.get("description", ""))
        label = item.get("label") or extract_label_from_description(desc)

        valid_boxes.append({
            "bbox": [round(abs_y1, 2), round(abs_x1, 2), round(abs_y2, 2), round(abs_x2, 2)],
            "description": desc,
            "label": label
        })

    return valid_boxes


PALETTE = [
    (231, 76, 60),
    (52, 152, 219),
    (46, 204, 113),
    (155, 89, 182),
    (241, 196, 15),
    (230, 126, 34),
    (26, 188, 156),
    (236, 240, 241),
]

def draw_bboxes_on_image(image: Image.Image, bboxes: List[Dict[str, Any]]) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, box in enumerate(bboxes):
        coords = box.get("bbox", [])
        if len(coords) != 4:
            continue
        y1, x1, y2, x2 = coords
        color = PALETTE[i % len(PALETTE)]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = box.get("label", "") or box.get("description", "")
        if label:
            text_str = label[:30]
            text_w = len(text_str) * 6 + 8
            text_h = 14
            draw.rectangle([x1, max(0, y1 - text_h), x1 + text_w, y1], fill=color)
            draw.text((x1 + 4, max(0, y1 - text_h) + 1), text_str, fill=(255, 255, 255), font=font)

    return annotated


# ==============================================================================
# 3. Context Margin & High-Res Crop Extraction
# ==============================================================================

def extract_context_crop(
    image: Image.Image,
    bbox: List[float],
    margin_ratio: float = 0.15,
    min_size: int = 224,
    upscale_factor: float = 1.0
) -> Image.Image:
    img_w, img_h = image.size
    y1, x1, y2, x2 = bbox

    box_w = x2 - x1
    box_h = y2 - y1

    margin_x = box_w * margin_ratio
    margin_y = box_h * margin_ratio

    exp_x1 = max(0, int(math.floor(x1 - margin_x)))
    exp_y1 = max(0, int(math.floor(y1 - margin_y)))
    exp_x2 = min(img_w, int(math.ceil(x2 + margin_x)))
    exp_y2 = min(img_h, int(math.ceil(y2 + margin_y)))

    if exp_x2 <= exp_x1:
        exp_x2 = min(img_w, exp_x1 + 10)
    if exp_y2 <= exp_y1:
        exp_y2 = min(img_h, exp_y1 + 10)

    crop = image.crop((exp_x1, exp_y1, exp_x2, exp_y2))

    crop_w, crop_h = crop.size
    target_scale = max(1.0, upscale_factor)
    if min_size > 0:
        min_dim = min(crop_w, crop_h)
        if min_dim < min_size:
            target_scale = max(target_scale, float(min_size) / float(max(1, min_dim)))

    if target_scale > 1.01:
        new_w = int(round(crop_w * target_scale))
        new_h = int(round(crop_h * target_scale))
        crop = crop.resize((new_w, new_h), resample=getattr(Image, "Resampling", Image).LANCZOS)

    return crop


# ==============================================================================
# 4. Visual-Likert-Scale Anchor Generator (SDXL DDIM One-Step FlowFixer)
# ==============================================================================

# Discrete Visual Likert Anchor Levels: (Level Name, Downscale Factor, Linear Calibrated Score)
VISUAL_LIKERT_LEVELS = [
    ("original", 1.0, 1.000),
    ("1.0x",     1.0, 0.875),
    ("0.875x",   0.875, 0.750),
    ("0.75x",    0.750, 0.625),
    ("0.625x",   0.625, 0.500),
    ("0.5x",     0.500, 0.375),
    ("0.375x",   0.375, 0.250),
    ("0.25x",    0.250, 0.125),
]


class VisualLikertAnchorGenerator:
    """
    Retrieves persistently cached visual Likert anchor crops from disk
    (precomputed via generate_visual_likert_anchors.py in .venv-flowfixer).
    """
    def __init__(self, cache_dir: Union[str, Path] = "outputs/visual_likert_anchors", device: str = "cuda"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

    def get_or_generate_anchors(
        self,
        ref_crop: Image.Image,
        category: str,
        asin: str,
        rubric_title: str
    ) -> List[Tuple[str, float, Image.Image]]:
        """
        Retrieves the 8 visual Likert anchors from disk cache.
        Returns: List of (level_name, score, anchor_crop)
        """
        safe_title = re.sub(r"[^\w\-_]", "_", rubric_title.lower().strip())
        anchor_dir = self.cache_dir / category / asin / safe_title

        anchors = []
        for level_name, scale_factor, score in VISUAL_LIKERT_LEVELS:
            anchor_file = anchor_dir / f"anchor_{level_name}.jpg"

            if anchor_file.exists():
                try:
                    img = Image.open(anchor_file).convert("RGB")
                    anchors.append((level_name, score, img))
                    continue
                except Exception as e:
                    print(f"[Warning] Error loading anchor {anchor_file}: {e}")

            # Fallback if anchor is missing before precomputation
            fallback_img = ref_crop.copy()
            anchors.append((level_name, score, fallback_img))

        return anchors


# ==============================================================================
# 5. Evaluator Backbones (CLIP, DINOv2, Qwen-reranker, Ours VLM Judge, Visual-Likert)
# ==============================================================================

class CLIPEvaluator:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        from transformers import CLIPModel, CLIPProcessor
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[Init] Loading CLIP Model ({model_name}) on {self.device}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.inference_mode()
    def compute_similarity_batch(self, ref_crops: List[Image.Image], sdg_crops: List[Image.Image]) -> List[float]:
        if not ref_crops or not sdg_crops:
            return []

        inputs_ref = self.processor(images=ref_crops, return_tensors="pt", padding=True).to(self.device)
        inputs_sdg = self.processor(images=sdg_crops, return_tensors="pt", padding=True).to(self.device)

        out_ref = self.model.get_image_features(**inputs_ref)
        out_sdg = self.model.get_image_features(**inputs_sdg)

        if hasattr(out_ref, "pooler_output") and out_ref.pooler_output is not None:
            ref_feat = out_ref.pooler_output
        elif hasattr(out_ref, "image_embeds") and out_ref.image_embeds is not None:
            ref_feat = out_ref.image_embeds
        elif isinstance(out_ref, torch.Tensor):
            ref_feat = out_ref
        else:
            ref_feat = out_ref[0]

        if hasattr(out_sdg, "pooler_output") and out_sdg.pooler_output is not None:
            sdg_feat = out_sdg.pooler_output
        elif hasattr(out_sdg, "image_embeds") and out_sdg.image_embeds is not None:
            sdg_feat = out_sdg.image_embeds
        elif isinstance(out_sdg, torch.Tensor):
            sdg_feat = out_sdg
        else:
            sdg_feat = out_sdg[0]

        ref_feat = F.normalize(ref_feat, p=2, dim=-1)
        sdg_feat = F.normalize(sdg_feat, p=2, dim=-1)

        sims = (ref_feat * sdg_feat).sum(dim=-1).cpu().tolist()
        return [max(0.0, min(1.0, (s + 1.0) / 2.0)) for s in sims]


class DINOEvaluator:
    def __init__(self, model_name: str = "facebook/dinov2-base", device: str = "cuda"):
        from transformers import AutoModel, AutoImageProcessor
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[Init] Loading DINOv2 Model ({model_name}) on {self.device}...")
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_name)

    @torch.inference_mode()
    def compute_similarity_batch(self, ref_crops: List[Image.Image], sdg_crops: List[Image.Image]) -> List[float]:
        if not ref_crops or not sdg_crops:
            return []

        inputs_ref = self.processor(images=ref_crops, return_tensors="pt").to(self.device)
        inputs_sdg = self.processor(images=sdg_crops, return_tensors="pt").to(self.device)

        out_ref = self.model(**inputs_ref)
        out_sdg = self.model(**inputs_sdg)

        ref_cls = F.normalize(out_ref.last_hidden_state[:, 0, :], p=2, dim=-1)
        sdg_cls = F.normalize(out_sdg.last_hidden_state[:, 0, :], p=2, dim=-1)

        sims = (ref_cls * sdg_cls).sum(dim=-1).cpu().tolist()
        return [max(0.0, min(1.0, (s + 1.0) / 2.0)) for s in sims]


class QwenRerankerEvaluator:
    """Evaluates crop pairs using Qwen3-VL-Reranker-2B via CrossEncoder."""
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-Reranker-2B", device: str = "cuda", reverse_image_order: bool = False):
        from sentence_transformers import CrossEncoder
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.reverse_image_order = reverse_image_order
        print(f"[Init] Loading Qwen3-VL-Reranker ({model_name}) on {self.device} (reverse_image_order={reverse_image_order})...")
        self.model = CrossEncoder(
            model_name,
            trust_remote_code=True,
            model_kwargs={'torch_dtype': torch.bfloat16} if torch.cuda.is_available() else {}
        )

    def compute_similarity_batch(
        self,
        ref_crops: List[Image.Image],
        sdg_crops: List[Image.Image],
        labels: Optional[List[str]] = None
    ) -> List[float]:
        if not ref_crops or not sdg_crops:
            return []

        pairs = list(zip(ref_crops, sdg_crops))
        if self.reverse_image_order:
            pairs = list(zip(sdg_crops, ref_crops))
        else:
            pairs = list(zip(ref_crops, sdg_crops))
        raw_scores = self.model.predict(pairs, batch_size=len(pairs), show_progress_bar=False)
        return [float(s) for s in raw_scores]


class VisualLikertQwenRerankerEvaluator:
    """
    Visual Likert Scale Evaluator using Qwen3-VL-Reranker-2B.
    Queries = SDG Crop, Documents = 8 Distorted Reference Anchors (original down to 0.25x).
    The best-matching anchor determines the Likert score (1.000 to 0.125).
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Reranker-2B",
        device: str = "cuda",
        anchor_cache_dir: Union[str, Path] = "outputs/visual_likert_anchors"
    ):
        from sentence_transformers import CrossEncoder
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        print(f"[Init] Loading Visual-Likert Qwen Reranker ({model_name}) on {self.device}...")
        self.model = CrossEncoder(
            model_name,
            trust_remote_code=True,
            model_kwargs={'torch_dtype': torch.bfloat16} if torch.cuda.is_available() else {}
        )
        self.anchor_gen = VisualLikertAnchorGenerator(cache_dir=anchor_cache_dir, device=device)

    def compute_similarity_batch(
        self,
        ref_crops: List[Image.Image],
        sdg_crops: List[Image.Image],
        labels: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        if not ref_crops or not sdg_crops:
            return [], []

        final_scores = []
        details = []

        for i, (ref_c, sdg_c) in enumerate(zip(ref_crops, sdg_crops)):
            meta = metadata[i] if metadata and i < len(metadata) else {}
            cat = meta.get("category", "common")
            asin = meta.get("asin", "item")
            lbl = labels[i] if labels and i < len(labels) else "main"

            # Retrieve/Generate the 8 discrete visual Likert anchors
            anchors = self.anchor_gen.get_or_generate_anchors(ref_c, category=cat, asin=asin, rubric_title=lbl)

            # Construct query-document pairs: (Query = SDG Crop, Document = Distorted Reference Anchor)
            pairs = [(sdg_c, anchor_crop) for _, _, anchor_crop in anchors]
            relevance_scores = self.model.predict(pairs, batch_size=len(pairs), show_progress_bar=False)

            best_idx = int(np.argmax(relevance_scores))
            best_level, best_score, _ = anchors[best_idx]

            final_scores.append(best_score)
            details.append({
                "status": f"Likert Anchor: {best_level}",
                "visual_evidence": (
                    f"Selected visual Likert anchor '{best_level}' with maximum relevance score "
                    f"({relevance_scores[best_idx]:.4f}) among 8 degradation levels."
                ),
                "selected_level": best_level,
                "anchor_scores": {lvl: round(float(s), 4) for (lvl, _, _), s in zip(anchors, relevance_scores)}
            })

        return final_scores, details


class VisualLikertVLMEvaluator:
    """
    Visual Likert Scale Evaluator using Vision-Language Models (Qwen3-VL-32B).
    Scaffold / Skeleton for multi-image Likert ranking & pairwise sweeps.
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-32B-Instruct",
        backend: str = "auto",
        anchor_cache_dir: Union[str, Path] = "outputs/visual_likert_anchors",
        **kwargs
    ):
        self.model_name = model_name
        self.backend = backend
        self.anchor_gen = VisualLikertAnchorGenerator(cache_dir=anchor_cache_dir)

    def evaluate_all_at_once(
        self,
        sdg_crop: Image.Image,
        anchor_crops: List[Tuple[str, float, Image.Image]]
    ) -> Tuple[float, Dict[str, Any]]:
        """Skeleton for presenting query crop alongside all distorted reference anchors simultaneously."""
        pass

    def evaluate_pairwise_sweep(
        self,
        sdg_crop: Image.Image,
        anchor_crops: List[Tuple[str, float, Image.Image]]
    ) -> Tuple[float, Dict[str, Any]]:
        """Skeleton for pairwise similarity comparison across each anchor level."""
        pass


FIVE_LEVEL_SCORES = {
    "Very Poor": 0.0,
    "Poor": 1.0,
    "Fair": 2.0,
    "Good": 3.0,
    "Excellent": 4.0,
    "Missing": 0.0,
    "Partial": 2.0,
    "Correct": 4.0,
}


def get_vlm_family(model_name: str) -> str:
    """Identifies the VLM model family based on checkpoint name or path."""
    name = (model_name or "").lower()
    if "internvl" in name:
        return "internvl"
    elif "glm" in name:
        return "glm"
    elif "qwen" in name:
        return "qwen"
    else:
        return "generic"


def adapt_prompt_for_reversed_order(prompt_text: str) -> str:
    """
    Adapts the evaluation prompt when image presentation order is reversed:
    Image 1 becomes the Generated Image and Image 2 becomes the Reference Image.
    Swaps all textual references between Image 1 (Reference) and Image 2 (Generated).
    """
    if not prompt_text:
        return prompt_text

    text = prompt_text

    # Two-stage placeholder replacement to prevent accidental collision
    pairs = [
        # Reference phrases (Image 1 -> Reference)
        (r"(?i)Reference Image \(Image\s*1\)", "__REF_IMG_P__"),
        (r"(?i)reference image \(Image\s*1\)", "__REF_IMG_P__"),
        (r"(?i)Image\s*1 \(Reference\)", "__REF_IMG_P__"),
        (r"(?i)Image\s*1 \(reference\)", "__REF_IMG_P__"),
        # Generated phrases (Image 2 -> Generated)
        (r"(?i)Generated Image \(Image\s*2\)", "__GEN_IMG_P__"),
        (r"(?i)generated image \(Image\s*2\)", "__GEN_IMG_P__"),
        (r"(?i)Image\s*2 \(Generated\)", "__GEN_IMG_P__"),
        (r"(?i)Image\s*2 \(generated\)", "__GEN_IMG_P__"),
    ]
    for pattern, placeholder in pairs:
        text = re.sub(pattern, placeholder, text)

    text = text.replace("__REF_IMG_P__", "Reference Image (Image 2)")
    text = text.replace("__GEN_IMG_P__", "Generated Image (Image 1)")
    return text


def format_vllm_judge_prompt(model_family: str, prompt_text: str, reverse_order: bool = False) -> str:
    """Formats the dual-image comparison prompt for vLLM according to model family conventions."""
    if reverse_order:
        effective_prompt = adapt_prompt_for_reversed_order(prompt_text)
        img1_label = "Generated Image (Image 1):"
        img2_label = "Reference Image (Image 2):"
    else:
        effective_prompt = prompt_text
        img1_label = "Reference Image (Image 1):"
        img2_label = "Generated Image (Image 2):"

    if model_family == "qwen":
        return (
            f"<|im_start|>user\n{effective_prompt}\n"
            f"{img1_label}\n<|vision_start|><|image_pad|><|vision_end|>\n"
            f"{img2_label}\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    elif model_family == "internvl":
        return (
            f"<|im_start|>user\n{effective_prompt}\n"
            f"{img1_label}\n<image>\n"
            f"{img2_label}\n<image><|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    elif model_family == "glm":
        return (
            f"[gMASK]<sop><|user|>\n{effective_prompt}\n"
            f"{img1_label}\n<|begin_of_image|><|end_of_image|>\n"
            f"{img2_label}\n<|begin_of_image|><|end_of_image|><|assistant|>\n"
        )
    else:
        return (
            f"<|im_start|>user\n{effective_prompt}\n"
            f"{img1_label}\n<image>\n"
            f"{img2_label}\n<image><|im_end|>\n"
            f"<|im_start|>assistant\n"
        )


class OursVLMEvaluator:
    """
    Evaluates crop pairs using 5-level VLM judging (Ours) based on user_prompt_evaluate_with_crops.txt.
    Supports multiple VLM families (Qwen, InternVL, GLM-4V, generic Vision2Seq).
    Scores: Very Poor=0.0, Poor=1.0, Fair=2.0, Good=3.0, Excellent=4.0 (normalized to 0.0-1.0).
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-32B-Instruct",
        prompt_path: Optional[Union[str, Path]] = "prompts/user_prompt_evaluate_with_crops.txt",
        backend: str = "auto",
        gpu_memory_utilization: float = 0.88,
        detector_engine: Optional[Any] = None,
        reverse_image_order: bool = False
    ):
        self.model_name = model_name
        self.backend = backend
        self.gpu_memory_utilization = gpu_memory_utilization
        self.reverse_image_order = reverse_image_order
        self.prompt_text = ""

        if prompt_path and Path(prompt_path).exists():
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.prompt_text = f.read().strip()
        else:
            self.prompt_text = (
                "You are an expert evaluator of subject identity preservation in commercial image generation.\n"
                "Evaluate the visual attribute in Image 1 (Reference) vs Image 2 (Generated).\n"
                "Assign a status: 'Very Poor' | 'Poor' | 'Fair' | 'Good' | 'Excellent' and return JSON:\n"
                '[{"title": "...", "status": "...", "visual_evidence": "..."}]'
            )

        self.engine_type = None
        self.vllm_engine = None
        self.vllm_sampling = None
        self.hf_model = None
        self.hf_processor = None

        # Only reuse detector engine if the model name matches exactly
        if (
            detector_engine
            and getattr(detector_engine, "engine_type", None) is not None
            and getattr(detector_engine, "model_name", None) == self.model_name
        ):
            self.engine_type = detector_engine.engine_type
            self.vllm_engine = detector_engine.vllm_engine
            self.vllm_sampling = detector_engine.vllm_sampling
            self.hf_model = detector_engine.hf_model
            self.hf_processor = detector_engine.hf_processor
        else:
            self._init_engine()

    def _init_engine(self):
        if self.engine_type is not None:
            return

        model_fam = get_vlm_family(self.model_name)

        if self.backend in ["auto", "vllm"]:
            try:
                from vllm import LLM, SamplingParams
                print(f"[Init Ours VLM] Initializing vLLM backend for {self.model_name} (family: {model_fam})...")
                self.vllm_engine = LLM(
                    model=self.model_name,
                    trust_remote_code=True,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    max_model_len=16384,
                    limit_mm_per_prompt={"image": 2}
                )
                self.vllm_sampling = SamplingParams(
                    temperature=0.0,
                    max_tokens=1024
                )
                self.engine_type = "vllm"
                return
            except Exception as e:
                print(f"[Engine Note] vLLM unavailable ({e}). Falling back to Transformers.")

        print(f"[Init Ours VLM] Initializing Hugging Face Transformers backend for {self.model_name} (family: {model_fam})...")
        from transformers import AutoProcessor, AutoTokenizer
        try:
            self.hf_processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception:
            try:
                self.hf_processor = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            except Exception as e:
                print(f"[Warning] Failed to load AutoProcessor/AutoTokenizer for {self.model_name}: {e}")
                self.hf_processor = None

        if model_fam == "qwen":
            cls = None
            try:
                from transformers import Qwen3VLForConditionalGeneration
                cls = Qwen3VLForConditionalGeneration
            except ImportError:
                try:
                    from transformers import Qwen2_5_VLForConditionalGeneration
                    cls = Qwen2_5_VLForConditionalGeneration
                except ImportError:
                    cls = None

            if cls is not None:
                try:
                    self.hf_model = cls.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.bfloat16,
                        device_map="auto"
                    ).eval()
                except Exception:
                    from transformers import AutoModelForVision2Seq
                    self.hf_model = AutoModelForVision2Seq.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.bfloat16,
                        device_map="auto"
                    ).eval()
            else:
                from transformers import AutoModelForVision2Seq
                self.hf_model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                ).eval()
        elif model_fam == "internvl":
            try:
                from transformers import AutoModel
                self.hf_model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    device_map="auto"
                ).eval()
            except Exception:
                from transformers import AutoModelForVision2Seq
                self.hf_model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="auto"
                ).eval()
        elif model_fam == "glm":
            try:
                from transformers import AutoModelForCausalLM
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="auto"
                ).eval()
            except Exception:
                from transformers import AutoModelForVision2Seq
                self.hf_model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="auto"
                ).eval()
        else:
            try:
                from transformers import AutoModelForVision2Seq
                self.hf_model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="auto"
                ).eval()
            except Exception:
                from transformers import AutoModel
                self.hf_model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="auto"
                ).eval()

        self.engine_type = "transformers"

    def parse_judge_output(self, raw_text: str, default_title: str = "") -> Tuple[float, str, str]:
        if not raw_text:
            return 0.0, "Missing", ""

        text = raw_text.strip()
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            text = json_match.group(1).strip()

        start_idx = text.find("[")
        end_idx = text.rfind("]")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            text = text[start_idx:end_idx + 1]
        elif text.find("{") != -1 and text.rfind("}") != -1:
            text = text[text.find("{"):text.rfind("}") + 1]

        status = "Fair"
        evidence = ""
        try:
            data = json.loads(text)
            if isinstance(data, list) and len(data) > 0:
                item = data[0]
            elif isinstance(data, dict):
                item = data
            else:
                item = {}
            status = str(item.get("status", item.get("verdict", "Fair"))).strip()
            evidence = str(item.get("visual_evidence", item.get("evidence", ""))).strip()
        except Exception:
            m_status = re.search(r'"status"\s*:\s*"([^"]+)"', raw_text, re.IGNORECASE)
            if m_status:
                status = m_status.group(1).strip()
            else:
                for lvl in ["Excellent", "Good", "Fair", "Poor", "Very Poor"]:
                    if lvl.lower() in raw_text.lower():
                        status = lvl
                        break

        score_val = 2.0
        for k, v in FIVE_LEVEL_SCORES.items():
            if k.lower() == status.lower():
                score_val = v
                break

        norm_score = score_val / 4.0
        return norm_score, status, evidence

    def compute_similarity_batch(
        self,
        ref_crops: List[Image.Image],
        sdg_crops: List[Image.Image],
        labels: Optional[List[str]] = None
    ) -> Tuple[List[float], List[Dict[str, str]]]:
        if not ref_crops or not sdg_crops:
            return [], []

        scores = []
        details = []

        model_fam = get_vlm_family(self.model_name)

        if self.engine_type == "vllm":
            vllm_inputs = []
            prompt = format_vllm_judge_prompt(model_fam, self.prompt_text, reverse_order=self.reverse_image_order)
            for ref_c, sdg_c in zip(ref_crops, sdg_crops):
                images = [sdg_c, ref_c] if self.reverse_image_order else [ref_c, sdg_c]
                vllm_inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {"image": images}
                })

            outputs = self.vllm_engine.generate(vllm_inputs, sampling_params=self.vllm_sampling)
            for i, out in enumerate(outputs):
                lbl = labels[i] if labels and i < len(labels) else ""
                raw_resp = out.outputs[0].text
                score, status, evidence = self.parse_judge_output(raw_resp, default_title=lbl)
                scores.append(score)
                details.append({"status": status, "visual_evidence": evidence})

        else:
            effective_prompt = adapt_prompt_for_reversed_order(self.prompt_text) if self.reverse_image_order else self.prompt_text
            for i, (ref_c, sdg_c) in enumerate(zip(ref_crops, sdg_crops)):
                lbl = labels[i] if labels and i < len(labels) else ""
                raw_resp = ""
                try:
                    if hasattr(self.hf_processor, "apply_chat_template"):
                        if self.reverse_image_order:
                            messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": effective_prompt},
                                        {"type": "text", "text": "Generated Image (Image 1):"},
                                        {"type": "image", "image": sdg_c},
                                        {"type": "text", "text": "Reference Image (Image 2):"},
                                        {"type": "image", "image": ref_c},
                                    ]
                                }
                            ]
                            images_arg = [[sdg_c, ref_c]]
                        else:
                            messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": effective_prompt},
                                        {"type": "text", "text": "Reference Image (Image 1):"},
                                        {"type": "image", "image": ref_c},
                                        {"type": "text", "text": "Generated Image (Image 2):"},
                                        {"type": "image", "image": sdg_c},
                                    ]
                                }
                            ]
                            images_arg = [[ref_c, sdg_c]]

                        text = self.hf_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = self.hf_processor(text=[text], images=images_arg, return_tensors="pt", padding=True).to(self.hf_model.device)
                        with torch.inference_mode():
                            generated_ids = self.hf_model.generate(**inputs, max_new_tokens=1024, do_sample=False, temperature=0.0)
                        gen_trimmed = generated_ids[0][len(inputs.input_ids[0]):]
                        raw_resp = self.hf_processor.decode(gen_trimmed, skip_special_tokens=True)
                    elif hasattr(self.hf_model, "chat"):
                        # Fallback for models providing chat() method (e.g., custom InternVL implementations)
                        if self.reverse_image_order:
                            question = (
                                f"{effective_prompt}\n"
                                f"Generated Image (Image 1): <image>\n"
                                f"Reference Image (Image 2): <image>"
                            )
                            first_c, second_c = sdg_c, ref_c
                        else:
                            question = (
                                f"{effective_prompt}\n"
                                f"Reference Image (Image 1): <image>\n"
                                f"Generated Image (Image 2): <image>"
                            )
                            first_c, second_c = ref_c, sdg_c

                        total_w = first_c.width + second_c.width
                        max_h = max(first_c.height, second_c.height)
                        composite = Image.new("RGB", (total_w, max_h), (255, 255, 255))
                        composite.paste(first_c, (0, 0))
                        composite.paste(second_c, (first_c.width, 0))
                        raw_resp, _ = self.hf_model.chat(
                            self.hf_processor,
                            composite,
                            question,
                            generation_config={"max_new_tokens": 1024, "do_sample": False}
                        )
                except Exception as e:
                    print(f"[Warning] Inference error on crop pair with {self.model_name}: {e}")
                    raw_resp = ""

                score, status, evidence = self.parse_judge_output(raw_resp, default_title=lbl)
                scores.append(score)
                details.append({"status": status, "visual_evidence": evidence})

        return scores, details


# ==============================================================================
# 6. VLM BBox Detection Engine (vLLM & Transformers Support)
# ==============================================================================

class VLMDetector:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-32B-Instruct",
        cache_file: Optional[Union[str, Path]] = None,
        backend: str = "auto",
        gpu_memory_utilization: float = 0.88
    ):
        self.model_name = model_name
        self.backend = backend
        self.gpu_memory_utilization = gpu_memory_utilization
        self.cache_file = Path(cache_file) if cache_file else None
        self.cache: Dict[str, List[Dict[str, Any]]] = {}
        self.load_cache()

        self.engine_type = None
        self.vllm_engine = None
        self.vllm_sampling = None
        self.hf_model = None
        self.hf_processor = None

    def load_cache(self):
        if self.cache_file and self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                print(f"[Cache] Loaded {len(self.cache)} cached detections from {self.cache_file}")
            except Exception as e:
                print(f"[Cache Warning] Failed to load cache from {self.cache_file}: {e}")
                self.cache = {}

    def save_cache(self):
        if self.cache_file:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)

    def _init_engine(self):
        if self.engine_type is not None:
            return

        if self.backend in ["auto", "vllm"]:
            try:
                from vllm import LLM, SamplingParams
                print(f"[Engine] Initializing high-throughput vLLM backend for {self.model_name}...")
                self.vllm_engine = LLM(
                    model=self.model_name,
                    trust_remote_code=True,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    max_model_len=16384,
                    limit_mm_per_prompt={"image": 2}
                )
                self.vllm_sampling = SamplingParams(
                    temperature=0.0,
                    max_tokens=2048
                )
                self.engine_type = "vllm"
                print("[Engine] vLLM backend successfully initialized.")
                return
            except Exception as e:
                print(f"[Engine Note] vLLM unavailable or failed to initialize ({e}). Falling back to Transformers.")

        print(f"[Engine] Initializing Hugging Face Transformers backend for {self.model_name}...")
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            self.hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            ).eval()
            self.hf_processor = AutoProcessor.from_pretrained(self.model_name)
        except Exception:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            self.hf_model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            ).eval()
            self.hf_processor = AutoProcessor.from_pretrained(self.model_name)

        self.engine_type = "transformers"
        print("[Engine] Transformers backend initialized.")

    def _get_rel_path(self, image_path: str) -> str:
        p_str = str(image_path).replace("\\", "/")

        markers = [
            "commercial-dreambench/",
            "samples/",
            "assets/",
            "outputs/",
            "data/"
        ]
        for marker in markers:
            if marker in p_str:
                if marker == "commercial-dreambench/":
                    return p_str.split(marker, 1)[1]
                return marker + p_str.split(marker, 1)[1]

        p = Path(image_path)
        try:
            return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except Exception:
            pass

        return p.name

    def get_cache_key(self, image_path: str, rubric_titles: List[str]) -> str:
        titles_str = "_".join(sorted(rubric_titles))
        p_str = str(image_path)
        abs_p = os.path.abspath(p_str)
        rel_p = self._get_rel_path(image_path)
        fname = Path(image_path).name

        # 1. Primary portable key (environment-agnostic)
        portable_h = hashlib.md5((rel_p + "_" + titles_str).encode("utf-8")).hexdigest()
        portable_key = f"{fname}_{portable_h}"
        if portable_key in self.cache:
            return portable_key

        # 2. Legacy candidates (for backward compatibility with unmigrated caches)
        legacy_candidates = [
            p_str,
            abs_p,
            abs_p.replace("/workspace/", "/root/Desktop/workspace/"),
            abs_p.replace("/root/Desktop/workspace/", "/workspace/"),
            f"/root/Desktop/workspace/woosung/commercial-dreambench/{rel_p}",
            f"/workspace/woosung/commercial-dreambench/{rel_p}",
            f"/root/Desktop/workspace/commercial-dreambench/{rel_p}",
            f"/workspace/commercial-dreambench/{rel_p}",
            f"/workspace/{rel_p}",
            f"/root/Desktop/workspace/{rel_p}",
        ]
        for cand in legacy_candidates:
            h = hashlib.md5((cand + "_" + titles_str).encode("utf-8")).hexdigest()
            k = f"{fname}_{h}"
            if k in self.cache:
                # Auto-alias legacy hit to portable key in memory so it persists portably
                self.cache[portable_key] = self.cache[k]
                return portable_key

        # Default canonical key for new detections: portable repo-relative key
        return portable_key

    def batch_detect_bboxes(
        self,
        tasks: List[Dict[str, Any]],
        visualize_dir: Optional[Path] = None,
        batch_size: int = 64,
        vllm_chunk_size: int = 1024
    ) -> Dict[str, List[Dict[str, Any]]]:
        results_map = {}
        pending_tasks = []

        for task in tasks:
            img_p = task["image_path"]
            rubrics = task["rubrics"]
            rubric_titles = [r['title'] for r in rubrics if 'title' in r]
            cache_key = self.get_cache_key(img_p, rubric_titles)
            task["cache_key"] = cache_key
            task["rubric_titles"] = rubric_titles

            if cache_key in self.cache:
                bboxes = self.cache[cache_key]
                results_map[cache_key] = bboxes
                if visualize_dir:
                    self._save_visualization(img_p, bboxes, visualize_dir, prefix=task.get("vis_prefix", ""))
            else:
                pending_tasks.append(task)

        if not pending_tasks:
            return results_map

        print(f"[VLM] Running detection for {len(pending_tasks)} uncached images with {self.model_name} (chunk size = {vllm_chunk_size})...")
        self._init_engine()

        if self.engine_type == "vllm":
            num_chunks = math.ceil(len(pending_tasks) / vllm_chunk_size)
            chunk_pbar = tqdm(
                range(0, len(pending_tasks), vllm_chunk_size),
                desc="vLLM BBox Chunks",
                total=num_chunks,
                dynamic_ncols=True
            )
            for c_start in chunk_pbar:
                chunk = pending_tasks[c_start : c_start + vllm_chunk_size]
                vllm_inputs = []
                valid_pending = []

                for t in chunk:
                    try:
                        img = Image.open(t["image_path"]).convert("RGB")
                        prompt = build_detection_prompt(t["rubrics"])
                        vllm_inputs.append({
                            "prompt": f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n",
                            "multi_modal_data": {"image": img}
                        })
                        valid_pending.append((t, img.width, img.height))
                    except Exception as e:
                        print(f"[Error] Failed to load image {t['image_path']}: {e}")

                if vllm_inputs:
                    outputs = self.vllm_engine.generate(vllm_inputs, sampling_params=self.vllm_sampling)
                    for (t, img_w, img_h), out in zip(valid_pending, outputs):
                        raw_resp = out.outputs[0].text
                        bboxes = parse_vlm_bbox_json(raw_resp, img_w, img_h)
                        self.cache[t["cache_key"]] = bboxes
                        results_map[t["cache_key"]] = bboxes
                        if visualize_dir:
                            self._save_visualization(t["image_path"], bboxes, visualize_dir, prefix=t.get("vis_prefix", ""))

                self.save_cache()
                del vllm_inputs, valid_pending
                gc.collect()

        else:
            num_batches = math.ceil(len(pending_tasks) / batch_size)
            batch_pbar = tqdm(
                range(0, len(pending_tasks), batch_size),
                desc="HF BBox Batches",
                total=num_batches,
                dynamic_ncols=True
            )
            for b_start in batch_pbar:
                batch_tasks = pending_tasks[b_start:b_start + batch_size]
                batch_texts = []
                batch_images = []
                batch_meta = []

                for t in batch_tasks:
                    try:
                        img = Image.open(t["image_path"]).convert("RGB")
                        prompt = build_detection_prompt(t["rubrics"])
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image", "image": t["image_path"]},
                                ]
                            }
                        ]
                        text = self.hf_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        batch_texts.append(text)
                        batch_images.append(img)
                        batch_meta.append((t, img.width, img.height))
                    except Exception as e:
                        print(f"[Error] Failed to process {t['image_path']}: {e}")

                if not batch_texts:
                    continue

                inputs = self.hf_processor(text=batch_texts, images=batch_images, return_tensors="pt", padding=True).to(self.hf_model.device)

                with torch.inference_mode():
                    generated_ids = self.hf_model.generate(
                        **inputs,
                        max_new_tokens=2048,
                        do_sample=False,
                        temperature=0.0
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                responses = self.hf_processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )

                for (t, img_w, img_h), resp_text in zip(batch_meta, responses):
                    bboxes = parse_vlm_bbox_json(resp_text, img_w, img_h)
                    self.cache[t["cache_key"]] = bboxes
                    results_map[t["cache_key"]] = bboxes
                    if visualize_dir:
                        self._save_visualization(t["image_path"], bboxes, visualize_dir, prefix=t.get("vis_prefix", ""))

                self.save_cache()
                del batch_texts, batch_images, batch_meta, inputs, generated_ids
                gc.collect()

        return results_map

    def _save_visualization(self, image_path: str, bboxes: List[Dict[str, Any]], visualize_dir: Path, prefix: str = ""):
        try:
            visualize_dir.mkdir(parents=True, exist_ok=True)
            img = Image.open(image_path).convert("RGB")
            annotated = draw_bboxes_on_image(img, bboxes)
            fname = f"annotated_{prefix}_{Path(image_path).name}" if prefix else f"annotated_{Path(image_path).name}"
            out_path = visualize_dir / fname
            annotated.save(out_path)
        except Exception as e:
            print(f"[Warning] Failed to save bbox overlay for {image_path}: {e}")


# ==============================================================================
# 7. Unified Evaluation Pipeline
# ==============================================================================

def run_bbox_crop_evaluation(
    eval_pairs: List[Dict[str, Any]],
    rubrics_map: Dict[str, List[Dict[str, str]]],
    detector: VLMDetector,
    metrics: List[str],
    judge_model: Optional[str] = None,
    reranker_model: str = "Qwen/Qwen3-VL-Reranker-2B",
    judge_prompt: str = "prompts/user_prompt_evaluate_with_crops.txt",
    anchor_cache_dir: str = "outputs/visual_likert_anchors",
    bbox_margin: float = 0.15,
    min_crop_size: int = 224,
    crop_upscale_factor: float = 1.0,
    visualize_dir: Optional[Path] = None,
    batch_size: int = 128,
    vllm_chunk_size: int = 1024,
    reverse_image_order: bool = False
) -> Dict[str, Any]:
    print(f"\n[Step 1/3] Indexing unique detection tasks across {len(eval_pairs)} image pairs...")
    detection_tasks = []
    task_keys_seen = set()

    for pair in eval_pairs:
        ref_path = pair["ref_image"]
        sdg_path = pair["sdg_image"]
        asin_name = pair.get("asin", "")
        cat_name = pair.get("category", "")
        rubric_key = pair.get("rubric_key", asin_name)
        rubrics = (
            rubrics_map.get(f"{cat_name}/{asin_name}")
            or rubrics_map.get(rubric_key)
            or rubrics_map.get(asin_name)
            or [{"title": "main object", "description": "primary subject"}]
        )

        prefix = f"{cat_name}_{asin_name}" if (cat_name and asin_name) else ""

        for path in [ref_path, sdg_path]:
            task_key = (path, rubric_key)
            if task_key not in task_keys_seen:
                task_keys_seen.add(task_key)
                detection_tasks.append({
                    "image_path": path,
                    "rubrics": rubrics,
                    "vis_prefix": prefix
                })

    detection_results = detector.batch_detect_bboxes(
        detection_tasks,
        visualize_dir=visualize_dir,
        batch_size=batch_size,
        vllm_chunk_size=vllm_chunk_size
    )

    clean_metrics = [m for m in (metrics or []) if m != "none"]
    if not clean_metrics:
        print("\n[Notice] No evaluation metrics requested (--metrics none). BBox detection and caching complete.")
        return {
            "total_pairs": len(eval_pairs),
            "total_detections_cached": len(detection_results),
            "detection_only": True
        }

    evaluators = {}
    if "clip" in clean_metrics:
        evaluators["clip"] = CLIPEvaluator()
    if "dino" in clean_metrics:
        evaluators["dino"] = DINOEvaluator()
    if "qwen_reranker" in clean_metrics:
        evaluators["qwen_reranker"] = QwenRerankerEvaluator(
            model_name=reranker_model,
            reverse_image_order=reverse_image_order
        )
    if "ours" in clean_metrics:
        chosen_judge_model = judge_model or detector.model_name
        evaluators["ours"] = OursVLMEvaluator(
            model_name=chosen_judge_model,
            prompt_path=judge_prompt,
            detector_engine=detector if chosen_judge_model == detector.model_name else None,
            reverse_image_order=reverse_image_order
        )
    for m_likert in ["visual-likert-scale_qwen_reranker", "visual-likert-scale_ref-distorted_qwen_reranker", "visual-likert-scale_crop-distorted_qwen_reranker"]:
        if m_likert in clean_metrics:
            c_dir = anchor_cache_dir
            if m_likert == "visual-likert-scale_ref-distorted_qwen_reranker" and anchor_cache_dir == "outputs/visual_likert_anchors":
                c_dir = "outputs/visual_likert_anchors_from_distorted_ref"
            evaluators[m_likert] = VisualLikertQwenRerankerEvaluator(
                model_name=reranker_model,
                anchor_cache_dir=c_dir
            )
    if "visual-likert-scale_vlm" in clean_metrics:
        evaluators["visual-likert-scale_vlm"] = VisualLikertVLMEvaluator(
            model_name=detector.model_name,
            anchor_cache_dir=anchor_cache_dir
        )

    # 2. Match Rubric Bounding Boxes (Memory-safe metadata only, crops extracted lazily)
    print(f"\n[Step 2/3] Pairing rubric bounding boxes...")
    crop_tasks_registry = []

    for pair_idx, pair in enumerate(eval_pairs):
        ref_path = pair["ref_image"]
        sdg_path = pair["sdg_image"]
        sample_key = pair["sample_key"]
        cat_name = pair.get("category", "")
        asin_name = pair.get("asin", "")
        rubric_key = pair.get("rubric_key", asin_name)
        rubrics = (
            rubrics_map.get(f"{cat_name}/{asin_name}")
            or rubrics_map.get(rubric_key)
            or rubrics_map.get(asin_name)
            or [{"title": "main object", "description": "primary subject"}]
        )
        rubric_titles = [r['title'] for r in rubrics if 'title' in r]

        ref_key = detector.get_cache_key(ref_path, rubric_titles)
        sdg_key = detector.get_cache_key(sdg_path, rubric_titles)

        ref_boxes = detection_results.get(ref_key, [])
        sdg_boxes = detection_results.get(sdg_key, [])

        for rubric_item in rubrics:
            r_title = rubric_item.get("title", "").strip().lower()

            matched_ref_box = None
            for b in ref_boxes:
                if b.get("label", "").strip().lower() == r_title or r_title in b.get("description", "").lower():
                    matched_ref_box = b
                    break

            matched_sdg_box = None
            for b in sdg_boxes:
                if b.get("label", "").strip().lower() == r_title or r_title in b.get("description", "").lower():
                    matched_sdg_box = b
                    break

            crop_tasks_registry.append({
                "pair_idx": pair_idx,
                "ref_image": ref_path,
                "sdg_image": sdg_path,
                "sample_key": sample_key,
                "rubric_title": rubric_item.get("title", ""),
                "category": pair.get("category", ""),
                "asin": pair.get("asin", ""),
                "ref_box": matched_ref_box.get("bbox") if matched_ref_box else None,
                "sdg_box": matched_sdg_box.get("bbox") if matched_sdg_box else None,
                "missing_in_sdg": (matched_sdg_box is None),
                "scores": {}
            })

    # 3. Memory-Safe Batched Metric Evaluation (Extracts crops on-the-fly in mini-batches)
    valid_indices = [i for i, item in enumerate(crop_tasks_registry) if not item["missing_in_sdg"]]
    print(f"\n[Step 3/3] Evaluating metrics across {len(crop_tasks_registry)} crop pairs ({len(valid_indices)} valid detections)...")

    if valid_indices:
        num_batches = math.ceil(len(valid_indices) / batch_size)
        crop_pbar = tqdm(
            range(0, len(valid_indices), batch_size),
            desc="Evaluating Crop Batches",
            total=num_batches,
            dynamic_ncols=True,
            unit="batch"
        )

        for b_start in crop_pbar:
            b_indices = valid_indices[b_start : b_start + batch_size]
            b_ref_crops = []
            b_sdg_crops = []
            b_labels = []
            b_metadata = []

            for idx in b_indices:
                t = crop_tasks_registry[idx]
                ref_img = Image.open(t["ref_image"]).convert("RGB")
                sdg_img = Image.open(t["sdg_image"]).convert("RGB")

                ref_bbox = t["ref_box"] if t["ref_box"] is not None else [0, 0, ref_img.height, ref_img.width]
                sdg_bbox = t["sdg_box"]

                ref_crop = extract_context_crop(
                    ref_img, ref_bbox,
                    margin_ratio=bbox_margin, min_size=min_crop_size, upscale_factor=crop_upscale_factor
                )
                sdg_crop = extract_context_crop(
                    sdg_img, sdg_bbox,
                    margin_ratio=bbox_margin, min_size=min_crop_size, upscale_factor=crop_upscale_factor
                )

                b_ref_crops.append(ref_crop)
                b_sdg_crops.append(sdg_crop)
                b_labels.append(t["rubric_title"])
                b_metadata.append({
                    "category": t.get("category", ""),
                    "asin": t.get("asin", ""),
                    "rubric_title": t.get("rubric_title", "")
                })

            for metric_name, evaluator in evaluators.items():
                if metric_name == "ours":
                    b_scores, b_details = evaluator.compute_similarity_batch(b_ref_crops, b_sdg_crops, b_labels)
                    for idx, s, det in zip(b_indices, b_scores, b_details):
                        crop_tasks_registry[idx]["scores"]["ours"] = s
                        crop_tasks_registry[idx]["status"] = det.get("status", "")
                        crop_tasks_registry[idx]["visual_evidence"] = det.get("visual_evidence", "")

                elif metric_name.startswith("visual-likert-scale_") and "reranker" in metric_name:
                    b_scores, b_details = evaluator.compute_similarity_batch(b_ref_crops, b_sdg_crops, b_labels, metadata=b_metadata)
                    for idx, s, det in zip(b_indices, b_scores, b_details):
                        crop_tasks_registry[idx]["scores"][metric_name] = s
                        if "visual_evidence" not in crop_tasks_registry[idx]:
                            crop_tasks_registry[idx]["status"] = det.get("status", "")
                            crop_tasks_registry[idx]["visual_evidence"] = det.get("visual_evidence", "")

                elif metric_name == "qwen_reranker":
                    b_scores = evaluator.compute_similarity_batch(b_ref_crops, b_sdg_crops, b_labels)
                    for idx, s in zip(b_indices, b_scores):
                        crop_tasks_registry[idx]["scores"][metric_name] = s

                else:
                    b_scores = evaluator.compute_similarity_batch(b_ref_crops, b_sdg_crops)
                    for idx, s in zip(b_indices, b_scores):
                        crop_tasks_registry[idx]["scores"][metric_name] = s

            del b_ref_crops, b_sdg_crops
            gc.collect()

    for item in crop_tasks_registry:
        if item["missing_in_sdg"]:
            for m in clean_metrics:
                item["scores"][m] = 0.0
            item["status"] = "Very Poor (Missing)"
            item["visual_evidence"] = "Attribute was not detected in the generated image (bounding box missing)."

    # 4. Aggregation (Sum of scores / max scores)
    summary_by_pair: Dict[int, Dict[str, Any]] = {}
    for item in crop_tasks_registry:
        p_idx = item["pair_idx"]
        if p_idx not in summary_by_pair:
            summary_by_pair[p_idx] = {
                "ref_image": item["ref_image"],
                "sdg_image": item["sdg_image"],
                "sample_key": item["sample_key"],
                "category": item.get("category", ""),
                "asin": item.get("asin", ""),
                "rubric_scores": [],
            }

        rubric_entry = {
            "rubric_title": item["rubric_title"],
            "missing_in_sdg": item["missing_in_sdg"],
            "scores": item["scores"]
        }
        if "status" in item:
            rubric_entry["status"] = item["status"]
        if "visual_evidence" in item:
            rubric_entry["visual_evidence"] = item["visual_evidence"]

        summary_by_pair[p_idx]["rubric_scores"].append(rubric_entry)

    overall_metric_totals: Dict[str, float] = {m: 0.0 for m in clean_metrics}
    total_rubrics_evaluated = len(crop_tasks_registry)

    for p_idx, p_data in summary_by_pair.items():
        pair_totals = {m: 0.0 for m in clean_metrics}
        n_rubrics = len(p_data["rubric_scores"])
        for r_entry in p_data["rubric_scores"]:
            for m in clean_metrics:
                pair_totals[m] += r_entry["scores"].get(m, 0.0)
                overall_metric_totals[m] += r_entry["scores"].get(m, 0.0)

        p_data["aggregated_scores"] = {
            m: (pair_totals[m] / max(1, n_rubrics)) for m in clean_metrics
        }

    final_summary = {
        "total_pairs": len(eval_pairs),
        "total_crop_evaluations": total_rubrics_evaluated,
        "mean_scores": {
            m: (overall_metric_totals[m] / max(1, total_rubrics_evaluated)) for m in clean_metrics
        },
        "pair_details": list(summary_by_pair.values())
    }

    return final_summary


# ==============================================================================
# 8. Dataset Builder (Reference vs SDG Variations)
# ==============================================================================

def find_image_pairs(
    ref_dir: Union[str, Path],
    sdg_dir: Union[str, Path],
    categories: Optional[List[str]] = None,
    asins: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Pairs each SDG variation image in `sdg_dir/<category>/<asin>/variation_*.jpg`
    with its corresponding reference image from `ref_dir/<category>/<asin>/reference.jpg`.
    Strictly ignores any reference or ground truth files inside sdg_dir.
    """
    ref_dir = Path(ref_dir)
    sdg_dir = Path(sdg_dir)

    pairs = []

    cat_dirs = sorted([d for d in sdg_dir.iterdir() if d.is_dir()]) if sdg_dir.exists() else []

    for cat_p in cat_dirs:
        cat_name = cat_p.name
        if categories and len(categories) > 0:
            if not any(c.lower() in cat_name.lower() for c in categories):
                continue

        asin_dirs = sorted([d for d in cat_p.iterdir() if d.is_dir()])
        for asin_p in asin_dirs:
            asin_name = asin_p.name
            if asins and len(asins) > 0:
                if not any(a.lower() in asin_name.lower() for a in asins):
                    continue

            # 1. Locate reference image in ref_dir (with fallback to sdg_dir)
            ref_image_path = None
            for r_name in ['reference.jpg', 'reference.png', 'ref.jpg', 'ref.png']:
                cand_ref = ref_dir / cat_name / asin_name / r_name
                if cand_ref.exists():
                    ref_image_path = cand_ref
                    break

            if not ref_image_path:
                for r_name in ['reference.jpg', 'reference.png', 'ref.jpg', 'ref.png']:
                    cand_ref = asin_p / r_name
                    if cand_ref.exists():
                        ref_image_path = cand_ref
                        break

            if not ref_image_path:
                continue

            # 2. Locate all generated variation images in asin_p (strictly matching variation_*)
            img_candidates = sorted(asin_p.glob("*.jpg")) + sorted(asin_p.glob("*.png")) + sorted(asin_p.glob("*.jpeg"))
            for img_p in img_candidates:
                stem_lower = img_p.stem.lower()
                # Strictly evaluate only generated variations (e.g. variation_1, variation_2)
                if not re.match(r"^variation_\d+", stem_lower):
                    continue

                sample_key = f"{cat_name}_{asin_name}_{img_p.stem}"
                pairs.append({
                    "ref_image": str(ref_image_path),
                    "sdg_image": str(img_p),
                    "category": cat_name,
                    "asin": asin_name,
                    "sample_key": sample_key,
                    "rubric_key": asin_name
                })

    return pairs


# ==============================================================================
# 9. Check Target Completion for --skip_if_done
# ==============================================================================

def get_metric_output_tag(
    metric_name: str,
    judge_model: Optional[str] = None,
    reverse_image_order: bool = False,
    output_tag_suffix: Optional[str] = None,
    append_crop_size: bool = False,
    min_crop_size: Optional[int] = None,
) -> str:
    """Returns the file/diagnostic tag for a metric, appending the judge model name for 'ours', '_image-reversed' if reversed, and crop size suffix if requested."""
    if metric_name == "ours" and judge_model:
        model_tag = Path(judge_model).name
        tag = f"ours-{model_tag}"
    else:
        tag = metric_name
    if reverse_image_order:
        tag = f"{tag}_image-reversed"
    if append_crop_size and min_crop_size is not None:
        tag = f"{tag}_crop-{min_crop_size}"
    if output_tag_suffix:
        suffix = output_tag_suffix if output_tag_suffix.startswith(("_", "-")) else f"_{output_tag_suffix}"
        tag = f"{tag}{suffix}"
    return tag


def filter_missing_eval_targets(
    rating_dir: Path,
    eval_pairs: List[Dict[str, Any]],
    metrics: List[str],
    judge_model: Optional[str] = None,
    reverse_image_order: bool = False,
    output_tag_suffix: Optional[str] = None,
    append_crop_size: bool = False,
    min_crop_size: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits eval_pairs into (missing_pairs, completed_pairs).
    A pair is considered complete only if it has a valid, non-null, non-NaN score
    across all specified metrics in rating_dir/{tag}_bbox-crop_results.json.
    """
    if not eval_pairs:
        return [], []
    if not metrics:
        return eval_pairs, []

    existing_ratings: Dict[str, Dict[str, Any]] = {}
    for m in metrics:
        tag = get_metric_output_tag(
            m,
            judge_model,
            reverse_image_order=reverse_image_order,
            output_tag_suffix=output_tag_suffix,
            append_crop_size=append_crop_size,
            min_crop_size=min_crop_size,
        )
        rating_file = rating_dir / f"{tag}_bbox-crop_results.json"
        # 1. Fall back to un-suffixed file if min_crop_size == 224 (all un-suffixed ratings are 224px crop size)
        if append_crop_size and min_crop_size == 224 and not rating_file.exists():
            base_tag = get_metric_output_tag(
                m,
                judge_model,
                reverse_image_order=reverse_image_order,
                output_tag_suffix=output_tag_suffix,
                append_crop_size=False,
            )
            base_file = rating_dir / f"{base_tag}_bbox-crop_results.json"
            if base_file.exists():
                rating_file = base_file
            elif not reverse_image_order and not output_tag_suffix and m == "ours" and judge_model in ["Qwen/Qwen3-VL-32B-Instruct", "Qwen3-VL-32B-Instruct", None]:
                legacy_file = rating_dir / f"{m}_bbox-crop_results.json"
                if legacy_file.exists():
                    rating_file = legacy_file

        # 2. Only fall back to legacy 'ours_bbox-crop_results.json' if not reversed, no crop tag, and using original default 32B model
        elif not reverse_image_order and not append_crop_size and not output_tag_suffix and not rating_file.exists() and m == "ours" and judge_model in ["Qwen/Qwen3-VL-32B-Instruct", "Qwen3-VL-32B-Instruct", None]:
            legacy_file = rating_dir / f"{m}_bbox-crop_results.json"
            if legacy_file.exists():
                rating_file = legacy_file

        if rating_file.exists():
            try:
                with open(rating_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        existing_ratings[m] = data
                    else:
                        existing_ratings[m] = {}
            except Exception:
                existing_ratings[m] = {}
        else:
            existing_ratings[m] = {}

    missing_pairs = []
    completed_pairs = []

    for p in eval_pairs:
        key = p.get("sample_key")
        is_complete = True
        for m in metrics:
            ratings = existing_ratings.get(m, {})
            if key not in ratings:
                is_complete = False
                break
            score = ratings[key]
            if score is None or not isinstance(score, (int, float)) or math.isnan(score):
                is_complete = False
                break

        if is_complete:
            completed_pairs.append(p)
        else:
            missing_pairs.append(p)

    return missing_pairs, completed_pairs


def check_eval_targets_completed(
    rating_dir: Path,
    eval_pairs: List[Dict[str, Any]],
    metrics: List[str],
    judge_model: Optional[str] = None,
    reverse_image_order: bool = False,
    output_tag_suffix: Optional[str] = None,
    append_crop_size: bool = False,
    min_crop_size: Optional[int] = None,
) -> bool:
    missing, _ = filter_missing_eval_targets(
        rating_dir,
        eval_pairs,
        metrics,
        judge_model=judge_model,
        reverse_image_order=reverse_image_order,
        output_tag_suffix=output_tag_suffix,
        append_crop_size=append_crop_size,
        min_crop_size=min_crop_size,
    )
    return len(missing) == 0


# ==============================================================================
# 10. CLI Entrypoint
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Efficient BBox-Crop Multi-Metric Evaluation")
    parser.add_argument("--ref_dir", type=str, default="", help="Directory containing reference images")
    parser.add_argument("--sdg_dir", type=str, default="", help="Directory containing SDG generated images")
    parser.add_argument("--pairs_json", type=str, default="", help="JSON list of {'ref_image': ..., 'sdg_image': ..., 'rubric_key': ...}")
    parser.add_argument("--rubrics_dir", type=str, default="assets/rubrics/amzn", help="Directory of rubric JSONs")
    parser.add_argument("--output_dir", type=str, default="outputs/bbox_crop_eval", help="Output directory for results")
    parser.add_argument("--rating_dir", type=str, default="", help="Rating directory where {metric}_bbox-crop_results.json will be saved")
    parser.add_argument("--cache_file", type=str, default="outputs/bbox_detections_cache.json", help="Persistent bbox cache JSON")
    parser.add_argument("--anchor_cache_dir", type=str, default="outputs/visual_likert_anchors", help="Directory to cache visual Likert distorted reference crops")
    parser.add_argument("--detector_model", type=str, default="Qwen/Qwen3-VL-32B-Instruct", help="VLM backbone for bbox detection (fixed default: Qwen/Qwen3-VL-32B-Instruct)")
    parser.add_argument("--judge_model", type=str, default=None, help="VLM backbone for Ours VLM judging (e.g. Qwen/Qwen3-VL-32B-Instruct, OpenGVLab/InternVL2_5-38B, THUDM/glm-4v-9b). Defaults to --vlm_model.")
    parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen3-VL-32B-Instruct", help="Default VLM backbone (default: Qwen/Qwen3-VL-32B-Instruct)")
    parser.add_argument("--reranker_model", type=str, default="Qwen/Qwen3-VL-Reranker-2B", help="Model backbone for Qwen Reranker evaluation (default: Qwen/Qwen3-VL-Reranker-2B)")
    parser.add_argument("--judge_prompt", type=str, default="prompts/user_prompt_evaluate_with_crops.txt", help="Prompt file path for Ours VLM judging")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "vllm", "transformers"], help="VLM inference backend")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.88, help="vLLM GPU memory utilization fraction")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["clip", "dino"],
        choices=["clip", "dino", "qwen_reranker", "ours", "visual-likert-scale_qwen_reranker", "visual-likert-scale_ref-distorted_qwen_reranker", "visual-likert-scale_crop-distorted_qwen_reranker", "visual-likert-scale_vlm", "none"],
        help="Metrics to evaluate"
    )
    parser.add_argument("-c", "--category", "--categories", dest="categories", nargs="*", default=None, help="Filter evaluation to specific categories")
    parser.add_argument("-a", "--asin", "--asins", dest="asins", nargs="*", default=None, help="Filter evaluation to specific ASINs")
    parser.add_argument("--skip_if_done", action="store_true", help="Skip execution if all target evaluation items are already done")
    parser.add_argument("--bbox_margin", type=float, default=0.15, help="Context expansion ratio around bbox (default: 0.15)")
    parser.add_argument("--min_crop_size", type=int, default=224, help="Minimum dimension for Lanczos upscaling (default: 224)")
    parser.add_argument("--crop_upscale", type=float, default=1.0, help="Multiplicative crop upscale factor (default: 1.0)")
    parser.add_argument("--save_visualizations", action="store_true", help="Save annotated images with bbox overlays")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for metric inference")
    parser.add_argument("--vllm_chunk_size", type=int, default=1024, help="Chunk size for vLLM detection batches to cap RAM usage")
    parser.add_argument("--reverse_image_order", action="store_true", help="Reverse image presentation order (Generated Image first, Reference Image second) to evaluate MLLM positional bias")
    parser.add_argument("--output_tag_suffix", type=str, default="", help="Custom suffix to append to output metric tags and result filenames")
    parser.add_argument("--append_crop_size", action="store_true", help="Append _crop-{min_crop_size} to output metric tags and result filenames")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rating_dir = Path(args.rating_dir) if args.rating_dir else out_dir
    rating_dir.mkdir(parents=True, exist_ok=True)

    vis_dir_name = "bbox_visualizations"
    if args.append_crop_size and args.min_crop_size is not None:
        vis_dir_name = f"{vis_dir_name}_crop-{args.min_crop_size}"
    if args.output_tag_suffix:
        suffix = args.output_tag_suffix if args.output_tag_suffix.startswith(("_", "-")) else f"_{args.output_tag_suffix}"
        vis_dir_name = f"{vis_dir_name}{suffix}"
    vis_dir = (out_dir / vis_dir_name) if args.save_visualizations else None
    clean_metrics = [m for m in (args.metrics or []) if m != "none"]

    # 1. Build Evaluation Pairs
    eval_pairs = []
    if args.pairs_json and os.path.exists(args.pairs_json):
        with open(args.pairs_json, "r", encoding="utf-8") as f:
            eval_pairs = json.load(f)
            for idx, p in enumerate(eval_pairs):
                if "sample_key" not in p:
                    p["sample_key"] = p.get("asin") or f"pair_{idx}"
    elif args.ref_dir and args.sdg_dir:
        eval_pairs = find_image_pairs(
            args.ref_dir,
            args.sdg_dir,
            categories=args.categories,
            asins=args.asins
        )
    else:
        print("[Main] No pairs provided. Please specify --pairs_json or --ref_dir and --sdg_dir.")
        eval_pairs = []

    print(f"[Main] Total pairs to evaluate: {len(eval_pairs)}")
    if args.categories:
        print(f"[Filter] Categories: {args.categories}")
    if args.asins:
        print(f"[Filter] ASINs: {args.asins}")

    if not eval_pairs and clean_metrics:
        print("[Warning] No matching evaluation pairs found. Exiting.")
        sys.exit(0)

    detector_model = getattr(args, "detector_model", "Qwen/Qwen3-VL-32B-Instruct") or "Qwen/Qwen3-VL-32B-Instruct"
    judge_model = getattr(args, "judge_model", None) or args.vlm_model or "Qwen/Qwen3-VL-32B-Instruct"

    # 2. Identify missing evaluation pairs
    if clean_metrics:
        missing_pairs, completed_pairs = filter_missing_eval_targets(
            rating_dir,
            eval_pairs,
            clean_metrics,
            judge_model=judge_model,
            reverse_image_order=args.reverse_image_order,
            output_tag_suffix=args.output_tag_suffix,
            append_crop_size=args.append_crop_size,
            min_crop_size=args.min_crop_size,
        )
        print(f"[Main] Status: {len(completed_pairs)} already completed, {len(missing_pairs)} missing or incomplete.")

        if args.skip_if_done:
            if not missing_pairs:
                print(f"[Skip] All {len(eval_pairs)} target items are verified complete in {rating_dir} for metrics {clean_metrics}. Skipping evaluation.")
                # If 224px crop size was matched via baseline file and crop-specific file doesn't exist, sync it
                if args.append_crop_size and args.min_crop_size == 224:
                    for m in clean_metrics:
                        crop_tag = get_metric_output_tag(
                            m,
                            judge_model,
                            reverse_image_order=args.reverse_image_order,
                            output_tag_suffix=args.output_tag_suffix,
                            append_crop_size=True,
                            min_crop_size=224,
                        )
                        crop_rating_file = rating_dir / f"{crop_tag}_bbox-crop_results.json"
                        if not crop_rating_file.exists():
                            base_tag = get_metric_output_tag(
                                m,
                                judge_model,
                                reverse_image_order=args.reverse_image_order,
                                output_tag_suffix=args.output_tag_suffix,
                                append_crop_size=False,
                            )
                            base_rating_file = rating_dir / f"{base_tag}_bbox-crop_results.json"
                            if not base_rating_file.exists() and m == "ours" and not args.reverse_image_order and not args.output_tag_suffix:
                                legacy_file = rating_dir / f"{m}_bbox-crop_results.json"
                                if legacy_file.exists():
                                    base_rating_file = legacy_file
                            if base_rating_file.exists():
                                try:
                                    import shutil
                                    shutil.copy2(base_rating_file, crop_rating_file)
                                    print(f"[Sync] Synced 224px baseline rating file {base_rating_file.name} -> {crop_rating_file.name}")
                                except Exception as e:
                                    print(f"[Warning] Could not sync 224px rating file: {e}")

                                base_diag = out_dir / f"evaluation_results_{base_tag}.json"
                                crop_diag = out_dir / f"evaluation_results_{crop_tag}.json"
                                if base_diag.exists() and not crop_diag.exists():
                                    try:
                                        import shutil
                                        shutil.copy2(base_diag, crop_diag)
                                        print(f"[Sync] Synced 224px baseline diagnostics {base_diag.name} -> {crop_diag.name}")
                                    except Exception:
                                        pass
                sys.exit(0)
            else:
                print(f"[Run] {len(missing_pairs)}/{len(eval_pairs)} targets need evaluation in {rating_dir}. Evaluating only missing items...")
                eval_pairs_to_run = missing_pairs
        else:
            eval_pairs_to_run = missing_pairs
    else:
        eval_pairs_to_run = eval_pairs

    if not eval_pairs_to_run and clean_metrics:
        print("[Main] All target evaluation pairs are already complete. Nothing to evaluate.")
        sys.exit(0)

    # 3. Load Rubrics
    print(f"[Main] Loading rubrics from {args.rubrics_dir}...")
    rubrics_map = load_all_rubrics(args.rubrics_dir)
    print(f"[Main] Loaded {len(rubrics_map)} rubric sets.")

    detector_model = getattr(args, "detector_model", "Qwen/Qwen3-VL-32B-Instruct") or "Qwen/Qwen3-VL-32B-Instruct"
    judge_model = getattr(args, "judge_model", None) or args.vlm_model or "Qwen/Qwen3-VL-32B-Instruct"

    # 4. Initialize VLM Detector (Fixed default: Qwen/Qwen3-VL-32B-Instruct)
    detector = VLMDetector(
        model_name=detector_model,
        cache_file=args.cache_file,
        backend=args.backend,
        gpu_memory_utilization=args.gpu_memory_utilization
    )

    # 5. Run Evaluation on Missing Pairs
    results = run_bbox_crop_evaluation(
        eval_pairs=eval_pairs_to_run,
        rubrics_map=rubrics_map,
        detector=detector,
        metrics=clean_metrics,
        judge_model=judge_model,
        reranker_model=args.reranker_model,
        judge_prompt=args.judge_prompt,
        anchor_cache_dir=args.anchor_cache_dir,
        bbox_margin=args.bbox_margin,
        min_crop_size=args.min_crop_size,
        crop_upscale_factor=args.crop_upscale,
        visualize_dir=vis_dir,
        batch_size=args.batch_size,
        vllm_chunk_size=args.vllm_chunk_size,
        reverse_image_order=args.reverse_image_order
    )

    # 6. Save Standard Flat Rating Files ({metric}_bbox-crop_results.json) & Merge Diagnostics
    if clean_metrics and not results.get("detection_only", False):
        for m in clean_metrics:
            m_tag = get_metric_output_tag(
                m,
                judge_model,
                reverse_image_order=args.reverse_image_order,
                output_tag_suffix=args.output_tag_suffix,
                append_crop_size=args.append_crop_size,
                min_crop_size=args.min_crop_size,
            )

            # Build metric-specific detailed result dictionary for new results
            metric_pair_details = []
            for p in results.get("pair_details", []):
                p_copy = {k: v for k, v in p.items() if k not in ("aggregated_scores", "rubric_scores")}
                p_copy["aggregated_scores"] = {m: p.get("aggregated_scores", {}).get(m, 0.0)}
                p_copy["rubric_scores"] = []
                for r_entry in p.get("rubric_scores", []):
                    r_copy = {k: v for k, v in r_entry.items() if k != "scores"}
                    r_copy["scores"] = {m: r_entry.get("scores", {}).get(m, 0.0)}
                    p_copy["rubric_scores"].append(r_copy)
                metric_pair_details.append(p_copy)

            m_details_file = out_dir / f"evaluation_results_{m_tag}.json"
            merged_pair_details = []
            if m_details_file.exists():
                try:
                    with open(m_details_file, "r", encoding="utf-8") as f:
                        old_diag = json.load(f)
                    if isinstance(old_diag, dict) and "pair_details" in old_diag:
                        new_sample_keys = {p["sample_key"] for p in metric_pair_details}
                        merged_pair_details = [
                            old_p for old_p in old_diag.get("pair_details", [])
                            if old_p.get("sample_key") not in new_sample_keys
                        ]
                except Exception:
                    merged_pair_details = []

            merged_pair_details.extend(metric_pair_details)
            total_merged_crops = sum(len(p.get("rubric_scores", [])) for p in merged_pair_details)
            total_merged_score = sum(
                r.get("scores", {}).get(m, 0.0)
                for p in merged_pair_details
                for r in p.get("rubric_scores", [])
            )
            merged_mean_score = (total_merged_score / max(1, total_merged_crops)) if total_merged_crops > 0 else 0.0

            metric_results = {
                "metric": m_tag,
                "total_pairs": len(merged_pair_details),
                "total_crop_evaluations": total_merged_crops,
                "mean_scores": {m_tag: round(float(merged_mean_score), 6)},
                "pair_details": merged_pair_details
            }

            with open(m_details_file, "w", encoding="utf-8") as f:
                json.dump(metric_results, f, indent=2)
            print(f"[Detailed] Saved {m_tag.upper()} detailed diagnostics ({len(merged_pair_details)} pairs) to: {m_details_file}")

            # Merge flat rating results
            m_rating_file = rating_dir / f"{m_tag}_bbox-crop_results.json"
            metric_ratings: Dict[str, float] = {}
            if m_rating_file.exists():
                try:
                    with open(m_rating_file, "r", encoding="utf-8") as f:
                        old_ratings = json.load(f)
                    if isinstance(old_ratings, dict):
                        metric_ratings = old_ratings
                except Exception:
                    metric_ratings = {}

            for p in results.get("pair_details", []):
                key = p["sample_key"]
                score = p.get("aggregated_scores", {}).get(m, 0.0)
                metric_ratings[key] = round(float(score), 6)

            with open(m_rating_file, "w", encoding="utf-8") as f:
                json.dump(metric_ratings, f, indent=4)
            print(f"[Rating] Saved {m_tag.upper()} rating results ({len(metric_ratings)} total entries) to: {m_rating_file}")

        print("\n=======================================================")
        print("           BBOX-CROP EVALUATION SUMMARY                ")
        print("=======================================================")
        print(f"Newly Evaluated Pairs       : {results.get('total_pairs', 0)}")
        print(f"Newly Evaluated Crops       : {results.get('total_crop_evaluations', 0)}")
        print("-------------------------------------------------------")
        for metric_name, score in results.get("mean_scores", {}).items():
            print(f"New Batch [{metric_name.upper():<30}]: Mean Score = {score:.4f}")
        print("=======================================================")
        print(f"Detailed diagnostics saved to: {out_dir}/evaluation_results_<metric>.json\n")


if __name__ == "__main__":
    main()
