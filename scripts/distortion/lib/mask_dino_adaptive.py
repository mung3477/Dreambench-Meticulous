import os
import cv2
import numpy as np
from PIL import Image
import json
import datetime

_detector = None

def visualize_dino_error(img, detections, box_threshold, output_path):
    """
    Draw all detected raw bounding boxes on the image, color-coded by why they were rejected.
    - Orange/Yellow: Score below box_threshold.
    - Red: Bounding box area ratio > 90% (covers the entire image frame).
    - Green: Valid detection (should not happen if this function is called under the no-valid-box error condition, but kept for completeness).
    """
    vis_img = img.copy()
    h, w, _ = vis_img.shape
    img_area = h * w

    for d in detections:
        score = d['score']
        box = d['box']
        label = d['label']

        xmin = max(0, int(box['xmin']))
        ymin = max(0, int(box['ymin']))
        xmax = min(w, int(box['xmax']))
        ymax = min(h, int(box['ymax']))

        box_area = (xmax - xmin) * (ymax - ymin)
        box_area_ratio = box_area / img_area

        if score < box_threshold:
            color = (0, 165, 255)  # Orange/Yellow in BGR
            status = "Low Score"
        elif box_area_ratio > 0.90:
            color = (0, 0, 255)  # Red in BGR
            status = "Full Frame"
        else:
            color = (0, 255, 0)  # Green in BGR
            status = "Valid"

        # Draw bounding box
        cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), color, 2)

        # Draw label text
        text = f"{label}: {score:.2f} ({status})"
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        # Background rect for text readability
        text_ymin = max(0, ymin - text_h - 4)
        cv2.rectangle(vis_img, (xmin, text_ymin), (xmin + text_w, ymin), color, -1)
        cv2.putText(
            vis_img,
            text,
            (xmin, ymin - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # Ensure output directory exists and save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, vis_img)


def get_detector(device=None):
    """Lazily load and cache the GroundingDINO pipeline on GPU."""
    global _detector
    if _detector is None:
        import os
        import logging
        import warnings
        import transformers
        transformers.logging.set_verbosity_error()
        logging.getLogger("transformers.models.grounding_dino.modeling_grounding_dino").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", message=".*You seem to be using the pipelines sequentially on GPU.*")
        warnings.filterwarnings("ignore", message=".*TORCH_CUDA_ARCH_LIST.*")
        import torch
        from transformers import pipeline

        if device is None:
            device = int(os.environ.get("DINO_GPU_DEVICE", 0))
        if torch.cuda.is_available() and device >= torch.cuda.device_count():
            device = 0

        # Load the zero-shot object detector on designated GPU
        _detector = pipeline(
            model="IDEA-Research/grounding-dino-tiny",
            task="zero-shot-object-detection",
            device=device
        )
    return _detector


def get_common_object_name(categorical_name: str) -> str:
    """
    Map a specific categorical name to a more common/generic object label
    to assist GroundingDINO detection.
    """
    name = categorical_name.lower().strip()

    # Predefined dictionary for specific or complex names
    mapping = {
        "3d printer": "printer",
        "adjustable wrench": "wrench",
        "area rug": "rug",
        "audio interface": "interface",
        "audio mixer": "mixer",
        "baby milestone blanket": "blanket",
        "barley grass powder": "powder",
        "bass drum head": "drum",
        "bath salts": "jar",
        "beard wax": "wax",
        "belt buckle": "buckle",
        "bike computer": "computer",
        "birthday banner": "banner",
        "bluetooth audio receiver": "receiver",
        "body butter": "cream",
        "body lotion": "lotion",
        "body oil": "oil",
        "callus remover gel": "gel",
        "camping chair": "chair",
        "car wash spray": "spray",
        "chef apron": "apron",
        "christmas ornament": "ornament",
        "christmas stocking": "stocking",
        "cleansing cream": "cream",
        "cleansing milk": "milk",
        "clip-on tuner": "tuner",
        "coconut oil": "oil",
        "current switch": "switch",
        "cutting die": "die",
        "desktop trash can": "trash can",
        "diamond painting kit": "painting",
        "digital hygrometer thermometer": "thermometer",
        "digital meat thermometer": "thermometer",
        "displayport cable": "cable",
        "dog blanket": "blanket",
        "dog collar": "collar",
        "door mat": "mat",
        "dried apricots": "bag",
        "dynamic microphone": "microphone",
        "eau de parfum": "perfume",
        "egg timer": "timer",
        "electric breast pump": "pump",
        "electric clothes dryer": "dryer",
        "electric shaver": "shaver",
        "essential oil": "oil",
        "essential oil diffuser": "diffuser",
        "eyelash glue": "glue",
        "face cream": "cream",
        "face mask": "mask",
        "face serum": "serum",
        "facial cleanser": "cleanser",
        "facial essence": "essence",
        "facial mask": "mask",
        "facial peel": "peel",
        "flower essence": "essence",
        "foaming body wash": "wash",
        "foot massager": "massager",
        "foot tambourine": "tambourine",
        "football mat": "mat",
        "frozen dessert": "box",
        "fruit juice blend": "juice",
        "game controller": "controller",
        "gaming headset": "headset",
        "gaming mouse pad": "mouse pad",
        "garden flag": "flag",
        "gas range": "stove",
        "gel polish": "polish",
        "gift card holder": "holder",
        "gift tag": "tag",
        "glass cleaner": "cleaner",
        "ground coffee": "coffee",
        "guitar effect pedal": "pedal",
        "guitar pedal": "pedal",
        "guitar preamp": "preamp",
        "gun stock finish": "finish",
        "hair activator": "activator",
        "hair conditioner": "conditioner",
        "hair growth treatment": "treatment",
        "hair mask": "mask",
        "hair stick": "stick",
        "hair straightener": "straightener",
        "hair volumizing powder": "powder",
        "hand cream": "cream",
        "homeopathic remedy": "remedy",
        "house flag": "flag",
        "indoor air quality monitor": "monitor",
        "inflatable snowman": "snowman",
        "insect killer": "spray",
        "insulated tumbler": "tumbler",
        "jigsaw puzzle": "puzzle",
        "kitchen cleaner": "cleaner",
        "kitchen hand towel": "towel",
        "laundry hamper": "hamper",
        "license plate cover": "cover",
        "lint bin": "bin",
        "lip crayon": "crayon",
        "lipo battery pack": "battery",
        "liquid herbal supplement": "supplement",
        "loose leaf tea": "tea",
        "makeup brush": "brush",
        "massage roller": "roller",
        "meal replacement shake": "shake",
        "metal coin": "coin",
        "metal shank buttons": "buttons",
        "metal sign": "sign",
        "micro sd card": "card",
        "microwave wax": "wax",
        "mountain bike": "bike",
        "mule sandals": "sandals",
        "multivitamin tablets": "tablets",
        "mushroom elixir mix": "powder",
        "mustache wax": "wax",
        "mustard seed oil": "oil",
        "night cream": "cream",
        "non alcoholic beverage pack": "beverage",
        "over the door hook": "hook",
        "pencil case": "case",
        "pendant necklace": "necklace",
        "permanent hair color": "box",
        "pest repellent spray": "spray",
        "phone grip": "grip",
        "phone ring grip": "grip",
        "playard mattress cover": "cover",
        "pocket holster": "holster",
        "pool ball rack": "rack",
        "pop up toy": "toy",
        "portable bluetooth speaker": "speaker",
        "portable dvd player": "dvd player",
        "portable ice maker": "ice maker",
        "poster print": "poster",
        "putty knife": "knife",
        "range hood": "range hood",
        "remote control": "remote",
        "roll-on perfume": "perfume",
        "rowing machine": "rowing machine",
        "running backpack": "backpack",
        "rv surge protector": "protector",
        "santa sack": "sack",
        "sd memory card": "card",
        "sea salt blend": "salt",
        "sea salt": "salt",
        "sesame oil": "oil",
        "shave oil": "oil",
        "sheet mask": "mask",
        "side by side refrigerator": "refrigerator",
        "skin care oil": "oil",
        "sleepsack": "sleepsack",
        "snack mix": "mix",
        "soil conditioner": "conditioner",
        "sonic facial brush": "brush",
        "spare tire cover": "cover",
        "sparkling beverage": "beverage",
        "spinach powder": "powder",
        "stable door plaque": "plaque",
        "state flag": "flag",
        "stroller bassinet": "bassinet",
        "stroller tag": "tag",
        "sugar free gum": "gum",
        "sugar scrub": "scrub",
        "sunscreen lotion": "lotion",
        "taco seasoning": "seasoning",
        "teeth whitening powder": "powder",
        "temperature and humidity sensor": "sensor",
        "thermal imaging camera": "camera",
        "throw pillow case": "pillowcase",
        "throw pillow cover": "pillow",
        "toilet paper": "toilet paper",
        "tortilla chips": "chips",
        "transistor radio": "radio",
        "turntable alignment protractor": "protractor",
        "usb flash drive": "drive",
        "uv led nail lamp": "lamp",
        "uv resin gel": "gel",
        "vinyl sticker": "sticker",
        "wall art print": "print",
        "wall tapestry": "tapestry",
        "water inlet valve": "valve",
        "waterproof phone case": "case",
        "wearable gimbal": "gimbal",
        "whey protein powder": "powder",
        "white noise machine": "machine",
        "wind chime": "wind chime",
        "world map tapestry": "tapestry",
        "yard flag": "flag",
        "zip around wallet": "wallet"
    }

    if name in mapping:
        return mapping[name]

    # Keyword-based fallbacks
    keywords = [
        "bottle", "tube", "painting", "bag", "shoes", "boots", "watch", "rug", "tapestry",
        "hat", "box", "case", "charger", "instrument", "electronics", "spray", "cream",
        "lotion", "cleaner", "pillow", "brush", "oil", "supplement", "soap", "glass",
        "tape", "gel", "powder", "tool", "device", "camera", "monitor", "sensor", "valve",
        "filter", "card", "backpack", "glove", "keyboard", "mouse", "headset", "speaker",
        "printer", "mixer", "timer", "dryer", "shaver", "diffuser", "massager", "heater",
        "controller", "stove", "range", "refrigerator", "oven"
    ]

    for kw in keywords:
        if kw in name:
            return kw

    return name


def generate_dino_adaptive_mask(
    image_path: str,
    categorical_name: str,
    block_size=25,
    min_area_ratio=0.00001,
    max_area_ratio=0.01,
    dilation_size=1,
    verbose=False,
    box_threshold=0.17,
    error_dir="/root/Desktop/workspace/woosung/flowfixer/error_reports",
    bbox_height_ratio=1.0,
    bbox_height_ratios=None,
    device=None,
    return_bbox_info=False
):
    """
    Detect the product bounding boxes using GroundingDINO, then run
    localized adaptive thresholding and contour masking ONLY within those bboxes.
    """
    # 1. Load image in grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 2. Run GroundingDINO object detector
    pil_img = Image.open(image_path).convert("RGB")
    detector = get_detector(device=device)

    # Map categorical_name to a more common object label to improve GroundingDINO detection
    common_name = get_common_object_name(categorical_name)

    # Candidate labels: specific categorical name, common name, and general fallbacks, sorted by specificity
    raw_candidates = [categorical_name, common_name, "product", "object"]
    candidate_labels = []
    for label in raw_candidates:
        if label and label.strip():
            lbl_clean = label.strip().lower()
            if lbl_clean not in candidate_labels:
                candidate_labels.append(lbl_clean)

    try:
        pipeline_threshold = min(0.15, box_threshold)
        detections = detector(pil_img, candidate_labels=candidate_labels, threshold=pipeline_threshold)
    except Exception as e:
        print(f"  [DINO] Detection error: {e}. Falling back to full image.")
        detections = []

    # 3. Filter valid bounding boxes (ignore full-frame matches > 90% area)
    # and build both fallback candidates (all non-full-frame) and valid boxes (above threshold) in a single pass
    valid_boxes = []
    fallback_candidates = []
    img_area = h * w
    for d in detections:
        box = d['box']
        box_area = (box['xmax'] - box['xmin']) * (box['ymax'] - box['ymin'])
        box_area_ratio = box_area / img_area
        # Skip boxes that cover almost the entire image (false-positive frame matches)
        if box_area_ratio > 0.90:
            continue

        fallback_candidates.append((d['score'], box))
        if d['score'] >= box_threshold:
            valid_boxes.append((d['score'], box))

    if not valid_boxes:
        if verbose:
            print(f"  [DINO] No product detected above threshold {box_threshold} or only full-frame matches found. Running fallback logic...")

        # Ensure error directory exists
        os.makedirs(error_dir, exist_ok=True)

        # Construct a clean descriptive prefix for the image
        normalized_path = os.path.normpath(image_path)
        path_parts = normalized_path.split(os.sep)
        if len(path_parts) >= 3:
            name_prefix = "_".join(path_parts[-3:])
        else:
            name_prefix = os.path.basename(image_path)
        name_prefix_no_ext, _ = os.path.splitext(name_prefix)
        vis_filename = f"{name_prefix_no_ext}_error.jpg"
        vis_path = os.path.join(error_dir, vis_filename)

        # Save error visualization
        try:
            visualize_dino_error(img, detections, box_threshold, vis_path)
            if verbose:
                print(f"  [DINO] Saved error visualization to: {vis_path}")
        except Exception as e:
            print(f"  [DINO] Failed to generate error visualization: {e}")

        # structured JSON logging
        json_path = os.path.join(error_dir, "errors.json")
        error_log = {
            "timestamp": datetime.datetime.now().isoformat(),
            "image_path": image_path,
            "categorical_name": categorical_name,
            "box_threshold": box_threshold,
            "detections": [
                {
                    "label": d["label"],
                    "score": float(d["score"]),
                    "box": {k: float(v) for k, v in d["box"].items()}
                }
                for d in detections
            ],
            "visualization_path": vis_path
        }

        existing_logs = []
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    existing_logs = json.load(f)
                    if not isinstance(existing_logs, list):
                        existing_logs = []
            except Exception:
                existing_logs = []

        existing_logs.append(error_log)

        try:
            with open(json_path, "w") as f:
                json.dump(existing_logs, f, indent=2)
        except Exception as e:
            print(f"  [DINO] Failed to write JSON error log: {e}")

        # Fallback using our prebuilt fallback_candidates
        if fallback_candidates:
            fallback_candidates.sort(key=lambda x: x[0], reverse=True)
            best_fallback_box = fallback_candidates[0][1]
            best_fallback_score = fallback_candidates[0][0]
            if verbose:
                print(f"  [DINO] No valid boxes above threshold {box_threshold}. Falling back to highest scoring raw detection ({best_fallback_score:.3f}): {best_fallback_box}")
            target_boxes = [best_fallback_box]
        else:
            if verbose:
                print("  [DINO] No detections or only full-frame matches found in raw detections. Falling back to entire image.")
            target_boxes = [{'xmin': 0, 'ymin': 0, 'xmax': w, 'ymax': h}]
    else:
        # Sort by confidence score descending and pick up to 4 highest scoring ones
        valid_boxes.sort(key=lambda x: x[0], reverse=True)
        selected_boxes = valid_boxes[:4]
        if verbose:
            print(f"  [DINO] Detected {len(valid_boxes)} bounding box(es). Selected top {len(selected_boxes)} box(es).")
            for idx, (score, box) in enumerate(selected_boxes):
                print(f"    Box {idx+1}: score={score:.3f}, box={box}")
        target_boxes = [item[1] for item in selected_boxes]


    # 4. Generate local mean on the full image (to avoid edge artifacts)
    if block_size % 2 == 0:
        block_size += 1
    local_mean = cv2.GaussianBlur(gray, (block_size, block_size), 0)

    is_multi = bbox_height_ratios is not None
    ratios = list(bbox_height_ratios) if is_multi else [bbox_height_ratio]
    masks = {r: np.zeros_like(gray) for r in ratios}
    img_area = h * w
    bbox_info_list = []

    # 5. Process each bounding box independently
    for i, box in enumerate(target_boxes):
        # Extract coordinates and clamp to image bounds
        xmin = max(0, int(box['xmin']))
        ymin = max(0, int(box['ymin']))
        xmax = min(w, int(box['xmax']))
        ymax = min(h, int(box['ymax']))

        if (xmax - xmin) <= 0 or (ymax - ymin) <= 0:
            continue

        # Crop the grayscale and local mean images
        gray_crop = gray[ymin:ymax, xmin:xmax]
        local_mean_crop = local_mean[ymin:ymax, xmin:xmax]

        # Compute local difference inside the bbox crop
        local_diff_crop = cv2.absdiff(gray_crop, local_mean_crop)

        # Dynamic threshold based on local variance INSIDE the crop
        mean_diff = np.mean(local_diff_crop)
        c_value = int(10 + 2.0 * mean_diff)
        if verbose:
            print(f"  [DINO Box {i+1}] mean_diff={mean_diff:.2f} -> Computed c_value={c_value}")

        # Binary thresholding and dilation on the crop
        _, thresh_crop = cv2.threshold(local_diff_crop, c_value, 255, cv2.THRESH_BINARY)

        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_size, dilation_size))
        dilated_crop = cv2.dilate(thresh_crop, dilate_kernel, iterations=1)

        # Find contours inside the crop
        contours, _ = cv2.findContours(dilated_crop, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            area_ratio = (cw * ch) / img_area
            if min_area_ratio <= area_ratio <= max_area_ratio:
                eff_ratios = {}
                for r in ratios:
                    effective_ch = max(1, int(ch * r))
                    eff_ratios[r] = (cw * effective_ch) / img_area
                    cv2.rectangle(
                        masks[r],
                        (xmin + cx, ymin + cy),
                        (xmin + cx + cw, ymin + cy + effective_ch),
                        255,
                        -1
                    )
                bbox_info_list.append({
                    "cw": int(cw),
                    "ch": int(ch),
                    "raw_area_ratio": float(area_ratio),
                    "effective_area_ratios": {float(r): float(v) for r, v in eff_ratios.items()}
                })

    res_masks = {r: Image.fromarray(masks[r]) for r in ratios} if is_multi else Image.fromarray(masks[bbox_height_ratio])
    if return_bbox_info:
        return res_masks, bbox_info_list
    else:
        return res_masks
