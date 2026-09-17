import os
import json
import re
import time
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

# Try importing vLLM engine
HAS_VLLM = False
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


def parse_any_json(text):
    """
    Robust JSON parser for LLM outputs that may contain markdown codeblocks or raw JSON strings.
    """
    cleaned = text.strip()

    # 1. Try extracting content inside markdown codeblocks ```json ... ```
    cb_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned, re.IGNORECASE)
    if cb_match:
        candidate = cb_match.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            cleaned = candidate

    # 2. Try direct parsing of the full cleaned text
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 3. Try outermost JSON array extraction [ ... ]
    array_match = re.search(r'(\[[\s\S]*\])', cleaned)
    if array_match:
        try:
            return json.loads(array_match.group(1).strip())
        except Exception:
            pass

    # 4. Try outermost single JSON object extraction { ... }
    obj_match = re.search(r'(\{[\s\S]*\})', cleaned)
    if obj_match:
        try:
            return json.loads(obj_match.group(1).strip())
        except Exception:
            pass

    # 5. Fallback: parse multiple unbracketed JSON objects into a list
    item_matches = re.findall(r'\{[\s\S]*?\}', cleaned)
    if item_matches:
        parsed_items = []
        for m in item_matches:
            try:
                parsed_items.append(json.loads(m))
            except Exception:
                pass
        if parsed_items:
            return parsed_items

    return json.loads(cleaned)



def build_aggregation_messages(rubrics, ref_image_path, gen_image_path, system_instruction_text):
    # return [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": system_instruction_text}
    #         ]
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": json.dumps(rubrics, indent=2)}
    #         ]
    #     },
    #     {
    #         "role": "assistant",
    #         "content": [
    #             {"type": "text", "text": "Understood. Now provide me a reference image and a generated image."}
    #         ]
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": "Reference Image (Image 1):"},
    #             {"type": "image", "image": ref_image_path},
    #             {"type": "text", "text": "Generated Image (Image 2):"},
    #             {"type": "image", "image": gen_image_path},
    #         ]
    #     }
    # ]
    return [
        {
        "role": "user",
        "content": [
            {"type": "text", "text": system_instruction_text},
            {"type": "text", "text": f"<Rubrics>{json.dumps(rubrics, indent=2)}</Rubrics>"}
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Reference Image (Image 1):"},
            {"type": "image", "image": ref_image_path},
            {"type": "text", "text": "Generated Image (Image 2):"},
            {"type": "image", "image": gen_image_path},
        ]
    }
    ]


def build_each_item_messages(rubric, ref_image_path, gen_image_path, system_instruction_text=None):
    if system_instruction_text is not None:
        single_item_rubric_str = json.dumps([rubric], indent=2)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_instruction_text},
                    {"type": "text", "text": f"<Rubrics>\n{single_item_rubric_str}\n</Rubrics>"}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Reference Image (Image 1):"},
                    {"type": "image", "image": ref_image_path},
                    {"type": "text", "text": "Generated Image (Image 2):"},
                    {"type": "image", "image": gen_image_path},
                ]
            }
        ]
    title = rubric.get("title", "")
    description = rubric.get("description", "")

    user_prompt_eval_context = """
        You are an expert visual quality assessor specializing in subject preservation for advertising.
        We are evaluating how well a subject depicted in a reference image (Image 1) is preserved in a generated commercial image (Image 2).
    """

    user_prompt_eval = f"""
<rubric_to_evaluate>
{{
  "title": "{title}",
  "description": "{description}",
}}
</rubric_to_evaluate>

<task>
Compare the subject in the reference image (Image 1) against the subject in the generated image (Image 2). Based on the visual rubric listed in <rubric_to_evaluate>:
1. Classify the importance of this visual attribute for defining the subject's identity: **Essential** (critical for recognition/branding), **Important** (highly noticeable feature), or **Optional** (minor/ornamental detail).
2. Determine its preservation status of the subject in the generated image: **Correct** (fully preserved), **Partial** (present but imperfect/incomplete), or **Missing** (not present or wrong).
3. Provide a brief analysis of the visual evidence.
</task>

<constraints>
- Evaluate exactly the rubric listed in <rubric_to_evaluate>.
- Use the reference image (Image 1) ONLY to locate and understand the visual attribute described in the rubric item.
- Evaluate the preservation and visibility of that attribute using ONLY the generated image (Image 2). The verdict must be based strictly on what is directly observable in the generated image (Image 2).
- Do NOT infer or assume missing visual details from the reference image (Image 1), prior knowledge, or brand familiarity. If an attribute cannot be directly observed with confidence in the generated image (Image 2), it must be treated as **NOT visible / Missing** in the verdict.
- Ground your assessment strictly in visual evidence.
- Format your response strictly as a valid JSON object matching the requested output format. Do not include any conversational text or formatting outside the JSON block.
</constraints>

<output_format>
Return your response as a valid JSON block matching this exact structure:
```json
{{
  "title": "{title}",
  "importance": "[Essential/Important/Optional]",
  "status": "[Correct/Partial/Missing]",
  "visual_evidence": "[Concise description of the attribute in the generated image]"
}}
```
</output_format>"""

    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt_eval_context}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Understood. Please provide the evaluation details and the images."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Reference Image (Image 1):"},
                {"type": "image", "image": ref_image_path},
                {"type": "text", "text": "Generated Image (Image 2):"},
                {"type": "image", "image": gen_image_path},
                {"type": "text", "text": user_prompt_eval},
            ]
        }
    ]


def build_revision_messages(rubric, item, ref_image_path, gen_image_path, prompt):
    title = item.get("title", rubric.get("title", ""))
    importance = item.get("importance", rubric.get("importance", "Essential"))
    initial_status = item.get("status", item.get("verdict", "Missing"))
    evidence = item.get("visual_evidence", "")
    description = rubric.get("description", "")

    review_context_prompt = """
        You are an expert visual quality assessor specializing in subject preservation for advertising.
        We are reviewing a discrepancy found during the visual quality assessment of a generated image (Image 2) compared to a reference image (Image 1).
    """

    review_prompt = f"""
Here is the context of the generation:
<generation_prompt>
{prompt}
</generation_prompt>

Here is the rubric item we are reviewing:
<rubric_item>
- Title: {title}
- Description: {description}
- Importance: {importance}
</rubric_item>

During the initial visual evaluation, the following verdict was made:
- Initial Status: {initial_status}
- Visual Evidence: {evidence}

Your task:
Analyze the `<generation_prompt>` and determine whether the visual discrepancy described in "Visual Evidence" is directly justified and expected due to the instructions in the prompt.

Use the following strict guidelines to determine if the discrepancy is justified:

1. **Direct and Unavoidable Physical Consequence**: A discrepancy is justified ONLY if it is a direct, unavoidable logical/physical result of an explicit action or state requested by the generation prompt (e.g., a cap must be removed because the prompt asks for a thumb pressing the nozzle, or the back of a bottle is hidden because the prompt requests a front shot).
2. **Do Not Excuse General Quality Failures**: General quality failures (such as blurry/unsharp text, warped logos, distorted letters, or low-resolution textures) are NEVER justified by prompts mentioning "artistic blur", "soft lighting", "shallow depth of field", or "natural setting". The subject itself must remain sharp and preserved.
3. **No Speculative Occlusions**: If the hand or another object is in the image, you must look at Image 2 to check if it actually covers the attribute. Do NOT claim the attribute is "naturally occluded" if the area containing the attribute is visible but simply rendered poorly/blurry.
4. **Be Strict**: If there is no clear and direct logical link between the prompt's instructions and the discrepancy, the discrepancy is NOT justified. Erring on the side of caution is preferred; do not make logical leaps or assume artistic intentions.

Decide if the discrepancy is justified:
- If YES (the discrepancy is a direct/unavoidable consequence of the prompt), set `is_justified` to true, and revise `revised_status` to "Correct".
- If NO (it is a generation failure, blur, distortion, or the occlusion does not actually cover the attribute), set `is_justified` to false, and set `revised_status` to match `initial_status`.

Format your response strictly as a valid JSON object matching the requested output format. Do not include any conversational text or formatting outside the JSON block.

<output_format>
Return your response as a valid JSON block matching this exact structure:
```json
{{
  "title": "{title}",
  "initial_status": "{initial_status}",
  "revised_status": "[Correct/Partial/Missing/Mismatch]",
  "is_justified": [true/false],
  "explanation": "[Provide a concise explanation of your reasoning, specifically explaining why the discrepancy is or is not a direct logical necessity of the generation prompt]"
}}
```
</output_format>"""

    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": review_context_prompt}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Understood. Please provide the review details and the images."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Reference Image (Image 1):"},
                {"type": "image", "image": ref_image_path},
                {"type": "text", "text": "Generated Image (Image 2):"},
                {"type": "image", "image": gen_image_path},
                {"type": "text", "text": review_prompt},
            ]
        }
    ]


def compute_score(eval_list_out):
    importance_weights = {"Essential": 1.0, "Important": 0.7, "Optional": 0.3}
    status_scores = {
        "Very Poor": 0.0,
        "Poor": 1.0,
        "Fair": 2.0,
        "Good": 3.0,
        "Excellent": 4.0,
        "Correct": 1.0,
        "Partial": 0.5,
        "Missing": 0.0,
        "Mismatch": -1.0,
        "Not Visible": 0.0,
        "Incorrect": 0.0
    }

    total_score = 0.0
    max_score = 0.0

    for item in eval_list_out:
        importance = item.get("importance", "Essential")
        status = item.get("status", item.get("verdict", "Incorrect"))
        weight = importance_weights.get(importance, 1.0)
        status_val = status_scores.get(status, 0.0)
        max_item_score = 4.0 if status in ["Very Poor", "Poor", "Fair", "Good", "Excellent"] else 1.0
        total_score += weight * status_val
        max_score += weight * max_item_score

    score = (total_score / max_score * 5.0) if max_score > 0 else 0.0
    percentage_score = (total_score / max_score * 100.0) if max_score > 0 else 0.0
    return score, percentage_score


def batch_generate_hf(batch_messages, model, processor, max_new_tokens=512):
    """
    Batched inference using standard HuggingFace PyTorch pipeline.
    """
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in batch_messages
    ]

    images_batch = []
    for msg in batch_messages:
        imgs = []
        for turn in msg:
            if isinstance(turn.get("content"), list):
                for elem in turn["content"]:
                    if elem.get("type") == "image":
                        img_val = elem.get("image")
                        if isinstance(img_val, str):
                            imgs.append(Image.open(img_val).convert("RGB"))
                        elif isinstance(img_val, Image.Image):
                            imgs.append(img_val)
        images_batch.append(imgs if imgs else None)

    inputs = processor(text=texts, images=images_batch, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    output_texts = []
    input_ids = inputs["input_ids"]
    for in_ids, out_ids in zip(input_ids, generated_ids):
        trimmed = out_ids[len(in_ids):]
        decoded = processor.decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        output_texts.append(decoded)

    return output_texts


def batch_generate_vllm(batch_items, vllm_engine, processor, max_new_tokens=None, chunk_size=1000):
    """
    High-throughput batched inference using vLLM engine with mini-batching
    to prevent CPU RAM OOM from holding too many PIL images in memory at once.
    """
    all_results = []

    for i in range(0, len(batch_items), chunk_size):
        chunk = batch_items[i : i + chunk_size]
        vllm_inputs = []
        sampling_params_list = []

        for item in chunk:
            messages = item["messages"]
            prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            images = []
            for turn in messages:
                if isinstance(turn.get("content"), list):
                    for elem in turn["content"]:
                        if elem.get("type") == "image":
                            img_path = elem.get("image")
                            if isinstance(img_path, str):
                                images.append(Image.open(img_path).convert("RGB"))
                            elif isinstance(img_path, Image.Image):
                                images.append(img_path)

            vllm_input = {
                "prompt": prompt_text,
                "multi_modal_data": {
                    "image": images
                }
            }
            vllm_inputs.append(vllm_input)

            item_max_tokens = item.get("max_tokens") if item.get("max_tokens") is not None else (max_new_tokens or 512)
            sampling_params_list.append(
                SamplingParams(
                    temperature=0.0,
                    max_tokens=item_max_tokens
                )
            )

        outputs = vllm_engine.generate(vllm_inputs, sampling_params=sampling_params_list, use_tqdm=True)
        all_results.extend([out.outputs[0].text for out in outputs])

    return all_results



def filter_evaluations(eval_list, args):
    mode_dir_name = f"{args.mode}-revised" if args.run_revision else args.mode
    pending_pairs = []
    cached_scores = {}
    cached_latencies = {}

    for pair in eval_list:
        m = pair["model"]
        cat = pair["category"]
        curr_asin = pair["asin"]
        target_base = pair["target_base"]
        result_file = os.path.join(args.rating_dir, m, "ours", mode_dir_name, cat, f"{curr_asin}_{target_base}.json")

        if args.skip_if_done and os.path.exists(result_file):
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    cached_res = json.load(f)
                    cached_scores.setdefault(m, []).append(cached_res["score"])
                    cached_latencies.setdefault(m, []).append(cached_res.get("latency_sec", 0.0))
            except Exception as e:
                print(f"[WARNING] Failed to load cached result {result_file}: {e}")
                pending_pairs.append(pair)
        else:
            pending_pairs.append(pair)

    return pending_pairs, cached_scores, cached_latencies


def main():
    parser = argparse.ArgumentParser(description="Evaluates DreamBench++ using efficient vLLM batched inference.")
    parser.add_argument("--model", type=str, nargs="+", default=None, help="Optional model subdirectory filter(s).")
    parser.add_argument("-c", "--category", type=str, default=None, help="Optional category filter.")
    parser.add_argument("-a", "--asin", type=str, default=None, help="Optional specific ASIN to evaluate.")
    parser.add_argument("--mode", type=str, default="each_item", choices=["aggregation", "each_item"], help="Evaluation mode.")
    parser.add_argument("-m", "--model_name", type=str, default="Qwen/Qwen3-VL-32B-Instruct", help="Name of local VLM model.")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="Device map specification.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for parallel prompt generation.")
    parser.add_argument("--use_vllm", action="store_true", help="Force use of vLLM backend engine for inference.")
    parser.add_argument("--data_dir", type=str, default="/root/Desktop/workspace/woosung/commercial-dreambench/assets/data/amzn", help="Root wordy data directory.")
    parser.add_argument("--samples_dir", type=str, default="/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn", help="Root amzn samples directory.")
    parser.add_argument("--rubrics_dir", type=str, default="/root/Desktop/workspace/woosung/commercial-dreambench/assets/rubrics/amzn", help="Root cached rubrics directory.")
    parser.add_argument("--rating_dir", type=str, default="/root/Desktop/workspace/woosung/commercial-dreambench/rating/amzn", help="Directory to save evaluations.")
    parser.add_argument("--root_dir", type=str, default="/root/Desktop/workspace/woosung/commercial-dreambench", help="Workspace root directory.")
    parser.add_argument("--evaluation_prompt", type=str, default="user_prompt_evaluate_with_rubric_objective.txt", help="Filename or path of evaluation prompt template.")
    parser.add_argument("--skip_if_done", action="store_true", help="Skip evaluation if the evaluation is already done.")
    parser.add_argument("--run_revision", action="store_true", help="Enable contextual evaluation revision based on prompt.")

    args = parser.parse_args()

    # Discover models to evaluate
    if args.model is not None:
        models = []
        for m in args.model:
            for sub_m in m.replace(',', ' ').split():
                models.append(sub_m)
    else:
        models = sorted([
            d for d in os.listdir(args.samples_dir)
            if os.path.isdir(os.path.join(args.samples_dir, d)) and not d.startswith(".")
        ])
    print(f"Discovered {len(models)} models to evaluate: {models}")

    # Gather category subdirs
    if args.category is not None:
        categories = [args.category]
    else:
        categories = sorted([
            d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d)) and d.startswith("raw_meta_")
        ])

    if os.path.isabs(args.evaluation_prompt) or os.path.exists(args.evaluation_prompt):
        user_eval_prompt_path = args.evaluation_prompt
    else:
        user_eval_prompt_path = os.path.join(args.root_dir, "prompts", args.evaluation_prompt)
    with open(user_eval_prompt_path, "r", encoding="utf-8") as f:
        user_prompt_eval_template = f.read().strip()

    # Build evaluation pairs list
    eval_list = []
    for m in models:
        for cat in categories:
            meta_json_path = os.path.join(args.data_dir, cat, "metadata.json")
            if not os.path.exists(meta_json_path):
                continue

            with open(meta_json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            for item in metadata:
                curr_asin = item.get("asin")
                if args.asin is not None and curr_asin != args.asin:
                    continue

                rubric_path = os.path.join(args.rubrics_dir, cat, f"{curr_asin}.json")
                if not os.path.exists(rubric_path):
                    continue

                ref_file = item.get("reference_file", f"{curr_asin}/reference.jpg")
                ref_image_path = os.path.abspath(os.path.join(args.data_dir, cat, ref_file))

                for var_file in item.get("variation_files", []):
                    var_rel_path = var_file.get("file")
                    gen_image_path = os.path.abspath(os.path.join(args.samples_dir, m, cat, var_rel_path))

                    if not os.path.exists(ref_image_path) or not os.path.exists(gen_image_path):
                        continue

                    eval_list.append({
                        "model": m,
                        "category": cat,
                        "asin": curr_asin,
                        "ref_image_path": ref_image_path,
                        "gen_image_path": gen_image_path,
                        "prompt": var_file.get("prompt", ""),
                        "rubric_path": rubric_path,
                        "target_base": os.path.splitext(os.path.basename(var_rel_path))[0]
                    })

    print(f"Found {len(eval_list)} target pairs to evaluate.")

    pending_pairs, cached_scores_map, cached_latencies_map = filter_evaluations(eval_list, args)
    if args.skip_if_done:
        num_skipped = len(eval_list) - len(pending_pairs)
        print(f"[SKIP CHECK] {num_skipped} / {len(eval_list)} evaluations already exist.")
        if len(pending_pairs) == 0 and len(eval_list) > 0:
            print("All target evaluation pairs already exist. Exiting early.")
            return

    # Decide Backend Engine (vLLM vs HuggingFace PyTorch Batching)
    use_vllm_engine = HAS_VLLM and args.use_vllm
    print(f"Loading local VLM processor for model: {args.model_name}...")
    processor = AutoProcessor.from_pretrained(args.model_name)

    vllm_engine = None
    hf_model_obj = None

    if use_vllm_engine:
        print(f"Initializing vLLM Engine backend for high-throughput batching...")
        try:
            vllm_engine = LLM(
                model=args.model_name,
                trust_remote_code=True,
                max_model_len=8192,
                limit_mm_per_prompt={"image": 2}
            )
            print("vLLM Engine initialized successfully!")
        except Exception as e:
            print(f"[WARNING] Failed to initialize vLLM engine ({e}). Falling back to PyTorch batched inference.")
            use_vllm_engine = False

    if not use_vllm_engine:
        print(f"Loading PyTorch model on {args.device} for batched inference (batch_size={args.batch_size})...")
        if "Qwen3" in args.model_name and Qwen3VLForConditionalGeneration is not None:
            hf_model_obj = Qwen3VLForConditionalGeneration.from_pretrained(
                args.model_name, torch_dtype=torch.bfloat16, device_map=args.device
            )
        else:
            try:
                hf_model_obj = AutoModelForCausalLM.from_pretrained(
                    args.model_name, torch_dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True
                )
            except Exception:
                from transformers import AutoModelForImageTextToText
                hf_model_obj = AutoModelForImageTextToText.from_pretrained(
                    args.model_name, torch_dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True
                )
        print("PyTorch model loaded successfully!")

    eval_targets = pending_pairs if args.skip_if_done else eval_list
    if not eval_targets:
        print("No pending evaluation targets found.")
        return

    print(f"\nProcessing {len(eval_targets)} total evaluation pairs across {len(models)} model(s)...")

    # Step 1: Flatten all rubric evaluation tasks into a single batch queue
    eval_queue = []
    pair_rubrics_map = {}

    for pair in eval_targets:
        m = pair["model"]
        curr_asin = pair["asin"]
        target_base = pair["target_base"]
        pair_id = f"{m}_{curr_asin}_{target_base}"

        with open(pair["rubric_path"], "r", encoding="utf-8") as f:
            rubrics = json.load(f)

        pair_rubrics_map[pair_id] = {
            "pair": pair,
            "rubrics": rubrics,
            "evaluations": [],
            "t_start": time.time()
        }

        if args.mode == "aggregation":
            msgs = build_aggregation_messages(
                rubrics, pair["ref_image_path"], pair["gen_image_path"], user_prompt_eval_template
            )
            eval_queue.append({
                "pair_id": pair_id,
                "item_index": 0,
                "messages": msgs,
                "max_tokens": 4096
            })
        else:
            for idx, rubric in enumerate(rubrics):
                msgs = build_each_item_messages(
                    rubric, pair["ref_image_path"], pair["gen_image_path"], user_prompt_eval_template
                )
                eval_queue.append({
                    "pair_id": pair_id,
                    "item_index": idx,
                    "rubric": rubric,
                    "messages": msgs,
                    "max_tokens": 512
                })

    # Step 2: Execute batched generation for the evaluation queue
    print(f"Running batched evaluation on {len(eval_queue)} total prompt tasks...")
    all_raw_responses = []

    if use_vllm_engine:
        raw_outputs = batch_generate_vllm(eval_queue, vllm_engine, processor)
        all_raw_responses = raw_outputs

    else:
        # Chunk into batches of size args.batch_size
        for i in tqdm(range(0, len(eval_queue), args.batch_size), desc="PyTorch Batched Generation"):
            chunk = eval_queue[i : i + args.batch_size]
            chunk_msgs = [c["messages"] for c in chunk]
            max_tokens = max(c["max_tokens"] for c in chunk)
            chunk_raw = batch_generate_hf(chunk_msgs, hf_model_obj, processor, max_new_tokens=max_tokens)
            all_raw_responses.extend(chunk_raw)

    # Step 3: Parse and aggregate initial evaluation results
    for task, raw_res in zip(eval_queue, all_raw_responses):
        pair_id = task["pair_id"]
        pair_entry = pair_rubrics_map[pair_id]

        try:
            eval_data = parse_any_json(raw_res)
        except Exception as e:
            print(f"[ERROR] JSON parse failed for task in {pair_id}: {e}")
            eval_data = {
                "title": task.get("rubric", {}).get("title", "Unknown"),
                "importance": "Essential",
                "status": "Missing",
                "visual_evidence": f"Failed to parse JSON response: {e}"
            }

        if isinstance(eval_data, list):
            pair_entry["evaluations"].extend(eval_data)
        else:
            pair_entry["evaluations"].append(eval_data)

    # Step 4: Optional prompt-based contextual revision phase
    if args.run_revision:
        revision_queue = []
        for pair_id, pair_entry in pair_rubrics_map.items():
            pair = pair_entry["pair"]
            rubrics = pair_entry["rubrics"]
            prompt = pair.get("prompt", "")

            for rubric, item in zip(rubrics, pair_entry["evaluations"]):
                status = item.get("status", item.get("verdict", "Missing"))
                if status != "Correct" and prompt:
                    rev_msgs = build_revision_messages(
                        rubric, item, pair["ref_image_path"], pair["gen_image_path"], prompt
                    )
                    revision_queue.append({
                        "pair_id": pair_id,
                        "rubric": rubric,
                        "initial_item": item,
                        "messages": rev_msgs
                    })

        if revision_queue:
            print(f"Running batched revisions on {len(revision_queue)} non-Correct items...")
            if use_vllm_engine:
                rev_raw_outputs = batch_generate_vllm(revision_queue, vllm_engine, processor)
            else:
                rev_raw_outputs = []
                for i in tqdm(range(0, len(revision_queue), args.batch_size), desc="PyTorch Batched Revision"):
                    chunk = revision_queue[i : i + args.batch_size]
                    chunk_msgs = [c["messages"] for c in chunk]
                    chunk_raw = batch_generate_hf(chunk_msgs, hf_model_obj, processor, max_new_tokens=512)
                    rev_raw_outputs.extend(chunk_raw)

            # Process revision verdicts
            rev_map = {}
            for rev_task, raw_rev_res in zip(revision_queue, rev_raw_outputs):
                pair_id = rev_task["pair_id"]
                try:
                    rev_data = parse_any_json(raw_rev_res)
                except Exception as e:
                    rev_data = {
                        "revised_status": rev_task["initial_item"].get("status", "Missing"),
                        "is_justified": False,
                        "explanation": f"Revision parsing failed: {e}"
                    }
                rev_map.setdefault(pair_id, []).append((rev_task["rubric"], rev_data))

            # Update evaluations with revision results
            for pair_id, rev_items in rev_map.items():
                pair_entry = pair_rubrics_map[pair_id]
                revised_list = []
                for orig_item in pair_entry["evaluations"]:
                    title = orig_item.get("title", "")
                    match_rev = next((r[1] for r in rev_items if r[0].get("title") == title), None)
                    if match_rev:
                        rev_status = match_rev.get("revised_status", orig_item.get("status", "Missing"))
                        explanation = match_rev.get("explanation", "")
                        is_justified = match_rev.get("is_justified", False)
                        was_revised = (rev_status != orig_item.get("status", "Missing"))

                        revised_item = {
                            "title": title,
                            "importance": orig_item.get("importance", "Essential"),
                            "status": rev_status,
                            "visual_evidence": orig_item.get("visual_evidence", ""),
                            "initial_status": orig_item.get("status", "Missing"),
                            "was_revised": was_revised,
                            "revision_explanation": explanation
                        }
                        revised_list.append(revised_item)
                    else:
                        revised_list.append(orig_item)
                pair_entry["evaluations"] = revised_list

    # Step 5: Save JSON evaluation outputs and collect stats per model
    output_dir_name = f"{args.mode}-revised" if args.run_revision else args.mode

    for m in models:
        output_dir = os.path.join(args.rating_dir, m, "ours", output_dir_name)
        os.makedirs(output_dir, exist_ok=True)

        scores = list(cached_scores_map.get(m, []))
        latencies = list(cached_latencies_map.get(m, []))
        processed_count = len(scores)

        model_pair_ids = [pid for pid, pentry in pair_rubrics_map.items() if pentry["pair"]["model"] == m]

        for pair_id in model_pair_ids:
            pair_entry = pair_rubrics_map[pair_id]
            pair = pair_entry["pair"]
            cat = pair["category"]
            curr_asin = pair["asin"]
            target_base = pair["target_base"]
            eval_latency = time.time() - pair_entry["t_start"]

            score, percentage_score = compute_score(pair_entry["evaluations"])
            result_file = os.path.join(output_dir, cat, f"{curr_asin}_{target_base}.json")

            eval_result = {
                "pair_id": pair_id,
                "model": m,
                "category": cat,
                "asin": curr_asin,
                "target_file": f"{curr_asin}/{target_base}.jpg",
                "score": score,
                "percentage_score": percentage_score,
                "latency_sec": eval_latency,
                "evaluations": pair_entry["evaluations"]
            }

            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(eval_result, f, indent=4, ensure_ascii=False)

            scores.append(score)
            latencies.append(eval_latency)
            processed_count += 1
            print(f"Evaluated [{m}] {curr_asin}_{target_base}: Score = {score:.2f}/5.0 ({percentage_score:.1f}%) in {eval_latency:.2f}s")

        if processed_count > 0:
            mean_score = sum(scores) / len(scores)
            mean_latency = sum(latencies) / len(latencies)
            summary = {
                "model": m,
                "mode": args.mode,
                "total_pairs_evaluated": processed_count,
                "mean_score": mean_score,
                "mean_latency_sec": mean_latency,
                "scores": scores,
                "latencies": latencies
            }
            summary_file = os.path.join(output_dir, "summary.json")
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=4)

            print("\n" + "=" * 50)
            print(f"EVALUATION STATISTICS SUMMARY FOR MODEL: {m} ({args.mode})")
            print("=" * 50)
            print(f"Total Evaluated Pairs: {processed_count}")
            print(f"Mean Score (out of 5.0): {mean_score:.3f}")
            print(f"Mean Latency per Evaluation: {mean_latency:.3f} seconds")
            print(f"Summary JSON saved to: {summary_file}")
            print("=" * 50 + "\n\n")


if __name__ == "__main__":
    os.environ["greedy"] = "true"
    os.environ["top_p"] = "0.8"
    os.environ["top_k"] = "20"
    os.environ["temperature"] = "0.7"
    os.environ["repetition_penalty"] = "1.0"
    os.environ["presence_penalty"] = "1.5"
    os.environ["out_seq_length"] = "16384"
    main()
