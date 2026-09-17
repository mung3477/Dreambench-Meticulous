import os
import json
import re
import time
import torch
import argparse
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

def generate_with_local_vlm(messages, model, processor, max_new_tokens=128):
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

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


def evaluate_aggregation(rubrics, ref_image_path, gen_image_path, model, processor, system_instruction_text):
    importance_weights = {
        "Essential": 1.0,
        "Important": 0.7,
        "Optional": 0.3
    }
    status_scores = {
        "Correct": 1.0,
        "Partial": 0.5,
        "Missing": 0.0,
        "Mismatch": -1.0,
        "Not Visible": 0.0
    }

    messages_eval = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_instruction_text}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(rubrics, indent=2)}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Understood. Now provide me a reference image and a generated image."}
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

    t_start = time.time()
    res_eval = generate_with_local_vlm(messages_eval, model, processor, max_new_tokens=2048)
    eval_latency = time.time() - t_start

    try:
        eval_data = parse_any_json(res_eval)
    except Exception as e:
        print(f"[ERROR] Failed to parse VLM response as JSON. Raw response:\n{res_eval}")
        raise e

    if isinstance(eval_data, list):
        eval_list_out = eval_data
    elif isinstance(eval_data, dict):
        eval_list_out = eval_data.get("evaluations", [eval_data])
    else:
        eval_list_out = [eval_data]

    total_score = 0.0
    max_score = 0.0
    for item in eval_list_out:
        importance = item.get("importance", "Essential")
        status = item.get("verdict", item.get("status", "Missing"))
        weight = importance_weights.get(importance, 1.0)
        status_val = status_scores.get(status, 0.0)
        total_score += weight * status_val
        max_score += weight

    score = 0.0
    percentage_score = 0.0
    if max_score > 0:
        score = (total_score / max_score) * 5.0
        percentage_score = (total_score / max_score) * 100.0

    return score, percentage_score, eval_list_out, eval_latency

def evaluate_each_item(rubrics, ref_image_path, gen_image_path, model, processor):
    importance_weights = {
        "Essential": 1.0,
        "Important": 0.7,
        "Optional": 0.3
    }
    status_scores = {
        "Correct": 1.0,
        "Partial": 0.5,
        "Missing": 0.0,
        "Mismatch": -1.0,
        "Not Visible": 0.0
    }

    eval_list_out = []
    total_score_ind = 0.0
    max_score_ind = 0.0
    t_start = time.time()

    for rubric in rubrics:
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

        messages_eval = [
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

        try:
            res_item = generate_with_local_vlm(messages_eval, model, processor, max_new_tokens=512)
            item_data = parse_any_json(res_item)
            eval_list_out.append(item_data)

            importance = item_data.get("importance", "Essential")
            status = item_data.get("status", "Missing")
            weight = importance_weights.get(importance, 1.0)
            status_val = status_scores.get(status, 0.0)
            total_score_ind += weight * status_val
            max_score_ind += weight
        except Exception as e:
            print(f"[WARNING] Error evaluating rubric '{title}': {e}")
            eval_list_out.append({
                "title": title,
                "importance": "Essential",
                "status": "Missing",
                "visual_evidence": f"Evaluation failed due to error: {e}"
            })
            max_score_ind += importance_weights.get("Essential", 1.0)

    eval_latency = time.time() - t_start
    score = 0.0
    percentage_score = 0.0
    if max_score_ind > 0:
        score = (total_score_ind / max_score_ind) * 5.0
        percentage_score = (total_score_ind / max_score_ind) * 100.0

    return score, percentage_score, eval_list_out, eval_latency

def revise_evaluations(eval_list_out, rubrics, ref_image_path, gen_image_path, model, processor, prompt):
    importance_weights = {
        "Essential": 1.0,
        "Important": 0.7,
        "Optional": 0.3
    }
    status_scores = {
        "Correct": 1.0,
        "Partial": 0.5,
        "Missing": 0.0,
        "Mismatch": -1.0,
        "Not Visible": 0.0
    }

    revised_eval_list = []
    total_score_revised = 0.0
    max_score_revised = 0.0

    print(f"### Running revision on {len(eval_list_out)} evaluated items using generation prompt...")

    for rubric, item in zip(rubrics, eval_list_out):
        title = item.get("title", rubric.get("title", ""))
        importance = item.get("importance", rubric.get("importance", "Essential"))
        initial_status = item.get("status", item.get("verdict", "Missing"))
        evidence = item.get("visual_evidence", "")
        description = rubric.get("description", "")

        if initial_status == "Correct":
            revised_item = {
                "title": title,
                "importance": importance,
                "status": "Correct",
                "visual_evidence": evidence,
                "initial_status": "Correct",
                "was_revised": False,
                "revision_explanation": "Initial status was Correct."
            }
            if "verdict" in item:
                revised_item["verdict"] = "Correct"
            revised_eval_list.append(revised_item)

            weight = importance_weights.get(importance, 1.0)
            total_score_revised += weight * 1.0
            max_score_revised += weight
            continue


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

        messages_review = [
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

        try:
            res_review = generate_with_local_vlm(messages_review, model, processor, max_new_tokens=512)
            review_data = parse_any_json(res_review)

            revised_status = review_data.get("revised_status", initial_status)
            is_justified = review_data.get("is_justified", False)
            explanation = review_data.get("explanation", "")
            was_revised = (revised_status != initial_status)

            revised_item = {
                "title": title,
                "importance": importance,
                "status": revised_status,
                "visual_evidence": evidence,
                "initial_status": initial_status,
                "was_revised": was_revised,
                "revision_explanation": explanation
            }
            if "verdict" in item:
                revised_item["verdict"] = revised_status
            revised_eval_list.append(revised_item)

            weight = importance_weights.get(importance, 1.0)
            status_val = status_scores.get(revised_status, 0.0)
            total_score_revised += weight * status_val
            max_score_revised += weight

            revised_str = f" [REVISED from {initial_status}]" if was_revised else ""
            print(f"- **{title}** ({importance}, Status: {revised_status}){revised_str}: score = {weight * status_val:.2f} / {weight:.2f}")
            if was_revised:
                print(f"  *Revision Reason:* {explanation}")
            else:
                print(f"  *Evidence:* {evidence}")
        except Exception as e:
            print(f"[WARNING] Failed to review rubric '{title}': {e}")
            revised_item = {
                "title": title,
                "importance": importance,
                "status": initial_status,
                "visual_evidence": evidence,
                "initial_status": initial_status,
                "was_revised": False,
                "revision_explanation": f"Failed to review: {e}"
            }
            if "verdict" in item:
                revised_item["verdict"] = initial_status
            revised_eval_list.append(revised_item)
            weight = importance_weights.get(importance, 1.0)
            status_val = status_scores.get(initial_status, 0.0)
            total_score_revised += weight * status_val
            max_score_revised += weight

    score = 0.0
    percentage_score = 0.0
    if max_score_revised > 0:
        score = (total_score_revised / max_score_revised) * 5.0
        percentage_score = (total_score_revised / max_score_revised) * 100.0

    return score, percentage_score, revised_eval_list

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
    parser = argparse.ArgumentParser(description="Evaluates DreamBench++ using compositional reasoning.")
    parser.add_argument("--model", type=str, nargs="+", default=None, help="Optional model subdirectory filter(s) (e.g. flux-kontext original).")
    parser.add_argument("-c", "--category", type=str, default=None, help="Optional category filter (e.g. raw_meta_All_Beauty).")
    parser.add_argument("-a", "--asin", type=str, default=None, help="Optional specific ASIN to evaluate.")
    parser.add_argument("--mode", type=str, default="each_item", choices=["aggregation", "each_item"], help="Evaluation mode.")
    parser.add_argument("-m", "--model_name", type=str, default="Qwen/Qwen3-VL-32B-Instruct", help="Name of local Qwen3-VL model.")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="Device map specification (e.g. cuda:0, cuda:1).")
    parser.add_argument("--data_dir", type=str, default="/root/Desktop/workspace/woosung/commercial-dreambench/assets/data/wordy", help="Root wordy data directory.")
    parser.add_argument("--samples_dir", type=str, default="/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn", help="Root amzn samples directory.")
    parser.add_argument("--rubrics_dir", type=str, default="/root/Desktop/workspace/woosung/commercial-dreambench/assets/rubrics/amzn", help="Root cached rubrics directory.")
    parser.add_argument("--rating_dir", type=str, default="/root/Desktop/workspace/woosung/commercial-dreambench/rating/amzn", help="Directory to save evaluations.")
    parser.add_argument("--root_dir", type=str, default="/root/Desktop/workspace/woosung/commercial-dreambench", help="Workspace root directory.")
    parser.add_argument("--skip_if_done", action="store_true", help="Skip evaluation if the evaluation is already done.")
    parser.add_argument("--run_revision", action="store_true", help="Enable contextual evaluation revision based on prompt.")

    args = parser.parse_args()

    # 1. Discover models to evaluate
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

    # 2. Gather category subdirs
    if args.category is not None:
        categories = [args.category]
    else:
        categories = sorted([
            d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d)) and d.startswith("raw_meta_")
        ])

    # 3. Load user aggregation prompt template
    user_eval_prompt_path = os.path.join(args.root_dir, "prompts", "user_prompt_evaluate_with_rubric.txt")
    print(f"Loading aggregation evaluate prompt from: {user_eval_prompt_path}")
    with open(user_eval_prompt_path, "r", encoding="utf-8") as f:
        user_prompt_eval = f.read().strip()

    # 4. Build list of evaluation pairs
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

    # 5. Filter pending evaluation pairs
    pending_pairs, cached_scores_map, cached_latencies_map = filter_evaluations(eval_list, args)

    if args.skip_if_done:
        num_skipped = len(eval_list) - len(pending_pairs)
        print(f"[SKIP CHECK] {num_skipped} / {len(eval_list)} evaluations already exist.")
        if len(pending_pairs) == 0 and len(eval_list) > 0:
            print("All target evaluation pairs already exist. Exiting early without loading model resources.")
            return

    # 6. Initialize the VLM model
    print(f"Loading local VLM model: {args.model_name} on {args.device}...")
    processor = AutoProcessor.from_pretrained(args.model_name)
    if "Qwen3" in args.model_name and Qwen3VLForConditionalGeneration is not None:
        model_obj = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16, device_map=args.device
        )
    else:
        try:
            model_obj = AutoModelForCausalLM.from_pretrained(
                args.model_name, torch_dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True
            )
        except Exception:
            from transformers import AutoModelForImageTextToText
            model_obj = AutoModelForImageTextToText.from_pretrained(
                args.model_name, torch_dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True
            )
    print(f"Model loaded successfully on {args.device}!")

    # 7. Process evaluations
    eval_targets = pending_pairs if args.skip_if_done else eval_list

    for m in models:
        if args.run_revision:
            output_dir = os.path.join(args.rating_dir, m, "ours", f"{args.mode}-revised")
        else:
            output_dir = os.path.join(args.rating_dir, m, "ours", args.mode)
        os.makedirs(output_dir, exist_ok=True)

        model_eval_pairs = [pair for pair in eval_targets if pair["model"] == m]

        if args.skip_if_done:
            scores = list(cached_scores_map.get(m, []))
            latencies = list(cached_latencies_map.get(m, []))
        else:
            scores = []
            latencies = []
        processed_count = len(scores)

        if not model_eval_pairs and processed_count == 0:
            continue

        for pair in tqdm(model_eval_pairs, desc=f"Evaluating model '{m}' ({args.mode})"):
            cat = pair["category"]
            curr_asin = pair["asin"]
            target_base = pair["target_base"]
            ref_image_path = pair["ref_image_path"]
            gen_image_path = pair["gen_image_path"]
            prompt = pair["prompt"]
            rubric_path = pair["rubric_path"]

            result_file = os.path.join(output_dir, cat, f"{curr_asin}_{target_base}.json")

            with open(rubric_path, "r", encoding="utf-8") as f:
                rubrics = json.load(f)

            try:
                # Path to the original (unrevised) evaluation result
                original_result_file = os.path.join(args.rating_dir, m, "ours", args.mode, cat, f"{curr_asin}_{target_base}.json")

                loaded_from_original = False
                if args.run_revision and os.path.exists(original_result_file):
                    try:
                        with open(original_result_file, "r", encoding="utf-8") as f:
                            orig_data = json.load(f)
                            eval_list_out = orig_data.get("evaluations", [])
                            eval_latency = orig_data.get("latency_sec", 0.0)
                            score = orig_data.get("score", 0.0)
                            percentage_score = orig_data.get("percentage_score", 0.0)
                            loaded_from_original = True
                            print(f"Loaded existing original evaluation for {curr_asin}_{target_base} to revise.")
                    except Exception as e:
                        print(f"[WARNING] Failed to load original evaluation from {original_result_file}: {e}")

                if not loaded_from_original:
                    if args.mode == "aggregation":
                        score, percentage_score, eval_list_out, eval_latency = evaluate_aggregation(
                            rubrics, ref_image_path, gen_image_path, model_obj, processor, user_prompt_eval
                        )
                    elif args.mode == "each_item":
                        score, percentage_score, eval_list_out, eval_latency = evaluate_each_item(
                            rubrics, ref_image_path, gen_image_path, model_obj, processor
                        )
                    else:
                        raise ValueError(f"Invalid mode: {args.mode}")

                if args.run_revision and prompt:
                    t_rev_start = time.time()
                    score, percentage_score, eval_list_out = revise_evaluations(
                        eval_list_out, rubrics, ref_image_path, gen_image_path, model_obj, processor, prompt
                    )
                    eval_latency += (time.time() - t_rev_start)

                eval_result = {
                    "pair_id": f"{curr_asin}_{target_base}",
                    "model": m,
                    "category": cat,
                    "asin": curr_asin,
                    "target_file": f"{curr_asin}/{target_base}.jpg",
                    "score": score,
                    "percentage_score": percentage_score,
                    "latency_sec": eval_latency,
                    "evaluations": eval_list_out
                }

                os.makedirs(os.path.dirname(result_file), exist_ok=True)
                with open(result_file, "w", encoding="utf-8") as f:
                    json.dump(eval_result, f, indent=4, ensure_ascii=False)

                scores.append(score)
                latencies.append(eval_latency)
                processed_count += 1
                print(f"Evaluated {curr_asin}_{target_base}: Score = {score:.2f}/5.0 ({percentage_score:.1f}%)")

            except Exception as e:
                print(f"[ERROR] Failed to evaluate {curr_asin}_{target_base}: {e}")

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

            print("\n" + "="*50)
            print(f"EVALUATION STATISTICS SUMMARY FOR MODEL: {m} ({args.mode})")
            print("="*50)
            print(f"Total Evaluated Pairs: {processed_count}")
            print(f"Mean Score (out of 5.0): {mean_score:.3f}")
            print(f"Mean Latency per Evaluation: {mean_latency:.3f} seconds")
            print(f"Summary JSON saved to: {summary_file}")
            print("="*50)

if __name__ == "__main__":
    # Standard generation environment variables
    os.environ["greedy"] = "false"
    os.environ["top_p"] = "0.8"
    os.environ["top_k"] = "20"
    os.environ["temperature"] = "0.7"
    os.environ["repetition_penalty"] = "1.0"
    os.environ["presence_penalty"] = "1.5"
    os.environ["out_seq_length"] = "16384"
    main()
