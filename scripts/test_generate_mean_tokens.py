#!/usr/bin/env python3
import argparse
import json
import os
from contextlib import contextmanager
from itertools import cycle, islice

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
    from peft import AutoPeftModelForCausalLM
except Exception:
    PeftModel = None
    AutoPeftModelForCausalLM = None


class RubricPromptBuilder:
    def __init__(self):
        self.rubric_role_prompt = (
            "# Role You are a top-tier Rubric Designer. Your sole task is to design JSON-formatted evaluation rubrics "
            "based on both the [Question] and the [Reference Answer] provided by the user.\n"
            "# Core Task\n"
            "1. Analyze [Question]: Understand every explicit and implicit requirement in the [Question].\n"
            "2. Leverage [Reference Answer]: Use the [Reference Answer] to capture nuanced expectations, desirable reasoning "
            "patterns, and formatting details that high-quality responses should exhibit. Treat it as authoritative context, not content to be copied.\n"
            "3. Create Rubrics: Following the [Evaluation Criteria Format] and [Design Rules] below, develop 3 to 25 evaluation criteria that ensure candidate answers respond to the [Question] and match the quality demonstrated in the [Reference Answer].\n"
            "4. Output Format: Must strictly follow the [Output Requirements] with no additional text.\n"
        )
        self.rubric_format_prompt = (
            "# [Evaluation Criteria Format]\n"
            "- Each criterion must contain the following fields:\n"
            "1. `title`: (String) A 2-5 word core summary.\n"
            "2. `description`: (String) A clear description of no more than 40 words or 5 sentences.\n"
            "3. `weight`: (Integer) A score between -1 and 10.\n\n"
            "Example output format (match this style exactly)\n"
            "[\n"
            '{"description": "Essential Criteria: The response must explicitly incorporate the 3% efficiency factor to determine the actual power output in visible photons from the 100-W bulb.", "title": "Efficiency Factor", "weight": 5},\n'
            '{"description": "Important Criteria: The response should correctly calculate the energy of a single photon using the formula E = hc/λ, with λ approximately equal to 5000 angstrom, ensuring proper usage of physical constants.", "title": "Photon Energy Calculation", "weight": 5}\n'
            "]\n"
            "Output Requirements (strict)\n"
            "* Output ONLY valid JSON.\n"
            "* Output must be a JSON array of criterion objects (no wrapper keys).\n"
            "* No markdown, no commentary, no code fences, no extra text.\n"
            "* Do not include anything besides the array.\n"
            "* Do not copy sentences from the Reference Answer; write fresh evaluative criteria.\n"
            "Design Rules\n"
            "* Prefer testable, concrete criteria over vague ones.\n"
            "* Use higher weights (7–10) for essential requirements.\n"
            "* Use mid weights (3–6) for strong-but-nonessential improvements.\n"
            "* Use at least one negative criterion (weight -1) for major violations (e.g., ignoring constraints, wrong format, hallucinating, contradicting the reference).\n"
            "* Avoid redundancy: each criterion should measure a distinct aspect.\n"
            "Now generate the rubric criteria for the provided Question and Reference Answer."
        )

    def build(self, question: str, reference_answer: str) -> str:
        return (
            f"{self.rubric_role_prompt}"
            f"# [Question]\n{question}\n"
            f"# [Reference Answer]\n{reference_answer}\n"
            f"{self.rubric_format_prompt}"
        )


def _normalize_text(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _extract_json_array(text: str) -> str:
    if text is None:
        return ""
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text.strip()


@contextmanager
def _disable_deepspeed_zero3_init():
    env_keys = {
        "ACCELERATE_USE_DEEPSPEED": "0",
        "DEEPSPEED_ZERO3_INIT_DISABLE": "1",
        "TRANSFORMERS_NO_DEEPSPEED": "1",
        "HF_DEEPSPEED_CONFIG_FILE": "",
        "ACCELERATE_DEEPSPEED_CONFIG_FILE": "",
    }
    saved_env = {k: os.environ.get(k) for k in env_keys}
    saved_hf_ref = None
    saved_hf_obj = None
    saved_hf_cfg = None
    try:
        for k, v in env_keys.items():
            if v == "":
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        try:
            import transformers.integrations.deepspeed as ds

            saved_hf_ref = getattr(ds, "_hf_deepspeed_config_weak_ref", None)
            saved_hf_obj = saved_hf_ref() if saved_hf_ref is not None else None
            saved_hf_cfg = getattr(ds, "_hf_deepspeed_config", None)

            if hasattr(ds, "unset_hf_deepspeed_config"):
                ds.unset_hf_deepspeed_config()
            if hasattr(ds, "_hf_deepspeed_config_weak_ref"):
                ds._hf_deepspeed_config_weak_ref = None
            if hasattr(ds, "_hf_deepspeed_config"):
                ds._hf_deepspeed_config = None
        except Exception:
            ds = None
        yield
    finally:
        if "ds" in locals() and ds is not None:
            if saved_hf_obj is not None and hasattr(ds, "set_hf_deepspeed_config"):
                ds.set_hf_deepspeed_config(saved_hf_obj)
            elif hasattr(ds, "_hf_deepspeed_config_weak_ref"):
                ds._hf_deepspeed_config_weak_ref = saved_hf_ref
            if hasattr(ds, "_hf_deepspeed_config"):
                ds._hf_deepspeed_config = saved_hf_cfg
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _load_pairs(args):
    if args.dataset_name:
        try:
            from datasets import load_dataset
        except Exception as e:
            raise ImportError("datasets is required for --dataset_name.") from e

        ds_kwargs = {}
        if args.dataset_config:
            ds_kwargs["name"] = args.dataset_config
        dataset = load_dataset(args.dataset_name, **ds_kwargs)
        if args.dataset_split not in dataset:
            raise ValueError(
                f"Split '{args.dataset_split}' not found in dataset. "
                f"Available splits: {list(dataset.keys())}"
            )
        split = dataset[args.dataset_split]
        if args.num_samples and args.num_samples > 0:
            split = split.shuffle(seed=args.seed).select(range(args.num_samples))

        def _get_first_present(feature, keys):
            for key in keys:
                if key in feature and feature[key] is not None:
                    return feature[key]
            return None

        pairs = []
        for ex in split:
            question = _get_first_present(ex, ["question", "problem", "prompt", "instruction", "input"])
            reference = _get_first_present(
                ex, ["reference_answer", "reference", "answer", "final_answer", "solution"]
            )
            if question is None or reference is None:
                continue
            pairs.append((question, reference))

        if not pairs:
            raise ValueError("No usable (question, reference_answer) pairs found in dataset.")
        return list(islice(cycle(pairs), args.num_samples))

    if args.input_jsonl:
        pairs = []
        with open(args.input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                question = obj.get("question") or obj.get("problem") or obj.get("prompt")
                reference = obj.get("reference_answer") or obj.get("reference") or obj.get("answer")
                if question is None or reference is None:
                    raise ValueError("Each JSONL line must include question+reference_answer (or aliases).")
                pairs.append((question, reference))
        if not pairs:
            raise ValueError("input_jsonl is empty.")
        return list(islice(cycle(pairs), args.num_samples))
    return [(args.question, args.reference_answer)] * args.num_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--question", default="What is 2+2?")
    parser.add_argument("--reference_answer", default="The answer is 4.")
    parser.add_argument("--input_jsonl", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_path", default="generated_responses.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_config_path = os.path.join(args.model_name_or_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        raise ValueError("adapter_config.json not found in model_name_or_path; expected a LoRA checkpoint.")
    with open(adapter_config_path, "r", encoding="utf-8") as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config.get("base_model_name_or_path")
    if base_model_name is None:
        raise ValueError("Unable to determine base_model_name_or_path from adapter_config.json.")

    if PeftModel is None:
        raise ImportError("peft is required to load the rubric LoRA checkpoint.")

    with _disable_deepspeed_zero3_init():
        if AutoPeftModelForCausalLM is not None:
            model = AutoPeftModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=model_dtype,
                trust_remote_code=args.trust_remote_code,
                attn_implementation=args.attn_implementation,
                low_cpu_mem_usage=False,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=model_dtype,
                trust_remote_code=args.trust_remote_code,
                attn_implementation=args.attn_implementation,
                low_cpu_mem_usage=False,
            )
            model = PeftModel.from_pretrained(base_model, args.model_name_or_path)

    model.to(args.device)
    model.eval()

    prompt_builder = RubricPromptBuilder()
    pairs = _load_pairs(args)
    prompts = []
    for q, ref in pairs:
        q_text = _normalize_text(q)
        ref_text = _normalize_text(ref)
        user_message = prompt_builder.build(q_text, ref_text)
        messages = [{"role": "user", "content": user_message}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    total_tokens = 0
    num_generated = 0

    with torch.no_grad(), open(args.output_path, "w", encoding="utf-8") as out_f:
        for start in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[start : start + args.batch_size]
            tokenized = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(args.device)

            input_ids = tokenized["input_ids"]
            prompt_lens = tokenized["attention_mask"].sum(dim=1).tolist()

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=tokenized.get("attention_mask", None),
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                pad_token_id=tokenizer.pad_token_id,
            )

            for i, output_ids in enumerate(outputs):
                prompt_len = int(prompt_lens[i])
                completion_ids = output_ids[prompt_len:]
                completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                completion_text = _extract_json_array(completion_text)
                num_tokens = int(len(completion_ids))
                total_tokens += num_tokens
                num_generated += 1

                record = {
                    "id": num_generated,
                    "prompt": batch_prompts[i],
                    "response": completion_text,
                    "num_tokens": num_tokens,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    mean_tokens = total_tokens / max(1, num_generated)
    print(f"Generated {num_generated} responses.")
    print(f"Mean response tokens: {mean_tokens:.2f}")
    print(f"Saved to: {args.output_path}")


if __name__ == "__main__":
    main()
