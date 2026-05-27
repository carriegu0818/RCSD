#!/usr/bin/env python
"""Fast standalone rubric cache generation with vLLM.

This mirrors the rubric prompt and cache layout used by opsd_train_reward.py,
but avoids launching the full reward-training entrypoint just to build rubrics.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any


QUESTION_KEYS = ["question", "problem", "prompt", "instruction", "input"]
REFERENCE_KEYS = ["reference_answer", "reference", "answer", "final_answer", "solution"]


class RubricPromptBuilder:
    """Same prompt as opsd_train_reward.py:RubricPromptBuilder."""

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


def _get_first_present(feature: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in feature and feature[key] is not None:
            return feature[key]
    return None


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in {"none", "null"}:
        return None
    return value


def _normalize_source_arg(value: str) -> str:
    aliases = {
        "natural": "natural_reasoning",
        "naturalreasoning": "natural_reasoning",
        "natural-reasoning": "natural_reasoning",
        "rar": "rar_science",
        "rarscience": "rar_science",
        "rar-science": "rar_science",
    }
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    normalized = aliases.get(normalized, normalized)
    valid_values = {"natural_reasoning", "rar_science"}
    if normalized not in valid_values:
        valid_display = ", ".join(sorted(valid_values))
        raise ValueError(f"Unsupported data_source={value!r}. Expected one of: {valid_display}.")
    return normalized


def _load_training_dataset(data_source: str) -> Any:
    from datasets import load_dataset

    if data_source == "natural_reasoning":
        dataset = load_dataset("facebook/natural_reasoning")
    elif data_source == "rar_science":
        dataset = load_dataset("anisha2102/RaR-Science")
    else:
        raise ValueError(f"Unsupported data_source={data_source!r}.")
    return dataset["train"]


def _has_reference_answer(example: dict[str, Any]) -> bool:
    reference_answer = _get_first_present(example, REFERENCE_KEYS)
    if reference_answer is None:
        return False
    reference_text = _normalize_text(reference_answer)
    return reference_text is not None and len(reference_text.strip()) > 0


def _extract_json_array(text: str) -> str:
    if text is None:
        return ""
    if isinstance(text, list):
        return json.dumps(text, ensure_ascii=False)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = cleaned.replace("<think>", "").replace("</think>", "").strip()
    decoder = json.JSONDecoder()

    search_from = 0
    while True:
        start = cleaned.find("[", search_from)
        if start == -1:
            break
        try:
            parsed, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            search_from = start + 1
            continue
        if isinstance(parsed, list):
            return json.dumps(parsed, ensure_ascii=False)
        search_from = start + 1

    return cleaned


def _is_valid_rubric_text(text: str) -> bool:
    if text is None or not str(text).strip():
        return False
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(parsed, list) or len(parsed) == 0:
        return False
    return all(isinstance(item, (dict, str)) and str(item).strip() for item in parsed)


def _read_adapter_config(model_path: str, force_lora: bool) -> dict[str, Any] | None:
    adapter_config_path = Path(model_path) / "adapter_config.json"
    if adapter_config_path.exists():
        with adapter_config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    if not force_lora:
        return None

    try:
        from peft import PeftConfig
    except Exception as exc:
        raise ImportError("--model_is_lora requires peft when adapter_config.json is not local.") from exc

    peft_config = PeftConfig.from_pretrained(model_path)
    if hasattr(peft_config, "to_dict"):
        return peft_config.to_dict()
    return dict(peft_config.__dict__)


def _load_tokenizer(
    candidates: list[str | None],
    trust_remote_code: bool,
) -> tuple[Any, str]:
    from transformers import AutoTokenizer

    errors = []
    for candidate in candidates:
        candidate = _normalize_optional_string(candidate)
        if candidate is None:
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                candidate,
                trust_remote_code=trust_remote_code,
                padding_side="left",
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer, candidate
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    joined = "\n".join(errors)
    raise RuntimeError(f"Unable to load a tokenizer from candidates. Errors:\n{joined}")


def _apply_chat_template(tokenizer: Any, user_message: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": user_message}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if not enable_thinking:
        try:
            return tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=False)
        except TypeError:
            pass
    return tokenizer.apply_chat_template(messages, **kwargs)


def _build_rubric_prompt(
    example: dict[str, Any],
    prompt_builder: RubricPromptBuilder,
    tokenizer: Any,
    enable_thinking: bool,
) -> str:
    question = _get_first_present(example, QUESTION_KEYS)
    reference_answer = _get_first_present(example, REFERENCE_KEYS)
    if question is None:
        raise ValueError("Missing question field for rubric generation.")
    if reference_answer is None:
        raise ValueError("Missing reference_answer field for rubric generation.")

    question_text = _normalize_text(question)
    reference_answer_text = _normalize_text(reference_answer)
    user_message = prompt_builder.build(question_text, reference_answer_text)
    return _apply_chat_template(tokenizer, user_message, enable_thinking=enable_thinking)


def _load_completed_records(records_path: Path) -> dict[int, str]:
    completed: dict[int, str] = {}
    if not records_path.exists():
        return completed
    with records_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                completed[int(record["idx"])] = str(record["rubric"])
            except Exception as exc:
                raise ValueError(f"Bad JSONL record in {records_path} line {line_number}: {exc}") from exc
    return completed


def _append_records(records_path: Path, records: list[dict[str, Any]]) -> None:
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _save_dataset_atomically(
    dataset: Any,
    cache_dir: Path,
    regenerate: bool,
    max_shard_size: str | None,
) -> None:
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = cache_dir.parent / f".{cache_dir.name}.tmp-{os.getpid()}-{int(time.time())}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    try:
        if max_shard_size:
            dataset.save_to_disk(str(tmp_dir), max_shard_size=max_shard_size)
        else:
            dataset.save_to_disk(str(tmp_dir))
    except TypeError:
        dataset.save_to_disk(str(tmp_dir))

    if cache_dir.exists():
        if not regenerate:
            shutil.rmtree(tmp_dir)
            raise FileExistsError(f"{cache_dir} already exists. Pass --regenerate to replace it.")
        shutil.rmtree(cache_dir)

    shutil.move(str(tmp_dir), str(cache_dir))


def _make_sampling_params(args: argparse.Namespace):
    from vllm import SamplingParams

    top_k = args.top_k if args.top_k and args.top_k > 0 else -1
    sampling_kwargs = {
        "n": 1,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": top_k,
        "max_tokens": args.max_new_tokens,
    }
    if args.stop:
        sampling_kwargs["stop"] = args.stop
    if args.guided_decoding_regex:
        try:
            from vllm.sampling_params import GuidedDecodingParams

            sampling_kwargs["guided_decoding"] = GuidedDecodingParams(
                backend=args.guided_decoding_backend,
                regex=args.guided_decoding_regex,
            )
        except Exception as exc:
            raise RuntimeError("Guided decoding was requested, but this vLLM install does not support it.") from exc
    return SamplingParams(**sampling_kwargs)


def _make_lora_request(adapter_path: str, base_model_name: str):
    from vllm.lora.request import LoRARequest

    try:
        return LoRARequest(
            lora_name="rubric",
            lora_int_id=1,
            lora_path=adapter_path,
            base_model_name=base_model_name,
        )
    except TypeError:
        return LoRARequest("rubric", 1, adapter_path)


def _make_llm(
    args: argparse.Namespace,
    model_name_or_path: str,
    tokenizer_name_or_path: str,
    adapter_config: dict[str, Any] | None,
):
    from vllm import LLM

    llm_kwargs = {
        "model": model_name_or_path,
        "tokenizer": tokenizer_name_or_path,
        "trust_remote_code": args.trust_remote_code,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "disable_log_stats": True,
        "swap_space": args.swap_space,
    }
    if args.max_model_len and args.max_model_len > 0:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.max_num_seqs and args.max_num_seqs > 0:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens and args.max_num_batched_tokens > 0:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if not args.disable_prefix_caching:
        llm_kwargs["enable_prefix_caching"] = True
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    if args.disable_custom_all_reduce:
        if "disable_custom_all_reduce" in inspect.signature(LLM).parameters:
            llm_kwargs["disable_custom_all_reduce"] = True
        else:
            print(
                "Warning: --disable_custom_all_reduce was set, but this vLLM version's "
                "LLM constructor does not expose disable_custom_all_reduce."
            )
    if adapter_config is not None:
        llm_kwargs.update(
            {
                "enable_lora": True,
                "max_loras": 1,
                "max_lora_rank": int(adapter_config.get("r", 64)),
            }
        )

    return LLM(**llm_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_source", default="rar_science", help="Dataset source: rar_science or natural_reasoning.")
    parser.add_argument("--rubric_model_path", default="Qwen/Qwen3-14B", help="Full model id/path or LoRA adapter path.")
    parser.add_argument("--rubric_base_model_name_or_path", default=None, help="Base model for LoRA rubric checkpoints.")
    parser.add_argument("--tokenizer_name_or_path", default=None, help="Optional tokenizer override.")
    parser.add_argument("--model_is_lora", action="store_true", help="Force rubric_model_path to be treated as a LoRA adapter.")
    parser.add_argument("--rubric_cache_dir", required=True, help="Output save_to_disk directory for generated rubric dataset.")
    parser.add_argument("--regenerate", action="store_true", help="Replace an existing cache after successful generation.")
    parser.add_argument("--resume", action="store_true", help="Resume from the JSONL work log if the job was interrupted.")
    parser.add_argument("--work_dir", default=None, help="Directory for incremental JSONL progress. Defaults beside cache dir.")
    parser.add_argument("--keep_work_dir", action="store_true", help="Keep the JSONL work directory after a successful save.")
    parser.add_argument("--sample_size", type=int, default=10000, help="Max examples to generate after shuffling. 0 means all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_num_proc", type=int, default=1)
    parser.add_argument("--prompt_max_length", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of prompts submitted to vLLM per outer batch. Default matches the original sbatch rubric_batch_size.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--stop", action="append", default=None, help="Optional stop string. Can be passed multiple times.")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max_model_len", type=int, default=0, help="vLLM max_model_len. Defaults to prompt+new tokens.")
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--max_num_batched_tokens", type=int, default=65536)
    parser.add_argument("--swap_space", type=float, default=4)
    parser.add_argument("--disable_prefix_caching", action="store_true")
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument(
        "--disable_custom_all_reduce",
        action="store_true",
        help="Disable vLLM's custom all-reduce kernel for tensor parallelism.",
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable_thinking",
        dest="enable_thinking",
        action="store_true",
        default=True,
        help="Let Qwen3 use thinking mode. This matches the original apply_chat_template default.",
    )
    thinking_group.add_argument(
        "--disable_thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable Qwen3 thinking mode. This changes the original prompt formatting behavior.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--guided_decoding_regex", default=None)
    parser.add_argument("--guided_decoding_backend", default="outlines")
    parser.add_argument("--filter_invalid", action="store_true", help="Drop invalid rubrics before saving.")
    parser.add_argument("--save_max_shard_size", default=None)
    return parser.parse_args()


def main() -> None:
    from tqdm.auto import tqdm

    args = parse_args()
    args.data_source = _normalize_source_arg(args.data_source)
    if not args.max_model_len:
        args.max_model_len = args.prompt_max_length + args.max_new_tokens

    cache_dir = Path(args.rubric_cache_dir)
    work_dir = Path(args.work_dir) if args.work_dir else Path(f"{args.rubric_cache_dir}_work")
    records_path = work_dir / "rubrics.jsonl"

    if cache_dir.exists() and not args.regenerate:
        print(f"Rubric cache already exists at {cache_dir}. Nothing to do.")
        return
    if work_dir.exists() and not args.resume:
        shutil.rmtree(work_dir)

    adapter_config = _read_adapter_config(args.rubric_model_path, force_lora=args.model_is_lora)
    is_lora_checkpoint = adapter_config is not None
    if is_lora_checkpoint:
        base_model_name = _normalize_optional_string(args.rubric_base_model_name_or_path) or _normalize_optional_string(
            adapter_config.get("base_model_name_or_path")
        )
        if base_model_name is None:
            raise ValueError(
                "Unable to determine the LoRA base model. Pass --rubric_base_model_name_or_path."
            )
        model_name_or_path = base_model_name
        tokenizer_candidates = [args.tokenizer_name_or_path, args.rubric_model_path, base_model_name]
    else:
        base_model_name = None
        model_name_or_path = args.rubric_model_path
        tokenizer_candidates = [args.tokenizer_name_or_path, args.rubric_model_path]

    tokenizer, tokenizer_name_or_path = _load_tokenizer(tokenizer_candidates, args.trust_remote_code)
    prompt_builder = RubricPromptBuilder()

    print("=" * 80)
    print("Rubric generation configuration")
    print("=" * 80)
    print(f"Data source: {args.data_source}")
    print(f"Rubric model path: {args.rubric_model_path}")
    if is_lora_checkpoint:
        print(f"LoRA base model: {base_model_name}")
    print(f"vLLM model: {model_name_or_path}")
    print(f"Tokenizer: {tokenizer_name_or_path}")
    print(f"Cache dir: {cache_dir}")
    print(f"Work dir: {work_dir}")
    print(f"Sample size: {args.sample_size}")
    print(f"Prompt max length: {args.prompt_max_length}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Qwen3 thinking enabled: {args.enable_thinking}")
    print("=" * 80)

    dataset = _load_training_dataset(args.data_source)
    print(f"Loaded {len(dataset)} train examples.")
    filter_kwargs = {}
    if args.dataset_num_proc and args.dataset_num_proc > 1:
        filter_kwargs["num_proc"] = args.dataset_num_proc
    dataset = dataset.filter(_has_reference_answer, desc="Filtering empty reference answers", **filter_kwargs)
    dataset = dataset.shuffle(seed=args.seed)
    if args.sample_size and args.sample_size > 0 and len(dataset) > args.sample_size:
        dataset = dataset.select(range(args.sample_size))
    if args.data_source == "rar_science" and "rubric" in dataset.column_names:
        dataset = dataset.remove_columns(["rubric"])
    print(f"Prepared {len(dataset)} examples for rubric generation.")

    completed = _load_completed_records(records_path) if args.resume else {}
    if completed:
        print(f"Resuming with {len(completed)} completed rubrics from {records_path}.")

    llm = _make_llm(args, model_name_or_path, tokenizer_name_or_path, adapter_config)
    sampling_params = _make_sampling_params(args)
    lora_request = _make_lora_request(args.rubric_model_path, model_name_or_path) if is_lora_checkpoint else None

    pending_indices = [idx for idx in range(len(dataset)) if idx not in completed]
    progress = tqdm(total=len(pending_indices), desc="Generating rubrics", dynamic_ncols=True)
    for start in range(0, len(pending_indices), args.batch_size):
        batch_indices = pending_indices[start : start + args.batch_size]
        prompts = [
            _build_rubric_prompt(
                dataset[int(idx)],
                prompt_builder=prompt_builder,
                tokenizer=tokenizer,
                enable_thinking=args.enable_thinking,
            )
            for idx in batch_indices
        ]
        tokenized = tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=args.prompt_max_length,
        )
        generate_kwargs = {
            "prompt_token_ids": tokenized["input_ids"],
            "sampling_params": sampling_params,
            "use_tqdm": False,
        }
        if lora_request is not None:
            generate_kwargs["lora_request"] = lora_request

        outputs = llm.generate(**generate_kwargs)
        records = []
        for idx, output in zip(batch_indices, outputs):
            text = output.outputs[0].text if output.outputs else ""
            records.append({"idx": int(idx), "rubric": _extract_json_array(text)})
        _append_records(records_path, records)
        completed.update({record["idx"]: record["rubric"] for record in records})
        progress.update(len(batch_indices))
    progress.close()

    missing = [idx for idx in range(len(dataset)) if idx not in completed]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} generated rubrics; first missing index: {missing[0]}")

    rubrics = [completed[idx] for idx in range(len(dataset))]
    if "rubric" in dataset.column_names:
        dataset = dataset.remove_columns(["rubric"])
    generated = dataset.add_column("rubric", rubrics)
    valid_flags = [_is_valid_rubric_text(rubric) for rubric in rubrics]
    valid_count = sum(valid_flags)
    print(f"Valid rubric JSON arrays: {valid_count}/{len(generated)}")

    if args.filter_invalid:
        generated = generated.add_column("rubric_valid", valid_flags)
        generated = generated.filter(lambda example: example["rubric_valid"], desc="Filtering invalid rubrics")
        generated = generated.remove_columns(["rubric_valid"])
        print(f"Kept {len(generated)} examples after invalid-rubric filtering.")

    _save_dataset_atomically(
        generated,
        cache_dir=cache_dir,
        regenerate=args.regenerate,
        max_shard_size=args.save_max_shard_size,
    )
    print(f"Saved rubric-augmented dataset to: {cache_dir}")

    if not args.keep_work_dir and work_dir.exists():
        shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()
