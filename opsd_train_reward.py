import json
import os
import random
import re
import wandb
from glob import glob
from contextlib import contextmanager

from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig
from opsd_trainer_reward import OPSDTrainer
from dataclasses import dataclass, field
from accelerate import PartialState

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


RUBRICHUB_SCIENCE_SOURCES = {"RaR_science.jsonl", "MegeScience.jsonl"}


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


def _get_first_present(feature, keys):
    for key in keys:
        if key in feature and feature[key] is not None:
            return feature[key]
    return None


def _normalize_text(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _normalize_optional_string(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in {"none", "null"}:
        return None
    return value


def _extract_prompt_text(prompt):
    if prompt is None:
        return None
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts = []
        for message in prompt:
            if isinstance(message, dict):
                content = message.get("content")
                if content is not None:
                    parts.append(_normalize_text(content))
            elif message is not None:
                parts.append(_normalize_text(message))
        return "\n".join(part for part in parts if part and part.strip()) or None
    if isinstance(prompt, dict):
        content = prompt.get("content")
        if content is not None:
            return _normalize_text(content)
    return _normalize_text(prompt)


def _normalize_rubrichub_rubrics(rubrics):
    normalized = []
    if not isinstance(rubrics, list):
        return rubrics
    for idx, rubric in enumerate(rubrics, start=1):
        if not isinstance(rubric, dict):
            normalized.append(rubric)
            continue
        description = rubric.get("description") or rubric.get("criterion") or rubric.get("criteria")
        weight = rubric.get("weight", rubric.get("points", rubric.get("score", 1)))
        title = rubric.get("title") or f"Criterion {idx}"
        normalized.append(
            {
                "title": _normalize_text(title),
                "description": _normalize_text(description),
                "weight": weight,
            }
        )
    return normalized


def _normalize_rubrichub_example(example):
    question = _extract_prompt_text(example.get("prompt") or example.get("query"))
    reward_model = example.get("reward_model") if isinstance(example.get("reward_model"), dict) else {}
    reference_answer = (
        example.get("reference_answer")
        or example.get("answer")
        or reward_model.get("ground_truth")
        or "No reference answer provided; use the rubric as the authoritative guidance."
    )
    rubrics = example.get("rubric_list") or example.get("Rubrics") or example.get("rubrics") or reward_model.get("rubrics")

    example["question"] = _normalize_text(question)
    example["reference_answer"] = _normalize_text(reference_answer)
    example["rubric_list"] = _normalize_rubrichub_rubrics(rubrics)
    return example


def _find_cached_rubrichub_parquet_files():
    cache_roots = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        cache_roots.append(os.path.join(hf_home, "hub"))
    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_hub_cache:
        cache_roots.append(hf_hub_cache)
    cache_roots.append(os.path.expanduser("~/.cache/huggingface/hub"))

    patterns = []
    for cache_root in dict.fromkeys(cache_roots):
        dataset_root = os.path.join(cache_root, "datasets--sojuL--RubricHub_v1", "snapshots", "*")
        patterns.extend(
            [
                os.path.join(dataset_root, "sft_RuFT", "*.parquet"),
            ]
        )
    return sorted({path for pattern in patterns for path in glob(pattern)})


def _minimal_rubrichub_row(example):
    normalized = _normalize_rubrichub_example(example)
    return {
        "question": normalized.get("question"),
        "reference_answer": normalized.get("reference_answer"),
        "rubric_list": normalized.get("rubric_list"),
        "data_source": "Science",
        "rubrichub_source": example.get("source"),
    }


def _load_rubrichub_dataset(sample_size: int = 0, seed: int = 42):
    files = _find_cached_rubrichub_parquet_files()
    if not files:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "RubricHub parquet files were not found in the Hugging Face cache, and "
                "huggingface_hub is unavailable for snapshot download."
            ) from exc
        snapshot_dir = snapshot_download(
            "sojuL/RubricHub_v1",
            repo_type="dataset",
            allow_patterns=["sft_RuFT/*.parquet"],
        )
        files = sorted(glob(os.path.join(snapshot_dir, "sft_RuFT", "*.parquet")))
    if not files:
        raise RuntimeError("No RubricHub parquet files found after dataset download.")

    import pyarrow.parquet as pq

    rng = random.Random(seed)
    rng.shuffle(files)

    rows = []
    for file_idx, path in enumerate(files):
        parquet_file = pq.ParquetFile(path)
        schema_names = parquet_file.schema_arrow.names
        columns = [name for name in ("source", "query", "answer", "rubrics") if name in schema_names]
        if not {"source", "query", "answer", "rubrics"}.issubset(columns):
            continue
        remaining_files = len(files) - file_idx
        file_sample_limit = None
        if sample_size and sample_size > 0:
            remaining_rows = sample_size - len(rows)
            if remaining_rows <= 0:
                break
            file_sample_limit = max(1, (remaining_rows + remaining_files - 1) // remaining_files)
        file_rows = 0
        for batch in parquet_file.iter_batches(columns=columns, batch_size=1024):
            for example in batch.to_pylist():
                if example.get("source") not in RUBRICHUB_SCIENCE_SOURCES:
                    continue
                if not example.get("answer") or not example.get("rubrics"):
                    continue
                row = _minimal_rubrichub_row(example)
                if sample_size and sample_size > 0:
                    rows.append(row)
                    file_rows += 1
                    if len(rows) >= sample_size or file_rows >= file_sample_limit:
                        break
                else:
                    rows.append(row)
            if sample_size and sample_size > 0 and (
                len(rows) >= sample_size or file_rows >= file_sample_limit
            ):
                break
    return Dataset.from_list(rows)


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


def _sanitize_rubric_example(example):
    sanitized = _extract_json_array(example.get("rubric"))
    example["rubric"] = sanitized
    example["rubric_valid"] = _is_valid_rubric_text(sanitized)
    return example


def _normalize_source_arg(value: str, field_name: str, aliases: dict[str, str], valid_values: set[str]) -> str:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    normalized = aliases.get(normalized, normalized)
    if normalized not in valid_values:
        valid_display = ", ".join(sorted(valid_values))
        raise ValueError(f"Unsupported {field_name}={value!r}. Expected one of: {valid_display}.")
    return normalized


def _load_training_dataset(data_source: str, sample_size: int = 0, seed: int = 42):
    if data_source == "natural_reasoning":
        dataset = load_dataset("facebook/natural_reasoning")
    elif data_source == "rar_science":
        dataset = load_dataset("anisha2102/RaR-Science")
    elif data_source == "rubrichub":
        return _load_rubrichub_dataset(sample_size=sample_size, seed=seed)
    else:
        raise ValueError(f"Unsupported data_source={data_source!r}.")
    return dataset["train"]


GENERIC_RUBRIC = [
    {
        "title": "Understand Problem",
        "description": "Understand the problem: Identify what is being asked and restate the goal clearly.",
        "weight": 1,
    },
    {
        "title": "Identify Information",
        "description": "Identify relevant information: Extract the important facts, quantities, constraints, definitions, and assumptions from the problem.",
        "weight": 1,
    },
    {
        "title": "Choose Strategy",
        "description": "Choose an appropriate solution strategy: Select a valid method, principle, formula, or reasoning approach for solving the problem.",
        "weight": 1,
    },
    {
        "title": "Execute Reasoning",
        "description": "Execute the reasoning carefully: Carry out the solution step by step, ensuring that each step follows logically from the previous one.",
        "weight": 1,
    },
    {
        "title": "Check Correctness",
        "description": "Check correctness and consistency: Verify calculations, units, assumptions, edge cases, and whether the answer satisfies the original question.",
        "weight": 1,
    },
    {
        "title": "Provide Final Answer",
        "description": "Provide the final answer clearly: State the final answer in a concise and unambiguous form.",
        "weight": 1,
    },
]


GENERIC_RUBRIC_TEXT = json.dumps(GENERIC_RUBRIC, ensure_ascii=False)


def _use_ground_truth_rubric(example):
    rubric_list = example.get("rubric_list")
    if rubric_list is None:
        raise ValueError("rubric_source='gt' requires a 'rubric_list' column in the dataset.")
    example["rubric"] = _normalize_text(rubric_list)
    return example


def _use_generic_rubric(example):
    example["rubric"] = GENERIC_RUBRIC_TEXT
    return example


def _build_reward_prompt_text(
    tokenizer,
    question: str,
    reference_answer: str,
    rubric: str,
    reason_first: bool,
    teacher_thinking: bool = True,
) -> str:
    if reason_first:
        user_message = (
            f"Question: {question}\n\n"
            f"Reference Answer:\n{reference_answer}\n\n"
            f"Rubric (JSON array):\n{rubric}\n"
            "\n\nThe reference answer and rubric above are authoritative. "
            "Briefly explain the key requirements, constraints, and reasoning expectations implied by them. "
            "Do NOT solve the question yet. Do NOT use <think> tags.\n"
        )
    else:
        user_message = (
            f"Question: {question}\n\n"
            f"Reference Answer:\n{reference_answer}\n\n"
            f"Rubric (JSON array):\n{rubric}\n\n"
            "Use the rubric and reference answer as guidance.\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        )
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=teacher_thinking,
    )


def _within_reward_prompt_limit(
    example,
    tokenizer,
    max_prompt_tokens: int,
    reason_first: bool,
    teacher_thinking: bool = True,
) -> bool:
    question = _normalize_text(_get_first_present(example, ["question", "problem", "prompt", "instruction", "input"]))
    reference_answer = _normalize_text(
        _get_first_present(example, ["reference_answer", "reference", "answer", "final_answer", "solution"])
    )
    rubric = _normalize_text(_get_first_present(example, ["rubric", "rubric_list"]))

    if question is None or reference_answer is None or rubric is None:
        return False

    prompt_text = _build_reward_prompt_text(
        tokenizer,
        question,
        reference_answer,
        rubric,
        reason_first,
        teacher_thinking=teacher_thinking,
    )
    prompt_tokens = tokenizer(
        prompt_text,
        padding=False,
        truncation=False,
        add_special_tokens=False,
    )["input_ids"]
    return len(prompt_tokens) <= max_prompt_tokens


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

            # Transformers >= 4.51 uses _hf_deepspeed_config_weak_ref for Zero-3 detection.
            saved_hf_ref = getattr(ds, "_hf_deepspeed_config_weak_ref", None)
            saved_hf_obj = saved_hf_ref() if saved_hf_ref is not None else None
            saved_hf_cfg = getattr(ds, "_hf_deepspeed_config", None)  # legacy name (older versions)

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


def _build_rubric_prompts(batch, prompt_builder, tokenizer):
    questions = _get_first_present(batch, ["question", "problem", "prompt", "instruction", "input"])
    reference_answers = _get_first_present(
        batch, ["reference_answer", "reference", "answer", "final_answer", "solution"]
    )
    if questions is None:
        raise ValueError("Missing question field for rubric generation.")
    if reference_answers is None:
        raise ValueError("Missing reference_answer field for rubric generation.")

    prompts = []
    for question, reference_answer in zip(questions, reference_answers):
        question_text = _normalize_text(question)
        reference_answer_text = _normalize_text(reference_answer)
        user_message = prompt_builder.build(question_text, reference_answer_text)
        messages = [{"role": "user", "content": user_message}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    return prompts


def generate_rubrics_for_dataset(
    dataset,
    rubric_model_path,
    rubric_base_model_name_or_path,
    rubric_cache_dir,
    state,
    distributed,
    rubric_prompt_max_length,
    rubric_max_new_tokens,
    rubric_temperature,
    rubric_top_p,
    rubric_top_k,
    rubric_batch_size,
    rubric_use_vllm,
    model_dtype,
    attn_implementation,
    trust_remote_code,
    vllm_mode=None,
    vllm_tensor_parallel_size=1,
    vllm_gpu_memory_utilization=0.6,
    vllm_server_host="localhost",
    vllm_server_port=8000,
    vllm_server_timeout=60,
    vllm_guided_decoding_regex=None,
):
    import torch

    device = state.device if state is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    rubric_dtype = model_dtype if device.type == "cuda" else torch.float32

    prompt_builder = RubricPromptBuilder()

    rubric_tokenizer = AutoTokenizer.from_pretrained(
        rubric_model_path,
        trust_remote_code=trust_remote_code,
        padding_side="left",
    )
    if rubric_tokenizer.pad_token is None:
        rubric_tokenizer.pad_token = rubric_tokenizer.eos_token

    adapter_config_path = os.path.join(rubric_model_path, "adapter_config.json")
    adapter_config = None
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)
    is_lora_checkpoint = adapter_config is not None
    base_model_name = rubric_model_path
    if is_lora_checkpoint:
        if PeftModel is None:
            raise ImportError("peft is required to load the rubric LoRA checkpoint.")
        base_model_name = _normalize_optional_string(rubric_base_model_name_or_path) or _normalize_optional_string(
            adapter_config.get("base_model_name_or_path")
        )
        if base_model_name is None:
            raise ValueError(
                "Unable to determine base_model_name_or_path from rubric adapter config. "
                "Pass --rubric_base_model_name_or_path for LoRA rubric checkpoints whose adapter_config.json "
                "does not record the base model."
            )

    rubric_model = None
    vllm_engine = None
    vllm_lora_request = None
    if rubric_use_vllm:
        from trl.import_utils import is_vllm_available

        if not is_vllm_available():
            raise ValueError("rubric_use_vllm is set but vLLM is not available.")
        if distributed and vllm_tensor_parallel_size and vllm_tensor_parallel_size > 1:
            raise ValueError(
                "rubric_use_vllm with distributed rubric generation currently requires "
                "vllm_tensor_parallel_size=1."
            )

        from vllm import LLM, SamplingParams
        from vllm.sampling_params import GuidedDecodingParams

        if is_lora_checkpoint:
            from vllm.lora.request import LoRARequest

        vllm_kwargs = {
            "model": base_model_name,
            "tokenizer": rubric_model_path,
            "trust_remote_code": trust_remote_code,
            "dtype": str(rubric_dtype).replace("torch.", ""),
            "tensor_parallel_size": vllm_tensor_parallel_size,
            "gpu_memory_utilization": vllm_gpu_memory_utilization,
        }
        if is_lora_checkpoint:
            lora_rank = adapter_config.get("r", 64)
            vllm_kwargs.update(
                {
                    "enable_lora": True,
                    "max_loras": 1,
                    "max_lora_rank": lora_rank,
                }
            )
        vllm_engine = LLM(**vllm_kwargs)
        if is_lora_checkpoint:
            vllm_lora_request = LoRARequest(
                lora_name="rubric",
                lora_int_id=1,
                lora_path=rubric_model_path,
                base_model_name=base_model_name,
            )
    else:
        with _disable_deepspeed_zero3_init():
            if is_lora_checkpoint:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=rubric_dtype,
                    trust_remote_code=trust_remote_code,
                    attn_implementation=attn_implementation or "flash_attention_2",
                    low_cpu_mem_usage=False,
                )
                rubric_model = PeftModel.from_pretrained(base_model, rubric_model_path)
            else:
                rubric_model = AutoModelForCausalLM.from_pretrained(
                    rubric_model_path,
                    torch_dtype=rubric_dtype,
                    trust_remote_code=trust_remote_code,
                    attn_implementation=attn_implementation or "flash_attention_2",
                    low_cpu_mem_usage=False,
                )
            rubric_model.to(device)

        rubric_model.eval()

    def _batched_generate(batch):
        prompts = _build_rubric_prompts(batch, prompt_builder, rubric_tokenizer)
        if rubric_use_vllm:
            if vllm_mode not in (None, "colocate"):
                raise ValueError(
                    f"rubric_use_vllm currently supports vllm_mode='colocate' only, got {vllm_mode}."
                )
            tokenized = rubric_tokenizer(
                prompts,
                padding=False,
                truncation=True,
                max_length=rubric_prompt_max_length,
            )
            prompt_token_ids = tokenized["input_ids"]
            guided_decoding = (
                GuidedDecodingParams(backend="outlines", regex=vllm_guided_decoding_regex)
                if vllm_guided_decoding_regex
                else None
            )
            sampling_params = SamplingParams(
                n=1,
                temperature=rubric_temperature,
                top_p=rubric_top_p,
                top_k=rubric_top_k if rubric_top_k and rubric_top_k > 0 else -1,
                max_tokens=rubric_max_new_tokens,
                guided_decoding=guided_decoding,
            )
            generate_kwargs = {
                "prompt_token_ids": prompt_token_ids,
                "sampling_params": sampling_params,
                "use_tqdm": False,
            }
            if vllm_lora_request is not None:
                generate_kwargs["lora_request"] = vllm_lora_request
            outputs = vllm_engine.generate(**generate_kwargs)
            decoded = [out.outputs[0].text if out.outputs else "" for out in outputs]
        else:
            tokenized = rubric_tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=rubric_prompt_max_length,
            ).to(device)

            with torch.no_grad():
                outputs = rubric_model.generate(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized.get("attention_mask", None),
                    max_new_tokens=rubric_max_new_tokens,
                    do_sample=True,
                    temperature=rubric_temperature,
                    top_p=rubric_top_p,
                    top_k=rubric_top_k,
                    pad_token_id=rubric_tokenizer.pad_token_id,
                )

            prompt_len = tokenized["input_ids"].shape[1]
            completions = outputs[:, prompt_len:]
            decoded = rubric_tokenizer.batch_decode(completions, skip_special_tokens=True)
        rubrics = [_extract_json_array(text) for text in decoded]
        return {"rubric": rubrics}

    if distributed and state is not None and state.num_processes > 1:
        if "__orig_idx__" not in dataset.column_names:
            dataset = dataset.add_column("__orig_idx__", list(range(len(dataset))))

        shard = dataset.shard(num_shards=state.num_processes, index=state.process_index)
        generated_shard = shard.map(
            _batched_generate,
            batched=True,
            batch_size=rubric_batch_size,
            desc=f"Generating rubrics (rank {state.process_index})",
        )

        shards_dir = os.path.join(rubric_cache_dir, "_shards")
        os.makedirs(shards_dir, exist_ok=True)
        shard_dir = os.path.join(shards_dir, f"shard_{state.process_index:05d}")
        generated_shard.save_to_disk(shard_dir)

        state.wait_for_everyone()

        if state.is_main_process:
            shard_paths = [
                os.path.join(shards_dir, f"shard_{i:05d}") for i in range(state.num_processes)
            ]
            shards = [load_from_disk(p) for p in shard_paths]
            combined = concatenate_datasets(shards)
            combined = combined.sort("__orig_idx__")
            combined = combined.remove_columns(["__orig_idx__"])
            os.makedirs(rubric_cache_dir, exist_ok=True)
            combined.save_to_disk(rubric_cache_dir)

        state.wait_for_everyone()
        generated = load_from_disk(rubric_cache_dir)
    else:
        generated = dataset.map(
            _batched_generate,
            batched=True,
            batch_size=rubric_batch_size,
            desc="Generating rubrics with rubric checkpoint",
        )

    # Free GPU memory early
    if "base_model" in locals():
        del base_model
    if rubric_model is not None:
        del rubric_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return generated


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extended script arguments with Thinking Machines loss option."""

    use_tinker_loss: bool = field(
        default=False,
        metadata={
            "help": "Use Thinking Machines style on-policy reverse KL loss instead of GKD's full-vocab JSD loss. "
            "This is much more memory efficient (O(1) vs O(vocab_size) per token)."
        },
    )
    fixed_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use the initial policy (step 0) as a fixed teacher. Only works with use_peft=True. "
            "The teacher will use the base model without LoRA adapters, while the student updates."
        },
    )
    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for this experiment. Will be used for both the output directory "
            "(appended to output_dir) and WandB run name. If not specified, will generate "
            "automatic name based on hyperparameters."
        },
    )
    data_source: str = field(
        default="natural_reasoning",
        metadata={
            "help": "Training dataset source. Supported values: natural_reasoning, rar_science."
        },
    )
    rubric_source: str = field(
        default="cache",
        metadata={
            "help": "Rubric source. Supported values: cache (generated/cached rubrics), "
            "gt (dataset rubric_list), generic (same general rubric for every example)."
        },
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the generated text so far. "
            "Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens."
        },
    )
    reason_first: bool = field(
        default=False,
        metadata={
            "help": "Let the teacher model first rationalize (generate rationalization explictly) about the given reasoning first then act as teacher."
        },
    )
    top_k_loss: int = field(
        default=0,
        metadata={
            "help": "Restrict the JSD loss to only the top-k tokens of the teacher distribution. Both student and "
            "teacher distributions are renormalized over these k tokens before computing JSD. "
            "Set to 0 (default) to use the full vocabulary."
        },
    )
    jsd_token_clip: float = field(
        default=0.05,
        metadata={
            "help": "Clip the JSD loss for each token to a maximum value. This can improve stability by preventing "
            "extremely high-loss stylistic tokens from dominating the training signal. Set to 0 for no clipping."
        },
    )
    use_ema_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use an exponential moving average (EMA) of student weights as the teacher. "
            "The EMA teacher is a smoothly-lagged version of the student, avoiding the teacher "
            "collapsing to the current policy (dynamic) or staying frozen (fixed_teacher). "
            "Mutually exclusive with fixed_teacher."
        },
    )
    ema_decay: float = field(
        default=0.999,
        metadata={
            "help": "EMA decay factor. Higher values make the teacher change more slowly. "
            "Typical range: 0.99–0.9999. Only used when use_ema_teacher=True."
        },
    )
    student_thinking: bool = field(
        default=True,
        metadata={
            "help": "Whether to enable Qwen3 thinking mode for the student during rollout. "
            "Default True to preserve existing reward-training scripts."
        },
    )
    teacher_thinking: bool = field(
        default=True,
        metadata={
            "help": "Whether to enable Qwen3 thinking mode for teacher privileged prompts and teacher reasoning. "
            "Default True to preserve existing reward-training scripts."
        },
    )
    rubric_model_path: str = field(
        default="/gpfs/radev/pi/ying_rex/sg2768/OPSD/outputs/qwen3_8b_rubric_fixteacher_temp12_lr2e5_gen4096/checkpoint-1000",
        metadata={
            "help": "Path or Hugging Face model id used to generate rubrics. Can be a full model or LoRA checkpoint."
        },
    )
    rubric_base_model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Optional base model for rubric LoRA checkpoints whose adapter_config.json is missing "
            "base_model_name_or_path. Ignored when rubric_model_path points to a full model."
        },
    )
    rubric_cache_dir: str = field(
        default=None,
        metadata={"help": "Optional path to save/load the dataset augmented with generated rubrics."},
    )
    regenerate_rubrics: bool = field(
        default=False,
        metadata={"help": "Regenerate rubrics even if a cached rubric dataset exists."},
    )
    rubric_prompt_max_length: int = field(
        default=4096,
        metadata={"help": "Max token length for rubric prompts before generation."},
    )
    rubric_sample_size: int = field(
        default=10000,
        metadata={"help": "Max number of examples to sample for rubric generation (0 = use all)."},
    )
    rubric_max_new_tokens: int = field(
        default=1024,
        metadata={"help": "Max new tokens to generate for each rubric."},
    )
    rubric_temperature: float = field(
        default=0.7,
        metadata={"help": "Sampling temperature for rubric generation."},
    )
    rubric_top_p: float = field(
        default=0.95,
        metadata={"help": "Top-p nucleus sampling for rubric generation."},
    )
    rubric_top_k: int = field(
        default=20,
        metadata={"help": "Top-k sampling for rubric generation (set 0 to disable)."},
    )
    rubric_batch_size: int = field(
        default=2,
        metadata={"help": "Batch size for rubric generation."},
    )
    rubric_distributed: bool = field(
        default=True,
        metadata={"help": "Shard rubric generation across all processes (use all GPUs)."},
    )
    rubric_use_vllm: bool = field(
        default=False,
        metadata={"help": "Use vLLM for rubric generation (LoRA via vLLM)."},
    )
    rubric_only: bool = field(
        default=False,
        metadata={"help": "Only generate/cache rubrics, then exit before training."},
    )
    max_reward_prompt_tokens: int = field(
        default=2048,
        metadata={"help": "Drop reward-training examples whose privileged prompt exceeds this many tokens."},
    )


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    script_args.data_source = _normalize_source_arg(
        script_args.data_source,
        field_name="data_source",
        aliases={
            "natural": "natural_reasoning",
            "naturalreasoning": "natural_reasoning",
            "rar": "rar_science",
            "rarscience": "rar_science",
            "rubric_hub": "rubrichub",
            "rubric-hub": "rubrichub",
        },
        valid_values={"natural_reasoning", "rar_science", "rubrichub"},
    )
    script_args.rubric_source = _normalize_source_arg(
        script_args.rubric_source,
        field_name="rubric_source",
        aliases={
            "ground_truth": "gt",
            "groundtruth": "gt",
            "general": "generic",
            "generic_rubric": "generic",
            "general_rubric": "generic",
        },
        valid_values={"cache", "gt", "generic"},
    )

    ################
    # WandB Run Name & Output Directory
    ################
    # Format learning rate (e.g., 2e-4 -> "2e-4" or 0.0002 -> "2e-4")
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")

    # Get number of processes from environment (set by accelerate launch)
    num_processes = int(os.environ.get("WORLD_SIZE", 1))

    # Calculate effective batch size
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    # Use custom run_config if provided, otherwise generate automatic name
    if script_args.run_config:
        full_wandb_run_config = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        # Append run_config to output_dir if it doesn't already end with it
        if not training_args.output_dir.endswith(script_args.run_config):
            from pathlib import Path

            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        # Extract model name from path (e.g., "Qwen3-1.7B" from "/home/siyanzhao/models/Qwen3-1.7B")
        model_name = model_args.model_name_or_path.split("/")[-1]

        # Create concise run name
        full_wandb_run_config = (
            f"opsd_{model_name}_"
            f"lr{lr_str}_"
            f"bs{effective_batch_size}_"
            f"tok{training_args.max_completion_length}"
        )

        # Add fixed_teacher to wandb name if enabled
        if script_args.fixed_teacher:
            full_wandb_run_config += "_fixteach"

    # Print configuration info
    print(f"\n{'='*80}")
    print(f"RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_config}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"Data Source: {script_args.data_source}")
    print(f"Rubric Source: {script_args.rubric_source}")
    print(f"Rubric Model Path: {script_args.rubric_model_path}")
    print(f"Student Thinking: {script_args.student_thinking}")
    print(f"Teacher Thinking: {script_args.teacher_thinking}")
    if script_args.rubric_base_model_name_or_path:
        print(f"Rubric Base Model Override: {script_args.rubric_base_model_name_or_path}")
    print(f"{'='*80}\n")

    ################
    # WandB Initialization
    ################
    # Validate fixed_teacher argument
    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError(
            "fixed_teacher=True requires use_peft=True. As the fixed teacher is implemented by disabling LoRA adapters."
        )

    # Only initialize wandb on main process (LOCAL_RANK 0 or not set)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            name=full_wandb_run_config,
            config={
                "model_name": model_args.model_name_or_path,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "max_completion_length": training_args.max_completion_length,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "lmbda": training_args.lmbda,
                "max_length": training_args.max_length,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "num_processes": num_processes,
                "use_tinker_loss": script_args.use_tinker_loss,
                "fixed_teacher": script_args.fixed_teacher,
                "data_source": script_args.data_source,
                "rubric_source": script_args.rubric_source,
                "rubric_model_path": script_args.rubric_model_path,
                "rubric_base_model_name_or_path": script_args.rubric_base_model_name_or_path,
                "top_k_loss": script_args.top_k_loss if script_args.top_k_loss > 0 else None,
                "use_ema_teacher": script_args.use_ema_teacher,
                "ema_decay": script_args.ema_decay if script_args.use_ema_teacher else None,
                "student_thinking": script_args.student_thinking,
                "teacher_thinking": script_args.teacher_thinking,
            },
        )

    ################
    # Model & Tokenizer
    ################
    import torch

    # Determine dtype - handle both old torch_dtype and new dtype attributes
    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        if isinstance(model_args.torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        else:
            model_dtype = model_args.torch_dtype
    elif hasattr(model_args, "dtype") and model_args.dtype is not None:
        model_dtype = model_args.dtype
    else:
        model_dtype = torch.bfloat16

    print(f"\n{'='*80}")
    print(f"Loading model with dtype: {model_dtype}")
    print(f"Using attention implementation: {model_args.attn_implementation or 'flash_attention_2'}")
    print(f"{'='*80}\n")

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs

    # No separate teacher model needed - we use the same model with privileged info

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    # Load the math dataset with ground truth solutions
    ################
    # Training
    ################
    # Add presence_penalty to training_args so it can be accessed in the trainer
    training_args.presence_penalty = script_args.presence_penalty
    # SFTTrainer requires a dataset_text_field; use question to avoid KeyError on tokenization.
    training_args.dataset_text_field = "question"

    sample_size = getattr(script_args, "rubric_sample_size", 0)
    seed = getattr(training_args, "seed", 42)
    train_dataset = _load_training_dataset(
        script_args.data_source,
        sample_size=sample_size if script_args.data_source == "rubrichub" else 0,
        seed=seed,
    )

    # Filter out entries with empty reference answers
    def _has_reference_answer(batch):
        refs = _get_first_present(
            batch, ["reference_answer", "reference", "answer", "final_answer", "solution"]
        )
        if refs is None:
            return [False] * len(_get_first_present(batch, ["question", "problem", "prompt", "instruction", "input"]))
        flags = []
        for ref in refs:
            if ref is None:
                flags.append(False)
                continue
            ref_text = _normalize_text(ref).strip()
            flags.append(len(ref_text) > 0)
        return flags

    train_dataset = train_dataset.filter(
        _has_reference_answer,
        batched=True,
        desc="Filtering empty reference answers",
    )

    # Sample a subset for rubric generation (or all if smaller / disabled)
    train_dataset = train_dataset.shuffle(seed=seed)
    if sample_size and sample_size > 0 and len(train_dataset) > sample_size:
        train_dataset = train_dataset.select(range(sample_size))

    # Generate rubric targets using the rubric checkpoint (cached on disk)
    state = PartialState()
    rubric_cache_dir = script_args.rubric_cache_dir
    if rubric_cache_dir is None:
        rubric_cache_dir = os.path.join(training_args.output_dir, "rubric_cache")

    if script_args.rubric_source == "generic":
        if state.is_main_process:
            print("Using the same generic rubric for every example.")
        train_dataset = train_dataset.map(
            _use_generic_rubric,
            desc="Applying generic rubric",
        )
    elif script_args.rubric_source == "gt":
        if "rubric_list" not in train_dataset.column_names:
            raise ValueError(
                f"rubric_source='gt' requires 'rubric_list' in the {script_args.data_source} training dataset."
            )
        if state.is_main_process:
            print("Using dataset-provided ground-truth rubrics from rubric_list.")
        train_dataset = train_dataset.map(
            _use_ground_truth_rubric,
            desc="Loading ground-truth rubrics from rubric_list",
        )
    else:
        if script_args.data_source == "rar_science" and "rubric" in train_dataset.column_names:
            train_dataset = train_dataset.remove_columns(["rubric"])
        if os.path.exists(rubric_cache_dir) and not script_args.regenerate_rubrics:
            if state.is_main_process:
                print(f"Loading cached rubric dataset from: {rubric_cache_dir}")
            state.wait_for_everyone()
            train_dataset = load_from_disk(rubric_cache_dir)
        else:
            if state.is_main_process:
                print(f"Generating rubrics using model/checkpoint: {script_args.rubric_model_path}")
                if script_args.rubric_distributed and state.num_processes > 1:
                    print(f"Rubric generation will use {state.num_processes} processes (all GPUs).")
                else:
                    print("Rubric generation will use a single process/GPU.")
            state.wait_for_everyone()

            if script_args.rubric_distributed and state.num_processes > 1:
                train_dataset = generate_rubrics_for_dataset(
                    train_dataset,
                    rubric_model_path=script_args.rubric_model_path,
                    rubric_base_model_name_or_path=script_args.rubric_base_model_name_or_path,
                    rubric_cache_dir=rubric_cache_dir,
                    state=state,
                    distributed=True,
                    rubric_prompt_max_length=script_args.rubric_prompt_max_length,
                    rubric_max_new_tokens=script_args.rubric_max_new_tokens,
                    rubric_temperature=script_args.rubric_temperature,
                    rubric_top_p=script_args.rubric_top_p,
                    rubric_top_k=script_args.rubric_top_k,
                    rubric_batch_size=script_args.rubric_batch_size,
                    rubric_use_vllm=script_args.rubric_use_vllm,
                    model_dtype=model_dtype,
                    attn_implementation=model_args.attn_implementation,
                    trust_remote_code=model_args.trust_remote_code,
                    vllm_mode=getattr(script_args, "vllm_mode", None),
                    vllm_tensor_parallel_size=getattr(script_args, "vllm_tensor_parallel_size", 1),
                    vllm_gpu_memory_utilization=getattr(
                        script_args, "vllm_gpu_memory_utilization", 0.6
                    ),
                    vllm_server_host=getattr(script_args, "vllm_server_host", "localhost"),
                    vllm_server_port=getattr(script_args, "vllm_server_port", 8000),
                    vllm_server_timeout=getattr(script_args, "vllm_server_timeout", 60),
                    vllm_guided_decoding_regex=getattr(
                        script_args, "vllm_guided_decoding_regex", None
                    ),
                )
                if state.is_main_process:
                    print(f"Saved rubric-augmented dataset to: {rubric_cache_dir}")
            else:
                if state.is_main_process:
                    train_dataset = generate_rubrics_for_dataset(
                        train_dataset,
                        rubric_model_path=script_args.rubric_model_path,
                        rubric_base_model_name_or_path=script_args.rubric_base_model_name_or_path,
                        rubric_cache_dir=rubric_cache_dir,
                        state=state,
                        distributed=False,
                        rubric_prompt_max_length=script_args.rubric_prompt_max_length,
                        rubric_max_new_tokens=script_args.rubric_max_new_tokens,
                        rubric_temperature=script_args.rubric_temperature,
                        rubric_top_p=script_args.rubric_top_p,
                        rubric_top_k=script_args.rubric_top_k,
                        rubric_batch_size=script_args.rubric_batch_size,
                        rubric_use_vllm=script_args.rubric_use_vllm,
                        model_dtype=model_dtype,
                        attn_implementation=model_args.attn_implementation,
                        trust_remote_code=model_args.trust_remote_code,
                        vllm_mode=getattr(script_args, "vllm_mode", None),
                        vllm_tensor_parallel_size=getattr(script_args, "vllm_tensor_parallel_size", 1),
                        vllm_gpu_memory_utilization=getattr(
                            script_args, "vllm_gpu_memory_utilization", 0.6
                        ),
                        vllm_server_host=getattr(script_args, "vllm_server_host", "localhost"),
                        vllm_server_port=getattr(script_args, "vllm_server_port", 8000),
                        vllm_server_timeout=getattr(script_args, "vllm_server_timeout", 60),
                        vllm_guided_decoding_regex=getattr(
                            script_args, "vllm_guided_decoding_regex", None
                        ),
                    )
                    os.makedirs(rubric_cache_dir, exist_ok=True)
                    train_dataset.save_to_disk(rubric_cache_dir)
                    print(f"Saved rubric-augmented dataset to: {rubric_cache_dir}")
                state.wait_for_everyone()
                if not state.is_main_process:
                    train_dataset = load_from_disk(rubric_cache_dir)

    if script_args.rubric_only:
        if state.is_main_process:
            print("Rubric-only mode enabled. Cached rubrics are ready; exiting before training.")
        raise SystemExit(0)

    pre_sanitize_count = len(train_dataset)
    train_dataset = train_dataset.map(
        _sanitize_rubric_example,
        desc="Sanitizing generated rubrics",
    )
    train_dataset = train_dataset.filter(
        lambda example: example.get("rubric_valid", False),
        desc="Filtering invalid rubrics",
    )
    invalid_removed = pre_sanitize_count - len(train_dataset)

    pre_length_filter_count = len(train_dataset)
    train_dataset = train_dataset.filter(
        lambda example: _within_reward_prompt_limit(
            example,
            tokenizer=tokenizer,
            max_prompt_tokens=script_args.max_reward_prompt_tokens,
            reason_first=script_args.reason_first,
            teacher_thinking=script_args.teacher_thinking,
        ),
        desc=f"Filtering reward prompts > {script_args.max_reward_prompt_tokens} tokens",
    )
    length_removed = pre_length_filter_count - len(train_dataset)

    if "rubric_valid" in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns(["rubric_valid"])

    print(
        f"Reward dataset cleanup: removed {invalid_removed} invalid rubrics and "
        f"{length_removed} overlength prompts; kept {len(train_dataset)} examples."
    )
    print(f"Final training dataset size after all filtering: {len(train_dataset)}")
    if state.is_main_process and wandb.run is not None:
        wandb.config.update(
            {"final_train_dataset_size": len(train_dataset)},
            allow_val_change=True,
        )

    trainer = OPSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        use_thinking_machines_loss=script_args.use_tinker_loss,
        fixed_teacher=script_args.fixed_teacher,
        reason_first=script_args.reason_first,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
        student_thinking=script_args.student_thinking,
        teacher_thinking=script_args.teacher_thinking,
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=True,
            temperature=training_args.temperature,
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    trainer.save_model(training_args.output_dir)
