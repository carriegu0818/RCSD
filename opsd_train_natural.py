import csv
import io
import json
import os
import random
import re
import urllib.request
from typing import Optional

import torch
import wandb

from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

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
from opsd_trainer_rar import OPSDTrainer
from dataclasses import dataclass, field

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


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
    teacher_prompt_tag: Optional[str] = field(
        default=None,
        metadata={
            "help": "Force teacher prompt mode: 'rubric', 'reference_answer', or 'cot'. "
            "If omitted, use dataset tags or auto-detect per example."
        },
    )
    gpqa_eval_interval: int = field(
        default=10,
        metadata={
            "help": "Evaluate GPQA every N steps. Set to 0 to disable."
        },
    )
    gpqa_eval_size: int = field(
        default=50,
        metadata={
            "help": "Number of GPQA examples to evaluate on (after shuffling)."
        },
    )
    gpqa_eval_seed: int = field(
        default=0,
        metadata={
            "help": "Random seed for GPQA sampling and permutations (matches simple-evals default)."
        },
    )
    gpqa_eval_max_new_tokens: int = field(
        default=64,
        metadata={
            "help": "Max new tokens for GPQA generation (letter choice)."
        },
    )
    gpqa_eval_repeats: int = field(
        default=1,
        metadata={
            "help": "Number of repeats per GPQA example (matches simple-evals n_repeats)."
        },
    )
    gpqa_variant: str = field(
        default="diamond",
        metadata={
            "help": "GPQA variant (diamond/main/extended)."
        },
    )
    gpqa_csv_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional local path or URL to the GPQA CSV. If unset, uses simple-evals public CSV URL."
        },
    )


GPQA_QUERY_TEMPLATE = """Answer the following multiple choice question.
The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.
{Question}
A) {A}
B) {B}
C) {C}
D) {D}"""

GPQA_ANSWER_PATTERN = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?")


def _normalize_text(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _extract_llm_response_text(responses):
    if responses is None:
        return None

    if isinstance(responses, str):
        stripped = responses.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
        return _extract_llm_response_text(parsed)

    if isinstance(responses, dict):
        response_text = responses.get("response")
        if response_text is not None:
            normalized = _normalize_text(response_text)
            if normalized and normalized.strip():
                return normalized
        for key in ("responses", "text", "content"):
            nested = _extract_llm_response_text(responses.get(key))
            if nested:
                return nested
        return None

    if isinstance(responses, list):
        for item in responses:
            extracted = _extract_llm_response_text(item)
            if extracted:
                return extracted
        return None

    normalized = _normalize_text(responses)
    if normalized and normalized.strip():
        return normalized
    return None


def _load_gpqa_rows(path_or_url: str) -> list[dict]:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        with urllib.request.urlopen(path_or_url) as response:
            data = response.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(data))
    else:
        with open(path_or_url, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
    rows = [row for row in reader]
    if not rows:
        raise ValueError(f"GPQA CSV appears empty: {path_or_url}")
    return rows


def _build_gpqa_examples(rows, num_examples: int, n_repeats: int, seed: int):
    rng = random.Random(seed)
    if num_examples:
        if n_repeats != 1:
            raise ValueError("gpqa_eval_repeats must be 1 when gpqa_eval_size is set.")
        if num_examples > len(rows):
            raise ValueError(
                f"Requested {num_examples} GPQA examples, but only {len(rows)} rows are available."
            )
        rows = rng.sample(rows, num_examples)

    rows = rows * max(1, n_repeats)
    examples = []

    for row in rows:
        try:
            question = str(row["Question"]).strip()
            correct = str(row["Correct Answer"]).strip()
            incorrect_1 = str(row["Incorrect Answer 1"]).strip()
            incorrect_2 = str(row["Incorrect Answer 2"]).strip()
            incorrect_3 = str(row["Incorrect Answer 3"]).strip()
        except KeyError as exc:
            raise ValueError(
                f"GPQA CSV is missing required columns. Expected columns like "
                f"'Question', 'Correct Answer', 'Incorrect Answer 1/2/3'. Missing: {exc}"
            ) from exc

        choices = [correct, incorrect_1, incorrect_2, incorrect_3]
        perm = rng.sample(range(4), 4)
        choices = [choices[i] for i in perm]
        correct_index = choices.index(correct)
        correct_letter = "ABCD"[correct_index]

        prompt = GPQA_QUERY_TEMPLATE.format(
            Question=question,
            A=choices[0],
            B=choices[1],
            C=choices[2],
            D=choices[3],
        )
        examples.append({"prompt": prompt, "answer": correct_letter})

    return examples


class GPQAEvalCallback(TrainerCallback):
    def __init__(self, trainer, tokenizer, eval_examples, interval, max_new_tokens):
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.eval_examples = eval_examples
        self.interval = interval
        self.max_new_tokens = max_new_tokens

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.interval <= 0:
            return
        if state.global_step == 0 or state.global_step % self.interval != 0:
            return
        self._evaluate_and_log(state.global_step)

    def _extract_letter(self, text: str):
        match = GPQA_ANSWER_PATTERN.search(text)
        if not match:
            return None
        return match.group(1)

    def _evaluate_and_log(self, step: int):
        import torch

        if not self.eval_examples:
            return

        model = self.trainer.accelerator.unwrap_model(self.trainer.model)
        was_training = model.training
        model.eval()
        try:
            correct = 0
            total = 0
            invalid = 0

            with torch.inference_mode():
                batch_size = 4
                for i in range(0, len(self.eval_examples), batch_size):
                    batch = self.eval_examples[i : i + batch_size]
                    prompts = [ex["prompt"] for ex in batch]
                    answers = [ex["answer"] for ex in batch]
                    if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
                        prompts = [
                            self.tokenizer.apply_chat_template(
                                [{"role": "user", "content": p}],
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                            for p in prompts
                        ]

                    original_padding_side = self.tokenizer.padding_side
                    self.tokenizer.padding_side = "left"
                    try:
                        inputs = self.tokenizer(
                            prompts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=self.trainer.args.max_length,
                        )
                    finally:
                        self.tokenizer.padding_side = original_padding_side

                    inputs = {k: v.to(self.trainer.accelerator.device) for k, v in inputs.items()}

                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                    )
                    gen_ids = output_ids[:, inputs["input_ids"].shape[1] :]
                    outputs = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

                    for pred, gold in zip(outputs, answers):
                        total += 1
                        letter = self._extract_letter(pred)
                        if letter is None:
                            invalid += 1
                            continue
                        if letter == gold:
                            correct += 1

            acc = correct / total if total else 0.0
            if self.trainer.accelerator.is_main_process:
                if wandb.run is not None:
                    wandb.log(
                        {"gpqa/acc": acc, "gpqa/total": total, "gpqa/invalid": invalid},
                        step=step,
                    )
                print(f"[GPQA] step={step} acc={acc:.4f} total={total} invalid={invalid}")
        finally:
            if was_training:
                model.train()
            self.trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

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
                "top_k_loss": script_args.top_k_loss if script_args.top_k_loss > 0 else None,
                "use_ema_teacher": script_args.use_ema_teacher,
                "ema_decay": script_args.ema_decay if script_args.use_ema_teacher else None,
                "gpqa_eval_interval": script_args.gpqa_eval_interval,
                "gpqa_eval_size": script_args.gpqa_eval_size,
                "gpqa_eval_seed": script_args.gpqa_eval_seed,
                "gpqa_eval_max_new_tokens": script_args.gpqa_eval_max_new_tokens,
                "gpqa_eval_repeats": script_args.gpqa_eval_repeats,
                "gpqa_variant": script_args.gpqa_variant,
                "gpqa_csv_path": script_args.gpqa_csv_path,
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
    # GPQA Eval Set
    ################
    gpqa_eval_examples = []
    is_main_process = os.environ.get("LOCAL_RANK", "0") == "0"
    if script_args.gpqa_eval_interval > 0 and is_main_process:
        gpqa_variant = script_args.gpqa_variant
        gpqa_csv_path = script_args.gpqa_csv_path or (
            f"https://openaipublic.blob.core.windows.net/simple-evals/gpqa_{gpqa_variant}.csv"
        )
        try:
            gpqa_rows = _load_gpqa_rows(gpqa_csv_path)
            gpqa_eval_examples = _build_gpqa_examples(
                gpqa_rows,
                script_args.gpqa_eval_size,
                script_args.gpqa_eval_repeats,
                script_args.gpqa_eval_seed,
            )
        except Exception as exc:
            raise ValueError(
                "Failed to load GPQA CSV. You can set --gpqa_csv_path to a local file "
                "or URL (see simple-evals gpqa_eval.py)."
                f" Details: {exc}"
            ) from exc

        source_desc = gpqa_csv_path if script_args.gpqa_csv_path else f"gpqa_{gpqa_variant}.csv"
        print(f"[GPQA] Loaded {len(gpqa_eval_examples)} examples from {source_desc}")

    if script_args.gpqa_eval_interval > 0 and torch.distributed.is_available() and torch.distributed.is_initialized():
        shared_eval_examples = [gpqa_eval_examples]
        torch.distributed.broadcast_object_list(shared_eval_examples, src=0)
        gpqa_eval_examples = shared_eval_examples[0]

    ################
    # Dataset
    ################
    # Load the Natural Reasoning dataset with rubric/reference answers
    ################
    # Training
    ################
    # Add presence_penalty to training_args so it can be accessed in the trainer
    training_args.presence_penalty = script_args.presence_penalty
    # SFTTrainer requires a dataset_text_field; use problem to avoid KeyError on tokenization.
    training_args.dataset_text_field = "problem"

    if script_args.teacher_prompt_tag not in (None, "rubric", "reference_answer", "cot"):
        raise ValueError("teacher_prompt_tag must be 'rubric', 'reference_answer', 'cot', or None.")

    dataset = load_dataset("facebook/natural_reasoning")

    def _normalize_example(example):
        problem = (
            example.get("question")
            or example.get("problem")
            or example.get("prompt")
            or example.get("instruction")
            or example.get("input")
            or example.get("query")
            or example.get("text")
        )
        context = example.get("context") or example.get("passage")
        if context and problem and context not in problem:
            problem = f"{context}\n\n{problem}"

        choices = example.get("choices") or example.get("options")
        if problem and choices:
            if isinstance(choices, dict):
                formatted = "\n".join(f"{k}. {v}" for k, v in choices.items())
                problem = f"{problem}\n\nChoices:\n{formatted}"
            elif isinstance(choices, list):
                if choices and isinstance(choices[0], dict):
                    formatted = "\n".join(
                        f"{c.get('label', idx)}. {c.get('text', c.get('content', ''))}"
                        for idx, c in enumerate(choices)
                    )
                else:
                    formatted = "\n".join(f"{idx}. {c}" for idx, c in enumerate(choices))
                problem = f"{problem}\n\nChoices:\n{formatted}"

        rubric = example.get("rubric", None)
        if rubric is None and "rubric_list" in example:
            rubric = example.get("rubric_list")
        if rubric is None:
            rubric = example.get("rationale") or example.get("explanation") or example.get("reasoning")

        reference_answer = (
            example.get("reference_answer")
            or example.get("solution")
            or example.get("answer")
            or example.get("final_answer")
            or example.get("output")
            or example.get("target")
            or example.get("label")
        )
        if reference_answer is None and "answers" in example:
            reference_answer = example.get("answers")
        cot_response = _extract_llm_response_text(example.get("responses"))
        if rubric is not None and not isinstance(rubric, str):
            rubric = json.dumps(rubric, ensure_ascii=False)
        reference_answer = _normalize_text(reference_answer)
        teacher_prompt_tag = example.get("teacher_prompt_tag") or example.get("teacher_tag") or example.get(
            "teacher_mode"
        )
        if script_args.teacher_prompt_tag is not None:
            teacher_prompt_tag = script_args.teacher_prompt_tag
        elif teacher_prompt_tag is None:
            if rubric is not None and reference_answer is not None:
                teacher_prompt_tag = "rubric"
            elif rubric is not None:
                teacher_prompt_tag = "rubric"
            elif reference_answer is not None:
                teacher_prompt_tag = "reference_answer"
            elif cot_response is not None:
                teacher_prompt_tag = "cot"
            else:
                teacher_prompt_tag = None
        if teacher_prompt_tag == "cot":
            reference_answer = cot_response
        return {
            "problem": problem,
            "rubric": rubric,
            "reference_answer": reference_answer,
            "teacher_prompt_tag": teacher_prompt_tag,
        }

    def _has_reference_answer(example):
        return example.get("problem") and (example.get("reference_answer") is not None)

    train_dataset = dataset["train"].map(_normalize_example).filter(_has_reference_answer)
    if len(train_dataset) > 10000:
        train_dataset = train_dataset.shuffle(seed=training_args.seed).select(range(10000))

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
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
    )

    if gpqa_eval_examples and script_args.gpqa_eval_interval > 0:
        trainer.add_callback(
            GPQAEvalCallback(
                trainer=trainer,
                tokenizer=tokenizer,
                eval_examples=gpqa_eval_examples,
                interval=script_args.gpqa_eval_interval,
                max_new_tokens=script_args.gpqa_eval_max_new_tokens,
            )
        )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=True,
            temperature=training_args.temperature,
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train()

    trainer.save_model(training_args.output_dir)
