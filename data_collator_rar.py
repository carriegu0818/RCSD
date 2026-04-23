import json
from contextlib import contextmanager

import torch


class SelfDistillationDataCollator:
    """
    Data collator for self-distillation that creates both student and teacher inputs.

    Student: sees only the problem (with chat template)
    Teacher: sees problem + rubric, reference answer, or CoT response + transition prompt (with chat template)

    To enable batch-level operations (like original GKD), we pad prompts to the same length
    within each batch, and track the actual (unpadded) prompt lengths for loss masking.
    """

    def __init__(self, tokenizer, max_length=2048, reason_first=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reason_first = reason_first

        # Prompt for reasoning about the reference answer before teaching
        self.reason_first_prompt_reference = (
            "\n\nThe reference answer above is correct. "
            "Please explain why it is correct and outline the key reasoning steps implied by it. "
            "Do NOT use <think> tags. Do NOT derive your own solution from scratch. "
            "Simply analyze and explain the reference answer provided above.\n"
        )
        self.reason_first_prompt_cot = (
            "\n\nThe step-by-step response above is intended to solve the problem correctly. "
            "Please analyze its key reasoning steps and why the solution works. "
            "Do NOT use <think> tags. Do NOT derive your own solution from scratch. "
            "Simply analyze the response provided above.\n"
        )
        # Prompt for reasoning about the rubric before teaching
        self.reason_first_prompt_rubric = (
            "\n\nPlease summarize the key criteria and constraints in the rubric above. "
            "Do NOT use <think> tags. Do NOT solve the problem. "
            "Simply analyze the rubric provided above.\n"
        )
        # Prompt for transitioning to teaching mode after reasoning
        self.transition_prompt_reference = (
            "\n\nAfter understanding the reference answer and the rationale behind it, now articulate your own "
            "step-by-step reasoning that derives the same final answer to the problem below:\n"
        )
        self.transition_prompt_cot = (
            "\n\nAfter understanding the solution above and the reasoning behind it, now articulate your own "
            "step-by-step reasoning that solves the problem below:\n"
        )
        self.transition_prompt_rubric = (
            "\n\nAfter understanding the rubric criteria, now solve the problem step by step, ensuring your "
            "reasoning satisfies the rubric and boxed answer match the rubric's criteria on final answer:\n"
        )

        # Keep the shared tokenizer left-padded for generation/eval code paths.
        # The collator temporarily switches to right padding only while batching prompts.
        print(f"[DataCollator] Original padding_side: {self.tokenizer.padding_side}")
        print("[DataCollator] Using temporary right padding inside the collator")
        print(f"[DataCollator] Reason first mode: {self.reason_first}")

    def __call__(self, features):

        batch_size = len(features)

        # Prepare student and teacher prompts using chat template (matching evaluation)
        student_prompts = []
        teacher_prompts = []
        teacher_reasoning_prompts = []  # NEW: for reason_first mode
        teacher_transition_texts = []
        teacher_prompt_tags = []

        for feature in features:
            # Extract problem and teacher inputs from dataset
            problem = self._get_first_present(
                feature, ["problem", "question", "prompt", "instruction", "input"]
            )
            rubric = self._get_first_present(feature, ["rubric", "rubric_list"])
            reference_answer = self._get_first_present(
                feature, ["reference_answer", "solution", "answer", "final_answer"]
            )
            teacher_prompt_tag = self._get_first_present(
                feature, ["teacher_prompt_tag", "teacher_tag", "teacher_mode"]
            )

            if teacher_prompt_tag is None:
                if rubric is not None and reference_answer is not None:
                    raise ValueError(
                        "Both rubric and reference_answer are present, but no teacher_prompt_tag was provided."
                    )
                if rubric is not None:
                    teacher_prompt_tag = "rubric"
                elif reference_answer is not None:
                    teacher_prompt_tag = "reference_answer"
                else:
                    raise ValueError("Missing both rubric and reference_answer for teacher prompt.")

            if problem is None:
                raise ValueError("Missing problem field for teacher prompt.")

            if teacher_prompt_tag == "rubric":
                if rubric is None:
                    raise ValueError("teacher_prompt_tag='rubric' but rubric is missing.")
                teacher_extra_text = self._normalize_text(rubric)
                teacher_prompt_tags.append("rubric")
            elif teacher_prompt_tag == "reference_answer":
                if reference_answer is None:
                    raise ValueError("teacher_prompt_tag='reference_answer' but reference_answer is missing.")
                teacher_extra_text = self._normalize_text(reference_answer)
                teacher_prompt_tags.append("reference_answer")
            elif teacher_prompt_tag == "cot":
                if reference_answer is None:
                    raise ValueError("teacher_prompt_tag='cot' but response text is missing.")
                teacher_extra_text = self._normalize_text(reference_answer)
                teacher_prompt_tags.append("cot")
            else:
                raise ValueError(f"Unknown teacher_prompt_tag: {teacher_prompt_tag}")
            # Student prompt: just the problem with instruction (matching evaluation format)
            student_user_message = f"Problem: {problem}\n\nPlease reason step by step, and put your final answer (short answer) within \\boxed{{}}. . Only one \\boxed{{}} per response."
            student_messages = [{"role": "user", "content": student_user_message}]

            # Apply chat template for student (matching evaluation)
            student_prompt = self.tokenizer.apply_chat_template(
                student_messages, tokenize=False, add_generation_prompt=True
            )
            student_prompts.append(student_prompt)

            if self.reason_first:
                # Reasoning prompt: ask teacher to analyze rubric or reference answer
                if teacher_prompt_tags[-1] == "rubric":
                    reasoning_user_message = (
                        f"Problem: {problem}\n\n"
                        f"Here is the grading rubric (JSON list of criteria):\n"
                        f"{teacher_extra_text}\n\n"
                        f"{self.reason_first_prompt_rubric}"
                    )
                    transition_text = (
                        f"\n{self.transition_prompt_rubric}\n"
                        f"Please reason step by step, and put your final answer (short answer) within \\boxed{{}}. Only one \\boxed{{}} per response."
                    )
                elif teacher_prompt_tags[-1] == "cot":
                    reasoning_user_message = (
                        f"Problem: {problem}\n\n"
                        f"Here is a step-by-step response to this problem:\n"
                        f"{teacher_extra_text}\n\n"
                        f"{self.reason_first_prompt_cot}"
                    )
                    transition_text = (
                        f"\n{self.transition_prompt_cot}\n"
                        f"Please reason step by step, and put your final answer (short answer) within \\boxed{{}}. Only one \\boxed{{}} per response."
                    )
                else:
                    reasoning_user_message = (
                        f"Problem: {problem}\n\n"
                        f"Here is a reference answer to this problem (short, no steps):\n"
                        f"{teacher_extra_text}\n\n"
                        f"{self.reason_first_prompt_reference}"
                    )
                    transition_text = (
                        f"\n{self.transition_prompt_reference}\n"
                        f"Please reason step by step, and put your final answer (short answer) within \\boxed{{}}. Only one \\boxed{{}} per response."
                    )
                reasoning_messages = [{"role": "user", "content": reasoning_user_message}]
                reasoning_prompt = self.tokenizer.apply_chat_template(
                    reasoning_messages, tokenize=False, add_generation_prompt=True
                )
                teacher_reasoning_prompts.append(reasoning_prompt)
                teacher_transition_texts.append(transition_text)

                # Teacher prompt will be constructed during training after reasoning
                # For now, create placeholder (will be replaced in training_step)
                teacher_prompts.append("")  # Placeholder
            else:
                # Teacher prompt: use rubric or reference answer
                if teacher_prompt_tags[-1] == "rubric":
                    teacher_user_message = (
                        f"Problem: {problem}\n\n"
                        f"Here is the grading rubric (JSON list of criteria):\n{teacher_extra_text}\n\n"
                        f"Use this rubric to produce a correct answer.\n"
                        f"Please reason step by step, and put your final answer (short answer) within \\boxed{{}}. Only one \\boxed{{}} per response."
                    )
                elif teacher_prompt_tags[-1] == "cot":
                    teacher_user_message = (
                        f"Problem: {problem}\n\n"
                        f"Here is a step-by-step response to this problem:\n{teacher_extra_text}\n\n"
                        f"Use this response as a guide.\n"
                        f"Please reason step by step, and put your final answer (short answer) within \\boxed{{}}. Only one \\boxed{{}} per response."
                    )
                else:
                    teacher_user_message = (
                        f"Problem: {problem}\n\n"
                        f"Here is a reference answer to this problem (short, no steps):\n{teacher_extra_text}\n\n"
                        f"Use this reference answer as a guide.\n"
                        f"Please reason step by step, and put your final answer (short answer) within \\boxed{{}}. Only one \\boxed{{}} per response."
                    )
                teacher_messages = [{"role": "user", "content": teacher_user_message}]

                # Apply chat template for teacher
                teacher_prompt = self.tokenizer.apply_chat_template(
                    teacher_messages, tokenize=False, add_generation_prompt=True
                )
                teacher_prompts.append(teacher_prompt)

        with self._temporary_padding_side("right"):
            # Tokenize WITHOUT padding first to get true lengths
            student_encoded_no_pad = self.tokenizer(
                student_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            student_prompt_lengths = [len(ids) for ids in student_encoded_no_pad["input_ids"]]

            # Find max lengths in this batch
            max_student_prompt_len = max(student_prompt_lengths)

            # Tokenize WITH padding to max length in batch
            student_encoded = self.tokenizer(
                student_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_student_prompt_len,
                return_tensors="pt",
            )

        result = {
            "student_prompts": student_encoded["input_ids"],
            "student_prompt_attention_mask": student_encoded["attention_mask"],
            "student_prompt_length": max_student_prompt_len,  # Single value for batch!
            # Keep individual lengths for proper masking
            "student_prompt_lengths_per_example": torch.tensor(student_prompt_lengths),
        }

        if self.reason_first:
            with self._temporary_padding_side("right"):
                # Tokenize reasoning prompts
                reasoning_encoded_no_pad = self.tokenizer(
                    teacher_reasoning_prompts,
                    padding=False,
                    truncation=True,
                    max_length=self.max_length,
                )
                reasoning_prompt_lengths = [len(ids) for ids in reasoning_encoded_no_pad["input_ids"]]
                max_reasoning_prompt_len = max(reasoning_prompt_lengths)

                reasoning_encoded = self.tokenizer(
                    teacher_reasoning_prompts,
                    padding="max_length",
                    truncation=True,
                    max_length=max_reasoning_prompt_len,
                    return_tensors="pt",
                )

                # Tokenize transition prompts (appended after reasoning)
                transition_encoded = self.tokenizer(
                    teacher_transition_texts,
                    padding="longest",
                    truncation=False,
                    return_tensors="pt",
                )

            result.update(
                {
                    "teacher_reasoning_prompts": reasoning_encoded["input_ids"],
                    "teacher_reasoning_attention_mask": reasoning_encoded["attention_mask"],
                    "teacher_reasoning_prompt_length": max_reasoning_prompt_len,
                    "teacher_transition_tokens": transition_encoded["input_ids"],
                    "teacher_prompt_tags": teacher_prompt_tags,
                }
            )
        else:
            with self._temporary_padding_side("right"):
                # Normal mode: tokenize teacher prompts
                teacher_encoded_no_pad = self.tokenizer(
                    teacher_prompts,
                    padding=False,
                    truncation=True,
                    max_length=self.max_length,
                )
                teacher_prompt_lengths = [len(ids) for ids in teacher_encoded_no_pad["input_ids"]]
                max_teacher_prompt_len = max(teacher_prompt_lengths)

                teacher_encoded = self.tokenizer(
                    teacher_prompts,
                    padding="max_length",
                    truncation=True,
                    max_length=max_teacher_prompt_len,
                    return_tensors="pt",
                )

            result.update(
                {
                    "teacher_prompts": teacher_encoded["input_ids"],
                    "teacher_prompt_attention_mask": teacher_encoded["attention_mask"],
                    "teacher_prompt_length": max_teacher_prompt_len,
                    "teacher_prompt_lengths_per_example": torch.tensor(teacher_prompt_lengths),
                    "teacher_prompt_tags": teacher_prompt_tags,
                }
            )

        return result

    @staticmethod
    def _get_first_present(feature, keys):
        for key in keys:
            if key in feature and feature[key] is not None:
                return feature[key]
        return None

    @staticmethod
    def _normalize_text(value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    @contextmanager
    def _temporary_padding_side(self, padding_side):
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side
        try:
            yield
        finally:
            self.tokenizer.padding_side = original_padding_side
