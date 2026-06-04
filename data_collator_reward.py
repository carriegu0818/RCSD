import json
import torch


class SelfDistillationDataCollator:
    """
    Data collator for self-distillation that creates both student and teacher inputs.

    Student: sees only the question (with chat template)
    Teacher: sees question + reference answer + rubric (with chat template)

    To enable batch-level operations (like original GKD), we pad prompts to the same length
    within each batch, and track the actual (unpadded) prompt lengths for loss masking.
    """

    def __init__(
        self,
        tokenizer,
        max_length=2048,
        reason_first=True,
        student_thinking=True,
        teacher_thinking=True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reason_first = reason_first
        self.student_thinking = student_thinking
        self.teacher_thinking = teacher_thinking

        # Prompt for reasoning about the rubric/reference answer before teaching
        self.reason_first_prompt = (
            "\n\nThe reference answer and rubric above are guidelines for the questions but may not be perfect. "
            "Briefly explain the key requirements, constraints, and reasoning expectations implied by them. "
            "Do NOT solve the question yet. Do NOT use <think> tags.\n"
        )
        # Prompt for transitioning to teaching mode after reasoning
        self.transition_prompt = (
           "\n\nThe reference and rubric above is provided only as private guidance to help you understand "
            "the correct reasoning path. Do NOT mention, cite, or refer to the reference solution, rubric, "
            "answer key, ground truth, or any external guidance in your response. Do NOT write phrases such as "
            "'given the solution', 'from the reference', 'according to the rubric', 'we know the answer is', "
            "or anything implying that you have seen the answer in advance. "
            "Instead, produce a natural, self-contained solution as if you are solving the problem independently "
            "from the original problem statement. Use your own reasoning process, derive the result step by step, "
            "and only present the final answer after the reasoning supports it. "
            "Think step by step, explore different approaches if useful, and don't be afraid to backtrack "
            "or reconsider if something doesn't work out. Put the final answer within \\boxed{}.\n"
            # "\n\nAfter understanding the rubric and reference answer, now solve the question step by step, "
            # "ensuring your reasoning satisfies the rubric. Put the final answer within \\boxed{}.\n"
        )

        # Set padding side explicitly for consistency
        print(f"[DataCollator] Original padding_side: {self.tokenizer.padding_side}")
        self.tokenizer.padding_side = "right"
        print(f"[DataCollator] Set padding_side to: {self.tokenizer.padding_side}")
        print(f"[DataCollator] Reason first mode: {self.reason_first}")
        print(f"[DataCollator] Student thinking mode: {self.student_thinking}")
        print(f"[DataCollator] Teacher thinking mode: {self.teacher_thinking}")

    def __call__(self, features):

        batch_size = len(features)

        # Prepare student and teacher prompts using chat template (matching evaluation)
        student_prompts = []
        teacher_prompts = []
        teacher_reasoning_prompts = []  # NEW: for reason_first mode

        for feature in features:
            # Extract question, reference answer, and rubric from dataset
            question = self._get_first_present(
                feature, ["question", "problem", "prompt", "instruction", "input"]
            )
            reference_answer = self._get_first_present(
                feature, ["reference_answer", "reference", "answer", "final_answer", "solution"]
            )
            rubric = self._get_first_present(feature, ["rubric", "rubric_list"])

            if question is None:
                raise ValueError("Missing question field for reward prompt.")
            if reference_answer is None:
                raise ValueError("Missing reference_answer field for reward prompt.")
            if rubric is None:
                raise ValueError("Missing rubric field for reward prompt.")

            question_text = self._normalize_text(question)
            reference_answer_text = self._normalize_text(reference_answer)
            rubric_text = self._normalize_text(rubric)

            # Student prompt: just the question with instruction (matching evaluation format)
            student_user_message = (
                f"Question: {question_text}\n\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            )
            student_messages = [{"role": "user", "content": student_user_message}]

            # Apply chat template for student (matching evaluation)
            student_prompt = self.tokenizer.apply_chat_template(
                student_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.student_thinking,
            )
            student_prompts.append(student_prompt)

            if self.reason_first:
                # Reasoning prompt: ask teacher to analyze rubric + reference answer
                reasoning_user_message = (
                    f"Question: {question_text}\n\n"
                    f"Reference Answer:\n{reference_answer_text}\n\n"
                    f"Rubric (JSON array):\n{rubric_text}\n"
                    f"{self.reason_first_prompt}"
                )
                reasoning_messages = [{"role": "user", "content": reasoning_user_message}]
                reasoning_prompt = self.tokenizer.apply_chat_template(
                    reasoning_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.teacher_thinking,
                )
                teacher_reasoning_prompts.append(reasoning_prompt)

                # Teacher prompt will be constructed during training after reasoning
                # For now, create placeholder (will be replaced in training_step)
                teacher_prompts.append("")  # Placeholder
            else:
                # Teacher prompt: privileged context (question + reference answer + rubric)
                teacher_user_message = (
                    f"Question: {question_text}\n\n"
                    f"Reference Answer:\n{reference_answer_text}\n\n"
                    f"Rubric (JSON array):\n{rubric_text}\n\n"
                    "Use the rubric and reference answer as guidance.\n"
                    "Please reason step by step, and put your final answer within \\boxed{}."
                )
                teacher_messages = [{"role": "user", "content": teacher_user_message}]

                # Apply chat template for teacher
                teacher_prompt = self.tokenizer.apply_chat_template(
                    teacher_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.teacher_thinking,
                )
                teacher_prompts.append(teacher_prompt)

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

            # Tokenize transition prompt (appended after reasoning)
            # Don't use chat template here - just the raw text
            transition_text = f"\n{self.transition_prompt}"
            transition_encoded = self.tokenizer(
                [transition_text] * batch_size,
                padding=False,
                truncation=False,
                return_tensors="pt",
            )

            result.update(
                {
                    "teacher_reasoning_prompts": reasoning_encoded["input_ids"],
                    "teacher_reasoning_attention_mask": reasoning_encoded["attention_mask"],
                    "teacher_reasoning_prompt_length": max_reasoning_prompt_len,
                    "teacher_transition_tokens": transition_encoded["input_ids"],
                }
            )
        else:
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
