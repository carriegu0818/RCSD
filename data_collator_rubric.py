import json
import torch


class SelfDistillationDataCollator:
    """
    Data collator for self-distillation that creates both student and teacher inputs.

    Student: receives the question + reference answer with rubric-design instructions (chat template).
    Teacher: sees the same prompt plus the ground-truth rubric (and optional reasoning-to-rubric transition).

    To enable batch-level operations (like original GKD), we pad prompts to the same length
    within each batch, and track the actual (unpadded) prompt lengths for loss masking.
    """

    def __init__(self, tokenizer, max_length=2048, reason_first=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reason_first = reason_first

        # Shared rubric design prompt pieces
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
            '[\n'
            '{"description": "Essential Criteria: The response must explicitly incorporate the 3% efficiency factor to determine the actual power output in visible photons from the 100-W bulb.", "title": "Efficiency Factor", "weight": 5},\n'
            '{"description": "Important Criteria: The response should correctly calculate the energy of a single photon using the formula E = hc/λ, with λ approximately equal to 5000 Å, ensuring proper usage of physical constants.", "title": "Photon Energy Calculation", "weight": 5}\n'
            ']\n'
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

        # Prompts for reason-first mode
        self.reason_first_prompt_reference = (
            "\n\nSummarize how the Reference Answer shapes the expected rubric before drafting the rubric itself. "
            "Do not start generating the rubric yet."
        )
        self.reason_first_prompt_rubric = (
            "\n\nReview the ground-truth rubric above. Briefly explain how it reflects the Question and Reference Answer. "
            "Do not rewrite or expand the rubric yet."
        )
        self.transition_prompt_reference = (
            "\n\nAfter understanding the reference rubric and the rationale behind each step, now articulate your own step-by-step reasoning that derives the same final answer to the problem below:\n"
        )
        self.transition_prompt_rubric = (
            "\n\nAfter understanding the reference rubric and the rationale behind each step, now articulate your own step-by-step reasoning that derives the same final answer to the problem below:\n"
        )

        # Set padding side explicitly for consistency
        print(f"[DataCollator] Original padding_side: {self.tokenizer.padding_side}")
        self.tokenizer.padding_side = "right"
        print(f"[DataCollator] Set padding_side to: {self.tokenizer.padding_side}")
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
            problem = self._get_first_present(feature, ["problem", "question", "prompt", "instruction", "input"])
            rubric = self._get_first_present(feature, ["rubric"])
            reference_answer = self._get_first_present(
                feature, ["reference_answer", "solution", "answer", "final_answer"]
            )
            teacher_prompt_tag = self._get_first_present(
                feature, ["teacher_prompt_tag", "teacher_tag", "teacher_mode"]
            )

            if problem is None:
                raise ValueError("Missing problem field for rubric prompt.")

            if reference_answer is None:
                raise ValueError("Missing reference_answer field for rubric prompt.")

            if rubric is None:
                raise ValueError("Missing rubric field for teacher prompt.")

            if teacher_prompt_tag is None:
                teacher_prompt_tag = "rubric"
            if teacher_prompt_tag != "rubric":
                raise ValueError(f"teacher_prompt_tag must be 'rubric' for rubric distillation, got {teacher_prompt_tag}.")

            rubric_text = self._normalize_text(rubric)
            reference_answer_text = self._normalize_text(reference_answer)
            problem_text = self._normalize_text(problem)

            teacher_prompt_tags.append("rubric")

            # Student prompt: rubric generation instructions with question + reference answer
            student_user_message = self._build_rubric_prompt(problem_text, reference_answer_text)
            student_messages = [{"role": "user", "content": student_user_message}]

            # Apply chat template for student (matching evaluation)
            student_prompt = self.tokenizer.apply_chat_template(
                student_messages, tokenize=False, add_generation_prompt=True
            )
            student_prompts.append(student_prompt)

            if self.reason_first:
                # Reasoning prompt: ask teacher to analyze rubric or reference answer
                reasoning_user_message = (
                    f"{self._build_rubric_prompt(problem_text, reference_answer_text)}\n\n"
                    f"Ground Truth Rubric (JSON array):\n{rubric_text}\n"
                    f"{self.reason_first_prompt_rubric}"
                )
                transition_text = f"\n{self.transition_prompt_rubric}"
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
                teacher_user_message = (
                    f"{self._build_rubric_prompt(problem_text, reference_answer_text)}\n\n"
                    f"Ground Truth Rubric (JSON array):\n{rubric_text}\n"
                    "Use the ground-truth rubric as the authoritative target and output the rubric JSON array only."
                )
                teacher_messages = [{"role": "user", "content": teacher_user_message}]

                # Apply chat template for teacher
                teacher_prompt = self.tokenizer.apply_chat_template(
                    teacher_messages, tokenize=False, add_generation_prompt=True
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

    def _build_rubric_prompt(self, question, reference_answer):
        return (
            f"{self.rubric_role_prompt}"
            f"# [Question]\n{question}\n"
            f"# [Reference Answer]\n{reference_answer}\n"
            f"{self.rubric_format_prompt}"
        )
