#!/usr/bin/env python3
"""
LLM-as-a-Judge rubric-quality evaluation for GT/reference rubrics plus multiple generated rubric caches.

Example:

python run_rubric_quality_eval_hf.py \
  --gt_dataset anisha2102/RaR-Science \
  --candidate learned:/gpfs/radev/pi/ying_rex/sg2768/OPSD/outputs/rubric_cache/qwen3_8b_rubricgen_fixteacher_lr5e6_gen4096/data-00000-of-00001.arrow:question:rubric \
  --candidate qwen3_8b_direct:/gpfs/radev/pi/ying_rex/sg2768/OPSD/outputs/rubric_cache/qwen3_8b_rubricgen/data-00000-of-00001.arrow:question:learned \
  --num_examples 500 \
  --shuffle \
  --model gpt-5.4-mini \
  --output_dir rubric_quality_three_outputs

Candidate format:
  name:path[:question_col[:rubric_col]]

Important:
  For qwen3_8b_direct cache, use rubric_col=learned.
"""

import argparse
import glob
import json
import os
import random
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import pyarrow as pa
import pyarrow.ipc as ipc


DIMENSIONS = [
    "relevance",
    "correctness",
    "completeness",
    "specificity",
    "non_redundancy",
    "usefulness",
]


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of grading rubrics for scientific and mathematical reasoning questions.

Your task is to evaluate the quality of several rubrics for the same question. The rubrics include one GT/reference rubric and one or more generated rubrics.

You must judge each rubric independently. Do not assume the reference rubric is always better. A generated rubric can be better if it is more complete, specific, or useful.

Inputs:
- question: The original question.
- reference_answer: The ideal answer or gold solution.
- rubrics: A dictionary of rubric_name -> rubric content.

Evaluate each rubric on the following six dimensions using a 1-5 scale:

1. Relevance: Does the rubric directly match the question and expected answer?
1 = mostly irrelevant; 3 = partially relevant; 5 = highly relevant.

2. Correctness: Are the rubric criteria factually and logically correct?
1 = contains serious errors; 3 = mostly correct with issues; 5 = fully correct.

3. Completeness: Does the rubric cover the main concepts, calculations, reasoning steps, and final-answer requirements needed to judge a response?
1 = misses most key requirements; 3 = covers some key requirements; 5 = covers nearly all key requirements.

4. Specificity: Is the rubric specific to this exact question rather than generic?
1 = generic and reusable for many unrelated questions; 3 = partially question-specific; 5 = highly instance-specific.

5. Non-redundancy: Does the rubric avoid repetitive, overlapping, or bloated criteria?
1 = highly redundant; 3 = some redundancy; 5 = concise and non-overlapping.

6. Usefulness: Would this rubric help a judge accurately evaluate model responses to this question?
1 = not useful; 3 = moderately useful; 5 = very useful.

Important instructions:
- Score every rubric independently.
- Do not omit any rubric that appears in the input.
- If a rubric is weak, empty, redundant, incomplete, or irrelevant, still score it from 1 to 5.
- Penalize rubrics that are verbose but redundant.
- Penalize rubrics that mention correct general concepts but miss the actual grading target.
- Reward rubrics that clearly identify the correct answer, necessary reasoning, key pitfalls, and important domain-specific checks.
- Use the full 1-5 range.
- Output valid JSON only. Do not include markdown or explanation outside JSON.

Return exactly this JSON format:
{
  "scores": {
    "RUBRIC_NAME_1": {
      "relevance": <1-5>,
      "correctness": <1-5>,
      "completeness": <1-5>,
      "specificity": <1-5>,
      "non_redundancy": <1-5>,
      "usefulness": <1-5>,
      "overall": <average of the six scores>
    },
    "RUBRIC_NAME_2": {
      "relevance": <1-5>,
      "correctness": <1-5>,
      "completeness": <1-5>,
      "specificity": <1-5>,
      "non_redundancy": <1-5>,
      "usefulness": <1-5>,
      "overall": <average of the six scores>
    }
  },
  "best_rubric": "<one rubric name or tie>",
  "brief_reason": "<one concise sentence>"
}
"""


def load_arrow_table_pyarrow(path: str):
    """Load a HuggingFace Arrow file without relying on datasets metadata parsing."""
    with pa.memory_map(path, "r") as source:
        try:
            reader = ipc.open_stream(source)
            return reader.read_all()
        except pa.ArrowInvalid:
            source.seek(0)
            reader = ipc.open_file(source)
            return reader.read_all()


def arrow_table_to_rows(path: str) -> List[Dict[str, Any]]:
    table = load_arrow_table_pyarrow(path)
    print(f"\nLoaded Arrow with pyarrow: {path}")
    print("Rows:", table.num_rows)
    print("Columns:", table.column_names)
    data = table.to_pydict()
    rows = []
    for i in range(table.num_rows):
        rows.append({col: data[col][i] for col in table.column_names})
    return rows


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dump_jsonl(rows: Iterable[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_question(q: str) -> str:
    q = str(q or "")
    q = re.sub(r"\s+", " ", q).strip().lower()
    return q


def rubric_to_text(rubric: Any) -> str:
    if rubric is None:
        return ""

    if isinstance(rubric, str):
        s = rubric.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                return rubric_to_text(json.loads(s))
            except Exception:
                return s
        return s

    if isinstance(rubric, list):
        parts = []
        for i, item in enumerate(rubric, 1):
            if isinstance(item, dict):
                title = item.get("title", item.get("criterion", item.get("name", "")))
                desc = item.get("description", item.get("desc", item.get("text", "")))
                weight = item.get("weight", item.get("score", ""))
                if title or desc or weight != "":
                    parts.append(f"{i}. {title} (weight={weight}): {desc}")
                else:
                    parts.append(f"{i}. {json.dumps(item, ensure_ascii=False)}")
            else:
                parts.append(f"{i}. {str(item)}")
        return "\n".join(parts)

    if isinstance(rubric, dict):
        for k in [
            "rubric_list",
            "generated_rubric",
            "rubric",
            "rubrics",
            "criteria",
            "model_output",
            "response",
            "learned",
            "GT",
        ]:
            if k in rubric:
                return rubric_to_text(rubric[k])
        return json.dumps(rubric, ensure_ascii=False, indent=2)

    return str(rubric)


def short_text(x: Any, max_chars: int = 12000) -> str:
    s = rubric_to_text(x)
    if len(s) > max_chars:
        return s[:max_chars] + "\n...[TRUNCATED]"
    return s


def pick_col(
    row: Dict[str, Any],
    candidates: List[str],
    required: bool = True,
    label: str = "field",
) -> Any:
    for c in candidates:
        if c and c in row and row[c] not in [None, ""]:
            return row[c]
    if required:
        raise KeyError(
            f"Could not find {label}; tried columns {candidates}; available={list(row.keys())}"
        )
    return ""


def import_datasets():
    try:
        from datasets import load_dataset  # type: ignore
        return load_dataset
    except Exception as e:
        raise ImportError(
            "This script needs HuggingFace datasets. Install with: pip install datasets pyarrow"
        ) from e


def dataset_to_rows(ds: Any, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = []
    n = len(ds) if limit is None else min(len(ds), limit)
    for i in range(n):
        r = dict(ds[i])
        r["__row_idx__"] = i
        rows.append(r)
    return rows


def load_gt_rows(args) -> List[Dict[str, Any]]:
    if args.gt_jsonl:
        rows = load_jsonl(args.gt_jsonl)
        for i, r in enumerate(rows):
            r["__row_idx__"] = i
        return rows

    if args.gt_arrow:
        rows = arrow_table_to_rows(args.gt_arrow)
        for i, r in enumerate(rows):
            r["__row_idx__"] = i
        return rows

    load_dataset = import_datasets()
    kwargs = {
        "path": args.gt_dataset,
        "split": args.gt_split,
    }
    if args.hf_cache_dir:
        kwargs["cache_dir"] = args.hf_cache_dir
    if args.force_redownload:
        kwargs["download_mode"] = "force_redownload"

    ds = load_dataset(**kwargs)
    return dataset_to_rows(ds)


def parse_candidate(spec: str) -> Tuple[str, str, str, str]:
    parts = spec.split(":")
    if len(parts) < 2:
        raise ValueError("--candidate must be name:path[:question_col[:rubric_col]]")

    name = parts[0]
    path = parts[1]
    q_col = parts[2] if len(parts) >= 3 and parts[2] else "question"
    r_col = parts[3] if len(parts) >= 4 and parts[3] else "rubric"

    if not name:
        raise ValueError("candidate name cannot be empty")

    return name, path, q_col, r_col


def resolve_arrow_path(path: str) -> str:
    if os.path.isdir(path):
        final_arrow = os.path.join(path, "data-00000-of-00001.arrow")
        if os.path.exists(final_arrow):
            return final_arrow

        arrows = sorted(
            p for p in glob.glob(os.path.join(path, "*.arrow"))
            if not os.path.basename(p).startswith("cache-")
        )
        if not arrows:
            arrows = sorted(glob.glob(os.path.join(path, "*.arrow")))
        if not arrows:
            raise FileNotFoundError(f"No arrow files found in {path}")
        return arrows[0]

    return path


def load_candidate_rows(spec: str) -> Tuple[str, List[Dict[str, Any]]]:
    name, path, q_col, r_col = parse_candidate(spec)
    path = resolve_arrow_path(path)

    raw_rows = load_jsonl(path) if path.endswith(".jsonl") else arrow_table_to_rows(path)
    if not raw_rows:
        raise ValueError(f"No rows loaded for candidate {name}")

    available = list(raw_rows[0].keys())
    print(f"Candidate {name} columns:", available)

    if q_col not in available:
        raise ValueError(
            f"Candidate {name}: question column '{q_col}' not found. Available columns: {available}"
        )
    if r_col not in available:
        raise ValueError(
            f"Candidate {name}: rubric column '{r_col}' not found. Available columns: {available}"
        )

    rows = []
    skipped_empty = 0

    for i, row in enumerate(raw_rows):
        question = str(row[q_col])
        rubric = row[r_col]

        rubric_text = rubric_to_text(rubric).strip()
        if not question.strip() or not rubric_text:
            skipped_empty += 1
            continue

        rows.append(
            {
                "question": question,
                "rubric": rubric,
                "__candidate__": name,
                "__row_idx__": i,
                "__path__": path,
            }
        )

    print(f"Loaded candidate {name}: {len(rows)} rows; skipped_empty={skipped_empty}")
    return name, rows


def build_eval_inputs(
    gt_rows: List[Dict[str, Any]],
    candidates: Dict[str, List[Dict[str, Any]]],
    args,
) -> List[Dict[str, Any]]:
    gt_by_q = {}

    for i, r in enumerate(gt_rows):
        try:
            q = pick_col(
                r,
                [args.gt_question_col, "question", "prompt", "input"],
                label="GT question",
            )
            ref_rubric = pick_col(
                r,
                [args.gt_rubric_col, "rubric_list", "rubric", "reference_rubric", "rubrics"],
                label="GT rubric",
            )
        except Exception:
            continue

        if not rubric_to_text(ref_rubric).strip():
            continue

        rr = dict(r)
        rr["__row_idx__"] = i
        gt_by_q[normalize_question(q)] = rr

    cand_maps = {}
    for name, rows in candidates.items():
        m = {}
        for row in rows:
            key = normalize_question(row["question"])
            if key:
                m[key] = row
        cand_maps[name] = m

    overlap_keys = set(gt_by_q.keys())
    for m in cand_maps.values():
        overlap_keys &= set(m.keys())

    overlap_keys = list(overlap_keys)
    print(f"Overlaps across GT and all candidates before sampling: {len(overlap_keys)}")

    if args.shuffle:
        random.Random(args.seed).shuffle(overlap_keys)
    else:
        overlap_keys = sorted(overlap_keys, key=lambda k: gt_by_q[k].get("__row_idx__", 0))

    if args.num_examples:
        overlap_keys = overlap_keys[: args.num_examples]

    if not overlap_keys:
        raise ValueError("No overlapping examples across GT and all candidates. Check splits/columns.")

    inputs = []
    for j, key in enumerate(overlap_keys):
        gt = gt_by_q[key]
        q = pick_col(gt, [args.gt_question_col, "question", "prompt", "input"], label="question")
        ref_ans = pick_col(
            gt,
            [args.gt_reference_col, "reference_answer", "answer", "reference", "solution"],
            required=False,
            label="reference answer",
        )
        ref_rubric = pick_col(
            gt,
            [args.gt_rubric_col, "rubric_list", "rubric", "reference_rubric", "rubrics"],
            label="GT rubric",
        )

        rubrics = {args.reference_name: ref_rubric}
        candidate_row_indices = {}

        for name, m in cand_maps.items():
            row = m[key]
            rubrics[name] = row["rubric"]
            candidate_row_indices[name] = row.get("__row_idx__")

        inputs.append(
            {
                "id": gt.get("id", gt.get("__orig_idx__", gt.get("__row_idx__", j))),
                "question": q,
                "reference_answer": ref_ans,
                "rubrics": rubrics,
                "gt_row_idx": gt.get("__row_idx__"),
                "candidate_row_indices": candidate_row_indices,
            }
        )

    return inputs


def build_user_prompt(ex: Dict[str, Any]) -> str:
    rubric_names = list(ex["rubrics"].keys())

    chunks = []
    for name, rubric in ex["rubrics"].items():
        chunks.append(f"[{name}]\n{short_text(rubric)}")
    rubric_block = "\n\n".join(chunks)

    required_names = "\n".join(f"- {name}" for name in rubric_names)

    return f"""Evaluate the following rubrics for the same question.

You must return scores for exactly these rubric names:
{required_names}

Do not omit any rubric. If a rubric is weak, empty, redundant, incomplete, or irrelevant, still score it from 1 to 5.

[Question]
{ex.get("question", "")}

[Reference Answer]
{ex.get("reference_answer", "")}

[Rubrics]
{rubric_block}
"""


def parse_json_response(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]

    return json.loads(text)


def call_judge(
    client: OpenAI,
    model: str,
    user_prompt: str,
    temperature: float,
    max_retries: int,
) -> Dict[str, Any]:
    last_err = None

    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            if not os.environ.get("DISABLE_JSON_RESPONSE_FORMAT"):
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)
            return parse_json_response(response.choices[0].message.content)

        except Exception as e:
            last_err = e
            time.sleep(min(30, 2 ** attempt))

    raise RuntimeError(f"Judge call failed after {max_retries} retries: {last_err}")


def validate_scores(result: Dict[str, Any], rubric_names: List[str]) -> Dict[str, Any]:
    if "scores" not in result:
        raise ValueError(f"Missing key: scores; got keys={list(result.keys())}")

    if not isinstance(result["scores"], dict):
        raise ValueError("result['scores'] is not a dict")

    raw_scores = result["scores"]
    normalized_key_map = {
        str(k).strip().lower().replace("-", "_").replace(" ", "_"): k
        for k in raw_scores.keys()
    }

    fixed_scores = {}

    for name in rubric_names:
        norm_name = name.strip().lower().replace("-", "_").replace(" ", "_")

        possible_aliases = [norm_name]
        if norm_name in ["gt", "reference", "gt_reference"]:
            possible_aliases += ["reference", "gt", "gt_reference", "reference_rubric"]
        if norm_name == "qwen3_8b_direct":
            possible_aliases += ["qwen3_8b", "qwen_direct", "direct", "qwen3_direct"]
        if norm_name == "learned":
            possible_aliases += ["generated", "ours", "learned_rubric"]

        found_key = None
        for alias in possible_aliases:
            if alias in normalized_key_map:
                found_key = normalized_key_map[alias]
                break

        if found_key is None:
            raise ValueError(
                f"Missing rubric scores for {name}; got keys {list(raw_scores.keys())}"
            )

        scores = raw_scores[found_key]
        if not isinstance(scores, dict):
            raise ValueError(f"Scores for {name} are not a dict: {scores}")

        vals = []
        for dim in DIMENSIONS:
            if dim not in scores:
                raise ValueError(f"Missing dimension {dim} for {name}; got {scores}")

            val = float(scores[dim])
            val = max(1.0, min(5.0, val))
            scores[dim] = val
            vals.append(val)

        scores["overall"] = sum(vals) / len(vals)
        fixed_scores[name] = scores

    result["scores"] = fixed_scores

    if "best_rubric" not in result:
        best = max(rubric_names, key=lambda n: result["scores"][n]["overall"])
        result["best_rubric"] = best

    if "brief_reason" not in result:
        result["brief_reason"] = ""

    return result


def summarize(
    results: List[Dict[str, Any]],
    rubric_names: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []

    for name in rubric_names:
        row = {"rubric": name}
        for dim in DIMENSIONS + ["overall"]:
            row[dim] = sum(r["judge_result"]["scores"][name][dim] for r in results) / len(results)
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    win_rows = []
    for dim in DIMENSIONS + ["overall"]:
        counts = {name: 0 for name in rubric_names}
        ties = 0

        for r in results:
            vals = {
                name: r["judge_result"]["scores"][name][dim]
                for name in rubric_names
            }
            maxv = max(vals.values())
            winners = [n for n, v in vals.items() if abs(v - maxv) < 1e-9]

            if len(winners) == 1:
                counts[winners[0]] += 1
            else:
                ties += 1

        row = {
            "metric": dim,
            "ties": ties,
            "tie_rate": ties / len(results),
        }
        for name in rubric_names:
            row[f"{name}_wins"] = counts[name]
            row[f"{name}_win_rate"] = counts[name] / len(results)
        win_rows.append(row)

    win_df = pd.DataFrame(win_rows)

    flat_rows = []
    for r in results:
        row = {
            "id": r.get("id"),
            "gt_row_idx": r.get("gt_row_idx"),
            "best_rubric": r["judge_result"].get("best_rubric", ""),
            "brief_reason": r["judge_result"].get("brief_reason", ""),
        }

        candidate_row_indices = r.get("candidate_row_indices") or {}
        for cand_name, cand_idx in candidate_row_indices.items():
            row[f"{cand_name}_row_idx"] = cand_idx

        for name in rubric_names:
            scores = r["judge_result"]["scores"][name]
            for dim in DIMENSIONS + ["overall"]:
                row[f"{name}_{dim}"] = scores[dim]

        flat_rows.append(row)

    flat_df = pd.DataFrame(flat_rows)
    return summary_df, win_df, flat_df


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--gt_dataset", default="anisha2102/RaR-Science")
    parser.add_argument("--gt_split", default="train")
    parser.add_argument("--gt_arrow", default=None)
    parser.add_argument("--gt_jsonl", default=None)
    parser.add_argument("--gt_question_col", default="question")
    parser.add_argument("--gt_reference_col", default="reference_answer")
    parser.add_argument("--gt_rubric_col", default="rubric_list")
    parser.add_argument("--reference_name", default="GT")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--force_redownload", action="store_true")

    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate generated rubric spec: name:path[:question_col[:rubric_col]]. Can repeat.",
    )

    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--output_dir", default="rubric_quality_three_outputs")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading GT/reference rows...")
    gt_rows = load_gt_rows(args)
    print(f"GT rows: {len(gt_rows)}")

    print("Loading candidate generated-rubric rows...")
    candidates = {}
    for spec in args.candidate:
        name, rows = load_candidate_rows(spec)
        if name in candidates or name == args.reference_name:
            raise ValueError(f"Duplicate or reserved candidate name: {name}")
        candidates[name] = rows

    eval_inputs = build_eval_inputs(gt_rows, candidates, args)
    rubric_names = [args.reference_name] + list(candidates.keys())

    print(f"Aligned examples selected: {len(eval_inputs)}")
    print("Rubrics compared:", rubric_names)

    input_path = os.path.join(args.output_dir, "rubric_quality_inputs.jsonl")
    dump_jsonl(eval_inputs, input_path)

    with open(os.path.join(args.output_dir, "judge_rubric_quality_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(JUDGE_SYSTEM_PROMPT)

    print(f"Wrote inputs to {input_path}")

    if args.dry_run:
        print("Dry run complete. No LLM calls made.")
        return

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )

    valid_results = []
    invalid_results = []

    for ex in tqdm(eval_inputs, desc="Judging rubrics"):
        user_prompt = build_user_prompt(ex)

        ok = False
        last_error = None
        judge_result = None

        for attempt in range(args.max_retries):
            try:
                judge_result = call_judge(
                    client=client,
                    model=args.model,
                    user_prompt=user_prompt,
                    temperature=args.temperature,
                    max_retries=1,
                )
                judge_result = validate_scores(judge_result, rubric_names)
                ok = True
                break

            except Exception as e:
                last_error = str(e)
                print(
                    f"[Invalid judge output; retry {attempt + 1}/{args.max_retries}] "
                    f"id={ex.get('id')} error={last_error}"
                )

        if ok and judge_result is not None:
            valid_results.append(
                {
                    "id": ex.get("id"),
                    "gt_row_idx": ex.get("gt_row_idx"),
                    "candidate_row_indices": ex.get("candidate_row_indices"),
                    "judge_result": judge_result,
                }
            )
        else:
            invalid_results.append(
                {
                    "id": ex.get("id"),
                    "gt_row_idx": ex.get("gt_row_idx"),
                    "candidate_row_indices": ex.get("candidate_row_indices"),
                    "error": last_error,
                    "rubric_names": rubric_names,
                }
            )

    print(f"Valid judge outputs: {len(valid_results)}")
    print(f"Filtered invalid outputs: {len(invalid_results)}")

    dump_jsonl(valid_results, os.path.join(args.output_dir, "rubric_quality_judgments.jsonl"))
    dump_jsonl(invalid_results, os.path.join(args.output_dir, "rubric_quality_invalid_judgments.jsonl"))

    if not valid_results:
        raise ValueError(
            "No valid judge outputs after filtering. Try a stronger judge model, shorter prompts, "
            "or evaluate pairwise instead of all rubrics together."
        )

    summary_df, win_df, flat_df = summarize(valid_results, rubric_names)

    summary_df.to_csv(os.path.join(args.output_dir, "rubric_quality_summary.csv"), index=False)
    win_df.to_csv(os.path.join(args.output_dir, "rubric_quality_win_rate.csv"), index=False)
    flat_df.to_csv(os.path.join(args.output_dir, "rubric_quality_per_example.csv"), index=False)

    print("\nSummary:")
    print(summary_df.round(3).to_string(index=False))

    print("\nWin rates:")
    print(win_df.round(3).to_string(index=False))

    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()