"""Preprocess RubricHub_v1 RuRL/Science into an OPSD train split + a held-out test set.

RubricHub_v1 has two subsets:
  * sft_RuFT/  -- 6 sampled responses per query (what the earlier rubrichub run trained on;
                  only ~4,650 unique science questions).
  * RuRL/      -- one RL prompt per question, with a ground-truth rubric. The science file
                  RuRL/rurbichub_v1_Science.parquet has ~29,418 unique science questions and
                  was never touched by training.

This script:
  1. Downloads RuRL/rurbichub_v1_Science.parquet.
  2. Normalizes each row into the OPSD reward-training schema
     (question, reference_answer, rubric_list).
  3. Holds out a deterministic test set of --test-size questions, chosen to be disjoint
     from BOTH the earlier sft_RuFT training questions and the RaR-Science eval test set,
     so the held-out set is clean for every model.
  4. Saves the train split (datasets save_to_disk) and the test set as JSON (in the OPSD
     data dir and in the lm-open-science-evaluation datasets dir for the judge eval).

Run once before training:
    python preprocess_rurl_science.py
"""

import argparse
import json
import os
import random
import re

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset
from huggingface_hub import hf_hub_download

PLACEHOLDER_REFERENCE = (
    "No reference answer provided; use the rubric as the authoritative guidance."
)


# --- normalization helpers (kept consistent with opsd_train_reward.py) --------------- #
def _norm_text(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


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
                    parts.append(_norm_text(content))
            elif message is not None:
                parts.append(_norm_text(message))
        return "\n".join(p for p in parts if p and p.strip()) or None
    if isinstance(prompt, dict):
        content = prompt.get("content")
        if content is not None:
            return _norm_text(content)
    return _norm_text(prompt)


def _normalize_rubrics(rubrics):
    normalized = []
    if not isinstance(rubrics, list):
        return normalized
    for idx, rubric in enumerate(rubrics, start=1):
        if not isinstance(rubric, dict):
            continue
        description = rubric.get("description") or rubric.get("criterion") or rubric.get("criteria")
        weight = rubric.get("weight", rubric.get("points", rubric.get("score", 1)))
        title = rubric.get("title") or f"Criterion {idx}"
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            weight = 1.0
        normalized.append(
            {
                "title": _norm_text(title) or f"Criterion {idx}",
                "description": _norm_text(description) or "",
                "weight": weight,
            }
        )
    return normalized


def _canon(question):
    if not question:
        return None
    return re.sub(r"\s+", " ", question).strip().lower()


def _read_arrow_questions(path):
    """Read the 'question' column out of an HF-saved Arrow file via raw pyarrow IPC."""
    if not os.path.exists(path):
        return set()
    for opener in (pa.ipc.open_stream, pa.ipc.open_file):
        try:
            with pa.memory_map(path) as src:
                table = opener(src).read_all()
            return {_canon(q) for q in table.column("question").to_pylist() if q}
        except Exception:
            continue
    return set()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Preprocess RuRL/Science for OPSD training + eval.")
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=os.path.join(here, "data", "rurl_science"),
                        help="Where the train split (save_to_disk) and test.json are written.")
    parser.add_argument("--eval-test-path",
                        default="/gpfs/radev/pi/ying_rex/sg2768/lm-open-science-evaluation/datasets/rurl_science/test.json",
                        help="Copy of the test set for the judge eval framework.")
    parser.add_argument("--sft-cache-dir",
                        default=os.path.join(here, "rubric_cache", "rubrichub_science_30k"),
                        help="The earlier sft_RuFT training cache; its questions are excluded from the test set.")
    parser.add_argument("--rar-test-path",
                        default="/gpfs/radev/pi/ying_rex/sg2768/lm-open-science-evaluation/datasets/rar_science/test.json",
                        help="RaR-Science eval test set; overlapping questions are dropped to keep that eval clean.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild even if outputs exist.")
    args = parser.parse_args()

    train_dir = os.path.join(args.out_dir, "train")
    test_json = os.path.join(args.out_dir, "test.json")
    if os.path.exists(train_dir) and not args.overwrite:
        print(f"[skip] train split already exists at {train_dir} (use --overwrite to rebuild).")
        return

    print("Downloading RuRL/rurbichub_v1_Science.parquet ...")
    path = hf_hub_download("sojuL/RubricHub_v1", "RuRL/rurbichub_v1_Science.parquet", repo_type="dataset")
    rows_raw = pq.ParquetFile(path).read(
        columns=["prompt", "reward_model", "Rubrics", "data_source", "ability"]
    ).to_pylist()
    print(f"  loaded {len(rows_raw)} RuRL science rows")

    # Build normalized, deduped rows.
    seen = set()
    pool = []
    for ex in rows_raw:
        question = _norm_text(_extract_prompt_text(ex.get("prompt")))
        if not question:
            continue
        key = _canon(question)
        if key in seen:
            continue
        reward_model = ex.get("reward_model") if isinstance(ex.get("reward_model"), dict) else {}
        rubrics = ex.get("Rubrics") or reward_model.get("rubrics")
        rubric_list = _normalize_rubrics(rubrics)
        if not rubric_list:
            continue
        reference = reward_model.get("ground_truth")
        reference = _norm_text(reference) if reference else ""
        if not reference.strip():
            reference = PLACEHOLDER_REFERENCE
        seen.add(key)
        pool.append({
            "question": question,
            "reference_answer": reference,
            "rubric_list": rubric_list,
            "data_source": "Science",
            "source": "RuRL_science",
            "_key": key,
        })
    print(f"  {len(pool)} unique science questions with valid rubrics")

    # Exclusions for a clean held-out test set.
    sft_used = _read_arrow_questions(os.path.join(args.sft_cache_dir, "data-00000-of-00001.arrow"))
    print(f"  sft_RuFT previously-trained questions: {len(sft_used)}")
    rar_test = set()
    if os.path.exists(args.rar_test_path):
        rar_test = {_canon(r["question"]) for r in json.load(open(args.rar_test_path))}
    print(f"  RaR-Science eval test questions: {len(rar_test)}")

    # Drop RaR-Science-test overlaps from the whole pool (keeps that eval clean either way).
    pool = [r for r in pool if r["_key"] not in rar_test]
    # Test candidates must also be unseen by the earlier sft_RuFT training.
    test_candidates = [r for r in pool if r["_key"] not in sft_used]
    print(f"  pool after RaR-test exclusion: {len(pool)} | eligible test candidates: {len(test_candidates)}")
    if len(test_candidates) < args.test_size:
        raise SystemExit(f"Only {len(test_candidates)} eligible test candidates < requested {args.test_size}.")

    rng = random.Random(args.seed)
    test_keys = set(r["_key"] for r in rng.sample(test_candidates, args.test_size))
    test_rows = [r for r in pool if r["_key"] in test_keys]
    train_rows = [r for r in pool if r["_key"] not in test_keys]
    print(f"  SPLIT -> train={len(train_rows)}  test={len(test_rows)}")

    # --- save train split (OPSD training reads this via load_from_disk) --- #
    os.makedirs(args.out_dir, exist_ok=True)
    train_ds = Dataset.from_list([
        {k: r[k] for k in ("question", "reference_answer", "rubric_list", "data_source", "source")}
        for r in train_rows
    ])
    train_ds.save_to_disk(train_dir)
    print(f"  wrote train split -> {train_dir} ({len(train_ds)} rows)")

    # --- save test set as JSON (rubric judge eval format) --- #
    def _to_test_record(i, r):
        return {
            "id": f"rurl_science-{i}",
            "question": r["question"],
            "reference_answer": r["reference_answer"],
            "rubric": r["rubric_list"],
            "rubric_count": len(r["rubric_list"]),
            "data_source": "Science",
        }
    test_records = [_to_test_record(i, r) for i, r in enumerate(test_rows)]
    with open(test_json, "w") as f:
        json.dump(test_records, f, ensure_ascii=False)
    os.makedirs(os.path.dirname(args.eval_test_path), exist_ok=True)
    with open(args.eval_test_path, "w") as f:
        json.dump(test_records, f, ensure_ascii=False)
    print(f"  wrote test set -> {test_json}\n                 -> {args.eval_test_path} ({len(test_records)} rows)")
    print("Done.")


if __name__ == "__main__":
    main()
