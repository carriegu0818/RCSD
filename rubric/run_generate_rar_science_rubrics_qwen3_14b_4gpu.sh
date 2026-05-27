#!/bin/bash
#SBATCH --job-name=qwen3-rubrics
#SBATCH --output=/gpfs/radev/pi/ying_rex/sg2768/OPSD/slurm/%x-%j.out
#SBATCH --error=/gpfs/radev/pi/ying_rex/sg2768/OPSD/slurm/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --exclude=r4513u10n01,r4513u20n01,r4513u30n01,r4516u01n01,r4518u01n01,r4518u09n01,r4518u05n01,r4507u05n01

# Usage:
#   Run Qwen3-14B then Qwen3-8B sequentially:
#     sbatch scripts/run_generate_rar_science_rubrics_qwen3_14b_4gpu.sh
#
#   Run only Qwen3-14B:
#     RUBRIC_GENERATOR_SEQUENCE=qwen3_14b sbatch \
#       scripts/run_generate_rar_science_rubrics_qwen3_14b_4gpu.sh
#
#   Run only Qwen3-8B:
#     RUBRIC_GENERATOR_SEQUENCE=qwen3_8b sbatch --job-name=qwen3-8b-rubrics \
#       scripts/run_generate_rar_science_rubrics_qwen3_14b_4gpu.sh
#
set -euo pipefail

source /gpfs/radev/pi/ying_rex/sg2768/miniconda3/etc/profile.d/conda.sh
conda activate opsd

export CONDA_PKGS_DIRS=/gpfs/radev/pi/ying_rex/sg2768/.conda/pkgs
export CONDA_ENVS_PATH=/gpfs/radev/pi/ying_rex/sg2768/.conda/envs
export PIP_CACHE_DIR=/gpfs/radev/pi/ying_rex/sg2768/.cache/pip
export TMPDIR=/gpfs/radev/pi/ying_rex/sg2768/.tmp
export TEMP=/gpfs/radev/pi/ying_rex/sg2768/.tmp
export TMP=/gpfs/radev/pi/ying_rex/sg2768/.tmp
export HF_HOME=/gpfs/radev/pi/ying_rex/sg2768/.cache/huggingface
export HF_HUB_CACHE=/gpfs/radev/pi/ying_rex/sg2768/.cache/huggingface/hub
export HF_DATASETS_CACHE=/gpfs/radev/pi/ying_rex/sg2768/.cache/huggingface/datasets
export CUDA_HOME=/gpfs/radev/apps/avx512/software/CUDA/12.1.1
export XDG_CACHE_HOME=/gpfs/radev/pi/ying_rex/sg2768/.cache
export TORCH_HOME=/gpfs/radev/pi/ying_rex/sg2768/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/gpfs/radev/pi/ying_rex/sg2768/.cache/torch/inductor

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

module load GCC/12.2.0

mkdir -p "${TMPDIR}"

echo "==== GPU DEBUG ===="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "nvidia-smi -L:"
nvidia-smi -L || true
python - <<'PY'
import torch
print("torch.cuda.device_count():", torch.cuda.device_count())
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {p.name}, cc {p.major}.{p.minor}, {p.total_memory/1e9:.1f} GB")
PY
echo "==================="

cd /gpfs/radev/pi/ying_rex/sg2768/OPSD/rubric

DATA_SOURCE="${DATA_SOURCE:-rar_science}"
if [[ -n "${RUBRIC_GENERATOR_SEQUENCE:-}" ]]; then
    RAW_RUBRIC_GENERATOR_SEQUENCE="${RUBRIC_GENERATOR_SEQUENCE}"
elif [[ -n "${RUBRIC_GENERATOR_SIZE:-}" ]]; then
    RAW_RUBRIC_GENERATOR_SEQUENCE="${RUBRIC_GENERATOR_SIZE}"
else
    RAW_RUBRIC_GENERATOR_SEQUENCE="qwen3_14b,qwen3_8b"
fi
if [[ "${RAW_RUBRIC_GENERATOR_SEQUENCE}" == "both" || "${RAW_RUBRIC_GENERATOR_SEQUENCE}" == "all" ]]; then
    RAW_RUBRIC_GENERATOR_SEQUENCE="qwen3_14b,qwen3_8b"
fi
RAW_RUBRIC_GENERATOR_SEQUENCE="${RAW_RUBRIC_GENERATOR_SEQUENCE// /,}"
IFS=',' read -r -a RUBRIC_GENERATOR_PROFILES <<< "${RAW_RUBRIC_GENERATOR_SEQUENCE}"

USER_RUBRIC_GENERATOR_MODEL="${RUBRIC_GENERATOR_MODEL:-${RUBRIC_MODEL_PATH:-}}"
USER_RUBRIC_CACHE_DIR="${RUBRIC_CACHE_DIR:-}"
USER_RUBRIC_WORK_DIR="${RUBRIC_WORK_DIR:-}"
USER_RUBRIC_BATCH_SIZE="${RUBRIC_BATCH_SIZE:-}"

RUBRIC_SAMPLE_SIZE="${RUBRIC_SAMPLE_SIZE:-10000}"
RUBRIC_PROMPT_MAX_LENGTH="${RUBRIC_PROMPT_MAX_LENGTH:-4096}"
RUBRIC_MAX_NEW_TOKENS="${RUBRIC_MAX_NEW_TOKENS:-2048}"
RUBRIC_TEMPERATURE="${RUBRIC_TEMPERATURE:-1.0}"
RUBRIC_TOP_P="${RUBRIC_TOP_P:-0.95}"
RUBRIC_TOP_K="${RUBRIC_TOP_K:-20}"

VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.6}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-6144}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"

MODEL_IS_LORA="${MODEL_IS_LORA:-0}"
REGENERATE_RUBRICS="${REGENERATE_RUBRICS:-0}"
RESUME_RUBRICS="${RESUME_RUBRICS:-1}"
ENABLE_THINKING="${ENABLE_THINKING:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
FILTER_INVALID="${FILTER_INVALID:-0}"
GUIDED_DECODING_REGEX="${GUIDED_DECODING_REGEX:-}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}"

if [[ "${#RUBRIC_GENERATOR_PROFILES[@]}" -gt 1 ]]; then
    if [[ -n "${USER_RUBRIC_GENERATOR_MODEL}" ]]; then
        echo "Do not set RUBRIC_GENERATOR_MODEL/RUBRIC_MODEL_PATH when running multiple profiles." >&2
        exit 2
    fi
    if [[ -n "${USER_RUBRIC_CACHE_DIR}" || -n "${USER_RUBRIC_WORK_DIR}" ]]; then
        echo "Do not set RUBRIC_CACHE_DIR/RUBRIC_WORK_DIR when running multiple profiles; each profile needs its own cache." >&2
        exit 2
    fi
fi

echo "DATA_SOURCE=${DATA_SOURCE}"
echo "RUBRIC_GENERATOR_SEQUENCE=${RAW_RUBRIC_GENERATOR_SEQUENCE}"
echo "RUBRIC_SAMPLE_SIZE=${RUBRIC_SAMPLE_SIZE}"
echo "RUBRIC_PROMPT_MAX_LENGTH=${RUBRIC_PROMPT_MAX_LENGTH}"
echo "RUBRIC_MAX_NEW_TOKENS=${RUBRIC_MAX_NEW_TOKENS}"
echo "VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE}"
echo "VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION}"
echo "VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN}"
echo "VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS}"
echo "VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS}"
echo "MODEL_IS_LORA=${MODEL_IS_LORA}"
echo "REGENERATE_RUBRICS=${REGENERATE_RUBRICS}"
echo "RESUME_RUBRICS=${RESUME_RUBRICS}"
echo "ENABLE_THINKING=${ENABLE_THINKING}"
echo "DISABLE_CUSTOM_ALL_REDUCE=${DISABLE_CUSTOM_ALL_REDUCE}"

validate_rubric_cache() {
    local cache_dir="$1"
    RUBRIC_CACHE_DIR="${cache_dir}" python - <<'PY'
import os
from datasets import load_from_disk

cache_dir = os.environ["RUBRIC_CACHE_DIR"]
dataset = load_from_disk(cache_dir)
print(f"Validated rubric cache load from: {cache_dir}")
print(f"Rows: {len(dataset)}")
print(f"Columns: {dataset.column_names}")
if "rubric" not in dataset.column_names:
    raise SystemExit("Generated cache is missing the required 'rubric' column.")
sample = dataset[0]["rubric"] if len(dataset) else None
if not isinstance(sample, str):
    raise SystemExit("Generated cache 'rubric' column is not string-valued.")
PY
}

run_rubric_generation() {
    local profile="$1"
    local default_model
    local default_cache_dir
    local default_batch_size

    case "${profile}" in
        qwen3_14b|qwen3-14b|14b)
            profile="qwen3_14b"
            default_model="Qwen/Qwen3-14B"
            default_cache_dir="/gpfs/radev/pi/ying_rex/sg2768/OPSD/outputs/rubric_cache/qwen3_14b_rubricgen"
            default_batch_size=16
            ;;
        qwen3_8b|qwen3-8b|8b)
            profile="qwen3_8b"
            default_model="Qwen/Qwen3-8B"
            default_cache_dir="/gpfs/radev/pi/ying_rex/sg2768/OPSD/outputs/rubric_cache/qwen3_8b_rubricgen"
            default_batch_size=32
            ;;
        *)
            echo "Unsupported rubric generator profile=${profile}. Expected qwen3_14b or qwen3_8b." >&2
            exit 2
            ;;
    esac

    local rubric_generator_model="${USER_RUBRIC_GENERATOR_MODEL:-${default_model}}"
    local rubric_cache_dir="${USER_RUBRIC_CACHE_DIR:-${default_cache_dir}}"
    local rubric_work_dir="${USER_RUBRIC_WORK_DIR:-${rubric_cache_dir}_work}"
    local rubric_batch_size="${USER_RUBRIC_BATCH_SIZE:-${default_batch_size}}"
    local rubric_base_model_name_or_path="${RUBRIC_BASE_MODEL_NAME_OR_PATH:-}"
    local tokenizer_name_or_path="${TOKENIZER_NAME_OR_PATH:-}"

    mkdir -p "$(dirname "${rubric_cache_dir}")"

    echo "========================================================================"
    echo "Starting rubric generation profile=${profile}"
    echo "RUBRIC_GENERATOR_MODEL=${rubric_generator_model}"
    echo "RUBRIC_CACHE_DIR=${rubric_cache_dir}"
    echo "RUBRIC_WORK_DIR=${rubric_work_dir}"
    echo "RUBRIC_BATCH_SIZE=${rubric_batch_size}"
    echo "========================================================================"

    CMD=(
        python generate_rubrics_vllm.py
        --data_source "${DATA_SOURCE}"
        --rubric_model_path "${rubric_generator_model}"
        --rubric_cache_dir "${rubric_cache_dir}"
        --work_dir "${rubric_work_dir}"
        --sample_size "${RUBRIC_SAMPLE_SIZE}"
        --prompt_max_length "${RUBRIC_PROMPT_MAX_LENGTH}"
        --max_new_tokens "${RUBRIC_MAX_NEW_TOKENS}"
        --batch_size "${rubric_batch_size}"
        --temperature "${RUBRIC_TEMPERATURE}"
        --top_p "${RUBRIC_TOP_P}"
        --top_k "${RUBRIC_TOP_K}"
        --tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}"
        --gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
        --max_model_len "${VLLM_MAX_MODEL_LEN}"
        --max_num_seqs "${VLLM_MAX_NUM_SEQS}"
        --max_num_batched_tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}"
    )

    if [[ -n "${rubric_base_model_name_or_path}" ]]; then
        CMD+=(--rubric_base_model_name_or_path "${rubric_base_model_name_or_path}")
    fi
    if [[ -n "${tokenizer_name_or_path}" ]]; then
        CMD+=(--tokenizer_name_or_path "${tokenizer_name_or_path}")
    fi
    if [[ "${MODEL_IS_LORA}" == "1" ]]; then
        CMD+=(--model_is_lora)
    fi
    if [[ "${REGENERATE_RUBRICS}" == "1" ]]; then
        CMD+=(--regenerate)
    fi
    if [[ "${RESUME_RUBRICS}" == "1" ]]; then
        CMD+=(--resume)
    fi
    if [[ "${ENABLE_THINKING}" == "1" ]]; then
        CMD+=(--enable_thinking)
    else
        CMD+=(--disable_thinking)
    fi
    if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
        CMD+=(--trust_remote_code)
    fi
    if [[ "${FILTER_INVALID}" == "1" ]]; then
        CMD+=(--filter_invalid)
    fi
    if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "1" ]]; then
        CMD+=(--disable_custom_all_reduce)
    fi
    if [[ -n "${GUIDED_DECODING_REGEX}" ]]; then
        CMD+=(--guided_decoding_regex "${GUIDED_DECODING_REGEX}")
    fi

    "${CMD[@]}"
    validate_rubric_cache "${rubric_cache_dir}"
    echo "Finished rubric generation profile=${profile}"
}

for profile in "${RUBRIC_GENERATOR_PROFILES[@]}"; do
    if [[ -z "${profile}" ]]; then
        continue
    fi
    run_rubric_generation "${profile}"
done
