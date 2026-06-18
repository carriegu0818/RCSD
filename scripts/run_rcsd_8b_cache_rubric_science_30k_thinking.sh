#!/bin/bash

# Train Qwen3-8B model with cached rubrics.
# Dataset: RCSD/rubric_cache/rubrichub_science_30k




export CONDA_PKGS_DIRS=.conda/pkgs
export CONDA_ENVS_PATH=.conda/envs
export PIP_CACHE_DIR=.cache/pip
export TMPDIR=.tmp
export TEMP=.tmp
export TMP=.tmp
export HF_HOME=.cache/huggingface
export HF_HUB_CACHE=.cache/huggingface/hub
export HF_DATASETS_CACHE=.cache/huggingface/datasets
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>


export XDG_CACHE_HOME=.cache
export TORCH_HOME=.cache/torch
export TORCHINDUCTOR_CACHE_DIR=.cache/torch/inductor


cd RCSD

DATA_SOURCE="${DATA_SOURCE:-rubrichub}"
RUBRIC_SOURCE="${RUBRIC_SOURCE:-cache}"
RUN_TAG="${DATA_SOURCE//-/_}_${RUBRIC_SOURCE//-/_}"
RUN_CONFIG="qwen3_8b_lr5e6_4096_b0_${RUN_TAG}_30k_thinking"
RUBRIC_CACHE_DIR="${RUBRIC_CACHE_DIR:-RCSD/rubric_cache/rubrichub_science_30k}"

echo "DATA_SOURCE=${DATA_SOURCE}"
echo "RUBRIC_SOURCE=${RUBRIC_SOURCE}"
echo "RUN_CONFIG=${RUN_CONFIG}"
echo "RUBRIC_CACHE_DIR=${RUBRIC_CACHE_DIR}"



resume_args=()


# Train using cached rubrics (multi-GPU)
accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 8 \
    --gradient_accumulation_steps 4 \
    --main_process_port 11835 \
    rcsd_train_reward.py \
    --model_name_or_path Qwen/Qwen3-8B \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 1 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 4 \
    --output_dir RCSD/outputs \
    --run_config "${RUN_CONFIG}" \
    --num_train_epochs 3 \
    --max_completion_length 4096 \
    --save_steps 10 \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length 25000 \
    --beta 0 \
    --use_peft \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_tensor_parallel_size 1 \
    --temperature 1.2 \
    --top_p 0.95 \
    --top_k 20 \
    --top_k_loss 20 \
    --data_source "${DATA_SOURCE}" \
    --rubric_source "${RUBRIC_SOURCE}" \
    --rubric_cache_dir "${RUBRIC_CACHE_DIR}" \
    --student_thinking True \
    --teacher_thinking True \
    --jsd_token_clip 0.05 \
    --reason_first \
    --fixed_teacher \
    --wandb_entity <YOUR_WANDB_ENTITY> \
    --wandb_project RCSD \
    "${resume_args[@]}"
