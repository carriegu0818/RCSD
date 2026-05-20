#!/bin/bash
#SBATCH --job-name=grpo
#SBATCH --output=/gpfs/radev/pi/ying_rex/sg2768/OPSD/slurm/%x-%j-natural-reference-cot.out
#SBATCH --error=/gpfs/radev/pi/ying_rex/sg2768/OPSD/slurm/%x-%j-natural-reference-cot.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2         # request 4 GPUs
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --exclude=r4513u10n01,r4513u20n01,r4513u30n01,r4516u01n01,r4518u01n01,r4518u09n01,r4518u05n01,r4507u05n01


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

module load GCC/12.2.0

GPQA_EVAL_INTERVAL="${GPQA_EVAL_INTERVAL:-10}"
GPQA_EVAL_SIZE="${GPQA_EVAL_SIZE:-50}"
TEACHER_EXTRA_SOURCE="${TEACHER_EXTRA_SOURCE:-responses}"

case "${TEACHER_EXTRA_SOURCE}" in
    reference_answer)
        TEACHER_PROMPT_TAG="reference_answer"
        RUN_CONFIG_SUFFIX="natural_reasoning_reference"
        ;;
    responses)
        TEACHER_PROMPT_TAG="cot"
        RUN_CONFIG_SUFFIX="natural_reasoning_responses_cot"
        ;;
    *)
        echo "Unsupported TEACHER_EXTRA_SOURCE=${TEACHER_EXTRA_SOURCE}. Use 'reference_answer' or 'responses'." >&2
        exit 1
        ;;
esac

echo "Teacher extra source: ${TEACHER_EXTRA_SOURCE}"
echo "Teacher prompt tag: ${TEACHER_PROMPT_TAG}"
echo "GPQA eval interval: ${GPQA_EVAL_INTERVAL}"
echo "GPQA eval size: ${GPQA_EVAL_SIZE}"


cd /gpfs/radev/pi/ying_rex/sg2768/OPSD

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 2 \
    --gradient_accumulation_steps 16 \
    --main_process_port 12969 \
    opsd_train_natural.py \
    --model_name_or_path Qwen/Qwen3-8B \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 1 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 16 \
    --output_dir  /gpfs/radev/pi/ying_rex/sg2768/OPSD/outputs \
    --run_config qwen3_8b_gen4096_fixteacher_temp12_lr5e6_${RUN_CONFIG_SUFFIX} \
    --num_train_epochs 3 \
    --max_completion_length 4096 \
    --save_steps 10 \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length 20000 \
    --beta 0.5 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.2 \
    --top_p 0.95 \
    --top_k 20 \
    --teacher_prompt_tag "${TEACHER_PROMPT_TAG}" \
    --fixed_teacher \
    --wandb_entity sgu33-stanford-university \
    --wandb_project OPSD
    
#--teacher_prompt_tag rubric reference_answer \
