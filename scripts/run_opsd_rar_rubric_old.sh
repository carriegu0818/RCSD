#!/bin/bash
#SBATCH --job-name=opsd-rubric
#SBATCH --output=/gpfs/radev/pi/ying_rex/sg2768/OPSD/slurm/%x-%j-rubric.out
#SBATCH --error=/gpfs/radev/pi/ying_rex/sg2768/OPSD/slurm/%x-%j-rubric.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2         # request 4 GPUs
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --exclude=r4513u10n01,r4513u20n01,r4513u30n01,r4516u01n01,r4518u01n01,r4518u09n01,r4518u05n01


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


cd /gpfs/radev/pi/ying_rex/sg2768/OPSD

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 2 \
    --gradient_accumulation_steps 2 \
    --main_process_port 12949 \
    opsd_train_rubric.py \
    --model_name_or_path Qwen/Qwen3-4B \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 2 \
    --output_dir  /gpfs/radev/pi/ying_rex/sg2768/OPSD/outputs \
    --run_config qwen3_4b_rubric_fixteacher_temp12_lr2e5_gen3072 \
    --num_train_epochs 3 \
    --max_completion_length 3072 \
    --save_steps 50 \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length 25000 \
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
    --teacher_prompt_tag rubric \
    --reason_first \
    --fixed_teacher \
    --wandb_entity sgu33-stanford-university \
    --wandb_project OPSD
    
#--teacher_prompt_tag rubric reference_answer \
