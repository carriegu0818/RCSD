#!/bin/bash
#SBATCH --job-name=opsd-reward
#SBATCH --output=/gpfs/radev/pi/ying_rex/sg2768/OPSD/slurm/%x-%j-reward-17b.out
#SBATCH --error=/gpfs/radev/pi/ying_rex/sg2768/OPSD/slurm/%x-%j-reward-17b.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2         # request 2 GPUs
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
export WANDB_API_KEY=1b0ca0f0dc2e90d460addf9ccaeda83188ad001d


export XDG_CACHE_HOME=/gpfs/radev/pi/ying_rex/sg2768/.cache
export TORCH_HOME=/gpfs/radev/pi/ying_rex/sg2768/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/gpfs/radev/pi/ying_rex/sg2768/.cache/torch/inductor

module load GCC/12.2.0

echo "==== GPU DEBUG ===="
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
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


cd /gpfs/radev/pi/ying_rex/sg2768/OPSD

# # 1) Generate rubrics (2 GPUs, distributed HF generate)
# accelerate launch \
#     --config_file accelerate.yaml \
#     --num_processes 4 \
#     --main_process_port 12947 \
#     opsd_train_reward.py \
#     --model_name_or_path Qwen/Qwen3-8B \
#     --learning_rate 2e-5 \
#     --per_device_train_batch_size 1 \
#     --gradient_checkpointing \
#     --gradient_accumulation_steps 4 \
#     --output_dir  /gpfs/radev/pi/ying_rex/sg2768/OPSD/outputs \
#     --run_config qwen3_8b_reward_rubricref_fixteacher_temp12_lr2e5_gen4096_2gpu \
#     --num_train_epochs 3 \
#     --max_completion_length 4096 \
#     --save_steps 50 \
#     --logging_steps 2 \
#     --attn_implementation flash_attention_2 \
#     --torch_dtype bfloat16 \
#     --max_length 20000 \
#     --beta 0.5 \
#     --use_peft \
#     --lora_r 64 \
#     --lora_alpha 128 \
#     --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
#     --temperature 1.2 \
#     --top_p 0.95 \
#     --top_k 20 \
#     --rubric_model_path /gpfs/radev/pi/ying_rex/sg2768/OPSD/outputs/qwen3_8b_rubric_fixteacher_temp12_lr2e5_gen4096/checkpoint-1000 \
#     --rubric_cache_dir /gpfs/radev/pi/ying_rex/sg2768/OPSD/outputs/rubric_cache/qwen3_8b_rubric_fixteacher_temp12_lr2e5_gen4096 \
#     --rubric_sample_size 10000 \
#     --rubric_max_new_tokens 4096 \
#     --rubric_distributed \
#     --rubric_only \
#     --fixed_teacher \
#     --wandb_entity sgu33-stanford-university \
#     --wandb_project OPSD

# 2) Train using cached rubrics (multi-GPU)
accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 2 \
    --gradient_accumulation_steps 16 \
    --main_process_port 15949 \
    opsd_train_reward.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 1 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 16 \
    --output_dir  /gpfs/radev/pi/ying_rex/sg2768/OPSD/outputs \
    --run_config qwen3_17b_reward_rubricref_reasonfirst_full_temp12_lr2e5_gen4096_2gpu_v2 \
    --num_train_epochs 3 \
    --max_completion_length 4096 \
    --save_steps 20 \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length 25000 \
    --beta 0.5 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_tensor_parallel_size 1 \
    --temperature 1.2 \
    --top_p 0.95 \
    --top_k 20 \
    --rubric_model_path /gpfs/radev/pi/ying_rex/sg2768/OPSD_runtime/outputs/qwen3_8b_rubric_fixteacher_temp12_lr2e5_gen4096/checkpoint-1000 \
    --rubric_cache_dir /gpfs/radev/pi/ying_rex/sg2768/OPSD_runtime/outputs/rubric_cache/qwen3_8b_rubric_fixteacher_temp12_lr2e5_gen4096 \
    --rubric_sample_size 10000 \
    --rubric_max_new_tokens 1024 \
    --rubric_distributed \
    --jsd_token_clip 0.05 \
    --reason_first \
    --wandb_entity sgu33-stanford-university \
    --wandb_project OPSD
    
#--teacher_prompt_tag rubric reference_answer \
