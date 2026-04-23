#!/bin/bash
#SBATCH --output=/gpfs/radev/pi/ying_rex/sg2768/OPSD/slurm/%x-%j-eval.out
#SBATCH --error=/gpfs/radev/pi/ying_rex/sg2768/OPSD/slurm/%x-%j-eval.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2         # request 4 GPUs
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00


#conda activate opsd
source ~/.bashrc
conda activate /gpfs/radev/pi/ying_rex/sg2768/miniconda3/envs/verl_env_copy
#conda activate /gpfs/radev/home/slz4/.conda/envs/verl_env

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


cd /gpfs/radev/pi/ying_rex/sg2768/OPSD/eval

BASE_MODEL="Qwen/Qwen3-8B"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TP_SIZE=$(awk -F',' '{print NF}' <<< "$CUDA_DEVICES")


# # evaluate base model performance
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "aime25" \
    --val_n 1 \
    --temperature 1.1 \
    --tensor_parallel_size "$TP_SIZE" \
    --gpu_memory_utilization 0.4 \
    --checkpoint_dir /gpfs/radev/pi/ying_rex/sg2768/OPSD/outputs/qwen3_8b_reward_rubricref_reasonfirst_fixteacher_temp12_lr2e5_gen4096_2gpu_v2/checkpoint-50
wait

# EVALUTE OPSD
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "aime24" \
    --val_n 1 \
    --temperature 1.1 \
    --tensor_parallel_size "$TP_SIZE" \
    --gpu_memory_utilization 0.4 \
    --checkpoint_dir /gpfs/radev/pi/ying_rex/sg2768/OPSD/outputs/qwen3_8b_reward_rubricref_reasonfirst_fixteacher_temp12_lr2e5_gen4096_2gpu_v2/checkpoint-50
wait

# EVALUTE OPSD
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "hmmt25" \
    --val_n 1 \
    --temperature 1.1 \
    --tensor_parallel_size "$TP_SIZE" \
    --gpu_memory_utilization 0.4 \
    --checkpoint_dir /gpfs/radev/pi/ying_rex/sg2768/OPSD/outputs/qwen3_8b_reward_rubricref_reasonfirst_fixteacher_temp12_lr2e5_gen4096_2gpu_v2/checkpoint-50
wait

# # evaluate base model performance
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "math500" \
    --val_n 1 \
    --temperature 1.1 \
    --tensor_parallel_size "$TP_SIZE" \
    --gpu_memory_utilization 0.4 \
    --checkpoint_dir /gpfs/radev/pi/ying_rex/sg2768/OPSD/outputs/qwen3_8b_reward_rubricref_reasonfirst_fixteacher_temp12_lr2e5_gen4096_2gpu_v2/checkpoint-50
wait

# n=4 eval -----------
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "aime25" \
    --val_n 4 \
    --temperature 1.1 \
    --tensor_parallel_size "$TP_SIZE" \
    --gpu_memory_utilization 0.4 \
    --checkpoint_dir /gpfs/radev/pi/ying_rex/sg2768/OPSD/outputs/qwen3_8b_reward_rubricref_reasonfirst_fixteacher_temp12_lr2e5_gen4096_2gpu_v2/checkpoint-50
wait

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "aime24" \
    --val_n 4 \
    --temperature 1.1 \
    --tensor_parallel_size "$TP_SIZE" \
    --gpu_memory_utilization 0.4 \
    --checkpoint_dir /gpfs/radev/pi/ying_rex/sg2768/OPSD/outputs/qwen3_8b_reward_rubricref_reasonfirst_fixteacher_temp12_lr2e5_gen4096_2gpu_v2/checkpoint-50
wait

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "hmmt25" \
    --val_n 4 \
    --temperature 1.1 \
    --tensor_parallel_size "$TP_SIZE" \
    --gpu_memory_utilization 0.4 \
    --checkpoint_dir /gpfs/radev/pi/ying_rex/sg2768/OPSD/outputs/qwen3_8b_reward_rubricref_reasonfirst_fixteacher_temp12_lr2e5_gen4096_2gpu_v2/checkpoint-50
wait

# n=8 eval -----------
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "aime25" \
    --val_n 8 \
    --temperature 1.1 \
    --tensor_parallel_size "$TP_SIZE" \
    --gpu_memory_utilization 0.4 \
    --checkpoint_dir /gpfs/radev/pi/ying_rex/sg2768/OPSD/outputs/qwen3_8b_reward_rubricref_reasonfirst_fixteacher_temp12_lr2e5_gen4096_2gpu_v2/checkpoint-50
wait

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "aime24" \
    --val_n 8 \
    --temperature 1.1 \
    --tensor_parallel_size "$TP_SIZE" \
    --gpu_memory_utilization 0.4 \
    --checkpoint_dir /gpfs/radev/pi/ying_rex/sg2768/OPSD/outputs/qwen3_8b_reward_rubricref_reasonfirst_fixteacher_temp12_lr2e5_gen4096_2gpu_v2/checkpoint-50
wait

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "hmmt25" \
    --val_n 8 \
    --temperature 1.1 \
    --tensor_parallel_size "$TP_SIZE" \
    --gpu_memory_utilization 0.4 \
    --checkpoint_dir /gpfs/radev/pi/ying_rex/sg2768/OPSD/outputs/qwen3_8b_reward_rubricref_reasonfirst_fixteacher_temp12_lr2e5_gen4096_2gpu_v2/checkpoint-50
wait

# n=16 eval -----------
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "aime25" \
    --val_n 16 \
    --temperature 1.1 \
    --tensor_parallel_size "$TP_SIZE" \
    --checkpoint_dir /gpfs/radev/pi/ying_rex/sg2768/OPSD/outputs/qwen3_8b_reward_rubricref_reasonfirst_fixteacher_temp12_lr2e5_gen4096_2gpu_v2/checkpoint-50
wait

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "aime24" \
    --val_n 16 \
    --temperature 1.1 \
    --tensor_parallel_size "$TP_SIZE" \
    --checkpoint_dir /gpfs/radev/pi/ying_rex/sg2768/OPSD/outputs/qwen3_8b_reward_rubricref_reasonfirst_fixteacher_temp12_lr2e5_gen4096_2gpu_v2/checkpoint-50
wait

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "hmmt25" \
    --val_n 16 \
    --temperature 1.1 \
    --tensor_parallel_size "$TP_SIZE" \
    --checkpoint_dir /gpfs/radev/pi/ying_rex/sg2768/OPSD/outputs/qwen3_8b_reward_rubricref_reasonfirst_fixteacher_temp12_lr2e5_gen4096_2gpu_v2/checkpoint-50
wait