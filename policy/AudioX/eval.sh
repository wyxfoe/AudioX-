#!/bin/bash

# ============================================================
# AudioX (RobotX) Policy Evaluation Script for RoboTwin
#
# Usage:
#   bash eval.sh <task_name> <task_config> <ckpt_setting> <ckpt_path> <seed> <gpu_id>
#
# Example:
#   bash eval.sh beat_block_hammer demo_clean demo_clean \
#     /home/wyx/AudioX-/checkpoints/robotx/robotx-step=00048000.ckpt \
#     0 0
#
# Parameters:
#   task_name    - Task to evaluate (e.g., beat_block_hammer)
#   task_config  - Evaluation environment config (demo_clean / demo_randomized)
#   ckpt_setting - Training data config used during training
#   ckpt_path    - Path to model checkpoint (.ckpt / .pt / .safetensors)
#   seed         - Random seed
#   gpu_id       - GPU device ID
# ============================================================

policy_name=AudioX
task_name=${1}
task_config=${2}
ckpt_setting=${3}
ckpt_path=${4}
seed=${5}
gpu_id=${6}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mcheckpoint: ${ckpt_path}\033[0m"

cd ../.. # move to RoboTwin root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
  --overrides \
  --task_name ${task_name} \
  --task_config ${task_config} \
  --ckpt_setting ${ckpt_setting} \
  --ckpt_path ${ckpt_path} \
  --seed ${seed} \
  --policy_name ${policy_name}
