#!/bin/bash

# Get command line parameters with defaults
SAVE_DIR=${1:-"qwen2.5_base_metamath_main_gt_grpo_openrlhf"}
PROMPT_DATA=${2:-"MoeReward/combined_rlhf_dataset_grpo_metamath_main"}
REWARD_FUNC=${3:-"math_domain_reward_func.py"}
NORMALIZE_ESTIMATOR=${4:-"group_norm"}
MODEL=${5:-"Qwen/Qwen2-0.5B"}

# Start ray
ray start --head --node-ip-address 0.0.0.0 --num-gpus 4

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "./scripts/grpo/"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --pretrain ${MODEL} \
   --save_path ${SAVE_DIR} \
   --micro_train_batch_size 1 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 64 \
   --n_samples_per_prompt 4 \
   --max_epochs 1 \
   --prompt_max_len 512 \
   --max_samples 10000 \
   --generate_max_len 512 \
   --init_kl_coef 1e-3 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k3 \
   --advantage_estimator ${NORMALIZE_ESTIMATOR} \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 1e-6 \
   --prompt_data ${PROMPT_DATA} \
   --remote_rm_url ${REWARD_FUNC} \
   --label_key answer \
   --input_key prompt \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --packing_samples \
   --wandb_run_name ${SAVE_DIR} \
   --save_steps -1 \
   --enable_prefix_caching