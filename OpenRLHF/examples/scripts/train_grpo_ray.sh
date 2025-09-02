set -x

# reinforce++
ray start --head --node-ip-address 0.0.0.0 --num-gpus 4

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/fs/clip-emoji/tonyzhou/moe_reward/OpenRLHF/examples/scripts"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --pretrain /fs/clip-scratch/tonyzhou/moe_reward/sft_checkpoint/olmoe_sft_lora_metamath/ \
   --reward_pretrain  /fs/clip-scratch/tonyzhou/moe_reward/reward_checkpoint/olmoe_reward_lora_metamath/ \
   --save_path /fs/clip-emoji/tonyzhou/moe_reward/lora_checkpoint/olmoe_grpo_lora_metamath_ray \
   --micro_train_batch_size 2 \
   --train_batch_size 8 \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size 8 \
   --n_samples_per_prompt 4 \
   --max_epochs 1 \
   --prompt_max_len 256 \
   --max_samples 100000 \
   --generate_max_len 256 \
   --init_kl_coef 1e-3 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k3 \
   --advantage_estimator group_norm \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --prompt_data MoeReward/combined_rlhf_dataset \
   --input_key context_messages \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --packing_samples \
   --use_wandb 71cd2140fcca386a655548a54ae4debb3649c860 \
   --save_steps -1

# You could also try
#   --kl_estimator k2 \

# also supports --advantage_estimator rloo | reinforce_baseline