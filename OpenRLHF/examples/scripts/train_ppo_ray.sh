# start ray
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# submit jobs (change working_dir to the correct one)
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/home/syqian/moe_reward/OpenRLHF/examples/scripts"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 2 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 2 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain /home/syqian/moe_reward/sft_checkpoint/qwen1.5_sft_lora_metamath/ \
   --reward_pretrain /home/syqian/moe_reward/rewards_checkpoint/qwen1.5_reward_lora_metamath_merged/ \
   --save_path /home/syqian/moe_reward/lora_checkpoint/qwen1.5_ppo_lora_metamath_ray \
   --micro_train_batch_size 16 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 1024 \
   --max_samples 100000 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data MoeReward/combined_rlhf_dataset \
   --input_key context_messages \
   --apply_chat_template \
   --normalize_reward \
   --packing_samples \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --load_checkpoint 