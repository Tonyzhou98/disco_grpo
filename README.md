# BalDistill

**Paper**:  
Yuhang Zhou*, Jing Zhu*, Shengyi Qian, Zhuokai Zhao, Xiyao Wang, Xiaoyu Liu, Ming Li, Paiheng Xu, Wei Ai, Furong Huang

*DISCO Balances the Scales: Adaptive Domain- and Difficulty-Aware Reinforcement Learning on Imbalanced Data*
[Link to Paper](https://arxiv.org/abs/2505.15074)

### Citation (BibTeX):
```bibtex
@misc{zhou2025discobalancesscalesadaptive,
      title={DISCO Balances the Scales: Adaptive Domain- and Difficulty-Aware Reinforcement Learning on Imbalanced Data}, 
      author={Yuhang Zhou and Jing Zhu and Shengyi Qian and Zhuokai Zhao and Xiyao Wang and Xiaoyu Liu and Ming Li and Paiheng Xu and Wei Ai and Furong Huang},
      year={2025},
      eprint={2505.15074},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.15074}, 
}
```

## Usage

### Install

Same as OpenRLHF (https://github.com/OpenRLHF/OpenRLHF)
Then
```
cd OpenRLHF
pip install -e .
```
to enable DISCO

### Training

Sample training scripts and reward functions: `scripts/grpo/train_grpo_qwen2.5_gt_domain_weight.sh`

Sample command line to run the script

DISCO training for Gemma-2-2b-it on dataset with ARC as the majority
```
./train_grpo_qwen2.5_gt_domain_weight.sh {self_defined_checkpoint_save_path} MoeReward/combined_rlhf_dataset_grpo_arc_main arc_domain_reward_func.py group_norm_wo_std google/gemma-2-2b-it
```
You can change reward function file to `reward_func.py` for Dr.GRPO and change `group_norm_wo_std` to `group_norm` for naive GRPO

Different training dataset with different distributions:

MoeReward/combined_rlhf_dataset_grpo_metamath_main (https://huggingface.co/datasets/MoeReward/combined_rlhf_dataset_grpo_metamath_main) 

MoeReward/combined_rlhf_dataset_grpo_arc_main (https://huggingface.co/datasets/MoeReward/combined_rlhf_dataset_grpo_arc_main)

MoeReward/combined_rlhf_dataset_grpo_imdb_main (https://huggingface.co/datasets/MoeReward/combined_rlhf_dataset_grpo_imdb_main)

MoeReward/combined_rlhf_dataset_grpo_nq_main (https://huggingface.co/datasets/MoeReward/combined_rlhf_dataset_grpo_nq_main)

### Eval
```
python moe_eval.py --model_name {checkpoint_path}  --no_template  --batch_size 32 --ds nq
python moe_eval.py --model_name {checkpoint_path}  --no_template  --batch_size 32 --ds arc
python moe_eval.py --model_name {checkpoint_path}  --no_template  --batch_size 32 --ds imdb
python moe_eval.py --model_name {checkpoint_path}  --no_template  --batch_size 32 --ds math
python moe_eval.py --model_name {checkpoint_path}  --no_template  --batch_size 32 --ds mgsm8k
```