import torch
import re
from reward_func_utils import extract_math_answer, normalize_answer
import numpy as np


def reward_func(queries, prompts, labels):
    base_rewards = []
    domain_tags = []

    for query, prompt, label in zip(queries, prompts, labels):
        reward = 0.0
        gen_answer = query[len(prompt):]

        # Identify domain and compute base reward
        if "math problem" in prompt:
            domain = "math"
            gt_label = extract_math_answer(label)
            gen_ans = extract_math_answer(query)
            if gt_label == gen_ans:
                reward = 1.0

        elif "multiple-choice answers" in prompt:
            domain = "arc"
            temp = re.findall(r'\b[A-H]\b', gen_answer)
            if temp and temp[-1] == label:
                reward = 1.0

        elif "concise and accurate answer" in prompt:  # NQ
            domain = "nq"
            normalized_text = normalize_answer(gen_answer)
            answer_list = label.split(";")
            for single in answer_list:
                if normalize_answer(single) in normalized_text:
                    reward = 1.0
                    break

        elif "movie review" in prompt:
            domain = "imdb"
            gen_answer = gen_answer.replace("negative", "Negative").replace("positive", "Positive")
            temp = re.findall(r'\b(Negative|Positive)\b', gen_answer)
            if temp and temp[-1] == label:
                reward = 1.0

        else:
            domain = "unknown"

        base_rewards.append(reward)
        domain_tags.append(domain)

    final_rewards = []

    sc = sum(base_rewards) / len(base_rewards)
    # difficulty_weight = np.log(1 + 1 / (sc + 1e-5)) ** 2  # smooth log-scaling
    difficulty_weight = 1 / (sc + 1e-5)  # smooth log-scaling

    for i in range(len(base_rewards)):
        current_reward = base_rewards[i]
        weighted_group = current_reward * difficulty_weight
        final_rewards.append(weighted_group)

    return torch.tensor(final_rewards, dtype=torch.float)
