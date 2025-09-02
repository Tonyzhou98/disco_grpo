import re
import torch
from reward_func_utils import extract_math_answer, normalize_answer


def reward_func(queries, prompts, labels):
    rewards = []
    for query, prompt, label in zip(queries, prompts, labels):
        reward = 0.0
        gen_answer = query[len(prompt):]

        if "math problem" in prompt:
            gt_label = extract_math_answer(label)
            gen_ans = extract_math_answer(query)
            if gt_label == gen_ans:
                reward = 1.0

        elif "multiple-choice answers" in prompt:
            temp = re.findall(r'\b[A-H]\b', gen_answer)
            if temp and temp[-1] == label:
                reward = 1.0

        elif "concise and accurate answer." in prompt:
            normalized_text = normalize_answer(gen_answer)
            answer_list = label.split(";")

            for single in answer_list:
                if normalize_answer(single) in normalized_text:
                    reward = 1.0
                    break

        elif "movie review" in prompt:
            gen_answer = gen_answer.replace("negative", "Negative").replace("positive", "Positive")
            temp = re.findall(r'\b(Negative|Positive)\b', gen_answer)
            if temp and temp[-1] == label:
                reward = 1.0

        rewards.append(reward)

    return torch.tensor(rewards, dtype=torch.float)
