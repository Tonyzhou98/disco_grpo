import re
import ast
import os
import json
import torch
import random
import argparse
from pathlib import Path
from eval_utils import extract_math_answer, normalize_answer
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score

random.seed(42)

# from peft import (
#     LoraConfig,
#     get_peft_model,
#     PeftModel,
#     PeftConfig
# )


def adapter_config_exists(directory):
    directory_path = Path(directory)
    if not directory_path.is_dir():
        return False

    file_path = directory_path / "adapter_config.json"
    return file_path.is_file()


def process_math(dataset):
    prompts, labels = [], []
    for p, l in zip(dataset['problem'], dataset['solution']):
        prompt = (
                "Below is a math problem. Provide a detailed, step-by-step solution.\n\n"
                "### Problem:\n" + p + "\n\n"
                                       "### Answer:\n"
        )
        prompts.append(prompt)
        labels.append(l)
    return prompts, labels


def process_gsm8k(dataset):
    prompts, labels = [], []
    for q, a in zip(dataset['question'], dataset['answer']):
        prompt = (
                "Below is a math problem. Provide a detailed, step-by-step solution.\n\n"
                "### Problem:\n" + q + "\n\n"
                                       "### Answer:\n"
        )
        prompts.append(prompt)
        labels.append(a)
    return prompts, labels


def process_math500(dataset):
    prompts, labels = [], []
    for q, a in zip(dataset['problem'], dataset['answer']):
        prompt = (
                "Below is a math problem. Provide a detailed, step-by-step solution.\n\n"
                "### Problem:\n" + q + "\n\n"
                                       "### Answer:\n"
        )
        prompts.append(prompt)
        labels.append(a)
    return prompts, labels


def process_mbpp(dataset):
    prompts, labels = [], []
    for t, c in zip(dataset['text'], dataset['code']):
        prompt = (
                "Below is a programming task. "
                "Write a code response that fulfills the task requirements.\n\n"
                "### Task:\n" + t + "\n\n"
                                    "### Answer:\n"
        )
        prompts.append(prompt)
        labels.append(c)
    return prompts, labels


def process_imdb(dataset):
    prompts, labels = [], []
    for t, c in zip(dataset['text'], dataset['label']):
        prompt = (
            "Below is a movie review. Determine the sentiment of the review as Positive or Negative.\n\n"
            f"### Review:\n{t}\n\n"
            "### Answer:\n"
        )
        prompts.append(prompt)
        labels.append("Positive" if c == 1 else "Negative")

    pairs = list(zip(prompts, labels))
    sampled_pairs = random.sample(pairs, 1000)
    sampled_prompts, sampled_labels = zip(*sampled_pairs)

    return list(sampled_prompts), list(sampled_labels)


def extract_function_info(code):
    """
    Parse the given code and extract the function name and its docstring.
    Returns a tuple (function_name, docstring) if found, else (None, None).
    """
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            # Try to get the docstring in the standard position.
            docstring = ast.get_docstring(node)
            if docstring is None:
                # Fallback: search for any string literal in the function body.
                for child in node.body:
                    if (isinstance(child, ast.Expr) and
                            isinstance(child.value, ast.Constant) and
                            isinstance(child.value.value, str)):
                        docstring = child.value.value.strip()
                        break
            return func_name, docstring
    raise NotImplementedError


def process_human_eval(dataset):
    prompts, labels = [], []
    for p, c in zip(dataset['prompt'], dataset['canonical_solution']):
        # name, docstring = extract_function_info(p)
        # try:
        #     docstring = docstring[0].lower() + docstring[1:]
        # except:
        #     print(p)
        #     print(docstring)
        #     raise NotImplementedError
        # output_text = f"Write a Python function named `{name}` that {docstring}"
        prompt = (
                "Below is a programming task. "
                "Write a code response that fulfills the task requirements.\n\n"
                "### Task:\n" + p + "\n\n"
                                    "### Answer:\n"
        )
        prompts.append(prompt)
        labels.append(c)
    return prompts, labels


def process_boolq(dataset):
    prompts, labels = [], []
    for a, q, p in zip(dataset['answer'], dataset['question'], dataset['passage']):
        answer_str = "Yes" if a in [True, "true", "True"] else "No"
        prompt = (
                "Below is a question along with a passage for context. "
                "Provide a concise answer (Yes or No) that addresses the question accurately.\n\n"
                "### Question:\n" + q + "\n\n"
                                        "### Passage:\n" + p + "\n\n"
                                                               "### Answer:\n"
        )
        prompts.append(prompt)
        labels.append(answer_str)
    return prompts, labels


def process_sqa(dataset):
    prompts, labels = [], []
    for a, q in zip(dataset['answer'], dataset['question']):
        answer_str = "Yes" if a in [True, "true", "True"] else "No"
        prompt = (
            "Below is a question that requires multi-step reasoning. "
            "Provide a concise answer (Yes or No) and a detailed explanation before concluding with the correct answer.\n\n"
            f"### Question:\n{q}\n\n"
            "### Answer:\n"
        )
        prompts.append(prompt)
        labels.append(answer_str)
    return prompts, labels


def process_nq(dataset):
    prompts, labels = [], []
    for a, q in zip(dataset['answer'], dataset['question']):
        answer_str = ";".join(a)
        prompt = (
            "Below is a question that requires a concise and accurate answer. "
            "Provide a detailed explanation before concluding with the correct answer.\n\n"
            f"### Question:\n{q}\n\n"
            "### Answer:\n"
        )
        prompts.append(prompt)
        labels.append(answer_str)

    return prompts, labels


def process_arc(dataset):
    prompts, labels = [], []
    for c, q, a in zip(dataset['choices'], dataset['question'], dataset["answerKey"]):
        choices_field = c
        texts = choices_field.get("text", [])
        labels_list = choices_field.get("label", [])
        # Construct choice options like "A. dry palms"
        choices_options = [f"{lab}. {txt}" for lab, txt in zip(labels_list, texts)]
        choices_str = "\n".join(choices_options)
        prompt = (
                "Below is a question with multiple-choice answers. "
                "Choose the correct option based on your reasoning.\n\n"
                "### Question:\n" + q + "\n\n"
                                        "### Choices:\n" + choices_str + "\n\n"
                                                                         "### Answer:\n"
        )
        prompts.append(prompt)
        labels.append(a)
    return prompts, labels


def process_alpaca(dataset):
    prompts, labels = [], []
    for i, o in zip(dataset['instruction'], dataset['output']):
        prompt = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n" + i + "\n\n"
                                           "### Answer:\n"
        )
        prompts.append(prompt)
        labels.append(o)
    return prompts, labels


def get_eval_data(ds):
    if ds == "MATH":
        math_subsets = ["algebra", "counting_and_probability", "geometry",
                        "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
        prompts, labels = [], []
        for subset in math_subsets:
            d = load_dataset("EleutherAI/hendrycks_math", subset, split="test")
            sub_p, sub_l = process_math(d)
            prompts.extend(sub_p)
            labels.extend(sub_l)
        return prompts, labels
    elif ds == "gsm8k":
        d = load_dataset("openai/gsm8k", 'main', split="train")
        return process_gsm8k(d)
    elif ds == "MATH-500":
        d = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return process_math500(d)
    elif ds == "mbpp":
        d = load_dataset("google-research-datasets/mbpp", 'full', split="test")
        return process_mbpp(d)
    elif ds == "human_eval":
        d = load_dataset("openai/openai_humaneval", split="test")
        return process_human_eval(d)
    elif ds == "imdb":
        d = load_dataset("stanfordnlp/imdb", split="test")
        return process_imdb(d)
    elif ds == "boolq":
        d = load_dataset("google/boolq", split="validation")
        return process_boolq(d)
    elif ds == "sqa":
        d = load_dataset("ChilleD/StrategyQA", split="test")
        return process_sqa(d)
    elif ds == "nq":
        d = load_dataset("google-research-datasets/nq_open", split="validation")
        return process_nq(d)
    elif ds == "arc":
        d = load_dataset("allenai/ai2_arc", 'ARC-Challenge', split="test")
        return process_arc(d)
    elif ds == "alpaca":
        d = load_dataset("tatsu-lab/alpaca_eval", trust_remote_code=True, split="eval")
        return process_alpaca(d)
    elif ds == "train":
        d = load_dataset("MoeReward/combined_rlhf_dataset", trust_remote_code=True, split="train")
        return d['context_messages'], d['context_messages']
    else:
        raise NotImplementedError


def extract_qwen_response(result):
    """Extract response from Qwen-style chat template"""
    assistant_start_tag = "<|im_start|>assistant\n"
    assistant_end_tag = "<|im_end|>"
    result = result.replace("<|endoftext|>", "")

    # Find the last assistant tag (in case there are multiple)
    last_start_idx = result.rfind(assistant_start_tag)

    if last_start_idx != -1:
        # Get the position after the assistant tag
        content_start_idx = last_start_idx + len(assistant_start_tag)

        # Find the closing tag
        end_idx = result.find(assistant_end_tag, content_start_idx)

        if end_idx != -1:
            # Extract the content between the tags
            assistant_content = result[content_start_idx:end_idx].strip()
            return assistant_content
        else:
            # No end tag found, take everything after the start tag
            assistant_content = result[content_start_idx:].strip()
            return assistant_content

    return ""


def extract_llama_response(result):
    """Extract response from Llama-style chat template"""
    assistant_start_tag = "[INST] "
    assistant_end_tag = " [/INST]"
    response_start = "Assistant: "

    # Find the last instruction block
    last_inst_idx = result.rfind(assistant_start_tag)
    if last_inst_idx != -1:
        # Find the end of the instruction
        inst_end_idx = result.find(assistant_end_tag, last_inst_idx)
        if inst_end_idx != -1:
            # Find the start of the assistant response after the instruction
            response_idx = result.find(response_start, inst_end_idx)
            if response_idx != -1:
                # Extract everything after "Assistant: "
                content_start_idx = response_idx + len(response_start)
                assistant_content = result[content_start_idx:].strip()
                return assistant_content

    return ""


def extract_olmoe_response(result):
    """Extract response from OLMoE-style chat template"""
    assistant_start_tag = "<|assistant|>\n"
    assistant_end_tag = "<|endoftext|>"

    # Find the last assistant tag
    last_start_idx = result.rfind(assistant_start_tag)

    if last_start_idx != -1:
        # Get the position after the assistant tag
        content_start_idx = last_start_idx + len(assistant_start_tag)

        # Find the closing tag
        end_idx = result.find(assistant_end_tag, content_start_idx)

        if end_idx != -1:
            # Extract the content between the tags
            assistant_content = result[content_start_idx:end_idx].strip()
            return assistant_content
        else:
            # No end tag found, take everything after the start tag
            assistant_content = result[content_start_idx:].strip()
            return assistant_content

    return ""


def process_batch(model, tokenizer, batch_instructs, batch_labels, ds, output_file, model_name, apply_template=True):
    if apply_template:
        template_batch = []
        for i in batch_instructs:
            chat = [{"role": "user", "content": i}]
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            template_batch.append(prompt)
        batch_instructs = template_batch

    inputs = tokenizer(
        batch_instructs, return_tensors="pt", padding=True, truncation=True
    )

    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            do_sample=False,  # Deterministic for evaluation
            pad_token_id=tokenizer.pad_token_id
        )

        # print("Token id output:", outputs[0])

        if apply_template:
            full_batch_results = tokenizer.batch_decode(
                outputs.detach().cpu().numpy(), skip_special_tokens=False
            )
        else:
            full_batch_results = tokenizer.batch_decode(
                outputs.detach().cpu().numpy(), skip_special_tokens=True
            )
        # Trim the generated text to remove the input part.
        # print("Raw output:", full_batch_results[0])

        if apply_template:
            if 'qwen' in model_name.lower():
                batch_results = [extract_qwen_response(result) for result, instruct in
                                 zip(full_batch_results, batch_instructs)]
            elif "olmoe" in model_name.lower():
                batch_results = [extract_olmoe_response(result) for result, instruct in
                                 zip(full_batch_results, batch_instructs)]
            else:
                raise NotImplementedError

        else:
            batch_results = [result[len(instruct):] for result, instruct in zip(full_batch_results, batch_instructs)]

        # print(batch_instructs)
        # print("Trimmed batch output:", batch_results[0])
        # print(full_batch_results)
        # exit(-1)

    for label, instruct, result in zip(batch_labels, batch_instructs, batch_results):
        results_json = {'label': label, 'text': instruct, 'rationale': result}
        # print(results_json['rationale'])
        output_file.write(f"{json.dumps(results_json)}\n")

    return batch_results


def main():
    # Define model and dataset names.
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', type=str, default="MATH",
                        help='which dataset to fine-tune')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen1.5-MoE-A2.7B",
                        help='which model to fine-tune')
    parser.add_argument('--no_template', action='store_true')
    parser.add_argument('--cache_dir', type=str, default="/fs/clip-scratch/tonyzhou/moe_reward/qwen1.5_moe")

    args = parser.parse_args()
    ds = args.ds
    model_name = args.model_name
    cache_dir = args.cache_dir

    prompts, labels = get_eval_data(ds)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if adapter_config_exists(model_name):
        print("use lora checkpoint")
        with open(f"{model_name}/adapter_config.json", 'r') as file:
            data = json.load(file)
        pretrained_ckpt = data["base_model_name_or_path"]
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_ckpt,
            trust_remote_code=True,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            cache_dir=None if 'tonyzhou' not in args.model_name else args.cache_dir
        )
        model = PeftModel.from_pretrained(base_model, model_name)
        model = model.merge_and_unload()

    else:
        print("use full parameter checkpoint")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=None if 'tonyzhou' not in args.model_name else args.cache_dir,
            torch_dtype=torch.bfloat16,  # Use BF16 to save memory.
            device_map="cuda:0"
        )

    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    results = []
    added_labels = []

    batch_size = args.batch_size
    if not os.path.exists(f"{ds}_result"):
        os.makedirs(f"{ds}_result")

    if model_name[-1] == "/":
        model_name = model_name[: -1]

    result_output_file = open(f"{ds}_result/{model_name.split('/')[-1]}_test_result.jsonl", 'w')

    apply_template = False

    if "ppo" in model_name or "grpo" in model_name or "rlhf" in model_name \
            or "SFT" in model_name or "Instruct" in model_name:
        apply_template = True

    if args.no_template:
        apply_template = False

    if apply_template:
        print("train with chat template, use apply_chat_template in tokenizer to test.")

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_instructs = prompts[i: i + batch_size]
        batch_labels = labels[i: i + batch_size]
        try:
            results.extend(
                process_batch(model, tokenizer, batch_instructs, batch_labels, ds, result_output_file, model_name,
                              apply_template=apply_template))
            added_labels.extend(batch_labels)
        except Exception as e:
            print(e)
            continue

    if ds == "MATH" or ds == "gsm8k" or ds == "MATH-500":
        results = [extract_math_answer(r) for r in results]
        labels = [extract_math_answer(r) for r in added_labels]
        print(f"accuracy: {accuracy_score(labels, results)}")
    elif ds == "imdb":
        verbalized_results = []
        for gen_answer in results:
            gen_answer = gen_answer.replace("negative", "Negative").replace("positive", "Positive")
            temp = re.findall(r'\b(Negative|Positive)\b', gen_answer)
            if len(temp) != 0:
                verbalized_results.append(temp[0])
            else:
                verbalized_results.append(random.choice(["Negative", "Positive"]))
        print(f"accuracy: {accuracy_score(added_labels, verbalized_results)}")

    elif ds == "nq":
        hit = 0
        for r, l in zip(results, added_labels):
            normalized_text = normalize_answer(r)
            answer_list = l.split(";")
            for single_answer in answer_list:
                if normalize_answer(single_answer) in normalized_text:
                    hit += 1
                    break

        print(f"accuracy: {hit / len(results)}")

    elif ds == "boolq" or ds == "arc" or ds == "sqa":
        verbalized_results = []
        for r in results:
            r = r.split(".")[0]
            if ds == "arc":
                temp = re.findall(r'A|B|C|D|E|F|G|H', r)
            else:
                r.replace("yes", "Yes")
                r.replace("no", "No")
                temp = re.findall(r'Yes|No', r)
            if len(temp) != 0:
                verbalized_results.append(temp[-1])
            else:
                if ds == "arc":
                    verbalized_results.append(random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']))
                else:
                    verbalized_results.append(random.choice(["Yes", "No"]))
        print(f"accuracy: {accuracy_score(added_labels, verbalized_results)}")


if __name__ == '__main__':
    main()
