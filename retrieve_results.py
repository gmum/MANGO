import argparse
import json
import os
import traceback
from collections import OrderedDict

import language_tool_python
import numpy as np
import torch
import torch.nn.functional as F
import transformers

from src.attacks import USE
from src.utils import load_gpt2_from_dict


def get_bert_scores(output_path: str):
    with open(output_path, 'r') as fp:
        lines = [x.rstrip() for x in fp.readlines() if x.startswith('Bert scores:')]
        lines = [x.replace('Bert scores: precision ', '').replace('recall: ', '').replace('f1: ', '') for x in lines]
        scores = [x.split(', ') for x in lines]
    precision, recall, f1_list = zip(*scores)
    f1 = np.mean(list(map(float, f1_list)))
    f1_std = np.std(list(map(float, f1_list)))
    support = len(lines)

    return f1, f1_std, support


def _retrieve_texts(output_path: str):
    with open(output_path, 'r') as fp:
        lines = [x.strip() for x in fp.readlines()]
        clean_indices = [i for i, x in enumerate(lines) if x.startswith('CLEAN TEXT')]
        lines = [x.lower() for x in lines]
        if lines[clean_indices[0] + 1].startswith('premise:'):
            clean_texts = [{'premise': lines[i + 1].replace("premise: ", "", 1),
                            'hypothesis': lines[i + 2].replace("hypothesis: ", "", 1)}
                           for i in clean_indices]
            adv_texts = [{'premise': lines[i + 4].replace("premise: ", "", 1),
                          'hypothesis': lines[i + 2].replace("hypothesis: ", "", 1)}
                         for i in clean_indices]
        elif lines[clean_indices[0] + 1].startswith('premise : '):
            clean_texts = [lines[i + 1].replace("premise : ", "", 1).replace("hypothesis : ", "", 1) for i in
                           clean_indices]
            adv_texts = [lines[i + 3].replace("premise : ", "", 1).replace("hypothesis : ", "", 1) for i in
                         clean_indices]
        else:
            clean_texts = [lines[i + 1] for i in clean_indices]
            adv_texts = [lines[i + 3] for i in clean_indices]

    return clean_texts, adv_texts


def recalculate_cosine_scores(output_path: str, use):
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')

    clean_texts, adv_texts = _retrieve_texts(output_path)

    similarities = []
    for clean, adv in zip(clean_texts, adv_texts):
        if isinstance(clean, dict) and isinstance(adv, dict):
            clean = f'{clean["premise"]} {clean["hypothesis"]}'
            adv = f'{adv["premise"]} {adv["hypothesis"]}'
        clean = tokenizer.decode(tokenizer.encode(clean)[1:-1])
        adv = tokenizer.decode(tokenizer.encode(adv)[1:-1])
        similarities.append(use.compute_sim([clean], [adv]))

    sim = np.mean(similarities)
    sim_std = np.std(similarities)
    return sim, sim_std


def recalculate_perpl_scores(output_path: str, gpt2, device):
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size

    def calculate_perpl(text):
        batch = tokenizer([text],
                          return_tensors='pt',
                          max_length=512,
                          truncation=True).to(device)
        input_ids = batch['input_ids'].squeeze(0)
        one_hot_encodings = torch.zeros(size=(len(input_ids), vocab_size)).float().to(device)
        for i, _id in enumerate(input_ids):
            one_hot_encodings[i, _id] = 1.0
        with torch.no_grad():
            output = gpt2(**batch)
        one_hot_encodings = one_hot_encodings.unsqueeze(0)
        logits = output['logits']
        shift_logits = logits[:, :-1, :].contiguous()
        shift_coeffs = one_hot_encodings[:, 1:, :].contiguous()
        shift_logits = shift_logits[:, :, :shift_coeffs.size(2)]
        log_perpl = -(shift_coeffs * F.log_softmax(shift_logits, dim=-1)).sum(-1).mean(-1)
        return torch.exp(log_perpl).cpu().item()

    clean_texts, adv_texts = _retrieve_texts(output_path)

    gains = []
    for clean, adv in zip(clean_texts, adv_texts):
        if isinstance(clean, dict) and isinstance(adv, dict):
            clean = f'{clean["premise"]} {clean["hypothesis"]}'
            adv = f'{adv["premise"]} {adv["hypothesis"]}'
        clean_perpl = calculate_perpl(clean)
        adv_perpl = calculate_perpl(adv)
        gain = (adv_perpl - clean_perpl) / clean_perpl
        gains.append(gain)

    gain = np.mean(gains) * 100
    gain_std = np.std(gains) * 100
    return gain, gain_std


def _get_dataset_name(output_path: str):
    for d in dataset_to_labels.keys():
        if d in output_path:
            return d
    raise ValueError()


dataset_to_labels = {
    'ag_news': 4,
    'yelp': 2,
    'imdb': 2,
    'mnli_hypothesis': 3,
    'mnli_premise': 3
}

chunk_id = 0
chunk_size = 1000
start_index = chunk_id * chunk_size
end_index = start_index + chunk_size


def _get_labels_and_skips(goal_function, dataset_name):
    from textattack.goal_function_results import GoalFunctionResultStatus
    from textattack.shared import AttackedText
    from src.dataset import load_data

    dataset, num_labels = load_data(dataset_name, seed=0)
    dataset = dataset["train"]

    def _get_example(input):
        if 'mnli' in dataset_name:
            x = OrderedDict({'premise': input['premise'], 'hypothesis': input['hypothesis']})
            return x, (input['label'] + 1) % 3
        else:
            return input['text'], input['label']

    samples = [_get_example(dataset[i]) for i in range(start_index, end_index)]
    labels = [x[1] for x in samples]
    init_results = [goal_function.init_attack_example(AttackedText(text), label)[0] for text, label in samples]
    skips = [r.goal_status == GoalFunctionResultStatus.SKIPPED for r in init_results]
    return labels, skips


def get_base_results(result_path: str):
    try:
        with open(result_path, 'r') as fp:
            result = json.load(fp)
            keys = ['original_accuracy', 'adv_accuracy', 'success_rate', 'queries', 'similarities']
            results_dict = {}
            for k in keys:
                if isinstance(result[k], list):
                    results_dict[k] = np.mean(result[k])
                    results_dict[f'{k}_std'] = np.std(result[k])
                else:
                    results_dict[k] = result[k]
            return results_dict
    except:
        return {}


def retrieve_ground_truth_probability(output_path):
    with open(output_path, 'r') as fp:
        lines = [x.strip() for x in fp.readlines()]
        initial_idx = [i for i, x in enumerate(lines) if x.startswith('Initial logits tensor')]
    probs = []
    for ids in initial_idx:
        label_line = lines[ids - 2]
        final_line = lines[ids + 1]
        if 'skipped' in label_line.lower():
            continue
        label = int(label_line[0])
        final_list = eval(final_line.replace('Final logits tensor([', '').replace("], device='cuda:0')", ""))
        final_tensor = torch.tensor(final_list)
        final_pred = torch.softmax(final_tensor, dim=-1)
        probs.append(final_pred[label])
    prob = np.mean(probs) * 100
    prob_std = np.std(probs) * 100
    return prob, prob_std


def calculate_delta_grammar(output_path: str):
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
    tool = language_tool_python.LanguageTool('en-US')
    tool.disabled_categories.update({'TYPOS', 'CASING', 'TYPOGRAPHY'})
    clean_texts, adv_texts = _retrieve_texts(output_path)
    delta_errors = []
    for clean, adv in zip(clean_texts, adv_texts):
        if isinstance(clean, dict) and isinstance(adv, dict):
            clean = f'{clean["premise"]} {clean["hypothesis"]}'
            adv = f'{adv["premise"]} {adv["hypothesis"]}'
        clean = tokenizer.decode(tokenizer.encode(clean)[1:-1])
        adv = tokenizer.decode(tokenizer.encode(adv)[1:-1])
        clean_errors = tool.check(clean)
        adv_errors = tool.check(adv)
        delta_errors.append(len(adv_errors) - len(clean_errors))

    delta = np.mean(delta_errors)
    delta_std = np.std(delta_errors)
    return delta, delta_std


def pretty_print_results(results_dict):
    key_str = ', '.join(results_dict.keys())
    for k, v in results_dict.items():
        if isinstance(v, tuple):
            print(f'Tuple for {k}: {v}')
    values_str = ', '.join([f'{round(v, 3)}' if v else '-' for v in results_dict.values()])
    print(key_str)
    print(values_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir_path', required=True, type=str)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    dir_path = args.dir_path
    device = args.device

    use = USE()
    gpt2 = load_gpt2_from_dict("models/transformer_wikitext-103.pth",
                               output_hidden_states=True).to(device)
    gpt2.eval()

    for root, _, files in os.walk(dir_path):
        if 'output.txt' in files:
            print('#' * 50)
            print(root)
            output_path = os.path.join(root, 'output.txt')
            results_path = os.path.join(root, 'results.json')
            try:
                base_results = get_base_results(results_path)
                f1, f1_std, support = get_bert_scores(output_path)
                base_results['bert_score'] = f1
                base_results['bert_score_std'] = f1_std
                cos, cos_std = recalculate_cosine_scores(output_path, use)
                base_results['cos'] = cos
                base_results['cos_std'] = cos_std
                perpl, perpl_std = recalculate_perpl_scores(output_path, gpt2, device)
                base_results['delta_perpl'] = perpl
                base_results['delta_perpl_std'] = perpl_std
                ground_truth_prob, ground_truth_prob_std = retrieve_ground_truth_probability(output_path)
                base_results['ground_truth_prob'] = ground_truth_prob
                base_results['ground_truth_prob_std'] = ground_truth_prob_std
                delta_grammar, delta_grammar_std = calculate_delta_grammar(output_path)
                base_results['delta_grammar'] = delta_grammar
                base_results['delta_grammar_std'] = delta_grammar_std
                keys = ['adv_accuracy', 'ground_truth_prob', 'ground_truth_prob_std', 'cos', 'cos_std', 'bert_score',
                        'bert_score_std', 'delta_perpl', 'delta_perpl_std', 'delta_grammar', 'delta_grammar_std',
                        'queries', 'queries_std']
                final_results = {k: base_results[k] if k in base_results else None for k in keys}
                print(f'Support: {support}')
                pretty_print_results(final_results)
            except Exception as e:
                print(f'Some exception occured! {e}')
                traceback.print_exc()
