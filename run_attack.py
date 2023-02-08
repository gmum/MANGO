# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import csv
import json
import os
import pickle
import sys
import time
from collections import OrderedDict
from importlib import import_module

import textattack
import torch
import transformers
from bert_score import BERTScorer
from textattack.attack_recipes import BERTAttackLi2020, TextFoolerJin2019
from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult
from textattack.constraints.pre_transformation import InputColumnModification
from tqdm import tqdm

from search_methods.gradient_searchers.gradient_searcher import GradientSearcher
from show_results import get_log_dir
from src.attacks import build_baegarg2019, build_bert_attack, USE
from src.dataset import load_data
from src.utils import bool_flag

dataset_to_model = {
    'imdb': 'textattack/bert-base-uncased-imdb',
    'yelp': 'textattack/bert-base-uncased-yelp-polarity',
    'ag_news': 'textattack/bert-base-uncased-ag-news',
    'mnli_premise': 'textattack/bert-base-uncased-MNLI',
    'mnli_hypothesis': 'textattack/bert-base-uncased-MNLI',
    'test': 'textattack/bert-base-uncased-imdb'
}

dataset_to_idf_dict = {
    'imdb': 'idfs/imdb.pt',
    'yelp': 'idfs/yelp.pt',
    'ag_news': 'idfs/ag_news.pt',
    'mnli_premise': 'idfs/mnli.pt',
    'mnli_hypothesis': 'idfs/mnli.pt',
}


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=['ag_news', 'yelp', 'imdb', 'mnli_premise', 'mnli_hypothesis', 'test'],
                        required=True)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--log_dir", default=None, type=str)
    parser.add_argument("--combi_path", default=None, type=str)
    parser.add_argument("--data-folder", required=False, type=str)
    parser.add_argument("--dump_path", type=str)
    parser.add_argument("--print", action='store_true')
    parser.add_argument("--append", action='store_true')

    parser.add_argument("--target_dir", type=str)
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=1)

    # Attack parameters
    parser.add_argument("--attack", required=True, type=str)
    parser.add_argument("--bae-threshold", type=float, default=0.8)
    parser.add_argument("--query-budget", type=int, default=None)
    parser.add_argument("--radioactive", type=bool_flag)
    parser.add_argument("--targeted", type=bool_flag, default=True)
    parser.add_argument("--ckpt", type=str)

    return parser


def main(params):
    # Loading data
    log_dir = params.log_dir if params.log_dir else get_log_dir(params)
    os.makedirs(log_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset, num_labels = load_data(params.dataset, params.seed)
    dataset = dataset["train"]
    print(f"Loaded dataset {params.dataset}, that has {len(dataset)} rows")

    model, model_wrapper = get_model(device, num_labels, params.dataset, params.ckpt)

    # Create radioactive directions and modify classification layer to use those
    if params.radioactive:
        torch.manual_seed(params.seed)
        radioactive_directions = torch.randn(num_labels, 768)
        radioactive_directions /= torch.norm(radioactive_directions, dim=1, keepdim=True)
        print(radioactive_directions)
        model.classifier.weight.data = radioactive_directions.to(device)
        model.classifier.bias.data = torch.zeros(num_labels).to(device)

    start_index = params.chunk_id * params.chunk_size
    end_index = start_index + params.chunk_size

    # Creating attack
    print(f"Building {params.attack} attack")
    if params.attack == "custom":
        current_label = -1
        if params.targeted:
            current_label = dataset[start_index]['label']
            assert all([dataset[i]['label'] == current_label for i in range(start_index, end_index)])
        attack = build_bert_attack(model_wrapper, current_label)
    elif params.attack == "bae3":
        attack = build_baegarg2019(model_wrapper, threshold_cosine=0.2,
                                   query_budget=params.query_budget)
    elif params.attack == "bert-attack":
        assert params.query_budget is None
        attack = build_bert_attack(model_wrapper)
    elif params.attack == "text-fooler":
        assert params.query_budget is None
        attack = TextFoolerJin2019.build(model_wrapper)
    elif params.attack.endswith('.py'):
        module_name = params.attack.replace('.py', '').replace('/', '.')
        build = import_module(name=module_name).build_attack
        tensorboard_dir = f'{log_dir}/tensorboard'
        disable_column_name = None if 'mnli' not in params.dataset else params.dataset.split('_')[1]
        if params.combi_path:
            combination = torch.load(params.combi_path)
        else:
            combination = {}
        attack = build(model_wrapper, tensorboard_dir, disable_column_name, **combination)
        idf_dict = torch.load(dataset_to_idf_dict[params.dataset])
        attack.goal_function.add_idf_dict(idf_dict)
    else:
        raise NotImplementedError()

    if params.dataset == 'mnli_premise':
        constraint = InputColumnModification(["premise", "hypothesis"], {"hypothesis"})
        attack.pre_transformation_constraints.append(constraint)
    elif params.dataset == 'mnli_hypothesis':
        constraint = InputColumnModification(["premise", "hypothesis"], {"premise"})
        attack.pre_transformation_constraints.append(constraint)

    run_text_attack(attack, dataset, end_index, log_dir, model_wrapper, params.print, start_index, params.dataset)


def get_model(device, num_labels, dataset, ckpt=None):
    # Load model and tokenizer from HuggingFace
    model_class = transformers.AutoModelForSequenceClassification
    model_name = dataset_to_model[dataset]
    model = model_class.from_pretrained(model_name, num_labels=num_labels).to(device)
    if ckpt is not None:
        state_dict = torch.load(ckpt)
        model.load_state_dict(state_dict)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 512
    tokenizer.max_len = 512
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
    return model, model_wrapper


def run_text_attack(attack, dataset, end_index, log_dir, model_wrapper, do_print, start_index, dataset_name):
    def _get_example(input):
        if 'mnli' in dataset_name:
            x = OrderedDict({'premise': input['premise'], 'hypothesis': input['hypothesis']})
            return x, (input['label'] + 1) % 3
        else:
            return input['text'], input['label']

    # Launching attack
    begin_time = time.time()
    samples = [_get_example(dataset[i]) for i in range(start_index, end_index)]
    # Storing attacked text
    bert_scorer = BERTScorer(model_type="bert-base-uncased", idf=False)
    n_success = 0
    similarities = []
    queries = []
    use = USE()
    write_rows = []
    original_stdout = sys.stdout
    skipped_attacks = 0
    pbar = tqdm(enumerate(samples))
    n_failures = 0
    file_type = 'a' if params.append else 'w'
    with open(f'{log_dir}/output.txt', file_type) as output_file:
        if do_print:
            output_file = sys.stdout
        for i_result, sample in pbar:
            example, ground_truth = sample
            result = attack.attack(example, ground_truth)
            text = result.original_text()
            text = model_wrapper.tokenizer.decode(model_wrapper.tokenizer.encode(text)[1:-1])
            ptext = result.perturbed_text()
            ptext = model_wrapper.tokenizer.decode(model_wrapper.tokenizer.encode(ptext)[1:-1])
            i_data = start_index + i_result
            write_rows.append([dataset[i_data]['label'] + 1, ptext])

            precision, recall, f1 = [r.item() for r in bert_scorer.score([ptext], [text])]
            initial_logits = model_wrapper([text])
            final_logits = model_wrapper([ptext])

            sys.stdout = output_file
            print("")
            print(50 * "*")
            print("")
            print("True label ", dataset[i_data]['label'])
            print(f"CLEAN TEXT\n {text}")
            print(f"ADV TEXT\n {ptext}")
            if type(result) not in [SuccessfulAttackResult, FailedAttackResult]:
                print("WARNING: Attack neither succeeded nor failed...")
                skipped_attacks += 1
            print(result.goal_function_result_str())
            print(f"Bert scores: precision {precision:.2f}, recall: {recall:.2f}, f1: {f1:.2f}")
            print("Initial logits", initial_logits)
            print("Final logits", final_logits)
            print("Logits difference", final_logits - initial_logits)
            sys.stdout = original_stdout

            # Statistics
            n_success += 1 if type(result) is SuccessfulAttackResult else 0
            n_failures += 1 if type(result) is FailedAttackResult else 0
            queries.append(result.num_queries)
            similarities.append(use.compute_sim([text], [ptext]))

            pbar.set_postfix({'success/failures/skipped': f'{n_success}/{n_failures}/{skipped_attacks}'})

        sys.stdout = output_file
        print("Processing all samples took %.2f" % (time.time() - begin_time))
        original_accuracy = (len(samples) - skipped_attacks) / len(samples)
        adv_accuracy = n_failures / len(samples)
        success_rate = n_success / (len(samples) - skipped_attacks)
        avg_queries = sum(queries) / len(queries)
        avg_similarity = sum(similarities) / len(similarities)
        logs = {
            "original_accuracy": round(original_accuracy * 100, 1),
            "adv_accuracy": round(adv_accuracy * 100, 1),
            "success_rate": round(success_rate * 100, 1),
            "avg_queries": round(avg_queries),
            "queries": queries,
            "avg_similarity": round(avg_similarity, 3),
            "similarities": similarities,
        }
        print("__logs:" + json.dumps(logs))
        sys.stdout = original_stdout
    with open(f'{log_dir}/logs.csv', "w") as f:
        f = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        f.writerows(write_rows)
    with open(f'{log_dir}/results.json', "w") as f:
        json.dump(logs, f)

    return logs


if __name__ == "__main__":
    print("Using text attack from ", textattack.__file__)
    # Parse arguments
    parser = get_parser()
    params = parser.parse_args()
    # if not params.radioactive:
    #     assert params.ckpt is not None, "Should specify --ckpt if not radioactive."
    assert not (params.radioactive and not params.targeted), "Radioactive means targeted"

    # Run main code
    begin_time = time.time()
    main(params)
    print("Running program took %.2f" % (time.time() - begin_time))
