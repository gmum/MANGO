# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#

from datasets import load_dataset


def load_data(dataset_name: str, seed: int):
    if dataset_name == "ag_news":
        dataset = load_dataset("ag_news")
        num_labels = 4
    elif dataset_name == "imdb":
        dataset = load_dataset("imdb", ignore_verifications=True)
        num_labels = 2
    elif dataset_name == "yelp":
        dataset = load_dataset("yelp_polarity")
        num_labels = 2
    elif "mnli" in dataset_name:
        dataset = load_dataset("glue", "mnli")
        num_labels = 3
    elif dataset_name == "test":
        dataset = {"train": [{"text": "Cat and mice.", "label": 1}]}
        num_labels = 2
        return dataset, num_labels
    dataset = dataset.shuffle(seed=seed)

    return dataset, num_labels
