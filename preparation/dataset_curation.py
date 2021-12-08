"""
    Curation of datasets used in the paper
"""
import json
import os.path
import random

from tqdm import tqdm
import pandas as pd
import nltk
import logging
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
import datasets

from utils.log import setup_logger, log_map
from utils.tool_funcs import overlap
from configs import _seed, _size, _n_fold, _dataset_store, _movieqa_squad_json, _kgqa_squad_json, _squad_json

random.seed(_seed)

# disable globally the tqdm inside datasets loading
datasets.set_progress_bar_enabled(False)

logger = logging.getLogger(__name__)
setup_logger(logger)


def read_jsonlines(filename):
    jsonlines = []
    with open(filename) as f:
        for line in f:
            jsonlines.append(json.loads(line))
    return jsonlines


def write_jsonlines(json_objects, filename):
    with open(filename, 'w') as f:
        for jsonObj in json_objects:
            json.dump(jsonObj, f)
            f.write("\n")


def convert_jsonlines_to_dataset(filename) -> Dataset:
    json_objects = read_jsonlines(filename)
    return Dataset.from_pandas(pd.DataFrame.from_records(json_objects))


def squad_json_to_dataset(filename: str):
    """
        Dataset.from_json may cause errors
    """
    with open(filename) as f:
        data = json.load(f)["data"]
    questions, answers, contexts = [], [], []
    for idx, d in enumerate(data):
        for para in d["paragraphs"]:
            contexts.append(para["context"])
            for qa in para["qas"]:
                questions.append(qa["question"])
                answers.append({
                    "text": [qa["answers"][0]["text"]],
                    "answer_start": [qa["answers"][0]["answer_start"]]
                })
    df = pd.DataFrame({
        "question": questions,
        "context": contexts,
        "answers": answers,
    })
    return Dataset.from_pandas(df)


def _k_folds(ds):
    """
        Cross-Validation Setting
    :return:
    """
    splits = [i for i in range(_n_fold)]
    folds = {}
    for fold in splits:
        test_split = ds.shard(_n_fold, fold)
        train_split = concatenate_datasets([ds.shard(_n_fold, idx) for idx in splits if idx != fold])
        folds[f"fold-{fold + 1}"] = DatasetDict(
            {
                "train": train_split,
                "test": test_split
            }
        )
    return folds


def squad_curation():
    _dataset_name = "squad"
    logger.info(f"Prepare the {_dataset_name} dataset")
    squad_v1 = load_dataset(_dataset_name, split="train").shuffle(seed=_seed)
    squad_v1.remove_columns(["id", 'title']).to_json(os.path.join(_dataset_store, "squad_v1.json"))


def covidqa_curation():
    """
        covidqa doesn't contain the dev/test set.  => 5-fold cross-validation
    :return:
    """
    _dataset_name = "covid_qa_deepset"
    logger.info(f"Prepare the {_dataset_name} dataset")
    covidqa = load_dataset(_dataset_name, split="train").shuffle(seed=_seed)
    covidqa_subset = covidqa.select([i for i in range(_size)])
    return _k_folds(covidqa_subset.remove_columns(["document_id", "is_impossible", "id"]))


def cuadqa_curation():
    """
        Contract understanding datasets
    :return:
    """

    def is_question(q_str):
        wh_h_terms = ["who", "what", "when", "where", "why", "how", "is", "can", "does", "do"]
        words = nltk.word_tokenize(q_str)
        return overlap(wh_h_terms, words)

    def has_answer(answers_dict):
        return len(answers_dict["text"]) != 0

    _dataset_name = "cuad"
    logger.info(f"Prepare the {_dataset_name} dataset")
    # sample from the train set
    cuadqa = load_dataset(_dataset_name, split="train").shuffle(seed=_seed)
    cuadqa_subset = cuadqa.filter(
        # contain question wh-, how, ... and must be answerable
        lambda example: is_question(example["question"]) and has_answer(example["answers"])
    ).select([i for i in range(_size)])
    return _k_folds(cuadqa_subset.remove_columns(["id", "title"]))


def movieqa_curation():
    """
        MovieQA selfRC part does not contain the answer_start field
        We use a subset (2000 examples) manually annotated by people
        at least contain a master degree in Computer Science.
    :return:
    """
    _dataset_name = "duorc"
    logger.info(f"Prepare the {_dataset_name} dataset")
    movieqa_subset = squad_json_to_dataset(os.path.join(_dataset_store, _movieqa_squad_json)).shuffle(seed=_seed)
    return _k_folds(movieqa_subset)


def kgqa_curation():
    """
    """
    _dataset_name = "kgqa"
    logger.info(f"Prepare the {_dataset_name} dataset")
    kgqa_subset = squad_json_to_dataset(os.path.join(_dataset_store, _kgqa_squad_json)).shuffle(seed=_seed)
    return _k_folds(kgqa_subset)


def load_ds(dataset_name: str):
    if dataset_name == "covidqa":
        return covidqa_curation()
    elif dataset_name == "movieqa":
        return movieqa_curation()
    elif dataset_name == "cuadqa":
        return cuadqa_curation()
    elif dataset_name == "kgqa":
        return kgqa_curation()


def store_datasets_in_folds():
    """
        Store the folds of max_budget on disk
    """
    ds_list = [covidqa_curation(), movieqa_curation(), cuadqa_curation(), kgqa_curation()]
    # create the folders to put the folds
    _ = [os.makedirs(os.path.join(_dataset_store, name), exist_ok=True) for name in ds_names]
    _ = [os.makedirs(os.path.join(_dataset_store, name, f"budget-{max_budget_size}"), exist_ok=True) for name in
         ds_names]

    pbar = tqdm(zip(ds_list, ds_names), total=len(ds_names))
    for ds, name in pbar:
        pbar.set_description(f"Process and store dataset: {name}")
        for fold in range(_n_fold):
            train_split = ds[f"fold-{fold + 1}"]["train"]  # type: Dataset
            test_split = ds[f"fold-{fold + 1}"]["test"]  # type: Dataset

            train_split.to_json(
                os.path.join(_dataset_store, name, f"budget-{max_budget_size}", f"fold-{fold + 1}-train.json"),
            )
            test_split.to_json(
                os.path.join(_dataset_store, name, f"budget-{max_budget_size}", f"fold-{fold + 1}-test.json")
            )


def simulate_budgets():
    """
        sample from the budget-1600/fold-*-train.json
        use the same fold-*-test.json for evaluation
    """
    for ds_name in ds_names:
        for budget in budgets:
            os.makedirs(os.path.join(_dataset_store, ds_name, f"budget-{budget}"), exist_ok=True)
            for fold in range(_n_fold):
                ds = Dataset.from_json(
                    os.path.join(_dataset_store, ds_name, f"budget-{max_budget_size}", f"fold-{fold + 1}-train.json"),
                ).shuffle(seed=_seed)
                budget_ds = ds.select([i for i in range(budget)])
                budget_ds.to_json(
                    os.path.join(_dataset_store, ds_name, f"budget-{budget}", f"fold-{fold + 1}-train.json")
                )


def merge_finetuning_preparation():
    squad_whole = read_jsonlines(os.path.join(_dataset_store, "squad_v1.json"))
    for ds_name in ds_names:
        for budget in budgets:
            squad_subset = random.Random(_seed).sample(squad_whole, budget)
            for fold in range(_n_fold):
                ds_pool = read_jsonlines(
                    os.path.join(_dataset_store, ds_name, f"budget-{budget}", f"fold-{fold + 1}-train.json"))

                # 1. MP
                # 2. MPO
                for factor in oversample_factors:
                    mpo = ds_pool * factor + squad_subset
                    random.Random(_seed).shuffle(mpo)
                    write_jsonlines(
                        mpo, os.path.join(_dataset_store, ds_name, f"budget-{budget}",
                                          f"mpo-factor-{factor}-fold-{fold + 1}-train.json")
                    )
                # 3. MW
                # 4. MWO
                for factor in oversample_factors:
                    mwo = ds_pool * factor + squad_whole
                    random.Random(_seed).shuffle(mwo)
                    write_jsonlines(
                        mwo, os.path.join(_dataset_store, ds_name, f"budget-{budget}",
                                          f"mwo-factor-{factor}-fold-{fold + 1}-train.json"))


if __name__ == '__main__':
    # settings
    oversample_factors = [1, 3]
    budgets = [100, 200, 400, 800, 1200]
    max_budget_size = (_n_fold - 1) * _size // _n_fold
    ds_names = ["COVIDQA", "MOVIEQA", "CUADQA", "KGQA"]

    tasks = [
        squad_curation,
        store_datasets_in_folds,
        simulate_budgets,
        merge_finetuning_preparation
    ]
    task_names = [
        "1. Caching SQuAD Dataset: ",
        "2. Preparing 5-fold for 4 datasets and store to disk: ",
        "3. Sampling from 5-fold and create budget training splits: ",
        "4. Preparing for merge fine-tuning (MP, MW, MPO, MWO): "
    ]
    assert len(tasks) == len(task_names)

    pbar = tqdm(zip(tasks, task_names), total=len(task_names))

    for do_task, task_name in pbar:
        log_map(logger, "Status", {"Task": task_name})
        do_task()
