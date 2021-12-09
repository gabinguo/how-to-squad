import os

basedir: str = os.path.abspath(os.path.dirname(__file__))

# random seed
_seed: int = 6

# dataset configuration
_dataset_store: str = os.path.join(basedir, "dataset_store")
_movieqa_squad_json: str = "movieqa_2000.json"
_kgqa_squad_json: str = "kgqa_2000.json"
_squad_json: str = "train-v1.1.json"
_size: int = 2000
_n_fold: int = 5

# language modeling
_min_word_number_per_line: int = 8
