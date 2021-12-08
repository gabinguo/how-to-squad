import torch
import logging
import os
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer
from typing import Dict, Union

logger = logging.getLogger(__name__)


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: Union[PreTrainedTokenizer, AutoTokenizer], file_path: str, block_size: int = 512):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found."
        logger.info(f"Creating features from dataset file at {file_path}")

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if len(line) > 0 and not line.isspace()]

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index) -> Dict[str, torch.tensor]:
        return self.examples[index]