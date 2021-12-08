import logging
import argparse
from typing import Tuple, Union

from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForMaskedLM, PreTrainedTokenizer
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback, IntervalStrategy
from train.language_modeling.dataset_handler import LineByLineTextDataset
from utils.log import setup_logger, log_map
from configs import _seed

logger = logging.getLogger(__name__)
setup_logger(logger)


def _init_components(model_name: str) -> Tuple[Union[AutoTokenizer, PreTrainedTokenizer], AutoModelForMaskedLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    return tokenizer, model


def main(params: argparse.Namespace):
    # set random seed for reproduction
    set_seed(_seed)

    tokenizer, model = _init_components(params.model_name)

    train_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=params.train_file,
                                          block_size=params.block_size)
    # used for early stopping
    eval_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=params.eval_file, block_size=params.block_size)

    log_map(logger, "Train Info", {
        "Train (in lines)": len(train_dataset),
        "Test (in lines)": len(eval_dataset)
    })

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=params.mlm_probability
    )

    training_args = TrainingArguments(
        learning_rate=params.learning_rate,
        weight_decay=params.weight_decay,
        output_dir=params.output_folder,
        overwrite_output_dir=True,
        num_train_epochs=params.num_train_epochs,
        per_device_train_batch_size=params.batch_size,
        per_device_eval_batch_size=params.batch_size,
        evaluation_strategy=IntervalStrategy.EPOCH,
        eval_steps=params.save_steps,
        save_steps=params.save_steps,
        logging_steps=params.save_steps,
        prediction_loss_only=True,
        load_best_model_at_end=True
    )
    log_map(logger, "Training Argument", training_args.__dict__)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=params.early_stopping_patience)]
    )

    # kick off training
    logger.info("Masked Language Modeling (MLM) start...")
    trainer.train()

    # save the model
    logger.info(f"Save the model to {params.output_folder}")
    trainer.save_model(params.output_folder)
    tokenizer.save_pretrained(params.output_folder)


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description="Argument parser for training language models' parameters.")
    argument_parser.add_argument("--block_size", type=int, default=128, help="Block size to train the model.")
    argument_parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    argument_parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    argument_parser.add_argument("--save_steps", type=int, default=100_000,
                                 help="Save each time hits this steps number")
    argument_parser.add_argument("--mlm_probability", type=float, default=0.15, help="MLM probability")
    argument_parser.add_argument("--learning_rate", type=float, default=0, help="Learning Rate")
    argument_parser.add_argument("--weight_decay", type=float, default=3e-05, help="Weight Decay")
    argument_parser.add_argument("--early_stopping_patience", type=int, default=1, help="Early Stopping Patience")
    argument_parser.add_argument("--train_file", type=str, help="Txt file for training, content should be in lines.")
    argument_parser.add_argument("--eval_file", type=str, help="Txt file for evaluating, content should be in lines.")
    argument_parser.add_argument("--output_folder", type=str, help="Folder to output the logs and final model")
    argument_parser.add_argument("--model_name", type=str, default="roberta-base",
                                 help="Initial checkpoint for the training.")

    args = argument_parser.parse_args()
    main(args)

    """
    Train language model for Roberta-base 
    python train_lm.py --train_file train_file.txt \
                       --eval_file eval_file.txt \
                       --output_folder ./output_folder \
                       --batch_size 64
    """
