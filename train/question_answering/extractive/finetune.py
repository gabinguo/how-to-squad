import argparse
import json
import logging
import os.path

from preparation.dataset_processing import prepare_train_features, prepare_validation_features, \
    postprocess_qa_predictions
from preparation.dataset_curation import convert_jsonlines_to_dataset, write_jsonlines
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import TrainingArguments, Trainer, IntervalStrategy
from transformers import default_data_collator
from datasets import Dataset, DatasetDict, load_metric
from utils.log import setup_logger, log_map
import torch

data_collator = default_data_collator

logger = logging.getLogger(__name__)
setup_logger(logger)

num_gpus = torch.cuda.device_count()
log_map(logger, "GPU INFO", {"# of GPU Available": num_gpus})


def train_model(params):
    batch_size = params.batch_size
    num_epochs = params.num_epochs
    max_seq_len = params.max_seq_len
    doc_stride = params.doc_stride
    use_gpu = params.use_gpu
    output_dir = params.output_dir
    weight_decay = params.weight_decay
    model_ckpt = params.model_ckpt
    lr = params.lr

    device = torch.device("cuda") if use_gpu and torch.cuda.is_available() else torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_ckpt)
    tokenized_ds = dataset_dict.map(
        lambda examples: prepare_train_features(examples, tokenizer, max_seq_len, doc_stride), batched=True,
        remove_columns=dataset_dict["train"].column_names, desc="Tokenization (Train): "
    )

    training_args = TrainingArguments(
        output_dir,
        do_train=True,
        do_eval=False,
        evaluation_strategy=IntervalStrategy.NO,
        save_strategy=IntervalStrategy.NO,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay
    )
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=None,
        # eval_dataset=tokenized_ds["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)

    logger.info("Training finished.")

    # evaluation
    logger.info("Start evaluating on the test file... ")
    test_features = dataset_dict["test"].map(
        lambda examples: prepare_validation_features(examples, tokenizer, max_seq_len, doc_stride), batched=True,
        remove_columns=dataset_dict["test"].column_names, desc="Tokenization (Test): "
    )
    raw_predictions = trainer.predict(test_features)
    test_features.set_format(type=test_features.format["type"], columns=list(test_features.features.keys()))
    final_predictions = postprocess_qa_predictions(tokenizer, dataset_dict["test"], test_features,
                                                   raw_predictions.predictions,
                                                   max_answer_length=params.max_answer_length)
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in dataset_dict["test"]]
    result = metric.compute(predictions=formatted_predictions, references=references)
    logger.info("Evaluation finished.")
    log_map(logger, "EM & F1", result)

    logger.info("Save the results to disk")
    # write result to file for records
    write_jsonlines(formatted_predictions, os.path.join(output_dir, "predictions.json"))
    write_jsonlines([result], os.path.join(output_dir, "eval_metrics.json"))

    log_map(logger, "Status", {"-": "All Done :)"})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train arguments parser")
    parser.add_argument("--train_file", type=str, help="Filename to the train .json file")
    parser.add_argument("--test_file", type=str, help="Filename to the test .json file")
    parser.add_argument("--model_ckpt", type=str, help="Path to the model folder.")
    parser.add_argument("--output_dir", type=str, help="Path to put the trained model.")
    parser.add_argument("--doc_stride", type=int, default=128, help="Overlap between chunks.")
    parser.add_argument("--max_seq_len", type=int, default=384, help="Length limit for chunks.")
    parser.add_argument("--max_answer_length", type=int, default=48, help="Max length of the answers.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size of training.")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--use_gpu", type=bool, default=True, help="Flag of gpu / cpu")

    args = parser.parse_args()
    try:
        train_data = Dataset.from_json(args.train_file)
        test_data = Dataset.from_json(args.test_file)
    except Exception as e:
        logger.warning("Exception occurs when try to read .json using Dataset.from_json")
        logger.warning(e)
        train_data = convert_jsonlines_to_dataset(args.train_file)
        test_data = convert_jsonlines_to_dataset(args.test_file)

    dataset_dict = DatasetDict({
        "train": train_data,
        "test": test_data
    })

    # use squad metric for evaluating the EM and F1
    metric = load_metric("squad")

    train_model(args)
