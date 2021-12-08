import argparse
import logging

from preparation.dataset_processing import prepare_train_features, prepare_validation_features
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import TrainingArguments, Trainer, IntervalStrategy
from transformers import default_data_collator
from datasets import Dataset, DatasetDict
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

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModelForQuestionAnswering.from_pretrained(model_ckpt)
    train_features = dataset_dict["train"].map(
        lambda examples: prepare_train_features(examples, tokenizer, max_seq_len, doc_stride), batched=True,
        remove_columns=dataset_dict["train"].column_names, desc="Tokenization (Train): "
    )
    test_features = dataset_dict["test"].map(
        lambda examples: prepare_validation_features(examples, tokenizer, max_seq_len, doc_stride), batched=True,
        remove_columns=dataset_dict["test"].column_names, desc="Tokenization (Test): "
    )
    if num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=[index for index in range(num_gpus)]).cuda()
    elif use_gpu:
        model.to(device)

    training_args = TrainingArguments(
        output_dir,
        do_train=True,
        do_eval=False,
        evaluation_strategy=IntervalStrategy.NO,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay
    )
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_features,
        eval_dataset=None,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)

    # evaluation
    raw_predictions = trainer.predict(test_features)
    test_features.set_format(type=test_features.format["type"], columns=list(test_features.features.keys()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train arguments parser")
    parser.add_argument("--train_csv", type=str, help="Filename to the train .csv file")
    parser.add_argument("--test_csv", type=str, help="Filename to the test .csv file")
    parser.add_argument("--model_ckpt", type=str, help="Path to the model folder.")
    parser.add_argument("--output_dir", type=str, help="Path to put the trained model.")
    parser.add_argument("--doc_stride", type=int, default=128, help="Overlap between chunks.")
    parser.add_argument("--max_seq_len", type=int, default=384, help="Length limit for chunks.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size of training.")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--use_gpu", type=bool, default=True, help="Flag of gpu / cpu")

    args = parser.parse_args()
    dataset_dict = DatasetDict({
        "train": Dataset.from_csv(args.train_csv),
        "test": Dataset.from_csv(args.test_csv)
    })

    train_model(args)
