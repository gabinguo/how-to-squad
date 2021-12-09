# 📌 How to fine-tune the Reader in general

### 🔖 Requirements:

- Create virtual environments
    ```bash
    # virtualenv --python=<path to python(>=3.6)> ./venv
    virtualenv ./venv
    source ./venv/bin/activate
    ```
- Install pytorch [Link][1]
- Install Libraries
    ```bash
    pip3 install transformers==4.12.5
    pip3 install datasets==1.16.1
    pip3 install nltk
     ```

### 🔖 Dataset Curation

- CovidQA [Link][2]
- MovieQA [Link][3]
- CuadQA [Link][4]
- KGQA [Source][5]

```bash
  # activate the virtualenv
  python3 -m preparation.dataset_curation
```

### 🛠 Further pre-training a language model

```bash
python3 -m train.language_modeling.mlm \
            --model_name bert-base-uncased \
            --train_file ./train_in_lines.txt \
            --eval_file ./eval_in_lines.txt \
            --output_folder ./further_pretrained_lm \
            --batch_size 32 \
            --block_size 128 \
            --num_train_epochs 15
```

### 🛠 Fine-tune a model on QA datasets

```bash
export TRAIN_FILE=""
export TEST_FILE=""
export OUTPUT_DIR=""
python3 -m train.question_answering.finetune \
		   --train_file $TRAIN_FILE \
		   --test_file $TEST_FILE \
		   --model_ckpt roberta-base \
		   --output_dir $OUTPUT_DIR \
		   --batch_size 32 \
		   --num_epochs 2 \
		   --lr 2e-5
```

[1]: https://pytorch.org/get-started/locally/

[2]: https://huggingface.co/datasets/covid_qa_deepset

[3]: https://huggingface.co/datasets/covid_qa_deepset

[4]: https://huggingface.co/datasets/duorc

[5]: https://wikidata.org
