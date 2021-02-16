# BERT-ranking
A simple BERT-based ranking model. The score of a query-document-pair is computed from the output corresponding to the CLS token of BERT.

## Requirements
This code is tested with Python 3.8.3 and
* torch==1.7.1
* transformers==4.0.1
* pytorch-lightning==1.1.2
* h5py==2.10.0
* numpy==1.18.5
* tqdm==4.48.0

## Cloning
Clone this repository using `git clone --recursive` to get the submodule.

## Usage
The following datasets are currently supported:
* [ANTIQUE](https://ciir.cs.umass.edu/downloads/Antique/)
* [FiQA Task 2](https://sites.google.com/view/fiqa/home)
* [InsuranceQA V2](https://github.com/shuzi/insuranceQA)
* [TREC-DL 2019 Passage Ranking](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019)

### Preprocessing
First, preprocess your dataset:
```
usage: preprocess.py [-h] [--num_negatives NUM_NEGATIVES]
                     [--pw_num_negatives PW_NUM_NEGATIVES] [--pw_query_limit PW_QUERY_LIMIT]
                     [--random_seed RANDOM_SEED]
                     SAVE {antique,fiqa,insuranceqa,trecdl2019passage} ...

positional arguments:
  SAVE                  Where to save the results
  {antique,fiqa,insuranceqa,trecdl2019passage}
                        Choose a dataset

optional arguments:
  -h, --help            show this help message and exit
  --num_negatives NUM_NEGATIVES
                        Number of negatives per positive (pointwise training) (default: 1)
  --pw_num_negatives PW_NUM_NEGATIVES
                        Number of negatives per positive (pairwise training) (default: 16)
  --pw_query_limit PW_QUERY_LIMIT
                        Maximum number of training examples per query (pairwise training)
                        (default: 64)
  --random_seed RANDOM_SEED
                        Random seed (default: 123)
```

### Training and Evaluation
Use the training script to train a new model and save checkpoints:
```
usage: train.py [-h] [--accumulate_grad_batches ACCUMULATE_GRAD_BATCHES]
                [--max_epochs MAX_EPOCHS] [--gpus GPUS [GPUS ...]]
                [--val_check_interval VAL_CHECK_INTERVAL] [--save_top_k SAVE_TOP_K]
                [--limit_val_batches LIMIT_VAL_BATCHES]
                [--limit_train_batches LIMIT_TRAIN_BATCHES]
                [--limit_test_batches LIMIT_TEST_BATCHES] [--precision {16,32}]
                [--accelerator ACCELERATOR] [--bert_type BERT_TYPE] [--bert_dim BERT_DIM]
                [--dropout DROPOUT] [--lr LR] [--loss_margin LOSS_MARGIN]
                [--batch_size BATCH_SIZE] [--warmup_steps WARMUP_STEPS]
                [--training_mode {pointwise,pairwise}] [--rr_k RR_K]
                [--num_workers NUM_WORKERS] [--val_patience VAL_PATIENCE]
                [--save_dir SAVE_DIR] [--random_seed RANDOM_SEED]
                [--load_weights LOAD_WEIGHTS] [--test]
                DATA_DIR FOLD_NAME

positional arguments:
  DATA_DIR              Folder with all preprocessed files
  FOLD_NAME             Name of the fold (within DATA_DIR)

optional arguments:
  -h, --help            show this help message and exit
  --accumulate_grad_batches ACCUMULATE_GRAD_BATCHES
                        Update weights after this many batches (default: 1)
  --max_epochs MAX_EPOCHS
                        Maximum number of epochs (default: 20)
  --gpus GPUS [GPUS ...]
                        GPU IDs to train on (default: None)
  --val_check_interval VAL_CHECK_INTERVAL
                        Validation check interval (default: 1.0)
  --save_top_k SAVE_TOP_K
                        Save top-k checkpoints (default: 1)
  --limit_val_batches LIMIT_VAL_BATCHES
                        Use a subset of validation data (default: 9223372036854775807)
  --limit_train_batches LIMIT_TRAIN_BATCHES
                        Use a subset of training data (default: 9223372036854775807)
  --limit_test_batches LIMIT_TEST_BATCHES
                        Use a subset of test data (default: 9223372036854775807)
  --precision {16,32}   Floating point precision (default: 32)
  --accelerator ACCELERATOR
                        Distributed backend (accelerator) (default: ddp)
  --bert_type BERT_TYPE
                        BERT model (default: bert-base-uncased)
  --bert_dim BERT_DIM   BERT output dimension (default: 768)
  --dropout DROPOUT     Dropout percentage (default: 0.1)
  --lr LR               Learning rate (default: 3e-05)
  --loss_margin LOSS_MARGIN
                        Margin for pairwise loss (default: 0.2)
  --batch_size BATCH_SIZE
                        Batch size (default: 32)
  --warmup_steps WARMUP_STEPS
                        Number of warmup steps (default: 1000)
  --training_mode {pointwise,pairwise}
                        Training mode (default: pairwise)
  --rr_k RR_K           Compute MRR@k (validation) (default: 10)
  --num_workers NUM_WORKERS
                        Number of DataLoader workers (default: 16)
  --val_patience VAL_PATIENCE
                        Validation patience (default: 3)
  --save_dir SAVE_DIR   Directory for logs, checkpoints and predictions (default: out)
  --random_seed RANDOM_SEED
                        Random seed (default: 123)
  --load_weights LOAD_WEIGHTS
                        Load pre-trained weights before training (default: None)
  --test                Test the model after training (default: False)
```
Use the `--test` argument to run the model on the testset using the best checkpoint after training.
