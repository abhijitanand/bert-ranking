#! /usr/bin/env python3


import os
import argparse

from ranking_utils.util import create_temp_testset, rank

from model.bert import BertRanker
from model.datasets import ValTestDataset


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('DATA_FILE', help='Preprocessed file containing queries and documents')
    ap.add_argument('CHECKPOINT', help='Model checkpoint')
    ap.add_argument('RUNFILE', help='Runfile to re-rank (TREC format)')
    ap.add_argument('--out_file', default='result.tsv', help='Output TREC runfile')
    ap.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = ap.parse_args()

    print(f'loading {args.CHECKPOINT}...')
    kwargs = {'data_file': None, 'train_file': None, 'val_file': None, 'test_file': None,
              'training_mode': None, 'rr_k': None, 'num_workers': None, 'freeze_bert': True}
    model = BertRanker.load_from_checkpoint(args.CHECKPOINT, **kwargs)

    print('creating temporary testset...')
    fd, f = create_temp_testset(args.DATA_FILE, args.RUNFILE)
    ds = ValTestDataset(args.DATA_FILE, f, model.hparams['bert_type'])

    rank(model, ds, args.out_file, args.batch_size)

    print(f'removing {f}...')
    os.close(fd)
    os.remove(f)


if __name__ == '__main__':
    main()
