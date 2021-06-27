#! /usr/bin/env python3


import os
import argparse
from pathlib import Path

from ranking_utils.util import create_temp_testsets, rank, write_trec_eval_file

from model.bert import BertRanker
from model.datasets import ValTestDataset


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('DATA_FILE', help='Preprocessed file containing queries and documents')
    ap.add_argument('--checkpoints', nargs='+', required=True, help='Model checkpoints')
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--runfiles', nargs='+', help='Corresponding runfiles to re-rank (TREC format)')
    group.add_argument('--testsets', nargs='+', help='Corresponding testsets to rank')
    ap.add_argument('--out_file', default='result.tsv', help='Output TREC runfile')
    ap.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = ap.parse_args()

    if args.runfiles:
        print('creating temporary testsets...')
        temp_testsets = create_temp_testsets(args.DATA_FILE, args.runfiles)
        testsets = [f for _, f in temp_testsets]
    else:
        testsets = map(Path, args.testsets)

    results = {}
    for ckpt, f in zip(args.checkpoints, testsets):
        print(f'loading {ckpt}...')
        kwargs = {'data_file': None, 'train_file': None, 'val_file': None, 'test_file': None,
                  'training_mode': None, 'rr_k': None, 'num_workers': None, 'freeze_bert': True}
        model = BertRanker.load_from_checkpoint(ckpt, **kwargs)
        ds = ValTestDataset(args.DATA_FILE, f, model.hparams['bert_type'])
        results.update(rank(model, ds, args.batch_size))

    print(f'writing {args.out_file}...')
    write_trec_eval_file(Path(args.out_file), results, 'bert')

    if args.runfiles:
        for fd, f in temp_testsets:
            print(f'removing {f}...')
            os.close(fd)
            os.remove(f)


if __name__ == '__main__':
    main()
