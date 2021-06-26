from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple

import torch
from transformers import BertModel, AdamW, get_constant_schedule_with_warmup

from ranking_utils.lightning.base_ranker import BaseRanker

from model.datasets import PointwiseTrainDataset, PairwiseTrainDataset, ValTestDataset, Batch


class BertRanker(BaseRanker):
    """Vanilla BERT ranker.

    Args:
        hparams (Dict[str, Any]): All model hyperparameters
    """
    def __init__(self, hparams: Dict[str, Any]):
        train_ds = None
        if hparams.get('training_mode') == 'pointwise':
            train_ds = PointwiseTrainDataset(hparams['data_file'], hparams['train_file_pointwise'], hparams['bert_type'])
        elif hparams.get('training_mode') == 'pairwise':
            train_ds = PairwiseTrainDataset(hparams['data_file'], hparams['train_file_pairwise'], hparams['bert_type'])
        val_ds = None
        if hparams.get('val_file') is not None:
            val_ds = ValTestDataset(hparams['data_file'], hparams['val_file'], hparams['bert_type'])
        test_ds = None
        if hparams.get('test_file') is not None:
            test_ds = ValTestDataset(hparams['data_file'], hparams['test_file'], hparams['bert_type'])

        num_workers = hparams.get('num_workers')
        super().__init__(hparams, train_ds, val_ds, test_ds, hparams['loss_margin'], hparams['batch_size'], num_workers)

        self.bert = BertModel.from_pretrained(hparams['bert_type'], return_dict=True)
        self.dropout = torch.nn.Dropout(hparams['dropout'])
        self.classification = torch.nn.Linear(hparams['bert_dim'], 1)

        for p in self.bert.parameters():
            p.requires_grad = not hparams['freeze_bert']

    def forward(self, batch: Batch) -> torch.Tensor:
        """Compute the relevance scores for a batch.

        Args:
            batch (Batch): BERT inputs

        Returns:
            torch.Tensor: The output scores, shape (batch_size, 1)
        """
        cls_out = self.bert(*batch)['last_hidden_state'][:, 0]
        return self.classification(self.dropout(cls_out))

    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        """Create an AdamW optimizer using constant schedule with warmup.

        Returns:
            Tuple[List[Any], List[Any]]: The optimizer and scheduler
        """
        params_with_grad = filter(lambda p: p.requires_grad, self.parameters())
        opt = AdamW(params_with_grad, lr=self.hparams['lr'])
        sched = get_constant_schedule_with_warmup(opt, self.hparams['warmup_steps'])
        return [opt], [{'scheduler': sched, 'interval': 'step'}]

    @staticmethod
    def add_model_specific_args(ap: ArgumentParser):
        """Add model-specific arguments to the parser.

        Args:
            ap (ArgumentParser): The parser
        """
        ap.add_argument('--bert_type', default='bert-base-uncased', help='BERT model')
        ap.add_argument('--bert_dim', type=int, default=768, help='BERT output dimension')
        ap.add_argument('--dropout', type=float, default=0.1, help='Dropout percentage')
        ap.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
        ap.add_argument('--loss_margin', type=float, default=0.2, help='Margin for pairwise loss')
        ap.add_argument('--batch_size', type=int, default=32, help='Batch size')
        ap.add_argument('--warmup_steps', type=int, default=1000, help='Number of warmup steps')
        ap.add_argument('--freeze_bert', action='store_true', help='Do not update any weights of BERT (only train the classification layer)')
        ap.add_argument('--training_mode', choices=['pointwise', 'pairwise'], default='pairwise', help='Training mode')
        ap.add_argument('--num_workers', type=int, default=16, help='Number of DataLoader workers')
