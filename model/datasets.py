from pathlib import Path
from typing import Iterable, Tuple

import torch
from transformers import BertTokenizer

from ranking_utils.lightning.datasets import PairwiseTrainDatasetBase, PointwiseTrainDatasetBase, ValTestDatasetBase


Input = Tuple[str, str]
Batch = Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]
PointwiseTrainInput = Tuple[Input, int]
PointwiseTrainBatch = Tuple[Batch, torch.FloatTensor]
PairwiseTrainInput = Tuple[Input, Input]
PairwiseTrainBatch = Tuple[Batch, Batch]
ValTestInput = Tuple[int, int, Input, int]
ValTestBatch = Tuple[torch.LongTensor, torch.LongTensor, Batch, torch.LongTensor]


def _get_single_input(query: str, doc: str, char_limit: int = 10000) -> Input:
    """Return a (query, document) pair for BERT, making sure the strings are not empty and don't exceed a number of characters.

    Args:
        query (str): The query
        doc (str): The document
        char_limit (int, optional): Maximum number of characters. Defaults to 10000.

    Returns:
        Input: Non-empty query and document
    """
    # empty queries or documents might cause problems later on
    if len(query.strip()) == 0:
        query = '(empty)'
    if len(doc.strip()) == 0:
        doc = '(empty)'

    # limit characters to avoid tokenization bottlenecks
    return query[:char_limit], doc[:char_limit]


def _collate_bert(inputs: Iterable[Input], tokenizer: BertTokenizer) -> Batch:
    """Tokenize and collate a number of single BERT inputs, adding special tokens and padding.

    Args:
        inputs (Iterable[Input]): The inputs
        tokenizer (BertTokenizer): Tokenizer

    Returns:
        Batch: Input IDs, attention masks, token type IDs
    """
    queries, docs = zip(*inputs)
    inputs = tokenizer(queries, docs, padding=True, truncation=True)
    return torch.LongTensor(inputs['input_ids']), \
           torch.LongTensor(inputs['attention_mask']), \
           torch.LongTensor(inputs['token_type_ids'])


class PointwiseTrainDataset(PointwiseTrainDatasetBase):
    """Dataset for pointwise training.

    Args:
        data_file (Path): Data file containing queries and documents
        train_file (Path): Trainingset file
        bert_type (str): Type for the tokenizer
    """
    def __init__(self, data_file: Path, train_file: Path, bert_type: str):
        super().__init__(data_file, train_file)
        self.tokenizer = BertTokenizer.from_pretrained(bert_type)

    def get_single_input(self, query: str, doc: str) -> Input:
        """Create a single model input from a query and a document.

        Args:
            query (str): The query
            doc (str): The document

        Returns:
            Input: The model input
        """
        return _get_single_input(query, doc)

    def collate_fn(self, inputs: Iterable[PointwiseTrainInput]) -> PointwiseTrainBatch:
        """Collate a number of pointwise inputs.

        Args:
            inputs (Iterable[PointwiseTrainInput]): The inputs

        Returns:
            PointwiseTrainBatch: A batch of pointwise inputs
        """
        inputs_, labels = zip(*inputs)
        return _collate_bert(inputs_, self.tokenizer), torch.FloatTensor(labels)


class PairwiseTrainDataset(PairwiseTrainDatasetBase):
    """Dataset for pairwise training.

    Args:
        data_file (Path): Data file containing queries and documents
        train_file (Path): Trainingset file
        bert_type (str): Type for the tokenizer
    """
    def __init__(self, data_file: Path, train_file: Path, bert_type: str):
        super().__init__(data_file, train_file)
        self.tokenizer = BertTokenizer.from_pretrained(bert_type)

    def get_single_input(self, query: str, doc: str) -> Input:
        """Create a single model input from a query and a document.

        Args:
            query (str): The query
            doc (str): The document

        Returns:
            Input: The model input
        """
        return _get_single_input(query, doc)

    def collate_fn(self, inputs: Iterable[PairwiseTrainInput]) -> PairwiseTrainBatch:
        """Collate a number of pairwise inputs.

        Args:
            inputs (Iterable[PairwiseTrainInput]): The inputs

        Returns:
            PairwiseTrainBatch: A batch of pairwise inputs
        """
        pos_inputs, neg_inputs = zip(*inputs)
        return _collate_bert(pos_inputs, self.tokenizer), _collate_bert(neg_inputs, self.tokenizer)


class ValTestDataset(ValTestDatasetBase):
    """Dataset for BERT validation/testing.

    Args:
        data_file (Path): Data file containing queries and documents
        val_test_file (Path): Validationset/testset file
        bert_type (str): Type for the tokenizer
    """
    def __init__(self, data_file: Path, val_test_file: Path, bert_type: Path):
        super().__init__(data_file, val_test_file)
        self.tokenizer = BertTokenizer.from_pretrained(bert_type)

    def get_single_input(self, query: str, doc: str) -> Input:
        """Create a single model input from a query and a document.

        Args:
            query (str): The query
            doc (str): The document

        Returns:
            Input: The model input
        """
        return _get_single_input(query, doc)

    def collate_fn(self, val_test_inputs: Iterable[ValTestInput]) -> ValTestBatch:
        """Collate a number of validation/testing inputs.

        Args:
            val_test_inputs (Iterable[BertValInput]): The inputs

        Returns:
            ValTestBatch: A batch of validation inputs
        """
        q_ids, doc_ids, inputs, labels = zip(*val_test_inputs)
        return torch.LongTensor(q_ids), \
               torch.LongTensor(doc_ids), \
               _collate_bert(inputs, self.tokenizer), \
               torch.LongTensor(labels)
