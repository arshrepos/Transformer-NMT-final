"""Multi30k dataset loading and PyTorch Dataset/DataLoader utilities."""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from data.tokenizer import PAD_IDX


def load_multi30k_split(data_dir, split):
    """Load a Multi30k split from pre-downloaded text files.

    Expects files like: train.en, train.de, val.en, val.de, test.en, test.de
    """
    src_path = os.path.join(data_dir, f"{split}.en")
    tgt_path = os.path.join(data_dir, f"{split}.de")

    with open(src_path, "r", encoding="utf-8") as f:
        src_lines = [line.strip() for line in f if line.strip()]
    with open(tgt_path, "r", encoding="utf-8") as f:
        tgt_lines = [line.strip() for line in f if line.strip()]

    assert len(src_lines) == len(tgt_lines), (
        f"Mismatch: {len(src_lines)} src vs {len(tgt_lines)} tgt in {split}"
    )
    return src_lines, tgt_lines


class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_tokenizer, tgt_tokenizer):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_ids = self.src_tokenizer.encode(self.src_sentences[idx])
        tgt_ids = self.tgt_tokenizer.encode(self.tgt_sentences[idx])
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded


def get_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
