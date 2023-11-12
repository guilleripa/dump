from typing import Generator

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

device = torch.device("mps" if torch.cuda.is_available() else "cpu")

tokenizer = get_tokenizer("basic_english")


def yield_tokens(data_iter: pd.DataFrame) -> Generator[str, None, None]:
    for _, text, _ in data_iter.itertuples():
        yield tokenizer(text)


class CustomDataset:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx].values


def get_dataloader(data: pd.DataFrame) -> DataLoader:
    vocab = build_vocab_from_iterator(yield_tokens(data), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    def text_pipeline(x):
        return vocab(tokenizer(x))

    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]

        for _text, _label in batch:
            label_list.append(_label)
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))

        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device)

    dataset = CustomDataset(data)
    return DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_batch)
