from typing import Generator

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

device = torch.device("cpu")

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
        return self.data.iloc[idx]

    @property
    def vocab(self):
        if not hasattr(self, "_vocab"):
            vocab = build_vocab_from_iterator(
                yield_tokens(self.data), specials=["<unk>"]
            )
            vocab.set_default_index(vocab["<unk>"])
            self._vocab = vocab

        return self._vocab


def get_dataloader(data: pd.DataFrame, batch_size: int = 16) -> DataLoader:
    dataset = CustomDataset(data)
    vocab = dataset.vocab

    def text_pipeline(x):
        return vocab(tokenizer(x))

    def collate_batch(batch):
        """Collate function to pad the text to the maximum length in a batch.
        The batch is sorted in descending order of text length to minimize the
        amount of padding needed.
        """
        label_list, text_list, lengths = [], [], []
        for _text, _label in batch:
            label_list.append(_label)
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            if processed_text.size(0) == 0:
                lengths.append(torch.tensor(1))
                print("Empty text found!", _text, _label, processed_text)
            else:
                lengths.append(processed_text.size(0))

        text_list = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True)
        label_list = torch.tensor(label_list, dtype=torch.float32).reshape(-1, 1)

        # sort based on text lengths
        lengths = torch.tensor(lengths)
        _, perm_idx = lengths.sort(0, descending=True)

        text_list = text_list[perm_idx]
        label_list = label_list[perm_idx]
        lengths = lengths[perm_idx]

        return label_list.to(device), text_list.to(device), lengths

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )
