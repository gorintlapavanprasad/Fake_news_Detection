"""
Dataset and vocabulary utilities using plain PyTorch.

To avoid torchtext version issues, we create our own very small Vocab class.

What this file does:
- Tokenize text (simple whitespace)
- Build a vocabulary using Python's Counter
- Map each token to an integer id
- Provide a collate_fn that pads sequences in a batch
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def simple_tokenizer(text: str):
    """Very simple tokenizer: split by whitespace."""
    return text.split()


class Vocab:
    """
    Tiny vocabulary class, just enough for our project.

    We support:
      - len(vocab)
      - vocab[token]  -> index
      - vocab[PAD_TOKEN] to get pad index
    """

    def __init__(self, token_to_idx, unk_token: str = UNK_TOKEN):
        self.token_to_idx = token_to_idx
        self.unk_index = token_to_idx[unk_token]

    def __len__(self):
        return len(self.token_to_idx)

    def __getitem__(self, token: str) -> int:
        # If token not present, return index of UNK
        return self.token_to_idx.get(token, self.unk_index)


def build_vocab(train_texts, tokenizer=simple_tokenizer, max_size=30000, min_freq=2):
    """
    Build vocabulary from a list of training texts.

    Steps:
    - Tokenize each text
    - Count token frequencies
    - Keep tokens with frequency >= min_freq
    - Sort by frequency (high to low)
    - Truncate to max_size (keeping space for PAD and UNK)
    """
    counter = Counter()

    # Count tokens
    for text in train_texts:
        tokens = tokenizer(text)
        counter.update(tokens)

    # Filter by min_freq
    tokens = [tok for tok, freq in counter.items() if freq >= min_freq]

    # Sort tokens by frequency (descending), then alphabetically for stability
    tokens.sort(key=lambda t: (-counter[t], t))

    # We reserve two spots for PAD and UNK
    if max_size is not None:
        max_tokens = max_size - 2  # PAD and UNK already
        tokens = tokens[:max_tokens]

    # Build mapping: special tokens first
    token_to_idx = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for tok in tokens:
        if tok not in token_to_idx:
            token_to_idx[tok] = len(token_to_idx)

    vocab = Vocab(token_to_idx, unk_token=UNK_TOKEN)
    return vocab


class NewsDataset(Dataset):
    """
    Dataset for fake news articles.

    Each item returns:
      - token_ids: tensor of token indices (variable length)
      - label: 0/1 tensor
    """

    def __init__(self, df, vocab, tokenizer=simple_tokenizer, max_len: int = 400):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def _encode_text(self, text: str):
        tokens = self.tokenizer(text)
        # Map to indices and truncate to max_len
        token_ids = [self.vocab[token] for token in tokens][: self.max_len]
        return torch.tensor(token_ids, dtype=torch.long)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        token_ids = self._encode_text(text)
        return token_ids, torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    """
    Collate function for DataLoader.

    - Pads sequences in the batch with PAD_TOKEN index (which is 0)
    - Also returns original lengths, useful for packing sequences in LSTM
    """
    token_seqs, labels = zip(*batch)

    lengths = torch.tensor([len(seq) for seq in token_seqs], dtype=torch.long)

    padded_seqs = pad_sequence(token_seqs, batch_first=True, padding_value=0)

    labels = torch.stack(labels)

    return padded_seqs, lengths, labels