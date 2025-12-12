"""
BiLSTM + Attention model for fake news classification.

High-level picture:
- Embedding layer
- BiLSTM to capture context
- Attention layer to focus on important tokens
- Final linear layer to output probability of "real" vs "fake"
"""

import torch
import torch.nn as nn


class Attention(nn.Module):
    """
    A simple additive attention mechanism over LSTM outputs.

    inputs: outputs from BiLSTM of shape (batch, seq_len, hidden_dim * num_directions)
    output: context vector (batch, hidden_dim * num_directions) and attention weights (batch, seq_len)
    """

    def __init__(self, hidden_dim: int, bidirectional: bool = True):
        super().__init__()
        self.d = hidden_dim * (2 if bidirectional else 1)
        self.attn = nn.Linear(self.d, 1, bias=False)

    def forward(self, outputs, mask=None):
        # outputs: (batch, seq_len, d)
        # mask: (batch, seq_len) - 1 for valid tokens, 0 for padding
        scores = self.attn(outputs).squeeze(-1)  # (batch, seq_len)

        if mask is not None:
            # For padded positions, we put -inf so softmax will ignore them
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)  # (batch, seq_len)

        # Context is weighted sum of outputs
        context = torch.bmm(attn_weights.unsqueeze(1), outputs).squeeze(1)
        # context: (batch, d)

        return context, attn_weights


class BiLSTMAttentionModel(nn.Module):
    """
    Full BiLSTM + Attention classifier.

    Note: We use sigmoid + BCELoss for binary classification (fake vs real).
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = Attention(hidden_dim, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, lengths):
        """
        input_ids: (batch, seq_len)
        lengths: (batch,) actual lengths before padding
        """
        embedded = self.embedding(input_ids)  # (batch, seq_len, emb_dim)

        # Pack padded sequence so that LSTM ignores padding properly
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # outputs: (batch, seq_len, hidden_dim * num_directions)

        # Create mask: 1 for non-pad tokens, 0 for pad
        mask = (input_ids != 0).long()

        context, attn_weights = self.attention(outputs, mask=mask)
        context = self.dropout(context)

        logits = self.fc(context).squeeze(-1)
        probs = self.sigmoid(logits)

        return probs, attn_weights