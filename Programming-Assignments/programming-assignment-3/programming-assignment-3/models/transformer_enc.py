# transformer_enc_dec.py

import torch
import torch.nn as nn
from .attention import *

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, attention_type=None):
        super(TransformerEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.self_attentions = nn.ModuleList([ScaledDotAttention(
                                    hidden_size=hidden_size,
                                 ) for i in range(self.num_layers)])

        self.attention_mlps = nn.ModuleList([nn.Sequential(
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size),
                                 ) for i in range(self.num_layers)])

        self.positional_encodings = self.create_positional_encodings()
        self.norm = LayerNorm(hidden_size)


    def forward(self, inputs):
        """Forward pass of the encoder RNN.

        Arguments:
            inputs: Input token indexes across a batch for all time steps in the sequence. (batch_size x seq_len)

        Returns:
            annotations: The hidden states computed at each step of the input sequence. (batch_size x seq_len x hidden_size)
            hidden: The final hidden state of the encoder, for each sequence in a batch. (batch_size x hidden_size)
        """

        batch_size, seq_len = inputs.size()
        encoded = self.embedding(inputs)  # batch_size x seq_len x hidden_size

        # ------------
        # FILL THIS IN - START
        # ------------
        # Add positional embeddings from self.create_positional_encodings. (a'la https://arxiv.org/pdf/1706.03762.pdf, section 3.5)
        encoded = self.embedding(inputs)  # batch_size x seq_len x hidden_size
        encoded += self.positional_encodings[:seq_len]
        # ------------
        # FILL THIS IN - END
        # ------------

        annotations = encoded

        for i in range(self.num_layers):
            # ------------
            # FILL THIS IN - START
            # ------------
            new_annotations, self_attention_weights = self.self_attentions[i](annotations, annotations, annotations)
            residual_annotations = annotations + new_annotations
            new_annotations = self.attention_mlps[i](residual_annotations.view(-1, self.hidden_size)).view(batch_size, seq_len, self.hidden_size)
            annotations = residual_annotations + new_annotations
            annotations = self.norm(annotations)
            # ------------
            # FILL THIS IN - END
            # ------------
            pass

        # Transformer encoder does not have a last hidden layer.
        return annotations, None

    def create_positional_encodings(self, max_seq_len=1000):
      """Creates positional encodings for the inputs.

      Arguments:
          max_seq_len: a number larger than the maximum string length we expect to encounter during training

      Returns:
          pos_encodings: (max_seq_len, hidden_dim) Positional encodings for a sequence with length max_seq_len.
      """
      pos_indices = torch.arange(max_seq_len)[..., None]
      dim_indices = torch.arange(self.hidden_size//2)[None, ...]
      exponents = (2*dim_indices).float()/(self.hidden_size)
      trig_args = pos_indices / (10000**exponents)
      sin_terms = torch.sin(trig_args)
      cos_terms = torch.cos(trig_args)

      pos_encodings = torch.zeros((max_seq_len, self.hidden_size))
      pos_encodings[:, 0::2] = sin_terms
      pos_encodings[:, 1::2] = cos_terms

      return pos_encodings
