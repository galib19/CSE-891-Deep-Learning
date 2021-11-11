# gru.py

import torch
import torch.nn as nn

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super(AdditiveAttention, self).__init__()

        self.hidden_size = hidden_size

        # Create a two layer fully-connected network. Hint: Use nn.Sequential
        # hidden_size*2 --> hidden_size, ReLU, hidden_size --> 1

        # ------------
        # FILL THIS IN - START
        # ------------

        self.attention_network = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), #hidden_size*2 --> hidden_size
            nn.ReLU(),
            nn.Linear(hidden_size, 1) #hidden_size --> 1
        )

        # ------------
        # FILL THIS IN - END
        # ------------

        self.softmax = nn.Softmax(dim=1)

    def forward(self, queries, keys, values):
        """The forward pass of the additive attention mechanism.

        Arguments:
            queries: The current decoder hidden state. (batch_size x hidden_size)
            keys: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)
            values: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)

        Returns:
            context: weighted average of the values (batch_size x 1 x hidden_size)
            attention_weights: Normalized attention weights for each encoder hidden state. (batch_size x seq_len x 1)

            The attention_weights must be a softmax weighting over the seq_len annotations.
        """
        # You are free to follow the code template below, or do it a different way,
        # as long as the output is correct.
        # ------------
        # FILL THIS IN - START
        # ------------

        expanded_queries = queries.view(keys.size(0), -1, self.hidden_size).expand_as(keys)
        concat_inputs =  torch.cat((expanded_queries, keys), 2)
        unnormalized_attention = self.attention_network(concat_inputs)
        attention_weights = self.softmax(unnormalized_attention)
        context = torch.bmm(attention_weights.transpose(2, 1), values)

        # ------------
        # FILL THIS IN - END
        # ------------
        return context, attention_weights

class ScaledDotAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ScaledDotAttention, self).__init__()

        self.hidden_size = hidden_size

        self.Q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.scaling_factor = torch.rsqrt(torch.tensor(self.hidden_size, dtype= torch.float))

    def forward(self, queries, keys, values):
        """The forward pass of the scaled dot attention mechanism.

        Arguments:
            queries: The current decoder hidden state, 2D or 3D tensor. (batch_size x (k) x hidden_size)
            keys: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)
            values: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)

        Returns:
            context: weighted average of the values (batch_size x k x hidden_size)
            attention_weights: Normalized attention weights for each encoder hidden state. (batch_size x k x seq_len)

            The output must be a softmax weighting over the seq_len annotations.
        """

        # You are free to follow the code template below, or do it a different way,
        # as long as the output is correct.
        # ------------
        # FILL THIS IN - START
        # ------------

        if queries.dim() != 3:
            queries = self.Q(queries.unsqueeze(1))

        q = self.Q(queries)
        k = self.K(keys)
        v = self.V(values)
<<<<<<< HEAD
        unnormalized_attention = self.scaling_factor * torch.bmm(k, q.transpose(1, 2))
        attention_weights = self.softmax(unnormalized_attention)
        context = torch.bmm(attention_weights.transpose(1, 2), v)

        
=======
        unnormalized_attention = k.bmm(q.permute(0, 2, 1)) * self.scaling_factor
        attention_weights = self.softmax(unnormalized_attention.permute(0, 2, 1))
        context = torch.bmm(attention_weights, v)

        # unnormalized_attention = self.scaling_factor * torch.bmm(k, q.transpose(1, 2))
        # attention_weights = self.softmax(unnormalized_attention)
        # context = torch.bmm(attention_weights.transpose(1, 2), v)
>>>>>>> 8bb6b935a9f4f8a4fb2e894fee42573a34107e93

        # ------------
        # FILL THIS IN - END
        # ------------



        return context, attention_weights.transpose(1,2)

class CausalScaledDotAttention(nn.Module):
    def __init__(self, hidden_size):
        super(CausalScaledDotAttention, self).__init__()

        self.hidden_size = hidden_size
        self.neg_inf = -1e7

        self.Q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.scaling_factor = torch.rsqrt(torch.tensor(self.hidden_size, dtype= torch.float))

    def forward(self, queries, keys, values):
        """The forward pass of the scaled dot attention mechanism.

        Arguments:
            queries: The current decoder hidden state, 2D or 3D tensor. (batch_size x (k) x hidden_size)
            keys: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)
            values: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)

        Returns:
            context: weighted average of the values (batch_size x k x hidden_size)
            attention_weights: Normalized attention weights for each encoder hidden state. (batch_size x seq_len x k)

            The output must be a softmax weighting over the seq_len annotations.
        """

        # ------------
        # FILL THIS IN - START
        # ------------
        # You are free to follow the code template below, or do it a different way,
        # as long as the output is correct.
        batch_size, seq_len, hidden_size = keys.size()


        if queries.dim() != 3:
            queries = self.Q(queries.unsqueeze(1))

        q = self.Q(queries)
        k = self.K(keys)
        v = self.V(values)
        unnormalized_attention = self.scaling_factor * torch.bmm(k, q.transpose(1, 2))
        mask = torch.tril(torch.ones(batch_size, seq_len, seq_len, dtype=torch.uint8)).transpose(1, 2)
        unnormalized_attention[mask == 0] = self.neg_inf
        attention_weights = self.softmax(unnormalized_attention)
        context = torch.bmm(attention_weights.transpose(1, 2), v)

        # ------------
        # FILL THIS IN - END
        # ------------
        return context, attention_weights.transpose(1,2)
