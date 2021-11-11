# gru.py

import torch
import torch.nn as nn

class MyGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyGRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # You are free to follow the code template below, or do it a different way,
        # as long as the output is correct.
        # ------------
        # FILL THIS IN - START
        # ------------

        self.Wir = nn.Linear(input_size, hidden_size)
        self.Wiz = nn.Linear(input_size, hidden_size)
        self.Wig = nn.Linear(input_size, hidden_size)

        self.Whr = nn.Linear(hidden_size, hidden_size)
        self.Whz = nn.Linear(hidden_size, hidden_size)
        self.Whg = nn.Linear(hidden_size, hidden_size)

        # ------------
        # FILL THIS IN - END
        # ------------

    def forward(self, x, h_prev):
        """Forward pass of the GRU computation for one time step.

        Arguments
            x: batch_size x input_size
            h_prev: batch_size x hidden_size

        Returns:
            h_new: batch_size x hidden_size
        """

        # ------------
        # FILL THIS IN - START
        # ------------

        z = torch.sigmoid(self.Wiz(x) + self.Whz(h_prev))
        r = torch.sigmoid(self.Wir(x) + self.Whr(h_prev))
        g = torch.tanh(self.Wig(x) + r * self.Whg(h_prev))
        h_new = (1-z)*g + z*h_prev

        # ------------
        # FILL THIS IN - END
        # ------------
        return h_new
