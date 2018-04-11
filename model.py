import torch
import torch.nn as nn
import torch.autograd as autograd


class LSTM(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeds = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim)
        # input_dim = output_dim
        self.linear = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)
        self.hidden = self.initHidden()

    def forward(self, input, hidden):
        embeds = self.embeds(input)
        lstm_out, hidden = self.lstm(
            embeds.view(len(input), 1, -1), hidden)
        output = self.linear(lstm_out.view(len(input), -1))
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
