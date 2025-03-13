from torch import nn


class MusicRNN(nn.Module):

    def __init__(self, input_size=4, hidden_size=256, num_layers=4, output_size=4):
        super(MusicRNN, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out
