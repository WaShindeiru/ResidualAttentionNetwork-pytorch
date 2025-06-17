import torch.nn as nn
import torch.nn.functional as F

class cnn_lstm_(nn.Module):
    def __init__(self):
        super(cnn_lstm_, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        # LSTM
        self.lstm_input_size = 64  # liczba cech na krok
        self.sequence_length = 25

        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=64,
                            batch_first=True, num_layers=1)

        # Fully connected layer
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # [B, 32, 13, 13]
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))   # [B, 64, 5, 5]

        x = x.view(x.size(0), 64, 25)          # [B, 64, 25]
        x = x.permute(0, 2, 1)                 # [B, 25, 64]

        # LSTM
        out, _ = self.lstm(x)                  # [B, 25, 64]
        out = out[:, -1, :]

        # Klasyfikacja
        out = self.fc(out)                     # [B, 10]
        return out