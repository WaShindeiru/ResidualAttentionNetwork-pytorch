import torch.nn as nn
import torch.nn.functional as F

class CNN_(nn.Module):
    def __init__(self):
        super(CNN_, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 10)  # 10 klas

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> [batch, 32, 13, 13]
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))  # -> [batch, 64, 5, 5]
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x