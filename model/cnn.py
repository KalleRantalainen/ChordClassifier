import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, MaxPool2d, BatchNorm2d, ReLU, Linear, Sequential, Dropout2d, Dropout

class ChordCNN(nn.Module):
    def __init__(self, num_chords=70):
        super(ChordCNN, self).__init__()

        # Block 1
        self.block_1 = Sequential(
            Conv2d(in_channels=6, out_channels=32, kernel_size=(3,3), padding=1),
            ReLU(),
            BatchNorm2d(32),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout2d(p=0.2)
        )

        # Block 2
        self.block_2 = Sequential(
            Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1),
            ReLU(),
            BatchNorm2d(64),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout2d(p=0.2)
        )

        # Block 3
        self.block_3 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1),
            ReLU(),
            BatchNorm2d(128),
            Dropout2d(p=0.2)
        )

        # Block 4
        self.block_4 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=1),
            ReLU(),
            BatchNorm2d(256),
            Dropout2d(p=0.2)
        )

        self.fc1 = Linear(256 * 5 * 5, 512)
        self.fc2 = Linear(512, num_chords)

        self.dropout = Dropout(p=0.5)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x