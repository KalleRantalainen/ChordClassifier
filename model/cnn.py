import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, MaxPool2d, BatchNorm2d, ReLU, Linear, Sequential, Dropout2d, Dropout

class ChordCNN(nn.Module):
    def __init__(self, num_chords=70):
        super(ChordCNN, self).__init__()

        # Block 1
        self.block_1 = Sequential(
            Conv2d(in_channels=1, out_channels=32, kernel_size=(26,1)),
            ReLU(),
            #BatchNorm2d(32),
            #MaxPool2d(kernel_size=(3,1), stride=(3,1)),
            Dropout2d(p=0.4)
        )

        # Block 2
        self.block_2 = Sequential(
            Conv2d(in_channels=32, out_channels=64, kernel_size=(26,1)),
            ReLU(),
            #BatchNorm2d(64),
            #MaxPool2d(kernel_size=(3,1), stride=(3,1)),
            Dropout2d(p=0.4)
        )

        # Block 3
        self.block_3 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(26,1)),
            ReLU(),
            BatchNorm2d(128),
            #MaxPool2d(kernel_size=(3,1), stride=(3,1)),
            Dropout2d(p=0.4)
        )

        # Block 4
        # self.block_4 = Sequential(
        #     Conv2d(in_channels=128, out_channels=256, kernel_size=(3,1)),
        #     ReLU(),
        #     BatchNorm2d(256),
        #     MaxPool2d(kernel_size=(2,1), stride=(2,1)),
        #     Dropout2d(p=0.2)
        # )

        self.fc1 = Linear(6912, 512)
        self.fc2 = Linear(512, num_chords)

        self.dropout = Dropout(p=0.2)

    def forward(self, x):
        # Data is in shape (batch_size, freq_bands, time_steps), the 
        # cnn expects shape (channels, batch_size, freq_bands, time_steps)
        # -> need to unsqueeze before passing the data
        #print("X shape before:", x.shape)
        x = x.unsqueeze(1)
        #print("X shape after unsqueeze:", x.shape)
        x = self.block_1(x)
        #print("X shape after block 1:", x.shape)
        x = self.block_2(x)
        #print("X shape after block 2:", x.shape)
        x = self.block_3(x)
        #print("X shape after block 3:", x.shape)
        #x = self.block_4(x)
        #print("X shape after block 4:", x.shape)

        x = x.view(x.size(0), -1)
        #print("X after reshape:", x.shape)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        #print("X after fc:", x.shape)

        return x