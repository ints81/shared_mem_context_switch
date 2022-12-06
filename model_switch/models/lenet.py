# import sys
# import os
# sys.path.append(os.getcwd())
import torch.nn as nn


class Lenet(nn.Module):
    def __init__(self, num_classes):
        super(Lenet, self).__init__()

        self.conv1 = nn.Conv2d(3,16,3,1,padding=1,bias=False)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,32,3,1, padding=1,bias=False)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1,bias=False)
        self.relu3 = nn.ReLU()
        self.max_pool3 = nn.MaxPool2d(2,2)
        self.flatten1 = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(4*4*64, 500)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.max_pool3(x)

        x = self.flatten1(x)

        x = self.fc1(x)
        x = self.relu4(x)

        x = self.fc2(x)

        return x
