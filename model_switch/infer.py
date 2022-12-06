import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from models.lenet import Lenet


def eval(model, x_data):
    model.eval()

    y = model(x_data)

    return y


def main():
    batch_size = 128
    num_classes = 10
    dummy_x = torch.randn(batch_size, 3, 32, 32).cuda()

    model = Lenet(num_classes=10).cuda()

    eval(model, dummy_x)


if __name__ == "__main__":
    main()

