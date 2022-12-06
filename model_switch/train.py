import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from models.lenet import Lenet


def train_iter(model, loss_func, optimizer, x_data, y_data, num_iters):
    model.train()

    for i in range(num_iters):
        y = model(x_data)

        loss = loss_func(y, y_data)
        loss.backward()
        optimizer.step()

        print(f"Iter {i} done...")


def main():
    batch_size = 128
    num_classes = 10
    dummy_x = torch.randn(batch_size, 3, 32, 32).cuda()
    dummy_y = torch.randint(num_classes, (batch_size,)).cuda()

    model = Lenet(num_classes=10).cuda()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    warmup_iters = 10
    train_iter(model, loss_func, optimizer, dummy_x, dummy_y, warmup_iters)


if __name__ == "__main__":
    main()

