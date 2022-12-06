import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import socket
import pickle

import custom_wait
from ipc_utils import rebuild_tensor_from_ipc_info


class Lenet(nn.Module):
    def __init__(self, num_classes, sock):
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

        self.wait_flag = custom_wait.WaitCuda()
        self.sock = sock

    def eval(self):
        self.training = False

        register_msg = f'register/124.0.0.1/cuda:{torch.cuda.current_device()}'
        self.sock.send(register_msg.encode('utf-8'))

        get_flag_msg = f'get_flag/124.0.0.1'
        self.sock.send(get_flag_msg.encode('utf-8'))

        recv_msg_len = self.sock.recv(4)
        flag_ipc_info_msg = self.sock.recv(int.from_bytes(recv_msg_len, byteorder="little"))
        flag_ipc_info = pickle.loads(flag_ipc_info_msg)
        self.shared_flag = rebuild_tensor_from_ipc_info(flag_ipc_info)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)

        self.wait_flag(self.shared_flag)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)

        self.wait_flag(self.shared_flag)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.max_pool3(x)

        self.wait_flag(self.shared_flag)

        x = self.flatten1(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)

        return x


def eval(model, x_data):
    model.eval()

    y = model(x_data)

    return y


def main():
    HOST = '127.0.0.1'
    PORT = 9999
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

    batch_size = 128
    num_classes = 10
    dummy_x = torch.randn(batch_size, 3, 32, 32).cuda()

    model = Lenet(num_classes=10, sock=client_socket).cuda()

    eval(model, dummy_x)

    client_socket.close()


if __name__ == "__main__":
    main()

