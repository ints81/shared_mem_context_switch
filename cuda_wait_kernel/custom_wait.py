import torch
import torch.nn as nn

import custom_wait_cuda


class WaitCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, wait_flag):
        wait_cuda.forward(wait_flag)

    @staticmethod
    def backward(ctx, wait_flag):
        wait_cuda.backward(wait_flag)


class WaitCuda(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, wait_flag):
        WaitCudaFunction.apply(wait_flag)
