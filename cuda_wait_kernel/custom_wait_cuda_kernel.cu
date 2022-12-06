#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void wait_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> wait_flag) {
    volatile int inf = 1;
    while (inf) {
        if (wait_flag[0] == 1) {
            wait_flag[0] = 0;
            break;
        }
    }
}

void wait_cuda_forward(torch::Tensor wait_flag) {
    AT_DISPATCH_INTEGRAL_TYPES(wait_flag.type(), "wait_forward_cuda", ([&] {
        wait_cuda_kernel<scalar_t><<<1, 1>>>(
            wait_flag.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>());
    }));
}

void wait_cuda_backward(torch::Tensor wait_flag) {
    AT_DISPATCH_INTEGRAL_TYPES(wait_flag.type(), "wait_backward_cuda", ([&] {
        wait_cuda_kernel<scalar_t><<<1, 1>>>(
            wait_flag.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>());
    }));
}