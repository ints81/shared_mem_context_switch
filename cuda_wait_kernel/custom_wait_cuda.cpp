#include <torch/extension.h>

#include <vector>

void wait_cuda_forward(torch::Tensor wait_flag);
void wait_cuda_backward(torch::Tensor wait_flag);

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void wait_forward(torch::Tensor wait_flag) {
  CHECK_INPUT(wait_flag);

  return wait_cuda_forward(wait_flag);
}

void wait_backward(torch::Tensor wait_flag) {
  CHECK_INPUT(wait_flag);

  return wait_cuda_backward(wait_flag);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &wait_forward, "wait forward (CUDA)");
  m.def("backward", &wait_backward, "wait backward (CUDA)");
}
