#include <torch/extension.h>

at::Tensor fwht_cuda_frontend(at::Tensor X);

at::Tensor fwht(at::Tensor X) {
  TORCH_CHECK(X.device().type() == torch::kCUDA, "X must be a CUDA tensor");
  auto n = X.size(-1);
  auto log2N = long(log2(n));
  TORCH_CHECK(n == 1 << log2N, "n must be a power of 2");
  auto output = X.clone();  // Cloning makes it contiguous.
  fwht_cuda_frontend(output);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwht", &fwht, "Batched Fast Walsh-Hadamard transform");
}
