#include <torch/extension.h>
#include <vector>

// s'(z) = (1 - s(z)) * s(z)
torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

// tanh'(z) = 1 - tanh^2(z)
torch::Tensor d_tanh(torch::Tensor z) {
  return 1 - z.tanh().pow(2);
}

// elu'(z) = relu'(z) + { alpha * exp(z) if (alpha * (exp(z) - 1)) < 0, else 0}
torch::Tensor d_elu(torch::Tensor z, torch::Scalar alpha = 1.0) {
  auto e = z.exp();
  auto mask = (alpha * (e - 1)) < 0;
  return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e);
}

std::vector<torch::Tensor> fwht_forward(... TODO args here ...) {
  return {... return values here ... };
}

std::vector<torch::Tensor> fwht_backward(... TODO args here ...) {
  return {... return values here ... };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fwht_forward, "FWHT forward");
  m.def("backward", &fwht_backward, "FWHT backward");
}