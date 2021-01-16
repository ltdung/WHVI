#include <torch/extension.h>
#include <vector>

at::Tensor fwht(torch::Tensor x_in) {
    torch::Tensor x = torch::clone(x_in);  // NOT IN PLACE! If we skip this line, then autograd gets messed up. FIXME.
    int n = x.size(1);
    x = x.transpose(0, 1);  // Flip dimensions
    int h = 1;
    while (h < n) {
        for (int i = 0; i < n; i += (2 * h)) {
            for (int j = i; j < i + h; ++j) {
                auto tmp = x[j] - x[j + h];
                x[j] += x[j + h];
                x[j + h] = tmp;
            }
        }
        h *= 2;
    }
    x = x.transpose(0, 1);  // Flip back
    return x;
}

at::Tensor fwht_forward(torch::Tensor x) {
    return fwht(x);
}

torch::Tensor fwht_backward(torch::Tensor grad_output) {
    return fwht(grad_output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fwht_forward, "FWHT forward");
  m.def("backward", &fwht_backward, "FWHT backward");
}