#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> fwht_forward(torch::Tensor x) {
    int n = x.size(1);
    for (int batch_item_index = 0; batch_item_index < x.size(0); ++batch_item_index) {
        int h = 1;
        while (h < n) {
            for (int i = 0; i < n; i += (2 * h)) {
                for (int j = i; j < i + h; ++j) {
                    auto tmp = x[batch_item_index][j] - x[batch_item_index][j + h];
                    x[batch_item_index][j] += x[batch_item_index][j + h];
                    x[batch_item_index][j + h] = tmp;
                }
            }
            h *= 2;
        }
    }
    return {x};
}

std::vector<torch::Tensor> fwht_backward(torch::Tensor grad_output) {
    std::cout << "Got to the start" << std::endl;
    int n_batches = grad_output.size(0);
    std::cout << "Got to the start" << std::endl;
    int n = grad_output.size(1);
    std::cout << "Got to the start" << std::endl;
    std::vector<torch::Tensor> result(n_batches);
    for (int i = 0; i < n_batches; ++i) {
        std::cout << i << std::endl;
        result[i] = torch::eye(n);
    }
    return {result};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fwht_forward, "FWHT forward");
  m.def("backward", &fwht_backward, "FWHT backward");
}