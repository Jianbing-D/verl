#include <torch/torch.h>
#include <torch/extension.h>
#include <string>


#define KERNEL_WITH_DTYPE(dtype, DTYPE, ...)                    \
    if (torch::kFloat32 == dtype) {                             \
        using DTYPE = float;                                    \
        { __VA_ARGS__ }                                         \
    } else if (torch::kBFloat16 == dtype) {                     \
        using DTYPE = __nv_bfloat16;                            \
        { __VA_ARGS__ }                                         \
    } else {                                                    \
        TORCH_CHECK(false, "Unsupported dtype: ", dtype);       \
    }   

void forward_mainloop() {

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_mainloop", &forward_mainloop, "Linear Cross Entropy Forward Mainloop");
}