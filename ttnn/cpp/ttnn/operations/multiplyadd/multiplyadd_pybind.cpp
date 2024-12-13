#include <pybind11/pybind11.h>
#include "ttnn/cpp/pybind11/decorators.hpp"
#include "multiplyadd.hpp"

void bind_multiplyadd_operation(py::module& module) {
    auto doc = fmt::format(
        R"doc(
        Args:
            input_tensor1 (ttnn.Tensor): the first input tensor.
            input_tensor2 (ttnn.Tensor): the second input tensor.
            input_tensor3 (ttnn.Tensor): the third input tensor.

        Returns:
            ttnn.Tensor: the output tensor.

        multiplyadd(input_tensor1: ttnn.Tensor, input_tensor2: ttnn.Tensor, input_tensor3: ttnn.Tensor) -> ttnn.Tensor
        )doc");

    bind_registered_operation(
        module,
        ttnn::multiplyadd,
        doc,
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::multiplyadd)& self,
               const ttnn::Tensor& input_tensor1,
               const ttnn::Tensor& input_tensor2,
               const ttnn::Tensor& input_tensor3) -> ttnn::Tensor {
                return self(input_tensor1, input_tensor2, input_tensor3);
            },
            py::arg("input_tensor1"),
            py::arg("input_tensor2"),
            py::arg("input_tensor3")});
}

void py_module(py::module& module) { bind_multiplyadd_operation(module); };
