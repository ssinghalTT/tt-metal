// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_norm_backward_pybind.hpp"

#include "moreh_norm_backward.hpp"
#include "pybind11/decorators.hpp"

namespace ttnn::operations::moreh::moreh_norm_backward {
void bind_moreh_norm_backward_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::moreh_norm_backward,
        "Moreh Norm Backward Operation",
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::arg("output"),
            py::arg("output_grad"),
            py::arg("p"),
            py::kw_only(),
            py::arg("dim") = std::nullopt,
            py::arg("keepdim") = false,
            py::arg("input_grad") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
        });
}
}  // namespace ttnn::operations::moreh::moreh_norm_backward
