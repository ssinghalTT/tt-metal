#pragma once
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace ttnn::operations::multiplyadd {
void bind_multiplyadd_operation(py::module& module);

void py_module(py::module& module);
}  // namespace ttnn::operations::multiplyadd
