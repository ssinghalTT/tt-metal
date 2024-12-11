// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"

namespace py = pybind11;

namespace ttnn::operations::binary_ng {
namespace detail {
void bind_binary_ng_operation(py::module& module);
}

void py_module(py::module& module);
}  // namespace ttnn::operations::binary_ng
