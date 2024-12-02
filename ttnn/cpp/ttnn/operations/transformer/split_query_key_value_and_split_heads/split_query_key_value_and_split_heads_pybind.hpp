// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"

namespace ttnn::operations::transformer {

void py_bind_split_query_key_value_and_split_heads(pybind11::module& module);

}  // namespace ttnn::operations::transformer
