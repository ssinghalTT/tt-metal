// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <string>
#include <unordered_map>

namespace tt {

namespace tt_metal {
void dump_tensor(const std::string& file_name, const Tensor& tensor, const std::unordered_map<std::string, std::string>& strategy = {});
void dump_tensor(std::ostream& output_stream, const Tensor& tensor, const std::unordered_map<std::string, std::string>& strategy = {});

template <typename T>
Tensor load_tensor(const std::string& file_name, T device = nullptr);

template <typename T>
Tensor load_tensor(std::istream& input_stream, T device = nullptr);

void dump_memory_config(std::ostream& output_stream, const MemoryConfig& memory_config);
void dump_memory_config(const std::string& file_name, const MemoryConfig& memory_config);


}  // namespace tt_metalls

}  // namespace tt
