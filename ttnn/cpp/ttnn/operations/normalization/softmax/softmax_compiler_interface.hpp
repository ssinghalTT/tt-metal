// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <exception>
#include <functional>

#include "third_party/json/json.hpp"
#include "tt_metal/common/logger.hpp"
#include "ttnn/compiler_interface/compiler_interface.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"

namespace ttnn::compiler_interface {

std::tuple<bool, size_t, size_t, size_t> softmax_op_constraints(
    const OperandParams& input, const int dim_arg, const OperandParams& output) {
    // get_op_trace is a lambda that prepares input and output tensors, capturing graph trace of the op without
    // inputs allocation.
    auto get_op_trace = [](const OperandParams& input, const int dim_arg, const OperandParams& output) {
        nlohmann::json op_trace;

        // outer graph capture is used to avoid capturing and dispatching of dummy input tensor(s) creation
        {
            ttnn::graph::GraphProcessor::begin_graph_capture(ttnn::graph::GraphProcessor::RunMode::NO_DISPATCH);
            const auto input_tensor = create_device_tensor(
                std::get<ttnn::SimpleShape>(input),
                std::get<tt::tt_metal::DataType>(input),
                std::get<tt::tt_metal::Layout>(input),
                SingletonDeviceContext::get_instance().get_device(),
                std::get<tt::tt_metal::MemoryConfig>(input));

            // output tensor is created in the inner graph capture to capture its allocation
            {
                ttnn::graph::GraphProcessor::begin_graph_capture(ttnn::graph::GraphProcessor::RunMode::NO_DISPATCH);
                auto output_tensor = ttnn::softmax(input_tensor, dim_arg, std::get<tt::tt_metal::MemoryConfig>(output));
                // close inner graph capture, before output buffer is deallocated
                op_trace = ttnn::graph::GraphProcessor::end_graph_capture();
            }
            // close outer graph capture
            ttnn::graph::GraphProcessor::end_graph_capture();
        }

        // TODO(mbezulj) remove this debug print
        std::cout << graph::to_graphviz(op_trace) << std::endl;

        return op_trace;
    };

    try {
        auto op_trace = get_op_trace(input, dim_arg, output);
        return extract_data_from_trace(op_trace);
    } catch (const std::exception& e) {
        tt::log_debug(tt::LogOp, "compiler_interface - error: {}", e.what());
    }

    return std::make_tuple(false, 0, 0, 0);
}

}  // namespace ttnn::compiler_interface
