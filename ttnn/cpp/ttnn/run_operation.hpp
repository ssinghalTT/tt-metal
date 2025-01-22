// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <ttnn/tensor/tensor.hpp>

#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/operation.hpp"
#include <tt-metalium/device_impl.hpp>
#include <tt-metalium/type_name.hpp>

namespace tt::tt_metal {

namespace operation {

using ttnn::operations::experimental::auto_format::FormatParams;

template <class OutputTensors = Tensors>
OutputTensors run(
    DeviceOperation<OutputTensors>&& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors = {},
    const OptionalTensors& optional_output_tensors = {},
    uint8_t cq_id = 0);

template <typename ConcreteOperation>
inline auto run(
    ConcreteOperation&& concrete_op,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors = {},
    const OptionalTensors& optional_output_tensors = {},
    uint8_t cq_id = 0) -> ProgramOutputTensors<ConcreteOperation> {
    using OutputTensors = ProgramOutputTensors<ConcreteOperation>;
    if constexpr (detail::is_device_operation<ConcreteOperation>()) {
        auto operation = DeviceOperation(concrete_op);
        return run<OutputTensors>(
            std::move(operation), input_tensors, optional_input_tensors, optional_output_tensors, cq_id);
    } else {
        static_assert(tt::stl::concepts::always_false_v<ConcreteOperation>, "Unsupported Operation");
    }
}

template <class OutputTensors = Tensors>
OutputTensors run_without_autoformat(
    DeviceOperation<OutputTensors>&& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors = {},
    const OptionalTensors& optional_output_tensors = {},
    uint8_t cq_id = 0);
template <typename ConcreteOperation>
inline auto run_without_autoformat(
    ConcreteOperation&& concrete_op,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const std::vector<std::optional<Tensor>>& optional_output_tensors = {},
    uint8_t cq_id = 0) -> ProgramOutputTensors<ConcreteOperation> {
    using OutputTensors = ProgramOutputTensors<ConcreteOperation>;
    auto operation = DeviceOperation<OutputTensors>(concrete_op);
    return run_without_autoformat<OutputTensors>(
        std::move(operation), input_tensors, optional_input_tensors, optional_output_tensors, cq_id);
}

Tensors run_with_autoformat(
    DeviceOperation<Tensors>&& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors = {},
    const OptionalTensors& optional_output_tensors = {},
    const float pad_value = 0,
    const bool pad_c = false,
    uint8_t cq_id = 0);

template <typename ConcreteOperation>
inline auto run_with_autoformat(
    ConcreteOperation&& concrete_op,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const std::vector<std::optional<Tensor>>& optional_output_tensors = {},
    const float pad_value = 0,
    const bool pad_c = false,
    uint8_t cq_id = 0) -> Tensors {
    using OutputTensors = ProgramOutputTensors<ConcreteOperation>;
    auto operation = DeviceOperation<Tensors>(concrete_op);
    return run_with_autoformat(
        std::move(operation), input_tensors, optional_input_tensors, optional_output_tensors, pad_value, pad_c, cq_id);
}

Tensors run_with_autoformat(
    DeviceOperation<Tensors>&& operation,
    const Tensors& input_tensors,
    const std::vector<FormatParams>& input_formatting,
    const std::vector<Layout>& output_layouts,
    const OptionalConstTensors& optional_input_tensors = {},
    const std::vector<std::optional<FormatParams>>& optional_input_formatting = {},
    const OptionalTensors& optional_output_tensors = {},
    uint8_t cq_id = 0);
template <typename ConcreteOperation>
inline auto run_with_autoformat(
    ConcreteOperation&& concrete_op,
    const std::vector<Tensor>& input_tensors,
    const std::vector<FormatParams>& input_formatting,
    const std::vector<Layout>& output_layouts,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const std::vector<std::optional<FormatParams>>& optional_input_formatting = {},
    const OptionalTensors& optional_output_tensors = {},
    uint8_t cq_id = 0) -> ProgramOutputTensors<ConcreteOperation> {
    using OutputTensors = ProgramOutputTensors<ConcreteOperation>;
    auto operation = DeviceOperation<OutputTensors>(concrete_op);
    return run_with_autoformat(
        std::move(operation),
        input_tensors,
        input_formatting,
        output_layouts,
        optional_input_tensors,
        optional_input_formatting,
        optional_output_tensors,
        cq_id);
}

template <class Callable, class OutputType = Tensors>
void launch_op(
    Callable&& op_func,
    const Tensors input_tensors,
    OutputType& output_tensors,
    const OptionalConstTensors optional_input_tensors = {},
    const OptionalTensors optional_output_tensors = {},
    bool enable_autoformat_device = true);

void launch_with_autoformat(
    std::function<Tensors(const Tensors&, const OptionalConstTensors&, const OptionalTensors&)>&& op_func,
    const Tensors& input_tensors,
    Tensors& output_tensors,
    const OptionalConstTensors& optional_input_tensors = {},
    const OptionalTensors& optional_output_tensors = {});

std::vector<IDevice*> get_workers_for_op_output(
    const std::vector<Tensor>& inputs,
    const std::vector<std::optional<const Tensor>>& optional_inputs = {},
    bool enable_autoformat_device = true);

namespace detail {
IDevice* get_device(const Tensors& input_tensors, const OptionalConstTensors& optional_input_tensors = {});
}

}  // namespace operation

}  // namespace tt::tt_metal

#include "run_operation_inl.hpp"
