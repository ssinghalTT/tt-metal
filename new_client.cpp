#include <cxxabi.h>

#include <atomic>
#include <chrono>
#include <concepts>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <typeinfo>
#include <vector>
#include <zmq.hpp>

#include "ttnn/cpp/ttnn/tensor/serialization.hpp"
#include "ttnn/cpp/ttnn/tensor/types.hpp"
#include "tt_metal/impl/device/device_mesh.hpp"
// #include "ttnn/cpp/ttnn/multi_device.hpp"
#include "ttnn/cpp/ttnn/multi_server/client.hpp"
#include "ttnn/cpp/ttnn/multi_server/message.hpp"
#include "ttnn/cpp/ttnn/multi_server/server.hpp"
#include "ttnn/cpp/ttnn/multi_server/utils.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/device.hpp"
#include "ttnn/cpp/ttnn/operations/core/core.hpp"
#include "ttnn/cpp/ttnn/operations/creation.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/binary/binary.hpp"

using namespace ttnn::multi_server;

DistributedTensor run_model(DistributedTensor& tensor_a, DistributedTensor& tensor_b, ServerDevice& device) {
    DistributedTensor device_tensor_a = device.to_device(tensor_a);
    DistributedTensor device_tensor_b = device.to_device(tensor_b);

    DistributedTensor input_a = device.to_layout(device_tensor_a);
    DistributedTensor input_b = device.to_layout(device_tensor_b);

    DistributedTensor output_tensor;
    for (int i = 0; i < 10; i++) {
        output_tensor = server_device.add(input_a, input_b);
        input_b = output_tensor;

        output_tensor.print();
    }

    device.remove_tensors({input_a, input_b, output_tensor});
    return output_tensor;
}

int main() {
    auto server_device = ServerDevice("tcp://localhost:8086");
    DistributedTensor tensor_a = create_multi_server_tensor(
        ttnn::ones(ttnn::Shape(std::array<uint32_t, 2>{32, 32}), ttnn::bfloat16),
        tt::tt_metal::ReplicateTensor(2),
        server_device);

    DistributedTensor tensor_b = create_multi_server_tensor(
        ttnn::ones(ttnn::Shape(std::array<uint32_t, 2>{32, 32}), ttnn::bfloat16),
        tt::tt_metal::ReplicateTensor(2),
        server_device);

    DistributedTensor output_tensor = run_model(tensor_a, tensor_b, server_device);

    server_device.close_device();

    return 0;
}
