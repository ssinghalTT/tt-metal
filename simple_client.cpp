#include <zmq.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <string>
#include <vector>

#include <functional>
#include <memory>
#include <concepts>
#include <type_traits>


#include "ttnn/device.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "tt_eager/tensor/serialization.hpp"
#include "tt_eager/tensor/types.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/binary/binary.hpp"
#include "tt_metal/impl/device/multi_device.hpp"

#include <typeinfo>
#include <cxxabi.h>

#include <atomic>

#include "ttnn/cpp/ttnn/multi_server/message.hpp"
#include "ttnn/cpp/ttnn/multi_server/client.hpp"
#include "ttnn/cpp/ttnn/multi_server/server.hpp"


using namespace ttnn::multi_server;

void run_model(Client& client, Tensor& a, Tensor& b) {
    try {
        TensorId a_id = client.distribute_tensor(a); // RPC call
        TensorId b_id = client.distribute_tensor(b); // RPC call

        TensorId device_a_id = client.to_device(a_id);
        TensorId device_b_id = client.to_device(b_id);


        TensorId input_id_a = client.to_layout(device_a_id);
        TensorId input_id_b = client.to_layout(device_b_id);
        TensorId result_id;

        // Remote worker does function disable table lookup and executes the operation
        for (int i = 0; i < 10; i++) {
            std::cout << "[CLIENT] Running operation" << "add" << std::endl;
            result_id = client.run_operation("add", input_id_a, input_id_b);

            input_id_b = result_id;
            auto result = client.fetch_tensor(result_id);
            result.print();
        }

        client.remove_tensors({input_id_a, input_id_b, result_id});
        client.shutdown_server();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}


int main() {
    auto& device_ref = ttnn::open_device(0);

    ttnn::Shape shape(std::array<uint32_t, 2>{32, 32});

    auto a = ttnn::ones(shape, ttnn::bfloat16);
    auto b = ttnn::ones(shape, ttnn::bfloat16);

    Worker<ttnn::Device> worker("tcp://*:8086", &device_ref);
    std::thread server_worker_thread([&worker]() {
        worker.run();
    });
    auto client = Client("tcp://localhost:8086");
    run_model(client, a, b);

    server_worker_thread.join();
    ttnn::close_device(device_ref);

    return 0;
}
