#include <zmq.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <string>
#include <vector>

#include "ttnn/device.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "tt_eager/tensor/serialization.hpp"
#include "tt_eager/tensor/types.hpp"

// Function to send messages to the server in a loop
void send_messages(zmq::context_t& context) {
    zmq::socket_t send_socket(context, zmq::socket_type::dealer);
    send_socket.connect("tcp://localhost:8086");  // Port for sending messages

    for (int i = 1; i <= 100; ++i) {
        std::string config = "SHARD " + std::to_string(i);
        zmq::message_t config_message(config.data(), config.size());

        std::string tensor_info = "tensor_data_" + std::to_string(i);
        zmq::message_t tensor_message(tensor_info.data(), tensor_info.size());

        send_socket.send(config_message, zmq::send_flags::sndmore);
        send_socket.send(tensor_message, zmq::send_flags::none);

        // Optionally, add a delay to simulate processing time or network latency
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// Function to receive messages from the server
void receive_messages(zmq::context_t& context) {
    zmq::socket_t recv_socket(context, zmq::socket_type::dealer);
    recv_socket.connect("tcp://localhost:8087");  // Port for receiving messages

    while (true) {
        zmq::message_t identity;
        if (recv_socket.recv(identity, zmq::recv_flags::dontwait)) {
            zmq::message_t processed_tensor;
            auto result = recv_socket.recv(processed_tensor);
            if (!result) {
                std::cerr << "Failed to receive processed tensor" << std::endl;
                continue;
            }
            // Process the response
            std::string identity_str(static_cast<char*>(identity.data()), identity.size());
            std::string processed_tensor_str(static_cast<char*>(processed_tensor.data()), processed_tensor.size());

            // Handle response
            std::cout << "Received response: " << identity_str << std::endl;
            std::cout << "Processed tensor: " << processed_tensor_str << std::endl;
        }
        // Perform other tasks
    }
}

ttnn::Tensor get_tensor(ttnn::Tensor& t, ttnn::Device& device) {

    std::ostringstream oss;
    tt::tt_metal::dump_tensor(oss, t);
    std::istringstream iss(oss.str());
    Tensor deserialized_tensor = tt::tt_metal::load_tensor(iss, &device);

    return deserialized_tensor;
}

Tensor execute(ttnn::Tensor& a, ttnn::Tensor& b, ttnn::Device& device) {

    auto t_a = ttnn::to_device(a, &device, ttnn::types::DRAM_MEMORY_CONFIG);
    auto t_b = ttnn::to_device(b, &device, ttnn::types::DRAM_MEMORY_CONFIG);

    auto tt_a = ttnn::to_layout(t_a, ttnn::TILE_LAYOUT, std::nullopt, std::nullopt, &device);//, std::nullopt, ttnn::types::DRAM_MEMORY_CONFIG, &device);
    auto tt_b = ttnn::to_layout(t_b, ttnn::TILE_LAYOUT, std::nullopt, std::nullopt, &device);//, std::nullopt, ttnn::types::DRAM_MEMORY_CONFIG, &device);


    auto c = ttnn::add(tt_a, tt_b);
    auto d = ttnn::from_device(c);

    return d;
}

int main() {
    const auto device_id = 0;
    auto& device = ttnn::open_device(device_id);
    std::array<uint32_t, 2> dimensions = {32, 32};
    ttnn::Shape shape(dimensions);

    auto a = ttnn::ones(shape, ttnn::bfloat16);
    auto b = ttnn::ones(shape, ttnn::bfloat16);


    // Decode Step
    auto t_a = get_tensor(a, device);
    auto t_b = get_tensor(b, device);

    auto d = execute(t_a, t_b, device);
    d.print();


    ttnn::close_device(device);
    /*


    zmq::context_t context(1);

    // Start sending thread
    std::thread sender_thread(send_messages, std::ref(context));

    // Start receiving thread
    std::thread receiver_thread(receive_messages, std::ref(context));

    // Join threads (optional, depending on whether you want main to wait for these threads)
    sender_thread.join();
    receiver_thread.join();
    */

    return 0;
}
