#include <zmq.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <sstream>

#include "ttnn/device.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/core.hpp"
#include "ttnn/operations/creation.hpp"

// Define the process_tensor function
zmq::message_t process_tensor(const zmq::message_t& tensor_message) {
    // Dummy processing logic
    std::string processed = "processed_" + std::string(static_cast<const char*>(tensor_message.data()), tensor_message.size());
    return zmq::message_t(processed.data(), processed.size());
}

struct MessagePackage {
    zmq::message_t identity;
    zmq::message_t processed_tensor;
};

std::queue<MessagePackage> response_queue;
std::mutex queue_mutex;
std::condition_variable queue_cv;

void receive_messages(zmq::context_t& context) {
    zmq::socket_t router_socket(context, zmq::socket_type::router);
    router_socket.bind("tcp://*:8086");  // Port for receiving messages

    while (true) {
        zmq::message_t identity;
        if (!router_socket.recv(identity, zmq::recv_flags::none)) {
            std::cerr << "Failed to receive identity message" << std::endl;
            continue; // Skip this iteration and try again
        }

        std::string identity_str(static_cast<char*>(identity.data()), identity.size());
        std::cout << "Received identity: " << identity_str << " size: " << identity.size() << std::endl;

        if (!router_socket.get(zmq::sockopt::rcvmore)) {
            std::cerr << "Expected more message parts after identity" << std::endl;
            continue; // Skip this iteration and try again
        }

        zmq::message_t config_message;
        if (!router_socket.recv(config_message, zmq::recv_flags::none)) {
            std::cerr << "Failed to receive config message" << std::endl;
            continue; // Skip this iteration and try again
        }

        if (!router_socket.get(zmq::sockopt::rcvmore)) {
            std::cerr << "Expected more message parts after config" << std::endl;
            continue; // Skip this iteration and try again
        }

        zmq::message_t tensor_message;
        if (!router_socket.recv(tensor_message, zmq::recv_flags::none)) {
            std::cerr << "Failed to receive tensor message" << std::endl;
            continue; // Skip this iteration and try again
        }

        std::string config_str(static_cast<char*>(config_message.data()), config_message.size());
        std::istringstream config_stream(config_str);
        std::string config_type;
        int shard_number;

        config_stream >> config_type >> shard_number;
        std::cout << "Received identity: " << identity_str << std::endl;
        std::cout << "Received config: " << config_type << " " << shard_number << std::endl;

        if (config_type == "SHARD") {
            std::cout << "Handling SHARD for shard number: " << shard_number << std::endl;
        } else if (config_type == "REPLICATE") {
            std::cout << "Handling REPLICATION" << std::endl;
        }

        zmq::message_t processed_tensor = process_tensor(tensor_message);

        // Add the response to the queue
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            response_queue.push({std::move(identity), std::move(processed_tensor)});
        }
        queue_cv.notify_one();
    }
}

void send_responses(zmq::context_t& context) {
    zmq::socket_t dealer_socket(context, zmq::socket_type::dealer);
    dealer_socket.bind("tcp://*:8087");  // Port for sending responses

    while (true) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        queue_cv.wait(lock, [] { return !response_queue.empty(); });

        auto response = std::move(response_queue.front());
        response_queue.pop();
        lock.unlock();

        dealer_socket.send(response.identity, zmq::send_flags::sndmore);
        dealer_socket.send(response.processed_tensor, zmq::send_flags::none);
    }
}

int main() {

    zmq::context_t context{1};

    // Start receiving thread
    std::thread receiver_thread(receive_messages, std::ref(context));

    // Start sending thread
    std::thread sender_thread(send_responses, std::ref(context));

    // Join threads (optional, depending on whether you want main to wait for these threads)
    receiver_thread.join();
    sender_thread.join();

    return 0;
}
