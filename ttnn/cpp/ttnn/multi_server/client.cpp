#include "ttnn/cpp/ttnn/multi_server/client.hpp"
#include "ttnn/cpp/ttnn/multi_server/utils.hpp"
#include <stdexcept>
#include <iostream>
#include <string_view>

namespace ttnn {
namespace multi_server {

uint64_t ServerDevice::next_device_id_ = 0;

std::unordered_map<uint64_t, ServerDevice*> ServerDevice::global_servers_;


ServerDevice* ServerDevice::get_device(uint64_t device_id)
{
    return ServerDevice::global_servers_.at(device_id);
}

constexpr std::string_view function_name(const char* name) {
    return std::string_view(name);
}

ServerDevice::ServerDevice(const std::string& server_address)
    : comm_(CommunicatorType::Client, server_address), device_id_(next_device_id_++) {

    this->initialize();


}

DistributedTensor ServerDevice::distribute_tensor(const ttnn::Tensor& tensor) {
    std::string tensor_data = encode_tensor(tensor);
    std::vector<uint8_t> tensor_data_bytes(tensor_data.begin(), tensor_data.end());
    std::vector<std::vector<uint8_t>> binarized_arguments = {tensor_data_bytes};
    comm_.send_message(Message{
        .type = MessageType::STORE_TENSOR,
        .operation = "",
        .tensor_ids = {},
        .binarized_arguments = binarized_arguments});
    Message response = comm_.receive_message();
    if (response.type != MessageType::RESPONSE_VALID) {
        throw std::runtime_error("Failed to store tensor");
    }
    auto tensor_ids = response.tensor_ids;
    return tensor_ids.at(0);
}

template <typename... Args>
DistributedTensor ServerDevice::run_operation(const std::string& op_name, Args... args) {
    comm_.send_message(Message{
        .type = MessageType::OPERATION,
        .operation = op_name,
        .tensor_ids = {args...},
        .binarized_arguments = {}
    });
    Message response = comm_.receive_message();
    if (response.type != MessageType::RESPONSE_VALID) {
        throw std::runtime_error("Operation failed");
    }
    auto tensor_ids = response.tensor_ids;
    return tensor_ids.at(0);
}

DistributedTensor ServerDevice::add(DistributedTensor a_id, DistributedTensor b_id) {
    return run_operation(std::string{function_name(__FUNCTION__)}, a_id, b_id);
}

DistributedTensor ServerDevice::to_device(DistributedTensor tensor_id) {
    comm_.send_message(Message{
        .type = MessageType::TO_DEVICE,
        .operation = "to_device",
        .tensor_ids = {tensor_id},
        .binarized_arguments = {}
    });
    Message response = comm_.receive_message();
    if (response.type != MessageType::RESPONSE_VALID) {
        throw std::runtime_error("To_device operation failed");
    }
    auto tensor_ids = response.tensor_ids;
    return tensor_ids.at(0);
}

DistributedTensor ServerDevice::to_layout(DistributedTensor tensor_id) {
    comm_.send_message(Message{
        .type = MessageType::TO_LAYOUT,
        .operation = "",
        .tensor_ids = {tensor_id},
        .binarized_arguments = {}
    });
    Message response = comm_.receive_message();
    if (response.type != MessageType::RESPONSE_VALID) {
        throw std::runtime_error("To_layout operation failed");
    }
    auto tensor_ids = response.tensor_ids;
    return tensor_ids.at(0);
}

ttnn::Tensor ServerDevice::fetch_tensor(DistributedTensor id) {
    comm_.send_message(Message{
        .type = MessageType::FETCH_TENSOR,
        .operation = "",
        .tensor_ids = {id},
        .binarized_arguments = {}
    });
    Message response = comm_.receive_message();
    if (response.type != MessageType::RESPONSE_VALID) {
        throw std::runtime_error("Failed to fetch tensor");
    }
    auto binarized_arguments = response.binarized_arguments;
    std::string binarized_arguments_string(reinterpret_cast<const char*>(binarized_arguments[0].data()), binarized_arguments[0].size());
    return decode_tensor(binarized_arguments_string);
}

void ServerDevice::remove_tensors(const std::vector<DistributedTensor>& ids) {
    comm_.send_message(Message{
        .type = MessageType::REMOVE_TENSOR,
        .operation = "",
        .tensor_ids = ids,
        .binarized_arguments = {}
    });
    Message response = comm_.receive_message();
    if (response.type != MessageType::RESPONSE_VALID) {
        throw std::runtime_error("Failed to remove tensors");
    }
}

void ServerDevice::close_device() {
    comm_.send_message(Message{
        .type = MessageType::KILL,
        .operation = "",
        .tensor_ids = {},
        .binarized_arguments = {}
    });
    Message response = comm_.receive_message();
}

bool ServerDevice::handshake() {
    comm_.send_message(Message{
        .type = MessageType::HANDSHAKE,
        .operation = "",
        .tensor_ids = {},
        .binarized_arguments = {}
    });
    Message response = comm_.receive_message();
    return (response.type == MessageType::RESPONSE_VALID);
}


void ServerDevice::initialize() {

    // Perform handshake
    if (this->handshake()) {
        std::cout << "Handshake successful. Registering device..." << std::endl;
        // Register the device only after successful handshake
        ServerDevice::global_servers_[this->device_id_] = this;
    } else {
        std::cerr << "Handshake failed. Exiting..." << std::endl;
        throw std::runtime_error("Handshake failed");
    }
}

}  // namespace multi_server
}  // namespace ttnn
