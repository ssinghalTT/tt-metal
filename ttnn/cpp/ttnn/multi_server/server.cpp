#include "ttnn/cpp/ttnn/multi_server/server.hpp"

//#include "tt_eager/tensor/serialization.hpp"
#include "ttnn/cpp/ttnn/operations/core/core.hpp"
#include "ttnn/cpp/ttnn/tensor/serialization.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "ttnn/cpp/ttnn/multi_server/message.hpp"
#include "ttnn/cpp/ttnn/multi_server/utils.hpp"
#include "ttnn/device.hpp"
#include <chrono>
#include <thread>
#include <iostream>
#include <type_traits>

#include "ttnn/cpp/ttnn/multi_device.hpp"

namespace ttnn {
namespace multi_server {

DistributedTensor TensorManager::store_tensor(ttnn::Tensor&& tensor) {
    auto server_tensor = DistributedTensor(next_id_++);
    tensors_[server_tensor.id] = std::move(tensor);
    return server_tensor;
}

ttnn::Tensor& TensorManager::get_tensor(DistributedTensor server_tensor) {
    auto it = tensors_.find(server_tensor.id);
    if (it == tensors_.end()) {
        throw std::runtime_error("Tensor not found");
    }
    return it->second;
}

void TensorManager::remove_tensor(DistributedTensor server_tensor) { tensors_.erase(server_tensor.id); }

template <typename DeviceType>
Worker<DeviceType>::Worker(const std::string& address) : comm_(CommunicatorType::Server, address) {
    if constexpr (std::is_same_v<DeviceType, tt::tt_metal::DeviceMesh>) {
        tt::log_info(tt::LogType::LogServer, "Using DeviceMesh");

        // TODO: deal with this raw heap allocation

        // Check if MMIO_OFFSET environment variable is set
        const char* mmio_offset_env = std::getenv("MMIO_OFFSET");
        size_t mmio_offset = 0;
        if (mmio_offset_env != nullptr) {
            try {
                mmio_offset = std::stoul(mmio_offset_env);
                log_info(tt::LogMetal, "Using MMIO_OFFSET from environment: {}", mmio_offset);
            } catch (const std::exception& e) {
                log_warning(tt::LogMetal, "Invalid MMIO_OFFSET value in environment. Using default: {}", mmio_offset);
            }
        }

        this->device_ =
            new tt::tt_metal::DeviceMesh(tt::tt_metal::DeviceGrid{1, 2}, tt::tt_metal::DeviceIds{0, 1}, 0, 0, 1, tt::tt_metal::DispatchCoreType::WORKER, mmio_offset);
    } else if constexpr (std::is_same_v<DeviceType, tt::tt_metal::Device>) {
        tt::log_info(tt::LogType::LogServer, "Using Single Device");
        const int device_id = 0;
        auto& device = ttnn::open_device(device_id);
        this->device_ = &device;
    } else {
        throw std::runtime_error("Unsupported device type");
    }

    // brute-force this for now and akhmed will find some magic
    this->dispatch_table.registerFunction("add", ttnn::add);
    // this->dispatch_table.registerFunction("matmul", ttnn::matmul);
    this->dispatch_table.registerFunction("to_device", ttnn::to_device<DeviceType>);
    this->dispatch_table.registerFunction("to_layout", ttnn::to_layout);
}

template <typename DeviceType>
Worker<DeviceType>::~Worker() {}

template <typename DeviceType>
ttnn::Tensor Worker<DeviceType>::execute(
    const std::string& op_name, ttnn::Tensor& a, ttnn::Tensor& b, DeviceType& device) {
    try {
        std::cout << "[SERVER] Executing operation: " << op_name << std::endl;
        std::cout << "[SERVER] Tensor A type: " << typeid(a).name() << std::endl;
        std::cout << "[SERVER] Tensor B type: " << typeid(b).name() << std::endl;
        std::cout << "[SERVER] Device type: " << typeid(device).name() << std::endl;

        if constexpr (std::is_same_v<DeviceType, ttnn::Device>) {
            return this->dispatch_table.invoke_operation(op_name, a, b);
        } else if constexpr (std::is_same_v<DeviceType, tt::tt_metal::DeviceMesh>) {
            auto first_device = device.get_devices()[0];
            return this->dispatch_table.invoke_operation(op_name, a, b);
        } else {
            throw std::runtime_error("Unsupported device type");
        }
    } catch (const std::bad_any_cast& e) {
        std::cerr << "[SERVER] Bad any cast in execute: " << e.what() << std::endl;
        std::cerr << "[SERVER] Operation: " << op_name << std::endl;
        std::cerr << "[SERVER] Tensor A type: " << typeid(a).name() << std::endl;
        std::cerr << "[SERVER] Tensor B type: " << typeid(b).name() << std::endl;
        std::cerr << "[SERVER] Device type: " << typeid(device).name() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "[SERVER] Error in execute: " << e.what() << std::endl;
        throw;
    }
}

template <typename DeviceType>
void Worker<DeviceType>::handle_store_tensor(const Message& message) {
    try {
        auto& binarized_arguments = message.binarized_arguments;
        std::cout << "[SERVER] Binarized arguments size: " << binarized_arguments.size() << std::endl;

        if (binarized_arguments.empty()) {
            throw std::runtime_error("No binarized arguments provided");
        }

        std::string binarized_arguments_string(
            reinterpret_cast<const char*>(binarized_arguments[0].data()), binarized_arguments[0].size());
        std::cout << "[SERVER] Binarized argument size: " << binarized_arguments_string.size() << " bytes" << std::endl;

        auto tensor = decode_tensor(binarized_arguments_string, this->device_);
        std::cout << "after decode" << std::endl;
        auto id = tensor_manager_.store_tensor(std::move(tensor));
        comm_.send_message(Message{
            .type = MessageType::RESPONSE_VALID,
            .tensor_ids = {id},
        });
    } catch (const std::exception& e) {
        std::cerr << "[SERVER] Error in handle_store_tensor: " << e.what() << std::endl;
        comm_.send_message(Message{
            .type = MessageType::RESPONSE_INVALID,
            .operation = "ERROR: " + std::string(e.what()),
            .tensor_ids = {},
            .binarized_arguments = {}});
    }
}

template <typename DeviceType>
void Worker<DeviceType>::handle_operation(const Message& message) {
    try {
        auto& input_a = tensor_manager_.get_tensor(message.tensor_ids[0]);
        auto& input_b = tensor_manager_.get_tensor(message.tensor_ids[1]);

        std::cout << "[SERVER] Operation received: " << message.operation << std::endl;
        std::cout << "[SERVER] Input tensor A type: " << typeid(input_a).name() << std::endl;
        std::cout << "[SERVER] Input tensor B type: " << typeid(input_b).name() << std::endl;
        std::cout << "[SERVER] Device type: " << typeid(*device_).name() << std::endl;

        auto result = execute(message.operation, input_a, input_b, *device_);
        auto result_id = tensor_manager_.store_tensor(std::move(result));

        comm_.send_message(Message{
            .type = MessageType::RESPONSE_VALID,
            .operation = "",
            .tensor_ids = {result_id},
            .binarized_arguments = {}});
    } catch (const std::bad_any_cast& e) {
        std::cerr << "[SERVER] Bad any cast error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[SERVER] Error in handle_operation: " << e.what() << std::endl;
    }
}

template <typename DeviceType>
void Worker<DeviceType>::handle_to_device(const Message& message) {
    if (message.tensor_ids.size() != 1) {
        throw std::runtime_error("Incorrect number of tensors for to_device");
        return;
    }
    auto& input = tensor_manager_.get_tensor(message.tensor_ids[0]);
    ttnn::Tensor result =
        this->dispatch_table.invoke_operation(message.operation, input, device_, ttnn::types::DRAM_MEMORY_CONFIG);

    auto result_id = tensor_manager_.store_tensor(std::move(result));
    comm_.send_message(Message{
        .type = MessageType::RESPONSE_VALID, .operation = "", .tensor_ids = {result_id}, .binarized_arguments = {}});
}

template <typename DeviceType>
void Worker<DeviceType>::handle_to_layout(const Message& message) {
    if (message.tensor_ids.size() != 1) {
        // Handle error: incorrect number of tensors
        return;
    }

    auto& input = tensor_manager_.get_tensor(message.tensor_ids[0]);
    ttnn::Tensor result;

    if constexpr (std::is_same_v<DeviceType, ttnn::Device>) {
        result = ttnn::to_layout(input, ttnn::TILE_LAYOUT, std::nullopt, std::nullopt, device_);
    } else if constexpr (std::is_same_v<DeviceType, tt::tt_metal::DeviceMesh>) {
        // For DeviceMesh, we might need to handle layout conversion differently
        // This is a placeholder implementation
        result = ttnn::to_layout(input, ttnn::TILE_LAYOUT, std::nullopt, std::nullopt, device_);
    }

    auto result_id = tensor_manager_.store_tensor(std::move(result));
    comm_.send_message(Message{
        .type = MessageType::RESPONSE_VALID, .operation = "", .tensor_ids = {result_id}, .binarized_arguments = {}});
}

template <typename DeviceType>
void Worker<DeviceType>::handle_fetch_tensor(const Message& message) {
    if (message.tensor_ids.size() != 1) {
        // Handle error: incorrect number of tensors
        return;
    }

    auto& tensor = tensor_manager_.get_tensor(message.tensor_ids[0]);
    auto host_tensor = ttnn::from_device(tensor);
    auto tensor_data = encode_tensor(host_tensor);
    std::vector<uint8_t> tensor_data_bytes(tensor_data.begin(), tensor_data.end());
    std::vector<std::vector<uint8_t>> binarized_arguments = {tensor_data_bytes};

    comm_.send_message(Message{
        .type = MessageType::RESPONSE_VALID,
        .operation = "",
        .tensor_ids = {},
        .binarized_arguments = binarized_arguments});
}

template <typename DeviceType>
void Worker<DeviceType>::handle_remove_tensor(const Message& message) {
    for (const auto& id : message.tensor_ids) {
        tensor_manager_.remove_tensor(id);
    }
    comm_.send_message(
        Message{.type = MessageType::RESPONSE_VALID, .operation = "", .tensor_ids = {}, .binarized_arguments = {}});
}

template <typename DeviceType>
void Worker<DeviceType>::handle_handshake(const Message& message) {
    comm_.send_message(
        Message{.type = MessageType::RESPONSE_VALID, .operation = "", .tensor_ids = {}, .binarized_arguments = {}});
}

template <typename DeviceType>
void Worker<DeviceType>::run() {
    auto last_message_time = std::chrono::high_resolution_clock::now();
    while (true) {
        Message message = comm_.receive_message();
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_message_time);
        last_message_time = current_time;
        std::cout << "[SERVER] Time elapsed since last message: " << elapsed.count() << " ms" << std::endl;

        try {
            switch (message.type) {
                case MessageType::STORE_TENSOR: handle_store_tensor(message); break;
                case MessageType::OPERATION: handle_operation(message); break;
                case MessageType::TO_DEVICE: handle_to_device(message); break;
                case MessageType::TO_LAYOUT: handle_to_layout(message); break;
                case MessageType::FETCH_TENSOR: handle_fetch_tensor(message); break;
                case MessageType::REMOVE_TENSOR: handle_remove_tensor(message); break;
                case MessageType::KILL:
                {
                    if constexpr (std::is_same_v<DeviceType, tt::tt_metal::DeviceMesh>) {
                        // For DeviceMesh, we need to delete the dynamically allocated object
                        delete device_;
                    } else if constexpr (std::is_same_v<DeviceType, ttnn::Device>) {
                        // For single Device, we need to close it
                        ttnn::close_device(*device_);
                    }
                    // Set the pointer to nullptr to avoid potential use-after-free
                    device_ = nullptr;

                    std::cout << "[SERVER] Received kill signal. Shutting down." << std::endl;
                    comm_.send_message(Message{
                        .type = MessageType::KILL, .operation = "KILLED", .tensor_ids = {}, .binarized_arguments = {}});

                    // Sleep for a short duration before returning
                    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                    return;
                }
                case MessageType::HANDSHAKE: handle_handshake(message); break;
                default: throw std::runtime_error("Unknown message type");
            }
        } catch (const std::exception& e) {
            std::cerr << "[SERVER] E1ggrror: " << e.what() << std::endl;
            std::cerr << "[SERVER] Error type: " << typeid(e).name() << std::endl;
            // comm_.send_message(Message(MessageType::RESPONSE_INVALID, "ERROR: " + std::string(e.what())));
        } catch (...) {
            std::cerr << "[SERVER] Unknown exception caught in run loop" << std::endl;
            // kcomm_.send_message(Message(MessageType::RESPONSE_INVALID, "ERROR: Unknown exception"));
        }
    }
}

template class Worker<tt::tt_metal::Device>;
template class Worker<tt::tt_metal::DeviceMesh>;

}  // namespace multi_server
}  // namespace ttnn
