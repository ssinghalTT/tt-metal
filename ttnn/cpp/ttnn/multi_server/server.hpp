#pragma once

#include <string>
#include <optional>
#include <unordered_map>
#include <memory>

#include "ttnn/cpp/ttnn/multi_server/utils.hpp"
#include "ttnn/cpp/ttnn/multi_server/message.hpp"
#include "ttnn/cpp/ttnn/multi_server/zmq_communicator.hpp"
#include "ttnn/cpp/ttnn/tensor/tensor.hpp"


namespace ttnn {
namespace multi_server {

class TensorManager {
public:
    DistributedTensor store_tensor(ttnn::Tensor&& tensor);
    ttnn::Tensor& get_tensor(DistributedTensor id);
    void remove_tensor(DistributedTensor id);

private:
    std::unordered_map<uint64_t, ttnn::Tensor> tensors_;
    uint64_t next_id_ = 1;
};


template <typename DeviceType>
class Worker {
public:
    Worker(const std::string& address);
    ~Worker();

    void run();

private:
    ttnn::Tensor execute(const std::string& op_name, ttnn::Tensor& a, ttnn::Tensor& b, DeviceType& device);
    void handle_store_tensor(const Message& message);
    void handle_operation(const Message& message);
    void handle_to_device(const Message& message);
    void handle_to_layout(const Message& message);
    void handle_fetch_tensor(const Message& message);
    void handle_remove_tensor(const Message& message);
    void handle_handshake(const Message& message);

    TensorManager tensor_manager_;
    FunctionDispatchTable dispatch_table;
    ServerMessageCommunicator comm_;
    DeviceType* device_;
};
} // namespace multi_server
} // namespace ttnn
