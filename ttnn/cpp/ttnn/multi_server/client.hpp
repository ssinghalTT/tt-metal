#pragma once
#include <string>
#include <vector>
#include <zmq.hpp>
#include "ttnn/cpp/ttnn/multi_server/message.hpp"
#include "ttnn/cpp/ttnn/multi_server/zmq_communicator.hpp"
#include "ttnn/cpp/ttnn/tensor/tensor.hpp"

namespace ttnn {
namespace multi_server {

class ServerDevice {
public:
    ServerDevice(const std::string& server_address);

    DistributedTensor distribute_tensor(const ttnn::Tensor& tensor);
    DistributedTensor to_device(DistributedTensor tensor_id);
    DistributedTensor to_layout(DistributedTensor tensor_id);
    ttnn::Tensor fetch_tensor(DistributedTensor id);
    void remove_tensors(const std::vector<DistributedTensor>& ids);
    void close_device();
    bool handshake();
    void initialize();

    template <typename... Args>
    DistributedTensor run_operation(const std::string& op_name, Args... args);

    // Operations
    DistributedTensor add(DistributedTensor a_id, DistributedTensor b_id);
    static ServerDevice* get_device(uint64_t device_id);

private:
    static uint64_t next_device_id_;
    static std::unordered_map<uint64_t, ServerDevice*> global_servers_;

    uint64_t device_id_;
    ServerDeviceMessageCommunicator comm_;
};

}  // namespace multi_server
}  // namespace ttnn
