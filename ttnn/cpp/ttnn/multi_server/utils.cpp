#include "ttnn/cpp/ttnn/multi_server/utils.hpp"
#include "ttnn/cpp/ttnn/tensor/serialization.hpp"

namespace ttnn {
namespace multi_server {

std::string encode_tensor(const ttnn::Tensor& t) {
    std::ostringstream oss;
    tt::tt_metal::dump_tensor(oss, t);
    return oss.str();
}

DistributedTensor create_multi_server_tensor(const ttnn::Tensor& t, const DistributedTensorConfig& strategy, ServerDevice& server_device) {
    TT_ASSERT(std::holds_alternative<tt::tt_metal::ReplicateTensor>(strategy));
    auto replicate = std::get<tt::tt_metal::ReplicateTensor>(strategy);

    std::vector<ttnn::Tensor> tensors;
    for (int i = 0; i < replicate.replication_factor; ++i) {
        tensors.push_back(t);
    }
    auto multi_device_tensor = tt::tt_metal::create_multi_device_tensor(
        tensors, tt::tt_metal::StorageType::MULTI_DEVICE_HOST, strategy);

    return server_device.distribute_tensor(multi_device_tensor);
}


}  // namespace multi_server
}  // namespace ttnn
