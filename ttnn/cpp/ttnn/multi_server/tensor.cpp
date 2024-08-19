#include "ttnn/cpp/ttnn/multi_server/tensor.hpp"
#include "ttnn/cpp/ttnn/multi_server/client.hpp"


namespace ttnn::multi_server {
    void DistributedTensor::print() const {

        std::cout << "Trying to fetch tensor from server device " << this->server_device_id << std::endl;
        auto device = ServerDevice::get_device(this->server_device_id);
        auto result = device->fetch_tensor(*this);
        result.print();
    }


} // namespace ttnn::multi_server
