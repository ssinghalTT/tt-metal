#pragma once

#include "third_party/json/json.hpp"
#include "tt_metal/tt_stl/reflection.hpp"  // Add this include

namespace ttnn {
namespace multi_server {

struct DistributedTensor {
    // Remove the default, and single-arg constructor
    DistributedTensor() : id(0), server_device_id(0) { }
    DistributedTensor(uint64_t id) : id(id), server_device_id(0) { }
    DistributedTensor(uint64_t id, uint64_t server_device_id) : id(id), server_device_id(server_device_id) { }

    uint64_t id;
    uint64_t server_device_id;

    void print() const;
};

} // namespace multi_server
} // namespace ttnn

namespace tt::stl::json {

template <>
struct to_json_t<ttnn::multi_server::DistributedTensor> {
    nlohmann::json operator()(const ttnn::multi_server::DistributedTensor &tensor) noexcept {
        return {{"id", tensor.id}, {"server_device_id", tensor.server_device_id}};
    }
};

template <>
struct from_json_t<ttnn::multi_server::DistributedTensor> {
    ttnn::multi_server::DistributedTensor operator()(const nlohmann::json& json_object) noexcept {
        return ttnn::multi_server::DistributedTensor(json_object["id"].get<uint64_t>(), json_object["server_device_id"].get<uint64_t>());
    }
};

} // namespace tt::stl::json
