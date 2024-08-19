#pragma once

#include <atomic>
#include <string>
#include <vector>

#include "third_party/json/json.hpp"
#include "tt_metal/tt_stl/reflection.hpp"  // Add this include
#include "ttnn/cpp/ttnn/multi_server/tensor.hpp"


namespace ttnn {
namespace multi_server {

using MessageId = uint64_t;

enum class MessageType {
    RESPONSE_INVALID,
    RESPONSE_VALID,
    STORE_TENSOR,
    OPERATION,
    TO_DEVICE,
    TO_LAYOUT,
    FETCH_TENSOR,
    REMOVE_TENSOR,
    KILL,
    HANDSHAKE
};

struct Message {
    MessageType type;
    std::string operation = "";
    std::vector<DistributedTensor> tensor_ids = {};
    std::vector<std::vector<uint8_t>> binarized_arguments = {};
};

}  // namespace multi_server
}  // namespace ttnn

namespace tt::stl::json {

    /**/

template <>
struct to_json_t<ttnn::multi_server::Message> {
    nlohmann::json operator()(const ttnn::multi_server::Message& message) noexcept
    {
        return {{"type", magic_enum::enum_name(message.type)},
                {"operation", to_json(message.operation)},
                {"tensor_ids", to_json(message.tensor_ids)},
                {"binarized_arguments", to_json(message.binarized_arguments)}};
    }
};

template <>
struct from_json_t<ttnn::multi_server::Message> {
    ttnn::multi_server::Message operator()(const nlohmann::json& json_object) noexcept
    {
        return ttnn::multi_server::Message(
            magic_enum::enum_cast<ttnn::multi_server::MessageType>(json_object["type"].get<std::string>()).value(),
            from_json<std::string>(json_object["operation"]),
            from_json<std::vector<ttnn::multi_server::DistributedTensor>>(json_object["tensor_ids"]),
            from_json<std::vector<std::vector<uint8_t>>>(json_object["binarized_arguments"]));
    }
};

}  // namespace tt::stl::json
