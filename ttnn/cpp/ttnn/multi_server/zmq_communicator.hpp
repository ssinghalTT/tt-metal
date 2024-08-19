#pragma once

#include <iomanip>   // std::setw
#include <iostream>  // std::cout, std::cerr, std::endl
#include <string>
#include <zmq.hpp>

#include "tt_metal/common/logger.hpp"
#include "tt_metal/tt_stl/reflection.hpp"
#include "ttnn/cpp/ttnn/multi_server/message.hpp"

namespace ttnn {
namespace multi_server {

enum class CommunicatorType { Client, Server };

// Serialize/Deserialize utilities
namespace io {

struct JsonMessageSerializer {
    static Message deserialize(const std::string& serialized) {
        return tt::stl::json::from_json_t<Message>()(nlohmann::json::parse(serialized));
    }

    static std::string serialize(const Message& message) { return tt::stl::json::to_json_t<Message>()(message).dump(); }
};

}  // namespace io

class ZmqServerDeviceCommunicator {
   public:
    ZmqServerDeviceCommunicator(const std::string& address) : context_(1), socket_(context_, zmq::socket_type::dealer) {
        socket_.set(zmq::sockopt::routing_id, "CLIENT");
        socket_.connect(address);
    }

    void send_message(std::string&& message) { socket_.send(zmq::buffer(std::move(message)), zmq::send_flags::none); }

    std::string receive_message() {
        zmq::message_t zmq_message;
        auto recv_result = socket_.recv(zmq_message);
        if (!recv_result) {
            throw std::runtime_error("Failed to receive message");
        }
        return std::string(static_cast<char*>(zmq_message.data()), zmq_message.size());
    }

   private:
    zmq::context_t context_;
    zmq::socket_t socket_;
};

class ZmqServerCommunicator {
   public:
    ZmqServerCommunicator(const std::string& address, std::string client_identity = "CLIENT") :
        context_(1),
        socket_(context_, zmq::socket_type::router),
        identity_(std::move(client_identity)),
        identity_buffer_(identity_.data(), identity_.size()) {
        socket_.set(zmq::sockopt::routing_id, identity_);
        socket_.bind(address);
    }

    void send_message(const std::string& message) {
        zmq::message_t payload(message.data(), message.size());

        socket_.send(identity_buffer_, zmq::send_flags::sndmore);
        socket_.send(payload, zmq::send_flags::none);
    }

    std::string receive_message() {
        zmq::recv_result_t recv_result;

        recv_result = socket_.recv(identity_message_);
        if (!recv_result) {
            tt::log_fatal(tt::LogType::LogServer, "Failed to receive identity");
            throw std::runtime_error("Failed to receive identity");
        }

        recv_result = socket_.recv(payload_message_);
        if (!recv_result) {
            tt::log_fatal(tt::LogType::LogServer, "Failed to receive payload");
            throw std::runtime_error("Failed to receive payload");
        }

        return std::string(static_cast<char*>(payload_message_.data()), payload_message_.size());
    }

   private:
    zmq::context_t context_;
    zmq::socket_t socket_;
    const std::string identity_;
    const zmq::const_buffer identity_buffer_;
    zmq::message_t identity_message_;
    zmq::message_t payload_message_;
};

template <typename CommunicatorImpl>
class MessageCommunicator {
   public:
    MessageCommunicator(CommunicatorType type, const std::string& address) :
        communicator_(address),
        logger_type_(type == CommunicatorType::Client ? tt::LogType::LogClient : tt::LogType::LogServer) {}

    void send_message(const Message& message) {
        auto serialized_message = io::JsonMessageSerializer::serialize(message);
        tt::log_debug(logger_type_, "Sending message {} of size: {}", serialized_message, serialized_message.size());
        communicator_.send_message(std::move(serialized_message));
    }

    Message receive_message() {
        auto serialized_message = communicator_.receive_message();
        tt::log_debug(logger_type_, "Received message {} of size: {}", serialized_message, serialized_message.size());
        return io::JsonMessageSerializer::deserialize(serialized_message);
    }

   private:
    tt::LogType logger_type_;
    CommunicatorImpl communicator_;
};

// Usage
using ServerDeviceMessageCommunicator = MessageCommunicator<ZmqServerDeviceCommunicator>;
using ServerMessageCommunicator = MessageCommunicator<ZmqServerCommunicator>;

}  // namespace multi_server
}  // namespace ttnn
