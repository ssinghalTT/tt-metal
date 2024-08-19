#include <iostream>
#include <string_view>
#include <array>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <algorithm>
#include <functional>

// Compile-time string type
template<size_t N>
struct FixedString {
    char value[N];

    constexpr FixedString(const char (&str)[N]) {
        std::copy_n(str, N, value);
    }

    constexpr operator std::string_view() const { return std::string_view(value, N - 1); }
};

template<size_t N>
FixedString(const char (&)[N]) -> FixedString<N>;

// Operation pair
template<typename Name, typename Func>
struct Operation {
    Name name;
    Func func;
};

// OperationRegistry to store operations
template<size_t Capacity>
struct OperationRegistry {
    Operation<FixedString<Capacity>, void(*)(void)> operations[Capacity];
    size_t size = 0;

    constexpr void add_operation(FixedString<Capacity> name, void(*func)(void)) {
        if (size < Capacity) {
            operations[size++] = {name, func};
        }
    }

    constexpr const auto& get_operations() const {
        return operations;
    }
};



// Helper to access the registry at compile-time
/*
template<size_t Capacity>
constexpr auto& get_registry() {
    static OperationRegistry<Capacity> registry;
    return registry;
}
*/

// Function to register an operation
template <typename Name, typename Func>
constexpr auto make_operation(Name name, Func&& func) {
    return Operation<Name, std::decay_t<Func>>{name, std::forward<Func>(func)};
}

// Primary template for function traits
template<typename T>
struct function_traits;

// Specialization for function types
template<typename R, typename... Args>
struct function_traits<R(Args...)> {
    using return_type = R;
    using args_tuple = std::tuple<Args...>;
    static constexpr std::size_t arity = sizeof...(Args);
};

// Specialization for function pointers
template<typename R, typename... Args>
struct function_traits<R(*)(Args...)> : function_traits<R(Args...)> {};

// Specialization for member function pointers
template<typename R, typename C, typename... Args>
struct function_traits<R(C::*)(Args...)> : function_traits<R(Args...)> {};

// Specialization for const member function pointers
template<typename R, typename C, typename... Args>
struct function_traits<R(C::*)(Args...) const> : function_traits<R(Args...)> {};

// Specialization for std::function
template<typename R, typename... Args>
struct function_traits<std::function<R(Args...)>> : function_traits<R(Args...)> {};

// Helper struct to handle const and & qualifiers
template<typename T>
struct remove_class {
    using type = T;
};

template<typename C, typename R, typename... Args>
struct remove_class<R(C::*)(Args...)> {
    using type = R(Args...);
};

template<typename C, typename R, typename... Args>
struct remove_class<R(C::*)(Args...) const> {
    using type = R(Args...);
};

// Specialization for functors and lambdas
template<typename T>
struct function_traits : function_traits<typename remove_class<decltype(&std::decay_t<T>::operator())>::type> {};

// Helper type alias for return type
template<typename T>
using function_return_type_t = typename function_traits<T>::return_type;

// Helper type alias for arguments tuple
template<typename T>
using function_args_tuple_t = typename function_traits<T>::args_tuple;

// Helper constant for arity
template<typename T>
inline constexpr std::size_t function_arity_v = function_traits<T>::arity;

// Operation Map
template<typename... Ops>
class OperationMap {
    std::tuple<Ops...> operations;

public:
    constexpr OperationMap(Ops... ops) : operations(std::move(ops)...) {}

    static constexpr size_t size() { return sizeof...(Ops); }

    template<typename... Args>
    constexpr auto invoke(std::string_view name, Args&&... args) const {
        return invoke_impl(name, std::index_sequence_for<Ops...>{}, std::forward<Args>(args)...);
    }

private:
    template<size_t... Is, typename... Args>
    constexpr auto invoke_impl(std::string_view name, std::index_sequence<Is...>, Args&&... args) const {
        using ResultVariant = std::variant<function_return_type_t<decltype(std::get<Is>(operations).func)>...>;
        ResultVariant result;
        bool found = false;
        (void)((name == std::string_view(std::get<Is>(operations).name) &&
                (result = ResultVariant(std::in_place_index<Is>, invoke_op<Is>(std::get<Is>(operations).func, std::forward<Args>(args)...)), found = true, true)) || ...);
        if (!found) {
            throw std::runtime_error("Operation not found");
        }
        return result;
    }

    template<size_t I, typename Func, typename... Args>
    constexpr auto invoke_op(Func&& func, Args&&... args) const {
        if constexpr (function_arity_v<Func> == sizeof...(Args)) {
            return func(std::forward<Args>(args)...);
        } else {
            return invoke_op_helper(std::forward<Func>(func), std::make_index_sequence<function_arity_v<Func>>{}, std::forward<Args>(args)...);
        }
    }

    template<typename Func, size_t... Is, typename... Args>
    constexpr auto invoke_op_helper(Func&& func, std::index_sequence<Is...>, Args&&... args) const {
        return func(std::get<Is>(std::forward_as_tuple(std::forward<Args>(args)...))...);
    }
};

// Helper function to create OperationMap
template<typename... Ops>
constexpr auto make_operation_map(Ops&&... ops) {
    return OperationMap<std::decay_t<Ops>...>(std::forward<Ops>(ops)...);
}

// Runtime sequence replay function
template<typename Map, typename... Args>
constexpr auto replay_sequence(const Map& map, const std::array<std::string_view, 3>& sequence, Args&&... args) {
    std::array<decltype(map.invoke(sequence[0], std::forward<Args>(args)...)), 3> results;

    for (size_t i = 0; i < sequence.size(); ++i) {
        results[i] = map.invoke(sequence[i], std::forward<Args>(args)...);
    }

    return results;
}

int main() {
    static constexpr auto Op1 = FixedString("Op1");
    static constexpr auto Op2 = FixedString("Op2");
    static constexpr auto Op3 = FixedString("Op3");

    constexpr auto op1 = make_operation(Op1, [](int x) { return x + 1; });
    constexpr auto op2 = make_operation(Op2, [](int x) { return x * 2; });
    constexpr auto op3 = make_operation(Op3, [](float x, float y) { return x + y; });

    constexpr auto operation_map = make_operation_map(op1, op2, op3);

    static_assert(operation_map.size() == 3, "Map should contain 3 operations");

    constexpr std::array<std::string_view, 3> sequence = {Op1, Op2, Op3};
    auto results = replay_sequence(operation_map, sequence, 5, 3.14f, 2.86f);

    for (const auto& result : results) {
        std::visit([](const auto& value) { std::cout << value << " "; }, result);
    }
    std::cout << std::endl;

    return 0;
}

/*
#include <zmq.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <string>
#include <vector>

#include <functional>
#include <memory>
#include <concepts>
#include <type_traits>


#include "ttnn/device.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "tt_eager/tensor/serialization.hpp"
#include "tt_eager/tensor/types.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/binary/binary.hpp"

#include <typeinfo>
#include <cxxabi.h>

template<typename T>
struct FunctionTraits;

template<typename R, typename... Args>
struct FunctionTraits<R(*)(Args...)> {
    using ReturnType = R;
    using ArgsTuple = std::tuple<Args...>;
    static constexpr size_t arity = sizeof...(Args);
};

template<typename C, typename R, typename... Args>
struct FunctionTraits<R(C::*)(Args...)> {
    using ReturnType = R;
    using ArgsTuple = std::tuple<Args...>;
    static constexpr size_t arity = sizeof...(Args);
};

template<typename C, typename R, typename... Args>
struct FunctionTraits<R(C::*)(Args...) const> {
    using ReturnType = R;
    using ArgsTuple = std::tuple<Args...>;
    static constexpr size_t arity = sizeof...(Args);
};

template<ttnn::operations::binary::BinaryOpType OpType, bool Inplace>
struct FunctionTraits<ttnn::decorators::operation_t<0, ttnn::operations::binary::ExecuteBinary<OpType, Inplace>>> {
    using ReturnType = ttnn::Tensor;
    using ArgsTuple = std::tuple<const ttnn::Tensor&, const ttnn::Tensor&>;
    static constexpr size_t arity = 2;
};

class FunctionDispatchTable {
private:
    class FunctionWrapperBase {
    public:
        virtual ~FunctionWrapperBase() = default;
        virtual std::any invoke(const std::vector<std::any>& args) const = 0;
        virtual const std::type_info& getReturnType() const = 0;
    };

    template<typename Func>
    class FunctionWrapper : public FunctionWrapperBase {
    private:
        Func func;

    public:
        FunctionWrapper(Func f) : func(std::move(f)) {}

        std::any invoke(const std::vector<std::any>& args) const override {
            return invokeHelper(args, std::make_index_sequence<FunctionTraits<Func>::arity>{});
        }

        const std::type_info& getReturnType() const override {
            return typeid(typename FunctionTraits<Func>::ReturnType);
        }

    private:
        template<size_t... Is>
        std::any invokeHelper(const std::vector<std::any>& args, std::index_sequence<Is...>) const {
            using ArgsTuple = typename FunctionTraits<Func>::ArgsTuple;
            if constexpr (std::is_void_v<typename FunctionTraits<Func>::ReturnType>) {
                func(std::any_cast<std::tuple_element_t<Is, ArgsTuple>>(args[Is])...);
                return {};
            } else {
                return func(std::any_cast<std::tuple_element_t<Is, ArgsTuple>>(args[Is])...);
            }
        }
    };

    std::unordered_map<std::string, std::unique_ptr<FunctionWrapperBase>> functions;

public:
    template<typename Func>
    void registerFunction(const std::string& name, Func&& f) {
        functions[name] = std::make_unique<FunctionWrapper<std::decay_t<Func>>>(std::forward<Func>(f));
    }

    template<typename... Args>
    std::optional<std::any> invoke(const std::string& name, Args&&... args) const {
        auto it = functions.find(name);
        if (it == functions.end()) {
            throw std::runtime_error("Function not found: " + name);
        }
        auto result = it->second->invoke({std::make_any<Args>(std::forward<Args>(args))...});
        const std::type_info& returnType = it->second->getReturnType();

        if (returnType == typeid(void)) {
            return std::nullopt;
        } else {
            return result;
        }
    }

    // Specialized invoke for ttnn::Tensor
    template<typename... Args>
    ttnn::Tensor invoke_operation(const std::string& name, Args&&... args) const {
        auto result = invoke(name, std::forward<Args>(args)...);
        if (!result) {
            throw std::runtime_error("Function does not return a value");
        }
        return std::any_cast<ttnn::Tensor>(*result);
    }

    template<typename... Args>
    auto operator()(const std::string& name, Args&&... args) const {
        return invoke(name, std::forward<Args>(args)...);
    }
};
// Function to send messages to the server in a loop
void send_messages(zmq::context_t& context) {
    zmq::socket_t send_socket(context, zmq::socket_type::dealer);
    send_socket.connect("tcp://localhost:8086");  // Port for sending messages

    for (int i = 1; i <= 100; ++i) {
        std::string config = "SHARD " + std::to_string(i);
        zmq::message_t config_message(config.data(), config.size());

        std::string tensor_info = "tensor_data_" + std::to_string(i);
        zmq::message_t tensor_message(tensor_info.data(), tensor_info.size());

        send_socket.send(config_message, zmq::send_flags::sndmore);
        send_socket.send(tensor_message, zmq::send_flags::none);

        // Optionally, add a delay to simulate processing time or network latency
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// Function to receive messages from the server
void receive_messages(zmq::context_t& context) {
    zmq::socket_t recv_socket(context, zmq::socket_type::dealer);
    recv_socket.connect("tcp://localhost:8087");  // Port for receiving messages

    while (true) {
        zmq::message_t identity;
        if (recv_socket.recv(identity, zmq::recv_flags::dontwait)) {
            zmq::message_t processed_tensor;
            auto result = recv_socket.recv(processed_tensor);
            if (!result) {
                std::cerr << "Failed to receive processed tensor" << std::endl;
                continue;
            }
            // Process the response
            std::string identity_str(static_cast<char*>(identity.data()), identity.size());
            std::string processed_tensor_str(static_cast<char*>(processed_tensor.data()), processed_tensor.size());

            // Handle response
            std::cout << "Received response: " << identity_str << std::endl;
            std::cout << "Processed tensor: " << processed_tensor_str << std::endl;
        }
        // Perform other tasks
    }
}

std::string encode_tensor(ttnn::Tensor& t) {
    std::ostringstream oss;
    tt::tt_metal::dump_tensor(oss, t);
    return oss.str();
}

ttnn::Tensor decode_tensor(std::string& encoded_tensor) {
    std::istringstream iss(encoded_tensor);
    return tt::tt_metal::load_tensor<Device*>(iss, nullptr);
}

ttnn::Tensor get_tensor(ttnn::Tensor& t, ttnn::Device* device) {
    auto encoded_tensor = encode_tensor(t);
    auto tensor = decode_tensor(encoded_tensor);
    return tensor;
}

Tensor execute(const std::string& op_name, ttnn::Tensor& a, ttnn::Tensor& b, ttnn::Device& device) {
    FunctionDispatchTable dispatch_table;
    dispatch_table.registerFunction("add", ttnn::add);
    //dispatch_table.registerFunction("to_device", FunctionWrapper(&ttnn::to_device));
    //dispatch_table.registerFunction("to_layout", FunctionWrapper(&ttnn::to_layout));
    //dispatch_table.registerFunction("from_device", FunctionWrapper(&ttnn::from_device));


    auto t_a = ttnn::to_device(a, &device, ttnn::types::DRAM_MEMORY_CONFIG);
    auto t_b = ttnn::to_device(b, &device, ttnn::types::DRAM_MEMORY_CONFIG);

    auto tt_a = ttnn::to_layout(t_a, ttnn::TILE_LAYOUT, std::nullopt, std::nullopt, &device);
    auto tt_b = ttnn::to_layout(t_b, ttnn::TILE_LAYOUT, std::nullopt, std::nullopt, &device);

    auto c = dispatch_table.invoke_operation(op_name, tt_a, tt_b);
    auto d = ttnn::from_device(c);

    return d;
}


class MessageCommunicator {
public:
    MessageCommunicator(const std::string& address, zmq::socket_type type)
        : context_(1), socket_(context_, type) {
        if (type == zmq::socket_type::rep) {
            socket_.bind(address);
        } else if (type == zmq::socket_type::req) {
            socket_.connect(address);
        }
    }

    void send_message(const std::string& message) {
        socket_.send(zmq::buffer(message), zmq::send_flags::none);
    }

    std::string receive_message() {
        zmq::message_t message;
        auto result = socket_.recv(message, zmq::recv_flags::none);
        return std::string(static_cast<char*>(message.data()), message.size());
    }

private:
    zmq::context_t context_;
    zmq::socket_t socket_;
};

class Worker {
public:
    Worker(const std::string& address, ttnn::Device* device)
    : device_(device), comm_(address, zmq::socket_type::rep) {}
    ~Worker() {}

    void run() {
        while (true) {
            std::string message = comm_.receive_message();
            if (message == "KILL") {
                std::cout << "[SERVER] Received kill signal. Shutting down." << std::endl;
                comm_.send_message("KILLED");
                break;
            }

            std::istringstream iss(message);
            std::string op_str, tensor1_str, tensor2_str;

            // Assume message format: "operation tensor1 tensor2"
            std::getline(iss, op_str, '|');
            std::getline(iss, tensor1_str, '|');
            std::getline(iss, tensor2_str, '|');

            // decode logic
            auto input_a = decode_tensor(tensor1_str);
            auto input_b = decode_tensor(tensor2_str);

            std::cout << "[SERVER] op_string received:" << op_str << std::endl;
            auto result = execute(op_str, input_a, input_b, *this->device_);
            auto result_str = encode_tensor(result);

            comm_.send_message(result_str);
        }
    }


private:
    MessageCommunicator comm_;
    ttnn::Device* device_;
};

int main() {
    auto& device_ref = ttnn::open_device(0);
    ttnn::Shape shape(std::array<uint32_t, 2>{32, 32});

    auto a = ttnn::ones(shape, ttnn::bfloat16);
    auto b = ttnn::ones(shape, ttnn::bfloat16);

    Worker worker("tcp://*:8086", &device_ref);
    std::thread server_worker_thread([&worker]() {
        worker.run();
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Give worker time to start

    MessageCommunicator client("tcp://localhost:8086", zmq::socket_type::req);
    std::string message = std::string("add") + "|" + encode_tensor(a) + "|" + encode_tensor(b);

    client.send_message(message);
    std::string result_str = client.receive_message();
    auto result = decode_tensor(result_str);
    result.print();

    // Send kill signal
    client.send_message("KILL");
    std::string kill_response = client.receive_message();
    std::cout << "Kill response: " << kill_response << std::endl;

    server_worker_thread.join();
    ttnn::close_device(device_ref);
}

// Function to send messages to the server in a loop
void send_messages(zmq::context_t& context) {
    zmq::socket_t send_socket(context, zmq::socket_type::dealer);
    send_socket.connect("tcp://localhost:8086");  // Port for sending messages

    for (int i = 1; i <= 100; ++i) {
        std::string config = "SHARD " + std::to_string(i);
        zmq::message_t config_message(config.data(), config.size());

        std::string tensor_info = "tensor_data_" + std::to_string(i);
        zmq::message_t tensor_message(tensor_info.data(), tensor_info.size());

        send_socket.send(config_message, zmq::send_flags::sndmore);
        send_socket.send(tensor_message, zmq::send_flags::none);

        // Optionally, add a delay to simulate processing time or network latency
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// Function to receive messages from the server
void receive_messages(zmq::context_t& context) {
    zmq::socket_t recv_socket(context, zmq::socket_type::dealer);
    recv_socket.connect("tcp://localhost:8087");  // Port for receiving messages

    while (true) {
        zmq::message_t identity;
        if (recv_socket.recv(identity, zmq::recv_flags::dontwait)) {
            zmq::message_t processed_tensor;
            auto result = recv_socket.recv(processed_tensor);
            if (!result) {
                std::cerr << "Failed to receive processed tensor" << std::endl;
                continue;
            }
            // Process the response
            std::string identity_str(static_cast<char*>(identity.data()), identity.size());
            std::string processed_tensor_str(static_cast<char*>(processed_tensor.data()), processed_tensor.size());

            // Handle response
            std::cout << "Received response: " << identity_str << std::endl;
            std::cout << "Processed tensor: " << processed_tensor_str << std::endl;
        }
    }
}


*/
