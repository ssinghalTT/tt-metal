#pragma once

#include <any>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include "ttnn/cpp/ttnn/decorators.hpp"
#include "ttnn/cpp/ttnn/operations/core/core.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/cpp/ttnn/tensor/serialization.hpp"
#include "ttnn/cpp/ttnn/multi_server/tensor.hpp"
#include "ttnn/cpp/ttnn/multi_server/client.hpp"

namespace ttnn {
namespace multi_server {

template<typename T>
struct FunctionTraits;
/*
template<reflect::fixed_string cpp_fully_qualified_name, typename registered_operation_t, bool auto_launch_op>
struct FunctionTraits<ttnn::decorators::registered_registered_operation_t<cpp_fully_qualified_name, registered_operation_t, auto_launch_op>> {
    using ReturnType = typename registered_operation_t::tensor_return_value_t;

    using ArgsTuple = std::conditional_t<
        requires { typename registered_operation_t::operation_attributes_t; typename registered_operation_t::tensor_args_t; },
        std::tuple<const typename registered_operation_t::operation_attributes_t&, const typename registered_operation_t::tensor_args_t&>,
        std::tuple<const ttnn::Tensor&, const ttnn::Tensor&> // fallback for binary operations
    >;

    static constexpr size_t arity = std::tuple_size_v<ArgsTuple>;
};
*/

// Specialization for function pointers
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

/*
template<reflect::fixed_string cpp_fully_qualified_name, typename registered_operation_t, bool auto_launch_op>
struct FunctionTraits<ttnn::decorators::registered_registered_operation_t<cpp_fully_qualified_name, registered_operation_t, auto_launch_op>> {
    using ReturnType = decltype(std::declval<registered_operation_t>().invoke(std::declval<typename registered_operation_t::tensor_args_t>()));

    using ArgsTuple = typename registered_operation_t::tensor_args_t;

    static constexpr size_t arity = std::tuple_size_v<ArgsTuple>;
};
*/

template<ttnn::operations::binary::BinaryOpType OpType, bool Inplace>
struct FunctionTraits<ttnn::decorators::registered_operation_t<
    reflect::fixed_string{"ttnn::add"},
    ttnn::operations::binary::BinaryOperationOverload<OpType, Inplace>,
    false
>> {
    using ReturnType = ttnn::Tensor;
    using ArgsTuple = std::tuple<const ttnn::Tensor&, const ttnn::Tensor&>;
    static constexpr size_t arity = std::tuple_size_v<ArgsTuple>;
};

template<>
struct FunctionTraits<ttnn::decorators::registered_operation_t<
    reflect::fixed_string{"ttnn::to_layout"},
    ttnn::operations::core::ToLayout,
    true
>> {
    using ReturnType = ttnn::Tensor;
    using ArgsTuple = std::tuple<
        const ttnn::Tensor&,
        const ttnn::Layout&,
        const std::optional<ttnn::DataType>&,
        const std::optional<ttnn::MemoryConfig>&,
        ttnn::Device*
    >;
    static constexpr size_t arity = std::tuple_size_v<ArgsTuple>;
};

// Specialization for the second overload of to_device
template<>
struct FunctionTraits<ttnn::Tensor (*)(const ttnn::Tensor&, tt::tt_metal::Device*, const std::optional<ttnn::MemoryConfig>&)> {
    using ReturnType = ttnn::Tensor;
    using ArgsTuple = std::tuple<const ttnn::Tensor&, tt::tt_metal::Device*, const std::optional<ttnn::MemoryConfig>&>;
    static constexpr size_t arity = 3;
};
template<>
struct FunctionTraits<ttnn::Tensor (*)(const ttnn::Tensor&, tt::tt_metal::DeviceMesh*, const std::optional<ttnn::MemoryConfig>&)> {
    using ReturnType = ttnn::Tensor;
    using ArgsTuple = std::tuple<const ttnn::Tensor&, tt::tt_metal::DeviceMesh*, const std::optional<ttnn::MemoryConfig>&>;
    static constexpr size_t arity = 3;
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
        template<typename T>
        struct is_optional : std::false_type {};

        template<typename T>
        struct is_optional<std::optional<T>> : std::true_type {};

        template<typename T>
        auto convertArg(const std::any& arg) const {
            if constexpr (is_optional<std::remove_cv_t<std::remove_reference_t<T>>>::value) {
                using OptionalType = std::remove_cv_t<std::remove_reference_t<T>>;
                using ValueType = typename OptionalType::value_type;

                if (arg.type() == typeid(std::nullopt_t)) {
                    if constexpr (std::is_reference_v<T>) {
                        static OptionalType nullopt_value;
                        return nullopt_value;
                    } else {
                        return OptionalType{};
                    }
                } else if (arg.type() == typeid(ValueType)) {
                    return OptionalType(std::any_cast<ValueType>(arg));
                } else if (arg.type() == typeid(OptionalType)) {
                    return std::any_cast<OptionalType>(arg);
                } else {
                    throw std::bad_any_cast();
                }
            } else {
                return std::any_cast<T>(arg);
            }
        }

        template<size_t... Is>
        std::any invokeHelper(const std::vector<std::any>& args, std::index_sequence<Is...>) const {
            using ArgsTuple = typename FunctionTraits<Func>::ArgsTuple;
            try {
                if constexpr (std::is_void_v<typename FunctionTraits<Func>::ReturnType>) {
                    func(convertArg<std::tuple_element_t<Is, ArgsTuple>>(args[Is])...);
                    return {};
                } else {
                    return func(convertArg<std::tuple_element_t<Is, ArgsTuple>>(args[Is])...);
                }
            } catch (const std::bad_any_cast& e) {
                std::cerr << "[ERROR] Bad any cast in invokeHelper:" << std::endl;
                ((std::cerr << "  Arg " << Is << " expected type: " << typeid(std::tuple_element_t<Is, ArgsTuple>).name()
                            << ", actual type: " << args[Is].type().name() << std::endl), ...);
                throw;
            }
        }

    };

    std::unordered_map<std::string, std::unique_ptr<FunctionWrapperBase>> functions;

public:
    template<typename Func>
    void registerFunction(const std::string& name, Func&& f) {
        functions[name] = std::make_unique<FunctionWrapper<std::decay_t<Func>>>(std::forward<Func>(f));
    }

    template<typename T>
    static std::any wrap_arg(T&& arg) {
        return std::forward<T>(arg);
    }

    template<typename... Args>
    std::optional<std::any> invoke(const std::string& name, Args&&... args) const {
        std::cout << "[DEBUG] Invoking function: " << name << std::endl;

        auto it = functions.find(name);
        if (it == functions.end()) {
            std::cerr << "[ERROR] Function not found: " << name << std::endl;
            throw std::runtime_error("Function not found: " + name);
        }

        std::cout << "[DEBUG] Function found in dispatch table" << std::endl;

        std::cout << "[DEBUG] Argument types:" << std::endl;
        int arg_index = 0;
        ((std::cout << "  Arg " << arg_index++ << ": " << typeid(Args).name() << std::endl), ...);

        std::vector<std::any> any_args = {std::make_any<Args>(std::forward<Args>(args))...};
        try {
            std::cout << "[DEBUG] Number of arguments: " << any_args.size() << std::endl;

            std::cout << "[DEBUG] Invoking function wrapper" << std::endl;
            for (size_t i = 0; i < any_args.size(); ++i) {
                std::cout << "  Arg " << i << " type: " << any_args[i].type().name() << std::endl;
            }
            auto result = it->second->invoke(any_args);
            std::cout << "[DEBUG] Function wrapper invoked successfully" << std::endl;

            const std::type_info& returnType = it->second->getReturnType();
            std::cout << "[DEBUG] Expected return type: " << returnType.name() << std::endl;
            std::cout << "[DEBUG] Actual return type: " << result.type().name() << std::endl;

            if (returnType == typeid(void)) {
                std::cout << "[DEBUG] Function returns void" << std::endl;
                return std::nullopt;
            } else {
                std::cout << "[DEBUG] Function returns a value" << std::endl;
                return result;
            }
        } catch (const std::bad_any_cast& e) {
                std::cerr << "[ERROR] Bad any cast in invoke: " << e.what() << std::endl;
                std::cerr << "[ERROR] Function: " << name << std::endl;
                std::cerr << "[ERROR] Expected argument types:" << std::endl;
                int arg_index = 0;
                ((std::cerr << "  Arg " << arg_index++ << ": " << typeid(Args).name() << std::endl), ...);
                std::cerr << "[ERROR] Actual argument types:" << std::endl;
                for (size_t i = 0; i < any_args.size(); ++i) {
                    std::cerr << "  Arg " << i << ": " << any_args[i].type().name() << std::endl;
                }
                throw;
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Exception in invoke: " << e.what() << std::endl;
            std::cerr << "[ERROR] Function: " << name << std::endl;
            throw;
        }
    }

    // Specialized invoke for ttnn::Tensor
    template<typename... Args>
    ttnn::Tensor invoke_operation(const std::string& name, Args&&... args) const {
        auto result = invoke(name, std::forward<Args>(args)...);
        if (!result) {
            throw std::runtime_error("Function does not return a value");
        }
        try {
            return std::any_cast<ttnn::Tensor>(*result);
        } catch (const std::bad_any_cast& e) {
            std::cerr << "[ERROR] Bad any cast in invoke_operation: " << e.what() << std::endl;
            std::cerr << "[ERROR] Actual type: " << result->type().name() << std::endl;
            throw;
        }
    }

    template<typename... Args>
    auto operator()(const std::string& name, Args&&... args) const {
        return invoke(name, std::forward<Args>(args)...);
    }
};


std::string encode_tensor(const ttnn::Tensor& t);

template <typename DeviceType = ttnn::Device>
ttnn::Tensor decode_tensor(const std::string& encoded_tensor, DeviceType* device = nullptr) {
    try {
        std::istringstream iss(encoded_tensor);
        auto result = tt::tt_metal::load_tensor(iss, device);
        std::cout << "[SERVER] Tensor decoded successfully" << std::endl;
        return result;
    } catch (const std::exception& e) {
        std::cerr << "[SERVER] Error in decode_tensor: " << e.what() << std::endl;
        throw;
    }
}

template <typename DeviceType>
ttnn::Tensor get_tensor(ttnn::Tensor& t, DeviceType* device) {
    auto encoded_tensor = encode_tensor(t);
    auto tensor = decode_tensor(encoded_tensor, device);
    return tensor;
}


DistributedTensor create_multi_server_tensor(const ttnn::Tensor& t, const DistributedTensorConfig& strategy, ServerDevice& client);


} // namespace multi_server
} // namespace ttnn
