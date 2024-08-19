// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/serialization.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>

#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"

namespace tt {

namespace tt_metal {

namespace detail {

static constexpr std::size_t SENTINEL_VALUE = std::numeric_limits<std::size_t>::max();

void dump_owned_storage(std::ostream& output_stream, const OwnedStorage& storage) {
    std::visit(
        [&output_stream]<typename T>(const owned_buffer::Buffer<T>& generic_buffer) {
            const auto buffer = owned_buffer::get_as<T>(generic_buffer);
            auto size = buffer.size();
            output_stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
            output_stream.write(reinterpret_cast<const char*>(buffer.begin()), sizeof(T) * size);
        },
        storage.buffer
    );
}

void dump_borrowed_storage(std::ostream& output_stream, const BorrowedStorage& storage) {
    std::visit(
        [&output_stream]<typename T>(const borrowed_buffer::Buffer<T>& generic_buffer) {
            const auto buffer = borrowed_buffer::get_as<T>(generic_buffer);
            auto size = buffer.size();
            output_stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
            output_stream.write(reinterpret_cast<const char*>(buffer.begin()), sizeof(T) * size);
        },
        storage.buffer
    );
}

void dump_multi_device_host_storage(std::ostream& output_stream, const MultiDeviceHostStorage& storage, const DistributedTensorConfig& strategy) {
    std::size_t num_buffers = storage.num_buffers();
    output_stream.write(reinterpret_cast<const char*>(&num_buffers), sizeof(std::size_t));

    // Use the user-specified strategy which defines how it gets distributed when mapped onto multi-device
    output_stream.write(reinterpret_cast<const char*>(&strategy), sizeof(DistributedTensorConfig));

    if (std::holds_alternative<ReplicateTensor>(strategy)) {
        std::visit(
            [&output_stream]<typename T>(const owned_buffer::Buffer<T>& generic_buffer) {
                const auto buffer = owned_buffer::get_as<T>(generic_buffer);
                auto size = buffer.size();
                output_stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
                output_stream.write(reinterpret_cast<const char*>(buffer.begin()), sizeof(T) * size);
            }, storage.get_buffer(0)
        );
        output_stream.write(reinterpret_cast<const char*>(&storage.shapes.at(0)), sizeof(Shape));

    } else {
        for (int i = 0; i < num_buffers; i++) {
            std::visit(
                [&output_stream]<typename T>(const owned_buffer::Buffer<T>& generic_buffer) {
                    const auto buffer = owned_buffer::get_as<T>(generic_buffer);
                    auto size = buffer.size();
                    output_stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
                    output_stream.write(reinterpret_cast<const char*>(buffer.begin()), sizeof(T) * size);
                }, storage.get_buffer(i)
            );
        }
        for (const auto& shape : storage.shapes) {
            output_stream.write(reinterpret_cast<const char*>(&shape), sizeof(Shape));
        }
    }
}

template<typename T>
OwnedStorage load_owned_storage(std::istream& input_stream) {
    std::size_t size = 0;
    input_stream.read(reinterpret_cast<char*>(&size), sizeof(std::size_t));
    auto buffer = owned_buffer::create<T>(size);
    input_stream.read(reinterpret_cast<char*>(buffer.begin()), sizeof(T) * size);
    return {buffer};

}

template<typename T>
MultiDeviceHostStorage load_multi_device_host_storage(std::istream& input_stream, DeviceMesh* device_mesh) {
    std::size_t num_buffers = 0;
    DistributedTensorConfig strategy;

    if (!input_stream.read(reinterpret_cast<char*>(&num_buffers), sizeof(std::size_t)) ||
        !input_stream.read(reinterpret_cast<char*>(&strategy), sizeof(DistributedTensorConfig))) {
        throw std::runtime_error("Failed to read num_buffers or strategy from input stream");
    }

    std::vector<OwnedBuffer> buffers;
    std::vector<Shape> shapes;
    if (std::holds_alternative<ReplicateTensor>(strategy)) {
        int replication_factor = std::get<ReplicateTensor>(strategy).replication_factor;
        std::size_t size = 0;

        if (!input_stream.read(reinterpret_cast<char*>(&size), sizeof(std::size_t))) {
            throw std::runtime_error("Failed to read buffer size from input stream");
        }

        std::cout << "debug print 0.2" << std::endl;
        auto buffer = owned_buffer::create<T>(size);
        std::cout << "debug print 0.5" << std::endl;
        // Check if the buffer is valid by ensuring it has a non-zero size
        if (buffer.size() == 0) {
            throw std::runtime_error("Failed to create buffer: buffer size is 0");
        }

        auto shape = Shape{};
        if (!input_stream.read(reinterpret_cast<char*>(buffer.begin()), sizeof(T) * size) ||
            !input_stream.read(reinterpret_cast<char*>(&shape), sizeof(Shape))) {
            throw std::runtime_error("Failed to read buffer data or shape from input stream");
        }
        buffers.push_back(std::move(buffer));
        shapes.push_back(shape);

        for (std::size_t i = 1; i < replication_factor; ++i) {
            std::visit([&buffers](auto&& arg) {
                using BufferType = std::decay_t<decltype(arg)>;
                buffers.push_back(BufferType{arg.get_ptr()});
            }, buffers[0]);
            shapes.push_back(shape);
        }
    } else {
        buffers.reserve(num_buffers);
        shapes.reserve(num_buffers);

        for (std::size_t i = 0; i < num_buffers; ++i) {
            std::size_t size = 0;
            if (!input_stream.read(reinterpret_cast<char*>(&size), sizeof(std::size_t))) {
                throw std::runtime_error("Failed to read buffer size from input stream");
            }

            auto buffer = owned_buffer::create<T>(size);
            // Check if the buffer is valid by ensuring it has a non-zero size
            if (buffer.size() == 0) {
                throw std::runtime_error("Failed to create buffer: buffer size is 0");
            }

            if (!input_stream.read(reinterpret_cast<char*>(buffer.begin()), sizeof(T) * size)) {
                throw std::runtime_error("Failed to read buffer data from input stream");
            }

            buffers.push_back(std::move(buffer));
        }
        for (std::size_t i = 0; i < num_buffers; ++i) {
            auto shape = Shape{};
            if (!input_stream.read(reinterpret_cast<char*>(&shape), sizeof(Shape))) {
                throw std::runtime_error("Failed to read shape from input stream");
            }
            shapes.push_back(shape);
        }
    }

    return {strategy, std::move(buffers), std::move(shapes)};
}


OwnedStorage load_owned_storage(std::istream& input_stream, DataType data_type) {
    if (data_type == DataType::UINT32 or data_type == DataType::BFLOAT8_B or data_type == DataType::BFLOAT4_B) {
        using T = std::uint32_t;
        return load_owned_storage<T>(input_stream);
    } else if (data_type == DataType::INT32) {
        using T = std::int32_t;
        return load_owned_storage<T>(input_stream);
    } else if (data_type == DataType::UINT8) {
        using T = std::uint8_t;
        return load_owned_storage<T>(input_stream);
    } else if (data_type == DataType::UINT16) {
        using T = std::uint16_t;
        return load_owned_storage<T>(input_stream);
    } else if (data_type == DataType::FLOAT32) {
        using T = float;
        return load_owned_storage<T>(input_stream);
    } else if (data_type == DataType::BFLOAT16) {
        using T = bfloat16;
        return load_owned_storage<T>(input_stream);
    } else {
        TT_THROW("Unsupported DataType");
    }
}


MultiDeviceHostStorage load_multi_device_host_storage(std::istream& input_stream, DataType data_type, DeviceMesh *device_mesh) {
    if (data_type == DataType::UINT32 or data_type == DataType::BFLOAT8_B or data_type == DataType::BFLOAT4_B) {
        using T = std::uint32_t;
        return load_multi_device_host_storage<T>(input_stream, device_mesh);
    } else if (data_type == DataType::UINT16) {
        using T = std::uint16_t;
        return load_multi_device_host_storage<T>(input_stream, device_mesh);
    } else if (data_type == DataType::FLOAT32) {
        using T = float;
        return load_multi_device_host_storage<T>(input_stream, device_mesh);
    } else if (data_type == DataType::BFLOAT16) {
        using T = bfloat16;
        return load_multi_device_host_storage<T>(input_stream, device_mesh);
    } else {
        TT_THROW("Unsupported DataType");
    }
}

Storage load_storage(std::istream& input_stream, DataType data_type, StorageType storage_type) {
    if (storage_type == StorageType::MULTI_DEVICE_HOST or storage_type == StorageType::MULTI_DEVICE) {
        std::cout << "loading multi-device host storage" << std::endl;
        return load_multi_device_host_storage(input_stream, data_type, nullptr);
    } else {
        std::cout << "loading single-device host storage" << std::endl;
        return load_owned_storage(input_stream, data_type);
    }
}

}  // namespace detail

void dump_tensor(std::ostream& output_stream, const Tensor& tensor, const std::unordered_map<std::string, std::string>& strategy) {
    auto shape = tensor.get_legacy_shape();
    auto data_type = tensor.get_dtype();
    auto layout = tensor.get_layout();
    auto storage_type = tensor.storage_type();

    output_stream.write(reinterpret_cast<const char*>(&detail::SENTINEL_VALUE), sizeof(std::size_t));
    output_stream.write(reinterpret_cast<const char*>(&VERSION_ID), sizeof(std::uint8_t));
    output_stream.write(reinterpret_cast<const char*>(&shape), sizeof(Shape));
    output_stream.write(reinterpret_cast<const char*>(&data_type), sizeof(DataType));
    output_stream.write(reinterpret_cast<const char*>(&layout), sizeof(Layout));
    output_stream.write(reinterpret_cast<const char*>(&storage_type), sizeof(StorageType));

    bool is_on_device = is_tensor_on_device_or_multidevice(tensor);
    bool has_memory_config = is_on_device;
    if (VERSION_ID >= 2) {
        output_stream.write(reinterpret_cast<const char*>(&has_memory_config), sizeof(bool));
        if (has_memory_config) {
            tt::tt_metal::dump_memory_config(output_stream, tensor.memory_config());
        }
    }

    Tensor tensor_to_dump = tensor;
    if (is_on_device) {
        tensor_to_dump = tensor_to_dump.cpu();
    }

    std::visit(
        [&output_stream, &strategy](const auto& storage) {

            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                detail::dump_owned_storage(output_stream, storage);
            }
            else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                detail::dump_borrowed_storage(output_stream, storage);
            }
            else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            }
            else if constexpr (std::is_same_v<StorageType, MultiDeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            }
            else if constexpr (std::is_same_v<StorageType, MultiDeviceHostStorage>) {
                auto distribute_config = (strategy.empty()) ? storage.strategy : get_distributed_tensor_config(strategy);
                detail::dump_multi_device_host_storage(output_stream, storage, distribute_config);
            }
            else {
                raise_unsupported_storage<StorageType>();
            }
        },
        tensor_to_dump.get_storage());
}

void dump_tensor(const std::string& file_name, const Tensor& tensor, const std::unordered_map<std::string, std::string>& strategy) {
    std::ofstream output_stream(file_name, std::ios::out | std::ios::binary);
    if (not output_stream) {
        throw std::runtime_error(fmt::format("Cannot open \"{}\"", file_name));
    }
    dump_tensor(output_stream, tensor, strategy);
}

template<typename T>
Tensor load_tensor(std::istream& input_stream, T device) {
    std::size_t read_sentinel;
    input_stream.read(reinterpret_cast<char*>(&read_sentinel), sizeof(read_sentinel));
    if (read_sentinel == detail::SENTINEL_VALUE) {
        std::uint8_t version_id;
        input_stream.read(reinterpret_cast<char*>(&version_id), sizeof(version_id));

        // Allow only backward compatible versions
        if (version_id > VERSION_ID) {
            throw std::runtime_error(fmt::format("Serialized tensor with version_id: {}. Loader version: {}", version_id, VERSION_ID));
        }
        auto shape = Shape{};
        DataType data_type;
        Layout layout;
        StorageType storage_type;
        input_stream.read(reinterpret_cast<char*>(&shape), sizeof(Shape));
        input_stream.read(reinterpret_cast<char*>(&data_type), sizeof(DataType));
        input_stream.read(reinterpret_cast<char*>(&layout), sizeof(Layout));
        input_stream.read(reinterpret_cast<char*>(&storage_type), sizeof(StorageType));

        bool has_memory_config = false;
        MemoryConfig memory_config = MemoryConfig{
            .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED, .buffer_type = BufferType::DRAM};

        if (version_id >= 2) {
            input_stream.read(reinterpret_cast<char*>(&has_memory_config), sizeof(bool));
            if (has_memory_config) {
                memory_config = tt::tt_metal::load_memory_config(input_stream);
            }
        }

        auto storage = detail::load_storage(input_stream, data_type, storage_type);

        auto tensor = Tensor(std::move(storage), shape, data_type, layout);
        if (device != nullptr) {
            tensor = tensor.to(static_cast<T>(device), memory_config);
        } else if (has_memory_config) {
            tt::log_warning("Memory config is ignored when loading the tensor because device is not provided");
        }
        return tensor;

    } else {
        input_stream.seekg(0, std::ios::beg); // No sentinel found, assume it's an older format and rewind

        auto shape = Shape{};
        DataType data_type;
        Layout layout;
        input_stream.read(reinterpret_cast<char*>(&shape), sizeof(Shape));
        input_stream.read(reinterpret_cast<char*>(&data_type), sizeof(DataType));
        input_stream.read(reinterpret_cast<char*>(&layout), sizeof(Layout));

        auto storage = detail::load_owned_storage(input_stream, data_type);
        auto tensor = Tensor(std::move(storage), shape, data_type, layout);
        if (device != nullptr) {
            tensor = tensor.to(device);
        }
        return tensor;
    }
}

template<typename T>
Tensor load_tensor(const std::string& file_name, T device) {
    std::ifstream input_stream(file_name, std::ios::in | std::ios::binary);
    if (not input_stream) {
        throw std::runtime_error(fmt::format("Cannot open \"{}\"", file_name));
    }
    return load_tensor(input_stream, device);
}

// Explicit instantiations
template Tensor load_tensor<Device*>(std::istream&, Device*);
template Tensor load_tensor<DeviceMesh*>(std::istream&, DeviceMesh*);
template Tensor load_tensor<Device*>(const std::string&, Device*);
template Tensor load_tensor<DeviceMesh*>(const std::string&, DeviceMesh*);
}  // namespace tt_metal

}  // namespace tt
