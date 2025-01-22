// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "umd/device/types/cluster_descriptor_types.h"
#include "host_api.hpp"
#include "dprint_server.hpp"
#include "device.hpp"
namespace tt {
namespace tt_metal::detail {

void CloseDevices(const std::map<chip_id_t, IDevice*>& devices);

}  // namespace tt_metal::detail

class DevicePool {
    friend void tt_metal::detail::CloseDevices(const std::map<chip_id_t, IDevice*>& devices);

public:
    DevicePool& operator=(const DevicePool&) = delete;
    DevicePool& operator=(DevicePool&& other) noexcept = delete;
    DevicePool(const DevicePool&) = delete;
    DevicePool(DevicePool&& other) noexcept = delete;

    static DevicePool& instance() noexcept {
        TT_ASSERT(_inst != nullptr, "Trying to get DevicePool without initializing it");
        return *_inst;
    }

    static void initialize(
        const std::vector<chip_id_t>& device_ids,
        const uint8_t num_hw_cqs,
        size_t l1_small_size,
        size_t trace_region_size,
        const tt_metal::DispatchCoreConfig& dispatch_core_config,
        tt::stl::Span<const std::uint32_t> l1_bank_remap = {}) noexcept;

    IDevice* get_active_device(chip_id_t device_id) const;
    std::vector<IDevice* > get_all_active_devices() const;
    bool close_device(chip_id_t device_id);
    void close_devices(const std::vector<IDevice*>& devices);
    bool is_device_active(chip_id_t id) const;
    void register_worker_thread_for_device(IDevice* device, std::thread::id worker_thread_id);
    void unregister_worker_thread_for_device(IDevice* device);
    const std::unordered_set<std::thread::id>& get_worker_thread_ids() const;

private:
    ~DevicePool();
    DevicePool();
    uint8_t num_hw_cqs;
    size_t l1_small_size;
    size_t trace_region_size;
    std::vector<uint32_t> l1_bank_remap;
    bool using_fast_dispatch;
    std::mutex lock;
    // TODO replace std::vector<std::unique_ptr<IDevice>> with stl::SlotMap<v1::DeviceKey, Device> when removing v0
    std::vector<std::unique_ptr<IDevice>> devices;
    // Used to track worker thread handles (1 worker thread created per device)
    // when we need to check if a call is made from an application thread or a
    // worker thread
    std::unordered_map<IDevice*, std::thread::id> device_to_worker_thread_id;
    std::unordered_set<std::thread::id> worker_thread_ids;
    std::thread::id device_pool_creation_thread_id;
    bool skip_remote_devices;
    std::unordered_set<uint32_t> firmware_built_keys;

    // Determine which CPU cores the worker threads need to be placed on for each device
    std::unordered_map<uint32_t, uint32_t> worker_thread_to_cpu_core_map;
    std::unordered_map<uint32_t, uint32_t> completion_queue_reader_to_cpu_core_map;
    void init_firmware_on_active_devices() const;
    void init_profiler_devices() const;
    void activate_device(chip_id_t id);
    void initialize_device(IDevice* dev) const;
    void add_devices_to_pool(const std::vector<chip_id_t>& device_ids);
    IDevice* get_device(chip_id_t id) const;
    static DevicePool* _inst;
};

}  // namespace tt
