// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>
#include <unordered_set>
#include <vector>

#include <sub_device_manager_tracker.hpp>

#include <device.hpp>
#include <allocator.hpp>
#include <buffer_constants.hpp>
#include <command_queue.hpp>
#include <hardware_command_queue.hpp>
#include <data_types.hpp>
#include <sub_device.hpp>
#include <sub_device_manager.hpp>
#include <sub_device_types.hpp>
#include <span.hpp>

namespace tt::tt_metal {

SubDeviceManagerTracker::SubDeviceManagerTracker(IDevice* device, std::unique_ptr<Allocator>&& global_allocator) :
    device_(device) {
    auto sub_device_manager = std::make_unique<SubDeviceManager>(device, std::move(global_allocator));
    default_sub_device_manager_ = sub_device_manager.get();
    active_sub_device_manager_ = default_sub_device_manager_;
    sub_device_managers_.insert_or_assign(sub_device_manager->id(), std::move(sub_device_manager));
}

SubDeviceManagerTracker::~SubDeviceManagerTracker() {
    active_sub_device_manager_ = nullptr;
    for (auto sub_device_manager = sub_device_managers_.begin(); sub_device_manager != sub_device_managers_.end();) {
        this->remove_sub_device_manager((sub_device_manager++)->first);
    }
    default_sub_device_manager_ = nullptr;
}

SubDeviceManagerId SubDeviceManagerTracker::create_sub_device_manager(
    tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) {
    auto sub_device_manager = std::make_unique<SubDeviceManager>(sub_devices, local_l1_size, device_);
    auto sub_device_manager_id = sub_device_manager->id();
    sub_device_managers_.insert_or_assign(sub_device_manager_id, std::move(sub_device_manager));
    return sub_device_manager_id;
}

std::tuple<SubDeviceManagerId, SubDeviceId> SubDeviceManagerTracker::create_sub_device_manager_with_fabric(
    tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) {
    auto fabric_sub_device = SubDevice(std::array{
        CoreRangeSet(),
        default_sub_device_manager_->sub_device(SubDeviceId{0}).cores(HalProgrammableCoreType::ACTIVE_ETH)});
    auto new_sub_devices = std::vector<SubDevice>(sub_devices.begin(), sub_devices.end());
    new_sub_devices.push_back(fabric_sub_device);
    auto fabric_sub_device_id = SubDeviceId{static_cast<uint32_t>(new_sub_devices.size() - 1)};
    auto sub_device_manager_id = this->create_sub_device_manager(new_sub_devices, local_l1_size);
    return {sub_device_manager_id, fabric_sub_device_id};
}

void SubDeviceManagerTracker::reset_sub_device_state(const std::unique_ptr<SubDeviceManager>& sub_device_manager) {
    auto num_sub_devices = sub_device_manager->num_sub_devices();
    for (uint8_t cq_id = 0; cq_id < device_->num_hw_cqs(); ++cq_id) {
        auto& hw_cq = device_->hw_command_queue(cq_id);
        // Only need to reset launch messages once, so reset on cq 0
        hw_cq.reset_worker_state(cq_id == 0, num_sub_devices, sub_device_manager->noc_mcast_unicast_data());
    }
    sub_device_manager->reset_sub_device_stall_group();
}

void SubDeviceManagerTracker::load_sub_device_manager(SubDeviceManagerId sub_device_manager_id) {
    TT_FATAL(!device_->using_slow_dispatch(), "Using sub device managers is unsupported with slow dispatch");
    if (active_sub_device_manager_->id() == sub_device_manager_id) {
        return;
    }
    if (active_sub_device_manager_->id() != default_sub_device_manager_->id()) {
        TT_FATAL(
            !active_sub_device_manager_->has_allocations(),
            "Cannot switch sub device managers while sub devices still have local allocations");
    }
    auto sub_device_manager = sub_device_managers_.find(sub_device_manager_id);
    TT_FATAL(sub_device_manager != sub_device_managers_.end(), "Sub device manager does not exist");
    this->reset_sub_device_state(sub_device_manager->second);
    const auto& default_allocator = default_sub_device_manager_->get_initialized_allocator(SubDeviceId{0});
    allocator::reset_allocator_size(*default_allocator, BufferType::L1);
    // Shrink the global allocator size to make room for sub-device allocators
    auto local_l1_size = sub_device_manager->second->local_l1_size();
    allocator::shrink_allocator_size(*default_allocator, BufferType::L1, local_l1_size, /*bottom_up=*/true);
    active_sub_device_manager_ = sub_device_manager->second.get();
}

void SubDeviceManagerTracker::clear_loaded_sub_device_manager() {
    this->load_sub_device_manager(default_sub_device_manager_->id());
}

void SubDeviceManagerTracker::remove_sub_device_manager(SubDeviceManagerId sub_device_manager_id) {
    if (active_sub_device_manager_ != nullptr) {
        TT_FATAL(
            sub_device_manager_id != active_sub_device_manager_->id(),
            "Cannot remove active sub device manager {}",
            sub_device_manager_id);
        TT_FATAL(
            sub_device_manager_id != default_sub_device_manager_->id(),
            "Cannot remove default sub device manager {}",
            sub_device_manager_id);
    }
    auto sub_device_manager = sub_device_managers_.find(sub_device_manager_id);
    TT_FATAL(sub_device_manager != sub_device_managers_.end(), "Sub device manager does not exist");
    sub_device_managers_.erase(sub_device_manager);
}

SubDeviceManager* SubDeviceManagerTracker::get_active_sub_device_manager() const { return active_sub_device_manager_; }

SubDeviceManager* SubDeviceManagerTracker::get_default_sub_device_manager() const {
    return default_sub_device_manager_;
}

SubDeviceManagerId SubDeviceManagerTracker::get_active_sub_device_manager_id() const {
    return active_sub_device_manager_->id();
}

SubDeviceManagerId SubDeviceManagerTracker::get_default_sub_device_manager_id() const {
    return default_sub_device_manager_->id();
}

}  // namespace tt::tt_metal
