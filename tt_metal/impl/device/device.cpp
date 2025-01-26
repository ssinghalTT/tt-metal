// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <device_impl.hpp>

#include <string>
#include <thread>
#include <tt_align.hpp>
#include "tt_metal/deprecated/device.hpp"
#include "common/core_assignment.hpp"
#include <host_api.hpp>
#include <trace.hpp>
#include <core_descriptor.hpp>
#include "tracy/Tracy.hpp"
#include <tt_metal.hpp>
#include <dprint_server.hpp>
#include "impl/debug/watcher_server.hpp"
#include "tt_metal/impl/dispatch/kernels/packet_queue_ctrl.hpp"
#include <utils.hpp>
#include <llrt.hpp>
#include <dev_msgs.h>
#include <device_pool.hpp>
#include <persistent_kernel_cache.hpp>
#include "tt_metal/tools/profiler/tt_metal_tracy.hpp"
#include <hal.hpp>
#include <hal_exp.hpp>
#include <sub_device.hpp>
#include <sub_device_manager_tracker.hpp>
#include <sub_device_manager.hpp>
#include <sub_device_types.hpp>
#include <span.hpp>
#include <types.hpp>
#include "impl/dispatch/topology.hpp"

namespace tt {

namespace tt_metal {

Device::Device(
    chip_id_t device_id, const uint8_t num_hw_cqs, size_t l1_small_size, size_t trace_region_size, tt::stl::Span<const std::uint32_t> l1_bank_remap, bool minimal, uint32_t worker_core, uint32_t completion_queue_reader_core) :
    id_(device_id), worker_thread_core_(worker_core), completion_queue_reader_core_(completion_queue_reader_core), work_executor_(worker_core, device_id) {
    ZoneScoped;
    this->initialize(num_hw_cqs, l1_small_size, trace_region_size, l1_bank_remap, minimal);
}

std::unordered_set<CoreCoord> Device::get_active_ethernet_cores(bool skip_reserved_tunnel_cores) const {
    return tt::Cluster::instance().get_active_ethernet_cores(this->id_, skip_reserved_tunnel_cores);
}

bool Device::is_active_ethernet_core(CoreCoord logical_core, bool skip_reserved_tunnel_cores) const {
    auto active_ethernet_cores = this->get_active_ethernet_cores(skip_reserved_tunnel_cores);
    return active_ethernet_cores.find(logical_core) != active_ethernet_cores.end();
}

std::unordered_set<CoreCoord> Device::get_inactive_ethernet_cores() const {
    return tt::Cluster::instance().get_inactive_ethernet_cores(this->id_);
}

bool Device::is_inactive_ethernet_core(CoreCoord logical_core) const {
    auto inactive_ethernet_cores = tt::Cluster::instance().get_inactive_ethernet_cores(this->id_);
    return inactive_ethernet_cores.find(logical_core) != inactive_ethernet_cores.end();
}

std::tuple<chip_id_t, CoreCoord> Device::get_connected_ethernet_core(CoreCoord eth_core) const {
    return tt::Cluster::instance().get_connected_ethernet_core(std::make_tuple(this->id_, eth_core));
}

std::vector<CoreCoord> Device::get_ethernet_sockets(chip_id_t connected_chip_id) const {
    return tt::Cluster::instance().get_ethernet_sockets(this->id_, connected_chip_id);
}

bool Device::is_mmio_capable() const {
    return tt::Cluster::instance().get_associated_mmio_device(this->id_) == this->id_;
}

CoreRangeSet Device::worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->sub_device(sub_device_id).cores(core_type);
}

uint32_t Device::num_worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->sub_device(sub_device_id).num_cores(core_type);
}

/* Get all dispatch cores associated with this device. On return, my_dispatch_cores contains dispatch cores used by
 * this device (split between cores on this device itself and if this is a remote device, the mmio device dispatch
 * cores being used by this device). On return, other_dispatch_cores contains dispatch cores on this device that are
 * used by other (remote) devices.
*/
void Device::get_associated_dispatch_virtual_cores(
    std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>> &my_dispatch_cores,
    std::unordered_map<chip_id_t,std::unordered_set<CoreCoord>> &other_dispatch_cores) {
    if (this->is_mmio_capable()) {
        for (const chip_id_t &device_id : tt::Cluster::instance().get_devices_controlled_by_mmio_device(this->id_)) {
            uint8_t num_hw_cqs = this->num_hw_cqs();
            uint16_t curr_channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
            CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device_id);
            for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
                if (device_id == this->id_) {
                    //mmio device.
                    bool dispatch_hd_allocated = false;
                    CoreCoord virtual_core_dispatch_hd;
                    if (dispatch_core_manager::instance().is_dispatcher_core_allocated(device_id, curr_channel, cq_id)) {
                        tt_cxy_pair dispatch_location = dispatch_core_manager::instance().dispatcher_core(device_id, curr_channel, cq_id);
                        virtual_core_dispatch_hd = this->virtual_core_from_logical_core(dispatch_location, dispatch_core_type);
                        my_dispatch_cores[this->id_].insert(virtual_core_dispatch_hd);
                        dispatch_hd_allocated = true;
                        log_debug(tt::LogMetal, "MMIO Device Dispatch core: Logical: {} - Physical: {}", dispatch_location.str(), virtual_core_dispatch_hd.str());
                    }
                    // Include dispatch_s in the dispatch core location set, if its not on the same core as dispatch_hd
                    if (dispatch_core_manager::instance().is_dispatcher_s_core_allocated(device_id, curr_channel, cq_id)) {
                        tt_cxy_pair dispatch_s_location = dispatch_core_manager::instance().dispatcher_s_core(device_id, curr_channel, cq_id);
                        CoreCoord virtual_core_dispatch_s = this->virtual_core_from_logical_core(dispatch_s_location, dispatch_core_type);
                        if ((!dispatch_hd_allocated) or (virtual_core_dispatch_s != virtual_core_dispatch_hd)) {
                            my_dispatch_cores[dispatch_s_location.chip].insert(virtual_core_dispatch_s);
                        }
                    }
                    if (dispatch_core_manager::instance().is_prefetcher_core_allocated(device_id, curr_channel, cq_id)) {
                        tt_cxy_pair prefetch_location = dispatch_core_manager::instance().prefetcher_core(device_id, curr_channel, cq_id);
                        CoreCoord virtual_core = this->virtual_core_from_logical_core(prefetch_location, dispatch_core_type);
                        my_dispatch_cores[this->id_].insert(virtual_core);
                        log_debug(tt::LogMetal, "MMIO Device Prefetch core: Logical: {} - Physical: {}", prefetch_location.str(), virtual_core.str());
                    }
                } else if (tt::DevicePool::instance().is_device_active(device_id)) {
                    //non mmio devices serviced by this mmio capable device.
                    //skip remote dispatch cores only if respective remote device is active.
                    if (dispatch_core_manager::instance().is_dispatcher_core_allocated(device_id, curr_channel, cq_id)) {
                        tt_cxy_pair dispatch_location = dispatch_core_manager::instance().dispatcher_core(device_id, curr_channel, cq_id);
                        CoreCoord virtual_core = this->virtual_core_from_logical_core(dispatch_location, dispatch_core_type);
                        other_dispatch_cores[this->id_].insert(virtual_core);
                        log_debug(tt::LogMetal, "Remote Device Dispatch core: Logical: {} - Physical: {} will keep running on MMIO Device.", dispatch_location.str(), virtual_core.str());
                    }
                    if (dispatch_core_manager::instance().is_prefetcher_core_allocated(device_id, curr_channel, cq_id)) {
                        tt_cxy_pair prefetch_location = dispatch_core_manager::instance().prefetcher_core(device_id, curr_channel, cq_id);
                        CoreCoord virtual_core = this->virtual_core_from_logical_core(prefetch_location, dispatch_core_type);
                        other_dispatch_cores[this->id_].insert(virtual_core);
                        log_debug(tt::LogMetal, "Remote Device Prefetch core: Logical: {} - Physical: {} will keep running on MMIO Device.", prefetch_location.str(), virtual_core.str());
                    }
                    if (dispatch_core_manager::instance().is_mux_core_allocated(device_id, curr_channel, cq_id)) {
                        tt_cxy_pair mux_location = dispatch_core_manager::instance().mux_core(device_id, curr_channel, cq_id);
                        CoreCoord virtual_core = this->virtual_core_from_logical_core(mux_location, dispatch_core_type);
                        other_dispatch_cores[this->id_].insert(virtual_core);
                        log_debug(tt::LogMetal, "Remote Device Mux core: Logical: {} - Physical: {} will keep running on MMIO Device.", mux_location.str(), virtual_core.str());
                    }
                    if (dispatch_core_manager::instance().is_demux_core_allocated(device_id, curr_channel, cq_id)) {
                        tt_cxy_pair demux_location = dispatch_core_manager::instance().demux_core(device_id, curr_channel, cq_id);
                        CoreCoord virtual_core = this->virtual_core_from_logical_core(demux_location, dispatch_core_type);
                        other_dispatch_cores[this->id_].insert(virtual_core);
                        log_debug(tt::LogMetal, "Remote Device Demux core: Logical: {} - Physical: {} will keep running on MMIO Device.", demux_location.str(), virtual_core.str());
                    }
                }
            }
        }
    } else {
        //remote device that is active
        uint8_t num_hw_cqs = this->num_hw_cqs();
        auto device_id = this->id_;
        uint16_t curr_channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
        CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device_id);
        for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
            if (dispatch_core_manager::instance().is_dispatcher_core_allocated(device_id, curr_channel, cq_id)) {
                tt_cxy_pair dispatch_location = dispatch_core_manager::instance().dispatcher_core(device_id, curr_channel, cq_id);
                CoreCoord virtual_core = this->virtual_core_from_logical_core(dispatch_location, dispatch_core_type);
                my_dispatch_cores[dispatch_location.chip].insert(virtual_core);
                log_debug(tt::LogMetal, "Remote Device Dispatch core: Logical: {} - Physical: {} will be reset on MMIO Device.", dispatch_location.str(), virtual_core.str());
            }
            if (dispatch_core_manager::instance().is_prefetcher_core_allocated(device_id, curr_channel, cq_id)) {
                tt_cxy_pair prefetch_location = dispatch_core_manager::instance().prefetcher_core(device_id, curr_channel, cq_id);
                CoreCoord virtual_core = this->virtual_core_from_logical_core(prefetch_location, dispatch_core_type);
                my_dispatch_cores[prefetch_location.chip].insert(virtual_core);
                log_debug(tt::LogMetal, "Remote Device Prefetch core: Logical: {} - Physical: {} will be reset on MMIO Device.", prefetch_location.str(), virtual_core.str());
            }
            if (dispatch_core_manager::instance().is_mux_core_allocated(device_id, curr_channel, cq_id)) {
                tt_cxy_pair mux_location = dispatch_core_manager::instance().mux_core(device_id, curr_channel, cq_id);
                CoreCoord virtual_core = this->virtual_core_from_logical_core(mux_location, dispatch_core_type);
                my_dispatch_cores[mux_location.chip].insert(virtual_core);
                log_debug(tt::LogMetal, "Remote Device Mux core: Logical: {} - Physical: {} will be reset on MMIO Device.", mux_location.str(), virtual_core.str());
            }
            if (dispatch_core_manager::instance().is_demux_core_allocated(device_id, curr_channel, cq_id)) {
                tt_cxy_pair demux_location = dispatch_core_manager::instance().demux_core(device_id, curr_channel, cq_id);
                CoreCoord virtual_core = this->virtual_core_from_logical_core(demux_location, dispatch_core_type);
                my_dispatch_cores[demux_location.chip].insert(virtual_core);
                log_debug(tt::LogMetal, "Remote Device Demux core: Logical: {} - Physical: {} will be reset on MMIO Device.", demux_location.str(), virtual_core.str());
            }
        }
        CoreCoord virtual_core;
        tt_cxy_pair mux_location = dispatch_core_manager::instance().mux_d_core(device_id, curr_channel, 0);
        virtual_core = this->virtual_core_from_logical_core(mux_location, dispatch_core_type);
        my_dispatch_cores[mux_location.chip].insert(virtual_core);
        tt_cxy_pair demux_location = dispatch_core_manager::instance().demux_d_core(device_id, curr_channel, 0);
        virtual_core = this->virtual_core_from_logical_core(demux_location, dispatch_core_type);
        my_dispatch_cores[demux_location.chip].insert(virtual_core);
        for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
            tt_cxy_pair prefetch_location =
                dispatch_core_manager::instance().prefetcher_d_core(device_id, curr_channel, cq_id);
            virtual_core = this->virtual_core_from_logical_core(prefetch_location, dispatch_core_type);
            my_dispatch_cores[prefetch_location.chip].insert(virtual_core);
        }
        for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
            tt_cxy_pair dispatch_location =
                dispatch_core_manager::instance().dispatcher_d_core(device_id, curr_channel, cq_id);
            virtual_core = this->virtual_core_from_logical_core(dispatch_location, dispatch_core_type);
            my_dispatch_cores[dispatch_location.chip].insert(virtual_core);
        }
        for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
            // Include dispatch_s in the dispatch core location set, if its not on the same core as dispatch_d
            tt_cxy_pair dispatch_location =
                dispatch_core_manager::instance().dispatcher_d_core(device_id, curr_channel, cq_id);
            virtual_core = this->virtual_core_from_logical_core(dispatch_location, dispatch_core_type);
            tt_cxy_pair dispatch_s_location =
                dispatch_core_manager::instance().dispatcher_s_core(device_id, curr_channel, cq_id);
            CoreCoord virtual_core_dispatch_s = this->virtual_core_from_logical_core(dispatch_s_location, dispatch_core_type);
            if (virtual_core_dispatch_s != virtual_core) {
                my_dispatch_cores[dispatch_s_location.chip].insert(virtual_core_dispatch_s);
            }
        }
    }
}

void Device::initialize_cluster() {
    ZoneScoped;
    if (llrt::RunTimeOptions::get_instance().get_clear_l1()) {
        this->clear_l1_state();
    }
    int ai_clk = tt::Cluster::instance().get_device_aiclk(this->id_);
    log_info(tt::LogMetal, "AI CLK for device {} is:   {} MHz", this->id_, ai_clk);
}

void Device::initialize_default_sub_device_state(size_t l1_small_size, size_t trace_region_size, tt::stl::Span<const std::uint32_t> l1_bank_remap) {
    // Create the default sub-device manager representing the entire chip
    const auto& compute_grid_size = this->compute_with_storage_grid_size();
    const auto& active_eth_cores = this->get_active_ethernet_cores(true);
    std::vector<CoreRange> active_eth_core_ranges;
    active_eth_core_ranges.reserve(active_eth_cores.size());
    for (const auto& core : active_eth_cores) {
        active_eth_core_ranges.emplace_back(core, core);
    }

    auto sub_devices = {SubDevice(std::array{
        CoreRangeSet(CoreRange({0, 0}, {compute_grid_size.x - 1, compute_grid_size.y - 1})),
        CoreRangeSet(std::move(active_eth_core_ranges))})};

    sub_device_manager_tracker_ = std::make_unique<SubDeviceManagerTracker>(
        this, this->initialize_allocator(l1_small_size, trace_region_size, l1_bank_remap), sub_devices);
}

std::unique_ptr<Allocator> Device::initialize_allocator(size_t l1_small_size, size_t trace_region_size, tt::stl::Span<const std::uint32_t> l1_bank_remap) {
    ZoneScoped;
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    const auto &dispatch_core_config = dispatch_core_manager::instance().get_dispatch_core_config(this->id_);
    CoreType dispatch_core_type = dispatch_core_config.get_core_type();
    // Construct allocator config from soc_desc
    // Take max alignment to satisfy NoC rd/wr constraints
    // Tensix/Eth -> PCIe/DRAM src and dst addrs must be L1_ALIGNMENT aligned
    // PCIe/DRAM -> Tensix/Eth src and dst addrs must be DRAM_ALIGNMENT aligned
    // Tensix/Eth <-> Tensix/Eth src and dst addrs must be L1_ALIGNMENT aligned
    const auto &logical_size = this->logical_grid_size();
    const auto &compute_size = this->compute_with_storage_grid_size();
    AllocatorConfig config(
        {.num_dram_channels = static_cast<size_t>(soc_desc.get_num_dram_channels()),
         .dram_bank_size = soc_desc.dram_bank_size,
         .dram_bank_offsets = {},
         .dram_unreserved_base =
             hal.get_dev_addr(HalDramMemAddrType::DRAM_BARRIER) + hal.get_dev_size(HalDramMemAddrType::DRAM_BARRIER),
         .dram_alignment = hal.get_alignment(HalMemType::DRAM),
         .l1_unreserved_base = align(
             hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::UNRESERVED),
             hal.get_alignment(HalMemType::DRAM)),
         .worker_grid = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(logical_size.x - 1, logical_size.y - 1))),
         .worker_l1_size = static_cast<size_t>(soc_desc.worker_l1_size),
         .storage_core_bank_size = get_storage_core_bank_size(id_, num_hw_cqs_, dispatch_core_config),
         .l1_small_size = tt::align(l1_small_size, hal.get_alignment(HalMemType::DRAM)),
         .trace_region_size = tt::align(trace_region_size, hal.get_alignment(HalMemType::DRAM)),
         .core_type_from_noc_coord_table = {},  // Populated later
         .worker_log_to_virtual_routing_x = tt::Cluster::instance().get_worker_logical_to_virtual_x(this->id()),
         .worker_log_to_virtual_routing_y = tt::Cluster::instance().get_worker_logical_to_virtual_y(this->id()),
         .l1_bank_remap = {l1_bank_remap.begin(), l1_bank_remap.end()},
         .compute_grid = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(compute_size.x - 1, compute_size.y - 1))),
         .l1_alignment = hal.get_alignment(HalMemType::L1),
         .disable_interleaved = false});
    TT_FATAL(config.l1_small_size < (config.storage_core_bank_size.has_value() ? config.storage_core_bank_size.value() : config.worker_l1_size - config.l1_unreserved_base),
            "Reserved size must be less than bank size");
    TT_FATAL(
        config.l1_small_size % config.l1_alignment == 0,
        "Reserved size must be aligned to L1 allocator alignment {}",
        config.l1_alignment);
    // Initialize dram_offsets from soc_descriptor
    for (auto channel = 0; channel < soc_desc.get_num_dram_channels(); channel++) {
        config.dram_bank_offsets.push_back(soc_desc.get_address_offset(channel));
    }
    // Initialize core_type_from_noc_coord_table table
    for (const auto& core: soc_desc.physical_cores) {
        config.core_type_from_noc_coord_table.insert({this->virtual_core_from_physical_core(core.first, core.second.type), AllocCoreType::Invalid});
    }

    for (const CoreCoord& core : tt::get_logical_compute_cores(id_, num_hw_cqs_, dispatch_core_config)) {
        this->compute_cores_.insert(core);
        const auto noc_coord = this->worker_core_from_logical_core(core);
        config.core_type_from_noc_coord_table[noc_coord] = AllocCoreType::ComputeAndStore;
    }
    for (const CoreCoord& core : tt::get_logical_storage_cores(id_, num_hw_cqs_, dispatch_core_config)) {
        this->storage_only_cores_.insert(core);
        const auto noc_coord = this->worker_core_from_logical_core(core);
        config.core_type_from_noc_coord_table[noc_coord] = AllocCoreType::StorageOnly;
    }
    for (const CoreCoord &core : tt::get_logical_dispatch_cores(id_, num_hw_cqs_, dispatch_core_config)) {
        const auto noc_coord = this->virtual_core_from_logical_core(core, dispatch_core_type);
        config.core_type_from_noc_coord_table[noc_coord] = AllocCoreType::Dispatch;
    }
    for (const auto &core : soc_desc.get_logical_ethernet_cores()) {
        this->ethernet_cores_.insert(core);
    }

    // L1_BANKING scheme creates 1 bank per DRAM core and splits up L1 such that there are power 2 num L1 banks
    // This is the only allocator scheme supported because kernel APIs assume num L1 banks are power of 2
    TT_ASSERT(this->allocator_scheme_ == MemoryAllocator::L1_BANKING);
    return std::make_unique<L1BankingAllocator>(config);
}

void Device::initialize_device_kernel_defines()
{
    // Clear previously stored defines, in case we are running with different configuration this time.
    // This is needed to handle the case where the number of L1 banks on GS can be changed in each run.
    this->device_kernel_defines_.clear();
    const size_t num_dram_banks = this->num_banks(BufferType::DRAM);
    const size_t num_l1_banks = this->num_banks(BufferType::L1);

    bool is_dram_pow2 = ceil(log2(num_dram_banks)) == log2(num_dram_banks);
    bool is_l1_pow2 = ceil(log2(num_l1_banks)) == log2(num_l1_banks);

    this->device_kernel_defines_.emplace("NUM_DRAM_BANKS", std::to_string(num_dram_banks));
    this->device_kernel_defines_.emplace("NUM_L1_BANKS", std::to_string(num_l1_banks));

    if (is_dram_pow2) {
        this->device_kernel_defines_.emplace("LOG_BASE_2_OF_NUM_DRAM_BANKS", std::to_string(static_cast<size_t>(log2(num_dram_banks))));
    } else {
        this->device_kernel_defines_.emplace("IS_NOT_POW2_NUM_DRAM_BANKS", "1");
    }
    if (is_l1_pow2) {
        this->device_kernel_defines_.emplace("LOG_BASE_2_OF_NUM_L1_BANKS", std::to_string(static_cast<size_t>(log2(num_l1_banks))));
    } else {
        this->device_kernel_defines_.emplace("IS_NOT_POW2_NUM_L1_BANKS", "1");
    }

    // TODO (abhullar): Until we switch to virtual coordinates, we need to pass physical PCIe coordinates to device
    //  because Blackhole PCIe endpoint is dependent on board type
    const metal_SocDescriptor& soc_d = tt::Cluster::instance().get_soc_desc(this->id());
    auto pcie_cores = soc_d.get_pcie_cores();
    auto grid_size = this->grid_size();

    CoreCoord pcie_core = pcie_cores.empty() ? grid_size : pcie_cores[0];

    this->device_kernel_defines_.emplace("PCIE_NOC_X", std::to_string(pcie_core.x));
    this->device_kernel_defines_.emplace("PCIE_NOC_Y", std::to_string(pcie_core.y));
}

void Device::initialize_build() {
    ZoneScoped;

    this->initialize_device_kernel_defines();
    this->build_env_.init(this->build_key(), this->arch(), this->device_kernel_defines_);

    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(this->id());
    uint32_t dispatch_message_addr =
        dispatch_constants::get(dispatch_core_type, this->num_hw_cqs_).get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE);

    uint32_t num_build_states = hal.get_num_risc_processors();

    auto init_helper = [this, dispatch_message_addr, num_build_states] (bool is_fw) -> JitBuildStateSet {
        std::vector<std::shared_ptr<JitBuildState>> build_states;

        build_states.resize(num_build_states);
        uint32_t programmable_core_type_count = hal.get_programmable_core_type_count();
        if (is_fw) {
            this->build_state_indices_.resize(programmable_core_type_count);
        }

        uint32_t index = 0;
        for (uint32_t programmable_core = 0; programmable_core < programmable_core_type_count; programmable_core++) {
            HalProgrammableCoreType core_type = magic_enum::enum_value<HalProgrammableCoreType>(programmable_core);
            uint32_t processor_class_count = hal.get_processor_classes_count(programmable_core);
            if (is_fw) {
                this->build_state_indices_[programmable_core].resize(processor_class_count);
            }
            for (uint32_t processor_class = 0; processor_class < processor_class_count; processor_class++) {
                auto compute_proc_class = magic_enum::enum_cast<HalProcessorClassType>(processor_class);
                bool is_compute_processor = compute_proc_class.has_value() and compute_proc_class.value() == HalProcessorClassType::COMPUTE;
                uint32_t processor_types_count = hal.get_processor_types_count(programmable_core, processor_class);
                if (is_fw) {
                    this->build_state_indices_[programmable_core][processor_class] = {index, processor_types_count};
                }
                for (uint32_t processor_type = 0; processor_type < processor_types_count; processor_type++) {
                    switch (core_type) {
                        case HalProgrammableCoreType::TENSIX: {
                            if (is_compute_processor) {
                                build_states[index] = std::make_shared<JitBuildCompute>(
                                    this->build_env_, JitBuiltStateConfig{.processor_id = processor_type, .is_fw=is_fw, .dispatch_message_addr=dispatch_message_addr});
                            } else {
                                // TODO: Make .processor_id = processor_type when brisc and ncrisc are considered one processor class
                                build_states[index] = std::make_shared<JitBuildDataMovement>(
                                    this->build_env_, JitBuiltStateConfig{.processor_id = processor_class, .is_fw=is_fw, .dispatch_message_addr=dispatch_message_addr});
                            }
                            break;
                        }
                        case HalProgrammableCoreType::ACTIVE_ETH: {
                            // Cooperative means active erisc FW needs to context switch to base FW
                            bool is_cooperative = this->arch() == ARCH::WORMHOLE_B0;
                            build_states[index] = std::make_shared<JitBuildActiveEthernet>(
                                this->build_env_,
                                JitBuiltStateConfig{
                                    .processor_id = processor_class,
                                    .is_fw = is_fw,
                                    .dispatch_message_addr = dispatch_message_addr,
                                    .is_cooperative = is_cooperative});
                            break;
                        }
                        case HalProgrammableCoreType::IDLE_ETH: {
                            build_states[index] = std::make_shared<JitBuildIdleEthernet>(
                                this->build_env_, JitBuiltStateConfig{.processor_id = processor_class, .is_fw=is_fw, .dispatch_message_addr=dispatch_message_addr});
                            break;
                        }
                        default:
                            TT_THROW("Unsupported programable core type {} to initialize build states", magic_enum::enum_name(core_type));
                    }
                    index++;
                }
            }
        }

       return build_states;
    };

    this->firmware_build_states_ = init_helper(true);
    this->kernel_build_states_ = init_helper(false);
}

void Device::build_firmware() {
    log_debug(tt::LogMetal, "Building base firmware for device {}", this->id_);
    ZoneScoped;

    jit_build_set(this->firmware_build_states_, nullptr);
}

void Device::initialize_device_bank_to_noc_tables(const HalProgrammableCoreType &core_type, CoreCoord virtual_core)
{
    const uint32_t dram_to_noc_sz_in_bytes = dram_bank_to_noc_xy_.size() * sizeof(uint16_t);
    const uint32_t l1_to_noc_sz_in_bytes = l1_bank_to_noc_xy_.size() * sizeof(uint16_t);
    const uint32_t dram_offset_sz_in_bytes = dram_bank_offset_map_.size() * sizeof(int32_t);
    const uint32_t l1_offset_sz_in_bytes = l1_bank_offset_map_.size() * sizeof(int32_t);

    const uint64_t mem_bank_to_noc_addr = hal.get_dev_addr(core_type, HalL1MemAddrType::BANK_TO_NOC_SCRATCH);
    const uint32_t mem_bank_to_noc_size = hal.get_dev_size(core_type, HalL1MemAddrType::BANK_TO_NOC_SCRATCH);

    TT_ASSERT((dram_to_noc_sz_in_bytes + l1_to_noc_sz_in_bytes + dram_offset_sz_in_bytes + l1_offset_sz_in_bytes) <= mem_bank_to_noc_size,
        "Size of bank_to_noc table is greater than available space");

    tt::Cluster::instance().write_core(&dram_bank_to_noc_xy_[0], dram_to_noc_sz_in_bytes, tt_cxy_pair(this->id(), virtual_core), mem_bank_to_noc_addr);
    uint64_t l1_noc_addr = mem_bank_to_noc_addr + dram_to_noc_sz_in_bytes;
    tt::Cluster::instance().write_core(&l1_bank_to_noc_xy_[0], l1_to_noc_sz_in_bytes, tt_cxy_pair(this->id(), virtual_core), l1_noc_addr);

    uint64_t dram_offset_addr = l1_noc_addr + l1_to_noc_sz_in_bytes;
    tt::Cluster::instance().write_core(&dram_bank_offset_map_[0], dram_offset_sz_in_bytes, tt_cxy_pair(this->id(), virtual_core), dram_offset_addr);
    uint64_t l1_offset_addr = dram_offset_addr + dram_offset_sz_in_bytes;
    tt::Cluster::instance().write_core(&l1_bank_offset_map_[0], l1_offset_sz_in_bytes, tt_cxy_pair(this->id(), virtual_core), l1_offset_addr);
}

void Device::initialize_firmware(const HalProgrammableCoreType &core_type, CoreCoord virtual_core, launch_msg_t *launch_msg, go_msg_t* go_msg) {
    ZoneScoped;

    this->initialize_device_bank_to_noc_tables(core_type, virtual_core);
    uint32_t core_type_idx = hal.get_programmable_core_type_index(core_type);
    uint32_t processor_class_count = hal.get_processor_classes_count(core_type);
    auto jit_build_config = hal.get_jit_build_config(core_type_idx, 0, 0); // Only the first risc needs to be programmed

    switch (core_type) {
        case HalProgrammableCoreType::TENSIX: {
            for (uint32_t processor_class = 0; processor_class < processor_class_count; processor_class++) {
                auto [build_idx, num_build_states] = this->build_processor_type_to_index(core_type_idx, processor_class);
                for (uint32_t riscv_id = build_idx; riscv_id < (build_idx + num_build_states); riscv_id++) {
                    ll_api::memory const& binary_mem = llrt::get_risc_binary(
                        firmware_build_states_[riscv_id]->get_target_out_path(""));
                    uint32_t fw_size = binary_mem.get_text_size();
                    if (riscv_id == 1) { // TODO: clean up how brisc/ncrisc are handled
                        // In this context, ncrisc_kernel_size16 is the size of the fw
                        launch_msg->kernel_config.ncrisc_kernel_size16 = (fw_size + 15) >> 4;
                    }
                    log_debug(LogDevice, "RISC {} fw binary size: {} in bytes", riscv_id, fw_size);

                    if (not llrt::RunTimeOptions::get_instance().get_skip_loading_fw())  {
                        llrt::test_load_write_read_risc_binary(binary_mem, this->id(), virtual_core, core_type_idx, processor_class, (riscv_id - build_idx));
                    }
                }
            }

            if (this->using_slow_dispatch()) {
                // Host always writes launch messages
                launch_msg->kernel_config.mode = DISPATCH_MODE_HOST;
            } else {
                std::vector<CoreCoord> physical_dispatch_cores = {};
                if (dispatch_core_manager::instance().get_dispatch_core_type(this->id()) == CoreType::WORKER) {
                    physical_dispatch_cores = this->worker_cores_from_logical_cores(dispatch_core_manager::instance().get_all_logical_dispatch_cores(this->id()));
                }
                if (std::find(physical_dispatch_cores.begin(), physical_dispatch_cores.end(), virtual_core) != physical_dispatch_cores.end()) {
                    // Dispatch cores - Host writes launch messages
                    launch_msg->kernel_config.mode = DISPATCH_MODE_HOST;
                } else {
                    // Worker cores - Dispatcher will write launch messages
                    launch_msg->kernel_config.mode = DISPATCH_MODE_DEV;
                }
            }

            break;
        }
        case HalProgrammableCoreType::ACTIVE_ETH:
        case HalProgrammableCoreType::IDLE_ETH: {
            bool is_idle_eth = core_type == HalProgrammableCoreType::IDLE_ETH;
            TensixSoftResetOptions reset_val = TENSIX_ASSERT_SOFT_RESET;
            if (not is_idle_eth) {
                // std::cout << "Active eth fw_launch_addr_value " << std::hex << jit_build_config.fw_launch_addr_value
                //           << " jit_build_config.fw_launch_addr " << jit_build_config.fw_launch_addr << std::endl;
                reset_val =
                    reset_val & static_cast<TensixSoftResetOptions>(
                                    ~std::underlying_type<TensixSoftResetOptions>::type(TensixSoftResetOptions::BRISC));
            } else {
                // std::cout << "Idle eth fw_launch_addr_value " << std::hex << jit_build_config.fw_launch_addr_value
                //           << " jit_build_config.fw_launch_addr " << jit_build_config.fw_launch_addr << std::endl;
            }
            // std::cout << "Assert risc reset val for eth " << std::hex << (uint32_t)reset_val << std::dec <<
            // std::endl;
            if (is_idle_eth or this->arch() == ARCH::BLACKHOLE) {
                tt::Cluster::instance().assert_risc_reset_at_core(tt_cxy_pair(this->id(), virtual_core), reset_val);
            }
            if (not llrt::RunTimeOptions::get_instance().get_skip_loading_fw()) {
                for (uint32_t processor_class = 0; processor_class < processor_class_count; processor_class++) {
                    auto [build_idx, num_build_states] = this->build_processor_type_to_index(core_type_idx, processor_class);
                    for (uint32_t eriscv_id = build_idx; eriscv_id < (build_idx + num_build_states); eriscv_id++) {
                        // std::cout << "Getting risc binary " <<
                        // firmware_build_states_[eriscv_id]->get_target_out_path("") << std::endl;
                        ll_api::memory const& binary_mem = llrt::get_risc_binary(
                            firmware_build_states_[eriscv_id]->get_target_out_path(""));
                        uint32_t fw_size = binary_mem.get_text_size();
                        log_debug(LogDevice, "ERISC fw binary size: {} in bytes", fw_size);
                        llrt::test_load_write_read_risc_binary(binary_mem, this->id(), virtual_core, core_type_idx, processor_class, (eriscv_id - build_idx));
                    }
                }
            }
            // Ethernet worker core. Launch messages will be sent by FD infra if it's enabled
            // Idle ethernet core. Used by FD infra. Host will write launch messages during init.
            launch_msg->kernel_config.mode = (this->using_slow_dispatch() or is_idle_eth) ? DISPATCH_MODE_HOST :  DISPATCH_MODE_DEV;
            break;
        }
        default:
            TT_THROW("Unsupported programable core type {} to initialize build states", magic_enum::enum_name(core_type));
    }

    tt::Cluster::instance().write_core(
        &jit_build_config.fw_launch_addr_value, sizeof(uint32_t), tt_cxy_pair(this->id_, virtual_core), jit_build_config.fw_launch_addr);

    // Initialize each entry in the launch_msg ring buffer with the correct dispatch mode - Cores that don't get a valid
    // launch_message during program execution need to at least have the correct dispatch mode.
    // When using Fast Dispatch on Tensix:
        // dispatch cores (Tensix) configured with DISPATCH_MODE_HOST
        // worker cores (Tensix and active eth) configured with DISPATCH_MODE_DEV
        // Idle Eth cores configured with DISPATCH_MODE_HOST but not used
    // When using Fast Dispatch on Idle Eth:
        // dispatch cores (Idle Eth) configured with DISPATCH_MODE_HOST
        // worker cores (Tensix and active eth) configured with DISPATCH_MODE_DEV
    // When using Slow Dispatch, all cores initialized with DISPATCH_MODE_HOST
    std::vector<launch_msg_t> init_launch_msg_data(launch_msg_buffer_num_entries, *launch_msg);
    tt::Cluster::instance().write_core(init_launch_msg_data.data(), launch_msg_buffer_num_entries * sizeof(launch_msg_t), tt_cxy_pair(this->id(), virtual_core), this->get_dev_addr(virtual_core, HalL1MemAddrType::LAUNCH));
    uint32_t go_addr = this->get_dev_addr(virtual_core, HalL1MemAddrType::GO_MSG);
    tt::Cluster::instance().write_core(go_msg, sizeof(go_msg_t), tt_cxy_pair(this->id(), virtual_core), go_addr);
    uint64_t launch_msg_buffer_read_ptr_addr = this->get_dev_addr(virtual_core, HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR);
    uint32_t zero = 0;
    tt::Cluster::instance().write_core(&zero, sizeof(uint32_t), tt_cxy_pair(this->id(), virtual_core), launch_msg_buffer_read_ptr_addr);
}

void Device::reset_cores() {
    ZoneScoped;

    auto kernel_still_running = [](launch_msg_t* launch_msg, go_msg_t *go_signal) {
        return (go_signal->signal) == RUN_MSG_GO && launch_msg->kernel_config.exit_erisc_kernel == 0;
    };

    auto mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->id_);
    // Assert worker cores + dispatch cores, in case they were in a bad state from before.
    std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>> dispatch_cores, other_dispatch_cores, device_to_early_exit_cores;
    go_msg_t go_msg;
    std::memset(&go_msg, 0, sizeof(go_msg_t));
    for (const auto& eth_core : this->get_active_ethernet_cores()) {
        CoreCoord virtual_core = this->ethernet_core_from_logical_core(eth_core);
        std::vector<uint32_t> data(sizeof(launch_msg_t) / sizeof(uint32_t));
        std::vector<uint32_t> go_signal_data(sizeof(go_msg_t) / sizeof(uint32_t));
        DeviceAddr launch_addr = hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::LAUNCH);
        DeviceAddr go_signal_addr = hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::GO_MSG);

        if (this->arch() == ARCH::BLACKHOLE) {
            TensixSoftResetOptions reset_val =
                TENSIX_ASSERT_SOFT_RESET &
                static_cast<TensixSoftResetOptions>(
                    ~std::underlying_type<TensixSoftResetOptions>::type(TensixSoftResetOptions::BRISC));
            tt::Cluster::instance().assert_risc_reset_at_core(tt_cxy_pair(this->id(), virtual_core), reset_val);
        }

        // uint32_t reset_val;
        // tt::Cluster::instance().read_reg(&reset_val, tt_cxy_pair(this->id_, physical_core), 0xFFB121B0);
        // std::cout << "Active eth core " << eth_core.str() << " in reset_cores reset val " << std::hex << reset_val <<
        // std::dec << std::endl;

        // TensixSoftResetOptions assert_reset_val = TENSIX_ASSERT_SOFT_RESET;
        // assert_reset_val = assert_reset_val &
        // static_cast<TensixSoftResetOptions>(~std::underlying_type<TensixSoftResetOptions>::type(TensixSoftResetOptions::BRISC));
        // std::cout << "\tAssert reset val " << std::hex << (uint32_t)assert_reset_val << std::dec << std::endl;

        // TensixSoftResetOptions deassert_reset_val = TENSIX_DEASSERT_SOFT_RESET &
        // static_cast<TensixSoftResetOptions>(~std::underlying_type<TensixSoftResetOptions>::type(TensixSoftResetOptions::TRISC0));
        // std::cout << "\tDeassert reset val " << std::hex << (uint32_t)deassert_reset_val << std::dec << std::endl;

        data = tt::llrt::read_hex_vec_from_core(this->id(), virtual_core, launch_addr, sizeof(launch_msg_t));
        go_signal_data = tt::llrt::read_hex_vec_from_core(this->id(), virtual_core, go_signal_addr, sizeof(go_msg_t));
        launch_msg_t* launch_msg = (launch_msg_t*)(&data[0]);
        go_msg_t* go_signal = (go_msg_t*)(&go_signal_data[0]);
        if (kernel_still_running(launch_msg, go_signal)) {
            log_info(
                tt::LogMetal,
                "While initializing Device {}, ethernet tunneler core {} on Device {} detected as still running, "
                "issuing exit signal.",
                this->id(),
                virtual_core.str(),
                this->id());
            launch_msg->kernel_config.exit_erisc_kernel = 1;
            llrt::write_launch_msg_to_core(this->id(), virtual_core, launch_msg, &go_msg, launch_addr, false);
            device_to_early_exit_cores[this->id()].insert(virtual_core);
        }
    }

    this->get_associated_dispatch_virtual_cores(dispatch_cores, other_dispatch_cores);
    // Ignore other_dispatch_cores, they will be reset by the devices that use them.
    for (auto &id_and_cores : dispatch_cores) {
        for (auto it = id_and_cores.second.begin(); it != id_and_cores.second.end(); it++) {
            const auto &virtual_core = *it;
            // For new FD init, we've already initialized dispatch cores on other devices, so don't reset here.
            if (id_and_cores.first != this->id())
                continue;

            // Only need to manually reset ethernet dispatch cores, tensix cores are all reset below.
            if (tt::Cluster::instance().is_ethernet_core(virtual_core, id_and_cores.first)) {
                // Ethernet cores won't be reset, so just signal the dispatch cores to early exit.
                std::vector<uint32_t> data(sizeof(launch_msg_t) / sizeof(uint32_t));
                std::vector<uint32_t> go_signal_data(sizeof(go_msg_t) / sizeof(uint32_t));
                DeviceAddr launch_addr = hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::LAUNCH);
                DeviceAddr go_signal_addr = hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::GO_MSG);
                data = tt::llrt::read_hex_vec_from_core(
                    id_and_cores.first, virtual_core, launch_addr, sizeof(launch_msg_t));
                go_signal_data = tt::llrt::read_hex_vec_from_core(
                    this->id(), virtual_core, go_signal_addr, sizeof(go_msg_t));
                launch_msg_t *launch_msg = (launch_msg_t *)(&data[0]);
                go_msg_t * go_signal = (go_msg_t *)(&go_signal_data[0]);
                if (kernel_still_running(launch_msg, go_signal)) {
                    log_info(
                        tt::LogMetal,
                        "While initializing device {}, ethernet dispatch core {} on Device {} detected as still running, issuing exit signal.",
                        this->id(),
                        virtual_core.str(),
                        id_and_cores.first);
                    launch_msg->kernel_config.exit_erisc_kernel = 1;
                    llrt::write_launch_msg_to_core(id_and_cores.first, virtual_core, launch_msg, &go_msg, launch_addr, false);
                    device_to_early_exit_cores[id_and_cores.first].insert(virtual_core);
                }
            }
        }
    }

    // Early exiting dispatch cores should show RUN_MSG_DONE when they exit.
    for (auto &id_and_cores : device_to_early_exit_cores) {
        const int timeout_ms = 10000; // 10 seconds for now
        if (!id_and_cores.second.empty()) {
            try {
                llrt::internal_::wait_until_cores_done(id_and_cores.first, RUN_MSG_GO, id_and_cores.second, timeout_ms);
            } catch (std::runtime_error &e) {
                log_warning(
                    "Detected dispatch kernels still running but failed to complete an early exit. This may happen "
                    "from time to time following a reset, continuing to FW intialization...");
            }
        }
    }

    // Reset Tensix cores
    CoreCoord grid_size = this->logical_grid_size();
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core = this->worker_core_from_logical_core(logical_core);

            // Don't reset dispatch cores for other devices, in case they're still running.
            if (other_dispatch_cores[this->id_].find(worker_core) == other_dispatch_cores[this->id_].end()) {
                if (this->storage_only_cores_.find(logical_core) == this->storage_only_cores_.end()) {
                    tt::Cluster::instance().assert_risc_reset_at_core(tt_cxy_pair(this->id(), worker_core));
                }
            }
        }
    }
}

void Device::initialize_and_launch_firmware() {
    ZoneScoped;

    launch_msg_t launch_msg;
    go_msg_t go_msg;
    std::memset(&launch_msg, 0, sizeof(launch_msg_t));
    go_msg.signal = RUN_MSG_INIT;

    // Populate core info, which will be written to device
    std::vector<uint32_t> core_info_vec(sizeof(core_info_msg_t) / sizeof(uint32_t));
    core_info_msg_t *core_info = (core_info_msg_t *) core_info_vec.data();

    const metal_SocDescriptor& soc_d = tt::Cluster::instance().get_soc_desc(this->id());
    uint64_t pcie_chan_base_addr = tt::Cluster::instance().get_pcie_base_addr_from_device(this->id());
    uint32_t num_host_channels = tt::Cluster::instance().get_num_host_channels(this->id());
    uint64_t pcie_chan_end_addr = pcie_chan_base_addr;
    for (int pcie_chan = 0; pcie_chan < num_host_channels; pcie_chan++) {
        pcie_chan_end_addr += tt::Cluster::instance().get_host_channel_size(this->id(), pcie_chan);
    }
    core_info->noc_pcie_addr_base = pcie_chan_base_addr;
    core_info->noc_pcie_addr_end = pcie_chan_end_addr;
    core_info->noc_dram_addr_base = 0;
    core_info->noc_dram_addr_end = soc_d.dram_core_size;

    const std::vector<CoreCoord> &pcie_cores = soc_d.get_pcie_cores();
    const std::vector<CoreCoord> &dram_cores = soc_d.get_dram_cores();
    const std::vector<CoreCoord> &eth_cores = soc_d.get_physical_ethernet_cores();
    // The SOC descriptor can list a dram core multiple times, depending on how GDDR is assigned to banks
    // Get a list of unique DRAM cores.
    std::unordered_set<CoreCoord> unique_dram_cores(dram_cores.begin(), dram_cores.end());
    TT_ASSERT(
        pcie_cores.size() + unique_dram_cores.size() + eth_cores.size() <= MAX_NON_WORKER_CORES,
        "Detected more pcie/dram/eth cores than fit in the device mailbox.");
    TT_ASSERT(
        eth_cores.size() <= MAX_VIRTUAL_NON_WORKER_CORES,
        "Detected more eth cores (virtual non-workers) than can fit in device mailbox.");
    for (int idx = 0; idx < MAX_NON_WORKER_CORES; idx++) {
        core_info->non_worker_cores[idx] = {CORE_COORD_INVALID, CORE_COORD_INVALID, AddressableCoreType::UNKNOWN};
    }
    for (int idx = 0; idx < MAX_VIRTUAL_NON_WORKER_CORES; idx++) {
        core_info->virtual_non_worker_cores[idx] = {CORE_COORD_INVALID, CORE_COORD_INVALID, AddressableCoreType::UNKNOWN};
    }

    int non_worker_cores_idx = 0;
    for (const CoreCoord &core : pcie_cores) {
        core_info->non_worker_cores[non_worker_cores_idx++] = {core.x, core.y, AddressableCoreType::PCIE};
    }
    for (const CoreCoord &core : unique_dram_cores) {
        core_info->non_worker_cores[non_worker_cores_idx++] = {core.x, core.y, AddressableCoreType::DRAM};
    }
    for (const CoreCoord &core : eth_cores) {
        core_info->non_worker_cores[non_worker_cores_idx++] = {core.x, core.y, AddressableCoreType::ETH};
    }
    if (hal.is_coordinate_virtualization_enabled()) {
        // Track Virtual Non Worker Cores (In this case only Eth) separately
        uint32_t virtual_non_worker_cores_idx = 0;
        for (const CoreCoord &core : eth_cores) {
            auto virtual_core = this->virtual_core_from_physical_core(core, CoreType::ETH);
            core_info->virtual_non_worker_cores[virtual_non_worker_cores_idx++] = {virtual_core.x, virtual_core.y, AddressableCoreType::ETH};
        }
    }

    // Determine which noc-coords are harvested
    // TODO(PGK/Almeet): fix this w/ new UMD
    std::vector<uint32_t> harvested_rows;
    uint32_t harvested_noc_rows = tt::Cluster::instance().get_harvested_rows(this->id());
    for (uint32_t y = 0; y < soc_d.grid_size.y; y++) {
        bool row_harvested = (harvested_noc_rows >> y) & 0x1;
        if (row_harvested) {
            harvested_rows.push_back(y);
        }
    }
    TT_ASSERT(harvested_rows.size() <= MAX_HARVESTED_ROWS, "Detected more harvested rows than fit in mailbox.");
    for (int idx = 0; idx < MAX_HARVESTED_ROWS; idx++) {
        core_info->harvested_y[idx] = (idx < harvested_rows.size()) ? harvested_rows[idx] : CORE_COORD_INVALID;
        // Populate harvested rows in virtual coordinate space if virtualization is supported by HW.
        // Harvested rows in the virtual space are placed at the end of the worker grid,
        if (hal.is_coordinate_virtualization_enabled() and idx < harvested_rows.size()) {
            core_info->virtual_harvested_y[idx] = (hal.get_virtual_worker_start_y() + this->logical_grid_size().y + harvested_rows.size() - (idx + 1));
        } else {
            core_info->virtual_harvested_y[idx] = CORE_COORD_INVALID;
        }
    }

    core_info->noc_size_x = soc_d.grid_size.x;
    core_info->noc_size_y = soc_d.grid_size.y;
    core_info->worker_grid_size_x = this->logical_grid_size().x;  // Grid size as virtual coords see it (workers only)
    core_info->worker_grid_size_y = this->logical_grid_size().y;

    // Download to worker cores
    log_debug("Initializing firmware");
    CoreCoord grid_size = this->logical_grid_size();
    std::unordered_set<CoreCoord> not_done_cores;

    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            if (!this->storage_only_cores_.count(logical_core)) {
                CoreCoord worker_core = this->worker_core_from_logical_core(logical_core);
                tt::llrt::write_hex_vec_to_core(
                    this->id(), worker_core, core_info_vec, this->get_dev_addr(worker_core, HalL1MemAddrType::CORE_INFO));
                this->initialize_firmware(HalProgrammableCoreType::TENSIX, worker_core, &launch_msg, &go_msg);
                not_done_cores.insert(worker_core);
            }
        }
    }

    // Clear erisc sync info
    for (const auto& eth_core : this->get_active_ethernet_cores()) {
        static std::vector<uint32_t> zero_vec_erisc_init(
            hal.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::APP_SYNC_INFO) / sizeof(uint32_t),
            0);
        CoreCoord virtual_core = this->ethernet_core_from_logical_core(eth_core);
        llrt::write_hex_vec_to_core(
            this->id(), virtual_core, zero_vec_erisc_init, hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::APP_SYNC_INFO));
    }

    bool init_aerisc = std::getenv("TT_METAL_INIT_AERISC") != nullptr;

    // Load erisc app base FW to eth cores on WH and active_erisc FW on second risc of BH active eth cores
    std::unordered_set<CoreCoord> bh_active_eth_cores;
    for (const auto& eth_core : this->get_active_ethernet_cores()) {
        CoreCoord phys_eth_core = this->ethernet_core_from_logical_core(eth_core);
        tt::llrt::write_hex_vec_to_core(
            this->id(), phys_eth_core, core_info_vec, this->get_dev_addr(phys_eth_core, HalL1MemAddrType::CORE_INFO));
        if (init_aerisc) {
            // std::cout << "Loading fw on active eth core " << eth_core.str() << std::endl;
            this->initialize_firmware(HalProgrammableCoreType::ACTIVE_ETH, phys_eth_core, &launch_msg, &go_msg);
            if (this->arch() == ARCH::BLACKHOLE) {
                bh_active_eth_cores.insert(phys_eth_core);
                // if (init_aeric) {
                not_done_cores.insert(phys_eth_core);
                // }
            }
        }
    }

    for (const auto &eth_core : this->get_inactive_ethernet_cores()) {
        CoreCoord phys_eth_core = this->ethernet_core_from_logical_core(eth_core);
        tt::llrt::write_hex_vec_to_core(
            this->id(), phys_eth_core, core_info_vec, this->get_dev_addr(phys_eth_core, HalL1MemAddrType::CORE_INFO));
        this->initialize_firmware(HalProgrammableCoreType::IDLE_ETH, phys_eth_core, &launch_msg, &go_msg);
        not_done_cores.insert(phys_eth_core);
    }

    // Barrier between L1 writes above and deassert below
    tt::Cluster::instance().l1_barrier(this->id());

    // Deassert worker cores
    TensixSoftResetOptions reset_val = TENSIX_DEASSERT_SOFT_RESET;
    for (const auto& worker_core : not_done_cores) {
        if (bh_active_eth_cores.find(worker_core) != bh_active_eth_cores.end()) {
            // bit 12 needs to be deasserted to run second erisc on BH
            reset_val = TENSIX_DEASSERT_SOFT_RESET &
                        static_cast<TensixSoftResetOptions>(
                            ~std::underlying_type<TensixSoftResetOptions>::type(TensixSoftResetOptions::TRISC0));
        }
        // if (worker_core.y == 1) {
        //     std::cout << "standard eth deassert val " << std::hex << (uint32_t)reset_val << std::dec << std::endl;
        // }
        tt::Cluster::instance().deassert_risc_reset_at_core(tt_cxy_pair(this->id(), worker_core), reset_val);
    }

    // Wait until fw init is done, ensures the next launch msg doesn't get
    // written while fw is still in init
    log_debug("Waiting for firmware init complete");
    const int timeout_ms = 10000; // 10 seconds for now
    try {
        llrt::internal_::wait_until_cores_done(this->id(), RUN_MSG_INIT, not_done_cores, timeout_ms);
    } catch (std::runtime_error &e) {
        TT_THROW("Device {} init: failed to initialize FW! Try resetting the board.", this->id());
    }
    log_debug("Firmware init complete");
}

void Device::clear_l1_state() {
    log_debug(tt::LogMetal, "Clearing L1 for device {}", this->id_);
    // Clear all clearable Tensix and Eth L1
    CoreCoord logical_grid_size = this->logical_grid_size();
    TT_ASSERT(this->l1_size_per_core() % sizeof(uint32_t) == 0);
    std::vector<uint32_t> zero_vec(this->l1_size_per_core() / sizeof(uint32_t), 0);
    constexpr uint32_t start_address = 0;
    for (uint32_t x = 0; x < logical_grid_size.x; x++) {
        for (uint32_t y = 0; y < logical_grid_size.y; y++) {
            CoreCoord logical_core(x, y);
            detail::WriteToDeviceL1(this, logical_core, start_address, zero_vec);
        }
    }

    // These L1 ranges are restricted becase UMD base routing FW uses L1 below FIRMWARE_BASE and
    // between TILE_HEADER_BUFFER_BASE to COMMAND_Q_BASE
    // Clear erisc sync info
    // for (const auto &eth_core : this->get_active_ethernet_cores()) {

    //     static const uint32_t max_l1_loading_size = hal.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH,
    //     HalL1MemAddrType::UNRESERVED) + hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH,
    //     HalL1MemAddrType::UNRESERVED);

    //     static std::vector<uint32_t> zero_vec_above_tile_header_buffer(
    //         (max_l1_loading_size - hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH,
    //         HalL1MemAddrType::TILE_HEADER_BUFFER)) / sizeof(uint32_t), 0);

    //     CoreCoord virtual_core = this->ethernet_core_from_logical_core(eth_core);

    //     llrt::write_hex_vec_to_core(
    //         this->id(),
    //         virtual_core,
    //         zero_vec_above_tile_header_buffer,
    //         hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::TILE_HEADER_BUFFER));
    // }

    // TODO: clear idle eriscs as well
}

bool Device::dispatch_s_enabled() const {
    // Dispatch_s is always enabled for Tensix Dispatch
    // Conditionally enabled for Ethernet Dispatch - If a single CQ is being used
    // This condition may be modified for BH
    return (this->num_hw_cqs() == 1 or dispatch_core_manager::instance().get_dispatch_core_type(this->id()) == CoreType::WORKER);
}

bool Device::distributed_dispatcher() const {
    // Ethernet dispatch with a single CQ. dispatch_s and dispatch_d are on different cores.
    return (this->num_hw_cqs() == 1 and dispatch_core_manager::instance().get_dispatch_core_type(this->id())  == CoreType::ETH);
}

void Device::compile_command_queue_programs() {
    ZoneScoped;
    auto command_queue_program_ptr = std::make_unique<Program>();
    auto mmio_command_queue_program_ptr = std::make_unique<Program>();
    if (this->is_mmio_capable()) {
        auto command_queue_program_ptr = create_and_compile_cq_program(this);
        this->command_queue_programs_.push_back(std::move(command_queue_program_ptr));
        // Since devices could be set up in any order, on mmio device do a pass and populate cores for tunnelers.
        if (tt::Cluster::instance().get_mmio_device_tunnel_count(this->id_) > 0) {
            tunnels_from_mmio_ = tt::Cluster::instance().get_tunnels_from_mmio_device(this->id_);
            for (auto& tunnel : tunnels_from_mmio_) {
                for (uint32_t tunnel_stop = 0; tunnel_stop < tunnel.size() - 1; tunnel_stop++) {
                    chip_id_t device_id = tunnel[tunnel_stop];
                    chip_id_t ds_device_id = tunnel[tunnel_stop + 1];
                    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(ds_device_id);
                    // Only one tunneler per connection, use CQ ID 0
                    dispatch_core_manager::instance().tunneler_core(device_id, ds_device_id, channel, 0);
                }
            }
        }
    } else {
        auto command_queue_program_ptr = create_and_compile_cq_program(this);
        this->command_queue_programs_.push_back(std::move(command_queue_program_ptr));
    }
}

// Writes issue and completion queue pointers to device and in sysmem and loads fast dispatch program onto dispatch cores
void Device::configure_command_queue_programs() {
    chip_id_t device_id = this->id();
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
    IDevice* mmio_device = tt::DevicePool::instance().get_active_device(mmio_device_id);

    std::vector<uint32_t> zero = {0x0}; // Reset state in case L1 Clear is disabled.
    std::vector<uint32_t> pointers;
    uint32_t cq_size = this->sysmem_manager().get_cq_size();
    TT_ASSERT(this->command_queue_programs_.size() == 1);

    Program& command_queue_program = *this->command_queue_programs_[0];
    uint8_t num_hw_cqs = this->num_hw_cqs();

    // Reset host-side command queue pointers for all channels controlled by this mmio device
    if (this->is_mmio_capable()) {
        for (chip_id_t serviced_device_id : tt::Cluster::instance().get_devices_controlled_by_mmio_device(device_id)) {
            uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(serviced_device_id);
            CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(mmio_device_id);
            uint32_t host_issue_q_rd_ptr = dispatch_constants::get(dispatch_core_type).get_host_command_queue_addr(CommandQueueHostAddrType::ISSUE_Q_RD);
            uint32_t host_issue_q_wr_ptr = dispatch_constants::get(dispatch_core_type).get_host_command_queue_addr(CommandQueueHostAddrType::ISSUE_Q_WR);
            uint32_t host_completion_q_wr_ptr = dispatch_constants::get(dispatch_core_type).get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_WR);
            uint32_t host_completion_q_rd_ptr = dispatch_constants::get(dispatch_core_type).get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_RD);
            uint32_t cq_start = dispatch_constants::get(dispatch_core_type).get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
            pointers.resize(cq_start/sizeof(uint32_t));
            for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
                // Reset the host manager's pointer for this command queue
                this->sysmem_manager_->reset(cq_id);

                pointers[host_issue_q_rd_ptr / sizeof(uint32_t)] = (cq_start + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4;
                pointers[host_issue_q_wr_ptr / sizeof(uint32_t)] = (cq_start + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4;
                pointers[host_completion_q_wr_ptr / sizeof(uint32_t)] = (cq_start + this->sysmem_manager_->get_issue_queue_size(cq_id) + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4;
                pointers[host_completion_q_rd_ptr / sizeof(uint32_t)] = (cq_start + this->sysmem_manager_->get_issue_queue_size(cq_id) + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4;

                tt::Cluster::instance().write_sysmem(pointers.data(), pointers.size() * sizeof(uint32_t), get_absolute_cq_offset(channel, cq_id, cq_size), mmio_device_id, get_umd_channel(channel));
            }
        }
    }

    // Write device-side cq pointers
    configure_dispatch_cores(this);

    // Run the cq program
    program_dispatch::finalize_program_offsets(command_queue_program, this);
    detail::ConfigureDeviceWithProgram(this, command_queue_program, true);
    tt::Cluster::instance().l1_barrier(this->id());
}

void Device::update_dispatch_cores_for_multi_cq_eth_dispatch() {
    // When running Multiple CQs using Ethernet Dispatch, we may need more dispatch cores than those allocated in the
    // core descriptor (ex: 2 CQs on N300 need 10 dispatch cores and the core descriptor only allocates 6).
    // Infer the remaining dispatch cores from the idle eth core list (this is device dependent).
    if (dispatch_core_manager::instance().get_dispatch_core_type(this->id()) == CoreType::ETH) {
        auto& dispatch_core_manager = dispatch_core_manager::instance();
        for (const auto& idle_eth_core : this->get_inactive_ethernet_cores()) {
            dispatch_core_manager.add_dispatch_core_to_device(this->id(), idle_eth_core);
        }
    }
}

void Device::init_command_queue_host() {
    using_fast_dispatch_ = true;
    sysmem_manager_ = std::make_unique<SystemMemoryManager>(this->id_, this->num_hw_cqs());
    hw_command_queues_.reserve(num_hw_cqs());
    sw_command_queues_.reserve(num_hw_cqs());
    for (size_t cq_id = 0; cq_id < num_hw_cqs(); cq_id++) {
        hw_command_queues_.push_back(std::make_unique<HWCommandQueue>(this, cq_id, dispatch_downstream_noc));
        sw_command_queues_.push_back(std::make_unique<CommandQueue>(this, cq_id));
    }
}

void Device::init_command_queue_device() {

    if (llrt::RunTimeOptions::get_instance().get_skip_loading_fw()) {
        detail::EnablePersistentKernelCache();
        this->compile_command_queue_programs();
        detail::DisablePersistentKernelCache();
    } else {
        this->compile_command_queue_programs();
    }

    TT_ASSERT(this->command_queue_programs_.size() == 1);
    this->configure_command_queue_programs();
    Program& command_queue_program = *this->command_queue_programs_[0];

    // TODO: should get a const ref
    std::vector<std::vector<CoreCoord>>logical_cores = command_queue_program.logical_cores();
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        const auto& logical_dispatch_cores = logical_cores[index];
        CoreType core_type = hal.get_core_type(index);
        for (const CoreCoord &logical_dispatch_core : logical_dispatch_cores) {
            launch_msg_t msg = command_queue_program.kernels_on_core(logical_dispatch_core, index)->launch_msg;
            go_msg_t go_msg = command_queue_program.kernels_on_core(logical_dispatch_core, index)->go_msg;
            CoreCoord virtual_core = this->virtual_core_from_logical_core(logical_dispatch_core, core_type);
            tt::llrt::write_launch_msg_to_core(this->id(), virtual_core, &msg, &go_msg, this->get_dev_addr(virtual_core, HalL1MemAddrType::LAUNCH));
        }
    }

    for (auto& hw_cq : this->hw_command_queues_) {
        hw_cq->set_num_worker_sems_on_dispatch(
            sub_device_manager_tracker_->get_active_sub_device_manager()->num_sub_devices());
        hw_cq->set_go_signal_noc_data_on_dispatch(
            sub_device_manager_tracker_->get_active_sub_device_manager()->noc_mcast_unicast_data());
    }
}

void Device::initialize_synchronous_sw_cmd_queue() {
    // Initialize a single Software Command Queue for SD, using passthrough mode.
    // This queue is used for all host bound functions using the Software CQ in SD mode.
    sw_command_queues_.reserve(num_hw_cqs());
    for (size_t cq_id = 0; cq_id < num_hw_cqs(); cq_id++) {
        sw_command_queues_.push_back(
            std::make_unique<CommandQueue>(this, cq_id, CommandQueue::CommandQueueMode::PASSTHROUGH));
    }
}

bool Device::initialize(const uint8_t num_hw_cqs, size_t l1_small_size, size_t trace_region_size, tt::stl::Span<const std::uint32_t> l1_bank_remap, bool minimal) {
    ZoneScoped;
    log_info(tt::LogMetal, "Initializing device {}. Program cache is {}enabled", this->id_, this->program_cache_.is_enabled() ? "": "NOT ");
    log_debug(tt::LogMetal, "Running with {} cqs ", num_hw_cqs);
    TT_FATAL(num_hw_cqs > 0 and num_hw_cqs <= dispatch_core_manager::MAX_NUM_HW_CQS, "num_hw_cqs can be between 1 and {}", dispatch_core_manager::MAX_NUM_HW_CQS);
    this->using_fast_dispatch_ = false;
    this->num_hw_cqs_ = num_hw_cqs;
    constexpr uint32_t harvesting_map_bits = 12;
    constexpr uint32_t num_hw_cq_bits = 8;
    constexpr uint32_t dispatch_core_axis_bits = 1;
    constexpr uint32_t dispatch_core_type_bits = 1;
    static_assert(dispatch_core_manager::MAX_NUM_HW_CQS <= (1 << num_hw_cq_bits));
    static_assert(static_cast<uint32_t>(DispatchCoreAxis::COUNT) <= (1 << dispatch_core_axis_bits));
    static_assert(static_cast<uint32_t>(DispatchCoreType::COUNT) <= (1 << dispatch_core_type_bits));
    static_assert(harvesting_map_bits + num_hw_cq_bits + dispatch_core_axis_bits + dispatch_core_type_bits <= sizeof(this->build_key_) * CHAR_BIT);

    // num_hw_cqs, dispatch_core_axis, dispatch_core_type all change the number of banks, so need to be part of the
    // build key since we have defines based on number of banks.
    const auto& dispatch_core_config = dispatch_core_manager::instance().get_dispatch_core_config(this->id_);
    this->build_key_ = (static_cast<uint32_t>(dispatch_core_config.get_dispatch_core_type()) << (harvesting_map_bits + num_hw_cq_bits + dispatch_core_axis_bits)) |
                       (static_cast<uint32_t>(dispatch_core_config.get_dispatch_core_axis()) << (harvesting_map_bits + num_hw_cq_bits)) |
                       (static_cast<uint32_t>(num_hw_cqs_) << harvesting_map_bits);
    if (not hal.is_coordinate_virtualization_enabled()) {
        // Coordinate virtualization is not enabled. For a single program, its associated binaries will vary across devices with different cores harvested.
        this->build_key_ = (this->build_key_) | tt::Cluster::instance().get_harvesting_mask(this->id());
    } else {
        // Coordinate Virtualization is enabled. Track only the number of harvested cores, instead of the exact harvesting configuration (this is not needed).
        this->build_key_ = (this->build_key_) | (std::bitset<harvesting_map_bits>(tt::Cluster::instance().get_harvesting_mask(this->id())).count());
    }
    this->initialize_cluster();
    this->initialize_default_sub_device_state(l1_small_size, trace_region_size, l1_bank_remap);
    this->initialize_build();
    this->generate_device_bank_to_noc_tables();

    // For minimal setup, don't initialize FW, watcher, dprint. They won't work if we're attaching to a hung chip.
    if (minimal)
        return true;

    // Mark initialized before compiling and sending dispatch kernels to device because compilation expects device to be initialized
    this->work_executor_.initialize();
    this->initialized_ = true;

    return true;
}

void Device::push_work(std::function<void()> work, bool blocking) {
    if (not this->initialized_) {
        log_warning("Attempting to push work to Device {} which is not initialized. Ignoring...", this->id_);
        return;
    }
    this->work_executor_.push_work(std::move(work), blocking);
}

bool Device::close() {
    log_info(tt::LogMetal, "Closing device {}", this->id_);
    if (not this->initialized_) {
        TT_THROW("Cannot close device {} that has not been initialized!", this->id_);
    }

    for (const std::unique_ptr<HWCommandQueue> &hw_command_queue : hw_command_queues_) {
        if (hw_command_queue->sysmem_manager().get_bypass_mode()) {
            hw_command_queue->record_end();
        }
        hw_command_queue->terminate();
    }

    std::cout << "terminated the HW CQs" << std::endl;
    this->work_executor_.reset();
    tt_metal::detail::DumpDeviceProfileResults(this, ProfilerDumpState::LAST_CLOSE_DEVICE);

    sub_device_manager_tracker_.reset(nullptr);

    std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>> not_done_dispatch_cores;
    std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>> cores_to_skip;
    this->get_associated_dispatch_virtual_cores(not_done_dispatch_cores, cores_to_skip);

    auto mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->id_);
    std::unordered_set<CoreCoord> wait_for_cores = not_done_dispatch_cores[mmio_device_id];

    std::cout << "About to wait for cores to be done, waiting on: " << std::endl;
    for (const auto& core : wait_for_cores) {
        std::cout << core.str() << "\t";
    }
    std::cout << "\n";
    llrt::internal_::wait_until_cores_done(mmio_device_id, RUN_MSG_GO, wait_for_cores);

    DprintServerDetach(this);
    watcher_detach(this);

    // Assert worker cores
    CoreCoord grid_size = this->logical_grid_size();
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core = this->worker_core_from_logical_core(logical_core);

            if (cores_to_skip[mmio_device_id].find(worker_core) == cores_to_skip[mmio_device_id].end()) {
                if (this->storage_only_cores_.find(logical_core) == this->storage_only_cores_.end()) {
                    tt::Cluster::instance().assert_risc_reset_at_core(tt_cxy_pair(this->id(), worker_core));
                }
            } else {
                log_debug(tt::LogMetal, "{} will not be Reset when closing Device {}", worker_core.str(), this->id());
            }
        }
    }

    std::cout << "asserted worker cores" << std::endl;

    if (this->arch() == ARCH::BLACKHOLE) {
        for (const auto& eth_core : this->get_active_ethernet_cores()) {
            CoreCoord phys_eth_core = this->ethernet_core_from_logical_core(eth_core);
            TensixSoftResetOptions reset_val =
                TENSIX_ASSERT_SOFT_RESET &
                static_cast<TensixSoftResetOptions>(
                    ~std::underlying_type<TensixSoftResetOptions>::type(TensixSoftResetOptions::BRISC));
            tt::Cluster::instance().assert_risc_reset_at_core(tt_cxy_pair(this->id(), phys_eth_core), reset_val);
        }
    }

    std::cout << "asserted active eth cores" << std::endl;

    if (this->id_ != mmio_device_id) {
        for (auto it = not_done_dispatch_cores[mmio_device_id].begin(); it != not_done_dispatch_cores[mmio_device_id].end(); it++) {
            const auto &virtual_core = *it;
            if(tt::Cluster::instance().is_ethernet_core(virtual_core, this->id_)) {
                log_debug(tt::LogMetal, "Ethernet dispatch core {} on Device {} is idle. Closing Device {}", virtual_core.str(), mmio_device_id, this->id());
            } else {
                log_debug(tt::LogMetal, "Resetting core {} on Device {} when closing Device {}", virtual_core.str(), mmio_device_id, this->id());
                tt::Cluster::instance().assert_risc_reset_at_core(tt_cxy_pair(mmio_device_id, virtual_core));
            }
        }
    }

    tt::Cluster::instance().l1_barrier(id_);

    this->compute_cores_.clear();
    this->storage_only_cores_.clear();
    this->ethernet_cores_.clear();
    this->disable_and_clear_program_cache();
    this->command_queue_programs_.clear();
    this->sw_command_queues_.clear();
    this->hw_command_queues_.clear();
    this->sysmem_manager_.reset();
    this->initialized_ = false;

    return true;
}

Device::~Device() {
    log_debug(tt::LogMetal, "Device {} destructor", this->id_);
    if (this->initialized_) {
        this->close();
    }
}

tt::ARCH Device::arch() const {
    return tt::Cluster::instance().arch();
}

int Device::num_dram_channels() const {
    return tt::Cluster::instance().get_soc_desc(id_).get_num_dram_channels();
}

uint32_t Device::l1_size_per_core() const {
    return tt::Cluster::instance().get_soc_desc(id_).worker_l1_size;
}
uint32_t Device::dram_size_per_channel() const {
    return tt::Cluster::instance().get_soc_desc(id_).dram_bank_size;
}

CoreCoord Device::grid_size() const {
    return tt::Cluster::instance().get_soc_desc(id_).grid_size;
}

CoreCoord Device::logical_grid_size() const {
    return tt::Cluster::instance().get_soc_desc(id_).worker_grid_size;
}

CoreCoord Device::dram_grid_size() const {
    return tt::Cluster::instance().get_soc_desc(id_).get_dram_grid_size();
}

CoreCoord Device::compute_with_storage_grid_size() const {
    const auto &dispatch_core_config = dispatch_core_manager::instance().get_dispatch_core_config(id_);
    return tt::get_compute_grid_size(id_, num_hw_cqs_, dispatch_core_config);
}

CoreType Device::core_type_from_physical_core(const CoreCoord &physical_coord) const {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    if (soc_desc.physical_cores.find(physical_coord) == soc_desc.physical_cores.end())
        TT_THROW("Physical core {} doesn't exist in metal_SocDescriptor.", physical_coord);

    return soc_desc.physical_cores.at(physical_coord).type;
}

CoreType Device::core_type_from_virtual_core(const CoreCoord &virtual_coord) const {
    if (tt::Cluster::instance().is_worker_core(virtual_coord, this->id_)) {
        return CoreType::WORKER;
    } else if (tt::Cluster::instance().is_ethernet_core(virtual_coord, this->id_)) {
        return CoreType::ETH;
    }
    return this->core_type_from_physical_core(virtual_coord);
}

CoreCoord Device::virtual_noc0_coordinate(uint8_t noc_index, CoreCoord coord) const {
    if (coord.x >= this->grid_size().x || coord.y >= this->grid_size().y) {
        // Coordinate already in virtual space: NOC0 and NOC1 are the same
        return coord;
    } else {
        const auto& grid_size = this->grid_size();
        // Coordinate in Physical NOC0 Space. Convert to Virtual.
        coord = this->virtual_core_from_physical_core(coord, this->core_type_from_physical_core(coord));
        // Derive virtual coord in noc_index space.
        CoreCoord virtual_coord = {
            hal.noc_coordinate(noc_index, grid_size.x, coord.x),
            hal.noc_coordinate(noc_index, grid_size.y, coord.y)
        };
        return virtual_coord;
    }
}

CoreCoord Device::virtual_noc_coordinate(uint8_t noc_index, CoreCoord coord) const {
     if (coord.x >= this->grid_size().x || coord.y >= this->grid_size().y) {
        // Coordinate already in virtual space: NOC0 and NOC1 are the same
        return coord;
    } else {
        const auto& grid_size = this->grid_size();
        // Coordinate passed in can be NOC0 or NOC1. The noc_index corresponds to
        // the system this coordinate belongs to.
        // Use this to convert to NOC0 coordinates and then derive Virtual Coords from it.
        CoreCoord physical_coord = {
            hal.noc_coordinate(noc_index, grid_size.x, coord.x),
            hal.noc_coordinate(noc_index, grid_size.y, coord.y)
        };
        return this->virtual_core_from_physical_core(physical_coord, this->core_type_from_physical_core(physical_coord));
    }
}

CoreCoord Device::physical_worker_core_from_logical_core(const CoreCoord &logical_core) const {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    return soc_desc.get_physical_tensix_core_from_logical(logical_core);
}

std::vector<CoreCoord> Device::worker_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const {
    std::vector<CoreCoord> worker_cores(logical_cores.size());
    for (std::size_t idx = 0; idx < logical_cores.size(); idx++)
        worker_cores[idx] = this->worker_core_from_logical_core(logical_cores[idx]);

    return worker_cores;
}

std::vector<CoreCoord> Device::ethernet_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const {
    std::vector<CoreCoord> eth_cores(logical_cores.size());
    for (std::size_t idx = 0; idx < logical_cores.size(); idx++) {
        eth_cores[idx] = this->ethernet_core_from_logical_core(logical_cores[idx]);
    }
    return eth_cores;
}

CoreCoord Device::virtual_core_from_logical_core(const CoreCoord &logical_coord, const CoreType& core_type) const {
    return tt::Cluster::instance().get_virtual_coordinate_from_logical_coordinates(this->id_, logical_coord, core_type);
}

CoreCoord Device::virtual_core_from_physical_core(const CoreCoord &physical_coord, const CoreType& core_type) const {
    return tt::Cluster::instance().get_virtual_coordinate_from_physical_coordinates(this->id_, physical_coord, core_type);
}

CoreCoord Device::worker_core_from_logical_core(const CoreCoord &logical_core) const {
    return this->virtual_core_from_logical_core(logical_core, CoreType::WORKER);
}

CoreCoord Device::ethernet_core_from_logical_core(const CoreCoord &logical_core) const {
    return this->virtual_core_from_logical_core(logical_core, CoreType::ETH);
}

CoreCoord Device::logical_core_from_ethernet_core(const CoreCoord &ethernet_core) const {
    return tt::Cluster::instance().get_logical_ethernet_core_from_virtual(this->id(), ethernet_core);
}

uint32_t Device::get_noc_unicast_encoding(uint8_t noc_index, const CoreCoord& core) const {
    auto virtual_noc_coord = this->virtual_noc0_coordinate(noc_index, core);
    return tt::tt_metal::hal.noc_xy_encoding(
        virtual_noc_coord.x,
        virtual_noc_coord.y
    );
}

uint32_t Device::get_noc_multicast_encoding(uint8_t noc_index, const CoreRange& cores) const {
    auto virtual_noc_start = this->virtual_noc0_coordinate(noc_index, cores.start_coord);
    auto virtual_noc_end = this->virtual_noc0_coordinate(noc_index, cores.end_coord);

    // NOC 1 mcasts from bottom left to top right, so we need to reverse the coords
    if (noc_index == 0) {
        return tt::tt_metal::hal.noc_multicast_encoding(
            virtual_noc_start.x,
            virtual_noc_start.y,
            virtual_noc_end.x,
            virtual_noc_end.y
        );
    } else {
        return tt::tt_metal::hal.noc_multicast_encoding(
            virtual_noc_end.x,
            virtual_noc_end.y,
            virtual_noc_start.x,
            virtual_noc_start.y
        );
    }
}

const std::unique_ptr<Allocator> &Device::get_initialized_allocator() const {
    return sub_device_manager_tracker_->get_default_sub_device_manager()->get_initialized_allocator(SubDeviceId{0});
}

const std::unique_ptr<Allocator> &Device::get_initialized_allocator(SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->get_initialized_allocator(sub_device_id);
}

uint32_t Device::num_sub_devices() const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->num_sub_devices();
}

uint32_t Device::num_banks(const BufferType &buffer_type) const {
    const auto& allocator = this->get_initialized_allocator();
    return allocator::num_banks(*allocator, buffer_type);
}

uint32_t Device::num_banks(const BufferType &buffer_type, SubDeviceId sub_device_id) const {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    return allocator::num_banks(*allocator, buffer_type);
}

uint32_t Device::bank_size(const BufferType &buffer_type) const {
    const auto& allocator = this->get_initialized_allocator();
    return allocator::bank_size(*allocator, buffer_type);
}

uint32_t Device::bank_size(const BufferType &buffer_type, SubDeviceId sub_device_id) const {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    return allocator::bank_size(*allocator, buffer_type);
}

uint32_t Device::dram_channel_from_bank_id(uint32_t bank_id) const {
    const auto& allocator = this->get_initialized_allocator();
    return allocator::dram_channel_from_bank_id(*allocator, bank_id);
}

uint32_t Device::dram_channel_from_bank_id(uint32_t bank_id, SubDeviceId sub_device_id) const {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    return allocator::dram_channel_from_bank_id(*allocator, bank_id);
}

CoreCoord Device::dram_core_from_dram_channel(uint32_t dram_channel) const {
    return tt::Cluster::instance().get_soc_desc(id_).get_preferred_worker_core_for_dram_channel(dram_channel);
}

CoreCoord Device::logical_core_from_dram_channel(uint32_t dram_channel) const {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    return tt::Cluster::instance().get_soc_desc(id_).get_logical_core_for_dram_channel(dram_channel);
}

uint32_t Device::dram_channel_from_logical_core(const CoreCoord& logical_core) const {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    return tt::Cluster::instance().get_soc_desc(id_).get_dram_channel_from_logical_core(logical_core);
}

int32_t Device::bank_offset(BufferType buffer_type, uint32_t bank_id) const {
    const auto& allocator = this->get_initialized_allocator();
    return allocator::bank_offset(*allocator, buffer_type, bank_id);
}

int32_t Device::bank_offset(BufferType buffer_type, uint32_t bank_id, SubDeviceId sub_device_id) const {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    return allocator::bank_offset(*allocator, buffer_type, bank_id);
}

CoreCoord Device::logical_core_from_bank_id(uint32_t bank_id) const {
    const auto& allocator = this->get_initialized_allocator();
    return allocator::logical_core_from_bank_id(*allocator, bank_id);
}

CoreCoord Device::logical_core_from_bank_id(uint32_t bank_id, SubDeviceId sub_device_id) const {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    return allocator::logical_core_from_bank_id(*allocator, bank_id);
}

const std::vector<uint32_t> &Device::bank_ids_from_dram_channel(uint32_t dram_channel) const {
    const auto& allocator = this->get_initialized_allocator();
    return allocator::bank_ids_from_dram_channel(*allocator, dram_channel);
}

const std::vector<uint32_t> &Device::bank_ids_from_dram_channel(uint32_t dram_channel, SubDeviceId sub_device_id) const {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    return allocator::bank_ids_from_dram_channel(*allocator, dram_channel);
}

const std::vector<uint32_t> &Device::bank_ids_from_logical_core(
    BufferType buffer_type, const CoreCoord &logical_core) const {
    const auto& allocator = this->get_initialized_allocator();
    return allocator::bank_ids_from_logical_core(*allocator, buffer_type, logical_core);
}

const std::vector<uint32_t> &Device::bank_ids_from_logical_core(
    BufferType buffer_type, const CoreCoord &logical_core, SubDeviceId sub_device_id) const {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    return allocator::bank_ids_from_logical_core(*allocator, buffer_type, logical_core);
}

allocator::Statistics Device::get_memory_allocation_statistics(const BufferType &buffer_type) const {
    const auto& allocator = this->get_initialized_allocator();
    return allocator::get_statistics(*allocator, buffer_type);
}

allocator::Statistics Device::get_memory_allocation_statistics(const BufferType &buffer_type, SubDeviceId sub_device_id) const {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    return allocator::get_statistics(*allocator, buffer_type);
}

uint32_t Device::get_allocator_alignment(const BufferType &buffer_type) const {
    const auto& allocator = this->get_initialized_allocator();
    return allocator::get_alignment(*allocator, buffer_type);
}

uint32_t Device::get_allocator_alignment(const BufferType &buffer_type, SubDeviceId sub_device_id) const {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    return allocator::get_alignment(*allocator, buffer_type);
}

size_t Device::get_l1_small_size() const {
    const auto& allocator = this->get_initialized_allocator();
    return allocator->config.l1_small_size;
}

size_t Device::get_l1_small_size(SubDeviceId sub_device_id) const {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    return allocator->config.l1_small_size;
}

void Device::dump_memory_blocks(const BufferType &buffer_type, std::ofstream &out) const {
    const auto& allocator = this->get_initialized_allocator();
    return allocator::dump_memory_blocks(*allocator, buffer_type, out);
}

void Device::dump_memory_blocks(const BufferType &buffer_type, std::ofstream &out, SubDeviceId sub_device_id) const {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    return allocator::dump_memory_blocks(*allocator, buffer_type, out);
}

MemoryBlockTable Device::get_memory_block_table(const BufferType& buffer_type) const {
    const auto& allocator = this->get_initialized_allocator();
    return allocator::get_memory_block_table(*allocator, buffer_type);
}

const std::unordered_set<Buffer *> &Device::get_allocated_buffers() const {
    const auto& allocator = this->get_initialized_allocator();
    return allocator::get_allocated_buffers(*allocator);
}

const std::unordered_set<Buffer *> &Device::get_allocated_buffers(SubDeviceId sub_device_id) const {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    return allocator::get_allocated_buffers(*allocator);
}

void Device::deallocate_buffers() {
    const auto& allocator = this->get_initialized_allocator();
    allocator::deallocate_buffers(*allocator);
}

void Device::deallocate_buffers(SubDeviceId sub_device_id) {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    allocator::deallocate_buffers(*allocator);
}

std::optional<DeviceAddr> Device::lowest_occupied_compute_l1_address() const {
    return sub_device_manager_tracker_->lowest_occupied_compute_l1_address();
}

std::optional<DeviceAddr> Device::lowest_occupied_compute_l1_address(tt::stl::Span<const SubDeviceId> sub_device_ids) const {
    return sub_device_manager_tracker_->lowest_occupied_compute_l1_address(sub_device_ids);
}

std::pair<int, int> Device::build_processor_type_to_index(uint32_t programmable_core, uint32_t processor_class) const {
    TT_ASSERT(programmable_core < this->build_state_indices_.size(),
        "Programmable core type {} is not included in the FW or Kernel build state", programmable_core);
    TT_ASSERT(processor_class < this->build_state_indices_[programmable_core].size(),
        "Processor class type {} is not included in the FW or Kernel build state", processor_class);
    return this->build_state_indices_[programmable_core][processor_class];
}

// Ideally the firmware getter would be private to the device, however, tests look for this
const JitBuildState& Device::build_firmware_state(uint32_t programmable_core, uint32_t processor_class, int i) const {
    return *(this->firmware_build_states_[build_processor_type_to_index(programmable_core, processor_class).first + i]);
}

const JitBuildState& Device::build_kernel_state(uint32_t programmable_core, uint32_t processor_class, int i) const {
    return *(this->kernel_build_states_[build_processor_type_to_index(programmable_core, processor_class).first + i]);
}

const JitBuildStateSubset Device::build_kernel_states(uint32_t programmable_core, uint32_t processor_class) const {
    std::pair<int, int> bptti = build_processor_type_to_index(programmable_core, processor_class);
    JitBuildStateSubset subset = {
        &this->kernel_build_states_[bptti.first],
        bptti.second
    };
    return subset;
}

const string Device::build_firmware_target_path(uint32_t programmable_core, uint32_t processor_class, int i) const {
    const JitBuildState& bs = build_firmware_state(programmable_core, processor_class, i);
    return bs.get_target_out_path("");
}

const string Device::build_kernel_target_path(uint32_t programmable_core, uint32_t processor_class, int i, const string& kernel_name) const {
    const JitBuildState& bs = build_kernel_state(programmable_core, processor_class, i);
    return bs.get_target_out_path(kernel_name);
}

HWCommandQueue& Device::hw_command_queue(size_t cq_id) {
    detail::DispatchStateCheck(true);
    TT_FATAL( cq_id < hw_command_queues_.size(), "cq_id {} is out of range", cq_id );
    TT_FATAL(this->is_initialized(), "Device has not been initialized, did you forget to call InitializeDevice?");
    return *hw_command_queues_[cq_id];
}

CommandQueue &Device::command_queue(size_t cq_id) {
    detail::DispatchStateCheck(using_fast_dispatch_);
    TT_FATAL( cq_id < sw_command_queues_.size(), "cq_id {} is out of range", cq_id );
    TT_FATAL(this->is_initialized(), "Device has not been initialized, did you forget to call InitializeDevice?");
    return *sw_command_queues_[cq_id];
}

bool Device::can_use_passthrough_scheduling() const {
    return this->work_executor_.use_passthrough();
}

void Device::synchronize() {
    if (not this->initialized_) {
        log_warning("Attempting to synchronize Device {} which is not initialized. Ignoring...", this->id_);
        return;
    }
    this->work_executor_.synchronize();
}

void Device::set_worker_mode(const WorkExecutorMode& mode) {
    this->work_executor_.set_worker_mode(mode);
}

void Device::enable_async(bool enable) {
    auto mode = enable ? WorkExecutorMode::ASYNCHRONOUS : WorkExecutorMode::SYNCHRONOUS;
    this->set_worker_mode(mode);
    // If a worker thread is spawned for a device, register/track it in a runtime structure.
    // If a worker thread is destroyed, remove it from the structure.
    // This is required for checking if a call is made from an application thread or a worker thread.
    // See InWorkerThread().
    if (enable) {
        tt::DevicePool::instance().register_worker_thread_for_device(this, this->work_executor_.get_worker_thread_id());
    } else {
        tt::DevicePool::instance().unregister_worker_thread_for_device(this);
    }
}

bool Device::using_slow_dispatch() const {
    return !using_fast_dispatch();
}

bool Device::using_fast_dispatch() const {
    return using_fast_dispatch_;
}

void Device::begin_trace(const uint8_t cq_id, const uint32_t tid) {
    ZoneScoped;
    TracyTTMetalBeginTrace(this->id(), tid);
    TT_FATAL(!this->hw_command_queues_[cq_id]->get_tid().has_value(), "CQ {} is already being used for tracing tid {}", (uint32_t)cq_id, tid);
    this->mark_allocations_safe();
    // Create an empty trace buffer here. This will get initialized in end_trace
    auto* active_sub_device_manager = sub_device_manager_tracker_->get_active_sub_device_manager();
    TT_FATAL(
        active_sub_device_manager->get_trace(tid) == nullptr,
        "Trace already exists for tid {} on device {}'s active sub-device manager {}",
        tid,
        this->id_,
        active_sub_device_manager->id());
    auto& trace_buffer = active_sub_device_manager->create_trace(tid);
    this->hw_command_queues_[cq_id]->record_begin(tid, trace_buffer->desc);
}

void Device::end_trace(const uint8_t cq_id, const uint32_t tid) {
    ZoneScoped;
    TracyTTMetalEndTrace(this->id(), tid);
    TT_FATAL(this->hw_command_queues_[cq_id]->get_tid() == tid, "CQ {} is not being used for tracing tid {}", (uint32_t)cq_id, tid);
    auto* active_sub_device_manager = sub_device_manager_tracker_->get_active_sub_device_manager();
    auto trace_buffer = active_sub_device_manager->get_trace(tid);
    TT_FATAL(
        trace_buffer != nullptr,
        "Trace instance {} must exist on device {}'s active sub-device manager {}",
        tid,
        this->id_,
        active_sub_device_manager->id());
    this->hw_command_queues_[cq_id]->record_end();
    Trace::initialize_buffer(this->command_queue(cq_id), trace_buffer);
    this->mark_allocations_unsafe();
}

void Device::replay_trace(const uint8_t cq_id, const uint32_t tid, const bool blocking) {
    ZoneScoped;
    TracyTTMetalReplayTrace(this->id(), tid);
    constexpr bool check = false;
    auto* active_sub_device_manager = sub_device_manager_tracker_->get_active_sub_device_manager();
    const auto& trace_buffer = active_sub_device_manager->get_trace(tid);
    TT_FATAL(
        trace_buffer != nullptr,
        "Trace instance {} must exist on device {}'s active sub-device manager {}",
        tid,
        this->id_,
        active_sub_device_manager->id());
    if constexpr (check) {
        Trace::validate_instance(*trace_buffer);
    }
    EnqueueTrace(this->command_queue(cq_id), tid, blocking);
}

void Device::release_trace(const uint32_t tid) {
    ZoneScoped;
    TracyTTMetalReleaseTrace(this->id(), tid);

    sub_device_manager_tracker_->get_active_sub_device_manager()->release_trace(tid);

    // Only enable allocations once all captured traces are released
    if (this->trace_buffers_size_ == 0) {
        this->mark_allocations_safe();
    }
}

std::shared_ptr<TraceBuffer> Device::get_trace(uint32_t tid) {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->get_trace(tid);
}

void Device::enable_program_cache() {
    log_info(tt::LogMetal, "Enabling program cache on device {}", this->id_);
    this->synchronize();
    program_cache_.enable();
}
void Device::disable_and_clear_program_cache() {
    log_info(tt::LogMetal, "Disabling and clearing program cache on device {}", this->id_);
    this->synchronize();
    if (this->program_cache_.is_enabled()) {
        program_cache_.disable();
    }
    program_cache_.clear();
}
std::size_t Device::num_program_cache_entries() {
    this->synchronize();
    return program_cache_.num_entries();
}

void Device::mark_allocations_unsafe() {
    tt::tt_metal::allocator::mark_allocations_unsafe(*this->get_initialized_allocator());
}

void Device::mark_allocations_safe() {
    tt::tt_metal::allocator::mark_allocations_safe(*this->get_initialized_allocator());
}

void Device::generate_device_bank_to_noc_tables()
{
    const size_t num_dram_banks = this->num_banks(BufferType::DRAM);
    std::vector<CoreCoord> dram_noc_coord_per_bank(num_dram_banks);
    dram_bank_offset_map_.clear();
    dram_bank_offset_map_.resize(num_dram_banks);
    for (unsigned bank_id = 0; bank_id < num_dram_banks; bank_id++) {
        dram_noc_coord_per_bank[bank_id] = this->dram_core_from_dram_channel(this->dram_channel_from_bank_id(bank_id));
        dram_bank_offset_map_[bank_id] = this->bank_offset(BufferType::DRAM, bank_id);
    }
    const size_t num_l1_banks = this->num_banks(BufferType::L1);
    std::vector<CoreCoord> l1_noc_coord_per_bank(num_l1_banks);
    l1_bank_offset_map_.clear();
    l1_bank_offset_map_.resize(num_l1_banks);
    for (unsigned bank_id = 0; bank_id < num_l1_banks; bank_id++) {
        l1_noc_coord_per_bank[bank_id] = this->worker_core_from_logical_core(this->logical_core_from_bank_id(bank_id));
        l1_bank_offset_map_[bank_id] = this->bank_offset(BufferType::L1, bank_id);
    }

    const metal_SocDescriptor& soc_d = tt::Cluster::instance().get_soc_desc(this->id());

    dram_bank_to_noc_xy_.clear();
    dram_bank_to_noc_xy_.reserve(tt::tt_metal::hal.get_num_nocs() * dram_noc_coord_per_bank.size());
    for (unsigned int noc = 0; noc < tt::tt_metal::hal.get_num_nocs(); noc++) {
        for (unsigned int bank_id = 0; bank_id < dram_noc_coord_per_bank.size(); bank_id++) {
            uint16_t noc_x = tt::tt_metal::hal.noc_coordinate(noc, soc_d.grid_size.x, dram_noc_coord_per_bank[bank_id].x);
            uint16_t noc_y = tt::tt_metal::hal.noc_coordinate(noc, soc_d.grid_size.y, dram_noc_coord_per_bank[bank_id].y);
            uint16_t xy = ((noc_y << tt::tt_metal::hal.get_noc_addr_node_id_bits()) | noc_x)
                          << tt::tt_metal::hal.get_noc_coord_reg_offset();
            dram_bank_to_noc_xy_.push_back(xy);
        }
    }

    l1_bank_to_noc_xy_.clear();
    l1_bank_to_noc_xy_.reserve(tt::tt_metal::hal.get_num_nocs() * l1_noc_coord_per_bank.size());
    for (unsigned int noc = 0; noc < tt::tt_metal::hal.get_num_nocs(); noc++) {
        for (unsigned int bank_id = 0; bank_id < l1_noc_coord_per_bank.size(); bank_id++) {
            auto l1_noc_coords = this->virtual_noc0_coordinate(noc, l1_noc_coord_per_bank[bank_id]);
            uint16_t noc_x = l1_noc_coords.x;
            uint16_t noc_y = l1_noc_coords.y;
            uint16_t xy = ((noc_y << tt::tt_metal::hal.get_noc_addr_node_id_bits()) | noc_x)
                          << tt::tt_metal::hal.get_noc_coord_reg_offset();
            l1_bank_to_noc_xy_.push_back(xy);
        }
    }
}

size_t Device::get_device_kernel_defines_hash() {
    return tt::utils::DefinesHash{}(this->device_kernel_defines_);
}

uint8_t Device::num_noc_mcast_txns(SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->num_noc_mcast_txns(sub_device_id);
}

uint8_t Device::num_noc_unicast_txns(SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->num_noc_unicast_txns(sub_device_id);
}

uint8_t Device::noc_data_start_index(SubDeviceId sub_device_id, bool mcast_data, bool unicast_data) const {
    if (mcast_data) {
        return sub_device_manager_tracker_->get_active_sub_device_manager()->noc_mcast_data_start_index(sub_device_id);
    } else if (unicast_data) {
        return sub_device_manager_tracker_->get_active_sub_device_manager()->noc_unicast_data_start_index(
            sub_device_id);
    } else {
        return 0;
    }
}

CoreCoord Device::virtual_program_dispatch_core(uint8_t cq_id) const {
    return this->hw_command_queues_[cq_id]->virtual_enqueue_program_dispatch_core;
}

// Main source to get NOC idx for dispatch core
NOC Device::dispatch_go_signal_noc() const {
    return this->dispatch_s_enabled() ? NOC::NOC_1 : NOC::NOC_0;
}

SubDeviceManagerId Device::get_active_sub_device_manager_id() const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->id();
}

SubDeviceManagerId Device::get_default_sub_device_manager_id() const {
    return sub_device_manager_tracker_->get_default_sub_device_manager()->id();
}

SubDeviceManagerId Device::create_sub_device_manager(tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) {
    return sub_device_manager_tracker_->create_sub_device_manager(sub_devices, local_l1_size);
}

std::tuple<SubDeviceManagerId, SubDeviceId> Device::create_sub_device_manager_with_fabric(tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) {
    return sub_device_manager_tracker_->create_sub_device_manager_with_fabric(sub_devices, local_l1_size);
}

void Device::load_sub_device_manager(SubDeviceManagerId sub_device_manager_id) {
    sub_device_manager_tracker_->load_sub_device_manager(sub_device_manager_id);
}

void Device::clear_loaded_sub_device_manager() { sub_device_manager_tracker_->clear_loaded_sub_device_manager(); }

void Device::remove_sub_device_manager(SubDeviceManagerId sub_device_manager_id) {
    sub_device_manager_tracker_->remove_sub_device_manager(sub_device_manager_id);
}

const std::vector<SubDeviceId> &Device::get_sub_device_ids() const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->get_sub_device_ids();
}

const std::vector<SubDeviceId> &Device::get_sub_device_stall_group() const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->get_sub_device_stall_group();
}

void Device::set_sub_device_stall_group(tt::stl::Span<const SubDeviceId> sub_device_ids) {
    sub_device_manager_tracker_->get_active_sub_device_manager()->set_sub_device_stall_group(sub_device_ids);
}

void Device::reset_sub_device_stall_group() {
    sub_device_manager_tracker_->get_active_sub_device_manager()->reset_sub_device_stall_group();
}

DeviceAddr Device::get_base_allocator_addr(const HalMemType &mem_type) const {
    const auto& allocator = this->get_initialized_allocator();
    return allocator::get_unreserved_base_address(*allocator, mem_type);
}

DeviceAddr Device::get_base_allocator_addr(const HalMemType &mem_type, SubDeviceId sub_device_id) const {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    return allocator::get_unreserved_base_address(*allocator, mem_type);
}

std::vector<CoreCoord> Device::get_optimal_dram_bank_to_logical_worker_assignment() {
    // Top level function that users (ex: Op Writers) can use to assign Tensix Worker cores
    // as DRAM readers or writers. Returns logical coordinates of optimally placed workers.
    // This function queries Physical Coordinates (only exposed directly to the Device class)
    // and passes them to logic in core_assignment.cpp to derive the most optimal core placement
    // based on architecture specific logic and Physical Grid configuration.
    if (not this->optimal_dram_bank_to_logical_worker_assignment_.size()) {
        uint32_t full_grid_size_x = this->grid_size().x;
        uint32_t full_grid_size_y = this->grid_size().y;

        auto compute_with_storage_grid_size = this->compute_with_storage_grid_size();
        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        // Get physical coordinates of DRAM Controller NOC end-points
        uint32_t num_dram_banks = this->num_dram_channels();
        std::vector<CoreCoord> dram_phy_coords;
        for (int i = 0; i < num_dram_banks; ++i) {
            dram_phy_coords.push_back(dram_core_from_dram_channel(i));
        }
        // Get all logical cores in the worker grid
        std::vector<CoreCoord> all_worker_cores_logical;
        for (int i = 0; i < num_cores_x; ++i) {
            for (int j = 0; j < num_cores_y; ++j) {
                all_worker_cores_logical.push_back(CoreCoord(i, j));
            }
        }
        // Get the physical rows and cols  (y, x) in the worker grid
        std::vector<uint32_t> worker_phy_y = std::vector<uint32_t>(num_cores_y);
        for (int i = 0; i < num_cores_y; ++i) {
            auto core_phy = this->physical_worker_core_from_logical_core(CoreCoord(0, i));
            worker_phy_y.at(i) = core_phy.y;
        }
        std::vector<uint32_t> worker_phy_x = std::vector<uint32_t>(num_cores_x);
        for (int i = 0; i < num_cores_x; ++i) {
            auto core_phy = this->physical_worker_core_from_logical_core(CoreCoord(i, 0));
            worker_phy_x.push_back(core_phy.x);
        }
        // Get optimal placement of worker cores interfacing with DRAM Controllers in physical coordinate space
        auto physical_worker_cores = get_optimal_dram_to_physical_worker_assignment(this->arch(), dram_phy_coords, full_grid_size_x, full_grid_size_y, worker_phy_x, worker_phy_y);
        // Convert to physical worker coordinates to logical. This gets returned to the user.
        for (int i = 0; i < physical_worker_cores.size(); ++i) {
            for (int j = 0; j < all_worker_cores_logical.size(); ++j) {
                auto core = this->physical_worker_core_from_logical_core(all_worker_cores_logical[j]);
                if (physical_worker_cores[i] == core) {
                    this->optimal_dram_bank_to_logical_worker_assignment_.push_back(all_worker_cores_logical[j]);
                }
            }
        }
    }
    return this->optimal_dram_bank_to_logical_worker_assignment_;
}

HalProgrammableCoreType Device::get_programmable_core_type(CoreCoord virtual_core) const {
    if (!tt::Cluster::instance().is_ethernet_core(virtual_core, this->id_)) {
        return HalProgrammableCoreType::TENSIX;
    }

    // Eth pcores have a different address, but only active ones.
    CoreCoord logical_core = this->logical_core_from_ethernet_core(virtual_core);
    if (this->is_active_ethernet_core(logical_core)) {
        return HalProgrammableCoreType::ACTIVE_ETH;
    }

    return HalProgrammableCoreType::IDLE_ETH;
}

// TODO: Find a better home for this function
// Extracts all the pairs of noc multicast encodings given a set of core ranges
std::vector<std::pair<transfer_info_cores, uint32_t>> Device::extract_dst_noc_multicast_info(const std::vector<CoreRange>& ranges, const CoreType core_type) {
    std::vector<std::pair<transfer_info_cores, uint32_t>> dst_noc_multicast_info;
    dst_noc_multicast_info.reserve(ranges.size());
    for (const CoreRange& core_range : ranges) {
        CoreCoord virtual_start = this->virtual_core_from_logical_core(core_range.start_coord, core_type);
        CoreCoord virtual_end = this->virtual_core_from_logical_core(core_range.end_coord, core_type);

        uint32_t num_receivers = core_range.size();
        dst_noc_multicast_info.push_back(std::make_pair(CoreRange(virtual_start, virtual_end), num_receivers));
    }
    return dst_noc_multicast_info;
}



size_t v1::GetNumAvailableDevices() { return tt::Cluster::instance().number_of_user_devices(); }

size_t v1::GetNumPCIeDevices() { return tt::Cluster::instance().number_of_pci_devices(); }

chip_id_t v1::GetPCIeDeviceID(chip_id_t device_id) {
    return tt::Cluster::instance().get_associated_mmio_device(device_id);
}

IDevice* v1::CreateDevice(chip_id_t device_id, CreateDeviceOptions options) {
    ZoneScoped;

    tt::DevicePool::initialize(
        {device_id},
        options.num_hw_cqs,
        options.l1_small_size,
        options.trace_region_size,
        options.dispatch_core_config,
        options.l1_bank_remap);

    return tt::DevicePool::instance().get_active_device(device_id);
}

bool v1::CloseDevice(IDevice* device) { return v0::CloseDevice(device); }

void v1::DeallocateBuffers(IDevice* device) { device->deallocate_buffers(); }

void v1::DumpDeviceProfileResults(IDevice* device) { detail::DumpDeviceProfileResults(device); }

ARCH v1::GetArch(IDevice* device) { return device->arch(); }

chip_id_t v1::GetId(IDevice* device) { return device->id(); }

int v1::GetNumDramChannels(IDevice* device) { return device->num_dram_channels(); }

std::uint32_t v1::GetL1SizePerCore(IDevice* device) { return device->l1_size_per_core(); }

CoreCoord v1::GetComputeWithStorageGridSize(IDevice* device) { return device->compute_with_storage_grid_size(); }

CoreCoord v1::GetDramGridSize(IDevice* device) { return device->dram_grid_size(); }

void v1::EnableProgramCache(IDevice* device) { device->enable_program_cache(); }

void v1::DisableAndClearProgramCache(IDevice* device) { device->disable_and_clear_program_cache(); }

void v1::PushWork(IDevice* device, std::function<void()> work, bool blocking) {
    device->push_work(std::move(work), blocking);
}

void v1::Synchronize(IDevice* device) { device->synchronize(); }

std::vector<CoreCoord> v1::GetEthernetSockets(IDevice* device, chip_id_t connected_chip_id) {
    return device->get_ethernet_sockets(connected_chip_id);
}

std::uint32_t v1::GetNumBanks(IDevice* device, BufferType buffer_type) { return device->num_banks(buffer_type); }

std::int32_t v1::GetBankOffset(IDevice* device, BufferType buffer_type, std::uint32_t bank_id) {
    return device->bank_offset(buffer_type, bank_id);
}

tt::stl::Span<const std::uint32_t> v1::BankIdsFromLogicalCore(
    IDevice* device, BufferType buffer_type, CoreCoord logical_core) {
    return device->bank_ids_from_logical_core(buffer_type, logical_core);
}

float v1::GetSfpuEps(IDevice* device) { return tt::tt_metal::experimental::hal::get_eps(); }

float v1::GetSfpuNan(IDevice* device) { return tt::tt_metal::experimental::hal::get_nan(); }

float v1::GetSfpuInf(IDevice* device) { return tt::tt_metal::experimental::hal::get_inf(); }

std::size_t v1::GetNumProgramCacheEntries(IDevice* device) { return device->num_program_cache_entries(); }

}  // namespace tt_metal

}  // namespace tt
