// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <circular_buffer.hpp>
#include <device.hpp>
#include <kernel.hpp>
#include <program_impl.hpp>
#include <worker_config_buffer.hpp>

namespace tt {

namespace tt_metal {

namespace program_dispatch {

struct ProgramDispatchMetadata {
    std::vector<ConfigBufferEntry> kernel_config_addrs;
    uint32_t sync_count;
    uint32_t stall_first;
    uint32_t stall_before_program;
};

uint32_t configure_rta_offsets_for_kernel_groups(
    uint32_t programmable_core_type_index,
    std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& kernels,
    std::vector<std::shared_ptr<KernelGroup>>& kernel_groups,
    uint32_t base_offset);

uint32_t configure_crta_offsets_for_kernel_groups(
    uint32_t programmable_core_type_index,
    std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& kernels,
    std::vector<std::shared_ptr<KernelGroup>>& kernel_groups,
    uint32_t crta_base_offset,
    std::array<uint32_t, DISPATCH_CLASS_MAX>& crta_offsets,
    std::array<uint32_t, DISPATCH_CLASS_MAX>& crta_sizes);

// Compute relative offsets (wrt the start of the kernel config ring buffer) and sizes of all
// program data structures in L1. Will be used when assembling dispatch commands for this program
template <typename T>
void finalize_program_offsets(T& workload_type, IDevice* device);

uint32_t finalize_rt_args(
    std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& kernels,
    std::vector<std::shared_ptr<KernelGroup>>& kernel_groups,
    uint32_t base_offset,
    uint32_t programmable_core_type_index,
    uint32_t& rta_offset,
    std::array<uint32_t, DISPATCH_CLASS_MAX>& crta_offsets,
    std::array<uint32_t, DISPATCH_CLASS_MAX>& crta_sizes);

uint32_t finalize_sems(
    uint32_t programmable_core_type_index,
    uint32_t sem_base_offset,
    const std::vector<Semaphore>& semaphores,
    uint32_t& semaphore_offset,
    uint32_t& semaphore_size);

uint32_t finalize_cbs(
    uint32_t programmable_core_type_index,
    std::vector<std::shared_ptr<KernelGroup>>& kernel_groups,
    uint32_t base_offset,
    uint32_t& cb_offset,
    uint32_t& cb_size,
    uint32_t& local_cb_size);

uint32_t finalize_kernel_bins(
    IDevice* device,
    uint32_t programmable_core_type_index,
    const std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& kernels,
    std::vector<std::shared_ptr<KernelGroup>>& kernel_groups,
    uint32_t base_offset,
    uint32_t& kernel_text_offset,
    uint32_t& kernel_text_size);

void insert_empty_program_dispatch_preamble_cmd(ProgramCommandSequence& program_command_sequence);

void insert_stall_cmds(ProgramCommandSequence& program_command_sequence, SubDeviceId sub_device_id, IDevice* device);

void assemble_runtime_args_commands(
    ProgramCommandSequence& program_command_sequence, Program& program, IDevice* device);

void assemble_device_commands(
    ProgramCommandSequence& program_command_sequence, Program& program, IDevice* device, SubDeviceId sub_device_id);

void initialize_worker_config_buf_mgr(WorkerConfigBufferMgr& config_buffer_mgr);

void reserve_space_in_kernel_config_buffer(
    WorkerConfigBufferMgr& config_buffer_mgr,
    const std::vector<uint32_t>& program_config_sizes,
    ProgramBinaryStatus program_binary_status,
    uint32_t num_program_workers,
    uint32_t expected_num_workers_completed,
    ProgramDispatchMetadata& dispatch_md);

void update_program_dispatch_commands(
    Program& program,
    ProgramCommandSequence& cached_program_command_sequence,
    uint32_t multicast_cores_launch_message_wptr,
    uint32_t unicast_cores_launch_message_wptr,
    uint32_t expected_num_workers_completed,
    CoreCoord dispatch_core,
    CoreType dispatch_core_type,
    SubDeviceId sub_device_id,
    const ProgramDispatchMetadata& dispatch_md,
    ProgramBinaryStatus program_binary_status,
    std::pair<bool, int> unicast_go_signal_update = {false, -1});

void write_program_command_sequence(
    const ProgramCommandSequence& program_command_sequence,
    SystemMemoryManager& manager,
    uint32_t command_queue_id,
    CoreType dispatch_core_type,
    bool stall_first,
    bool stall_before_program);

KernelHandle get_device_local_kernel_handle(KernelHandle kernel_handle);

template <typename WorkloadType, typename DeviceType>
uint32_t program_base_addr_on_core(
    WorkloadType& workload, DeviceType generic_device, HalProgrammableCoreType programmable_core_type);

}  // namespace program_dispatch

}  // namespace tt_metal

}  // namespace tt
