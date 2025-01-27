// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_metal.hpp>

#include "tt_metal/distributed/mesh_buffer.hpp"
#include "tt_metal/distributed/mesh_command_queue.hpp"
#include "tt_metal/distributed/mesh_workload.hpp"
#include "tt_metal/distributed/mesh_workload_utils.hpp"

namespace tt::tt_metal::distributed {

MeshWorkload::MeshWorkload() {
    // A MeshWorkload tracks maintains its own handles to kernels across all
    // encapsulated programs
    kernel_groups_.resize(hal.get_programmable_core_type_count());
    kernels_.resize(hal.get_programmable_core_type_count());
}

void MeshWorkload::add_program(const LogicalDeviceRange& device_range, Program&& program) {
    // Add a program to a MeshWorkload and tie it a specific logical device range
    programs_[device_range] = std::move(program);
    logical_device_ranges_.push_back(device_range);
}

void MeshWorkload::compile(MeshDevice* mesh_device) {
    // Multi-Step Compile:
    // 1. Compile Kernel Binaries
    // 2. Allocate and Validate CBs
    // 3. Finalize: Compute relative offsets for all data structures in L1
    for (auto& [device_range, program] : programs_) {
        program.compile(mesh_device);
        program.allocate_circular_buffers(mesh_device);
        tt::tt_metal::detail::ValidateCircularBufferRegion(program, mesh_device);
    }
    program_dispatch::finalize_program_offsets(*this, mesh_device);
}

void MeshWorkload::load_binaries(MeshCommandQueue& mesh_cq) {
    // Load binaries for all programs to their respective devices in
    // the Mesh. Only done when the MeshWorkload is enqueued for the first
    // time.
    auto* mesh_device = mesh_cq.device();
    if (program_binary_status_.size()) {
        TT_FATAL(
            program_binary_status_.find(mesh_device->id()) != program_binary_status_.end(),
            "Reusing MeshWorkloads across MeshDevices is currently not supported.");
        TT_FATAL(
            program_binary_status_.at(mesh_device->id()) == ProgramBinaryStatus::Committed,
            "Expected Program Biinaries to be committed to DRAM.");
    } else {
        // Allocate kernel binary buffers of max size across all devices, to ensure we have lock step allocation.
        uint32_t max_kernel_bin_buf_size = 0;
        for (auto& [device_range, program] : programs_) {
            uint32_t curr_kernel_bin_size = program.get_program_transfer_info().binary_data.size() * sizeof(uint32_t);
            max_kernel_bin_buf_size = std::max(max_kernel_bin_buf_size, curr_kernel_bin_size);
        }
        // Allocate a buffer for kernel binaries on each device.
        // Once MeshBuffer is available, allocate kernel bin MeshBuffer directly here
        for (auto device : mesh_device->get_devices()) {
            std::shared_ptr<Buffer> kernel_bin_buf = Buffer::create(
                device,
                max_kernel_bin_buf_size,
                HostMemDeviceCommand::PROGRAM_PAGE_SIZE,
                BufferType::DRAM,
                TensorMemoryLayout::INTERLEAVED,
                std::nullopt,
                false);
            kernel_bin_buffers_.insert(
                kernel_bin_buf);  // Tie the lifetime of kernel binary buffers to the MeshWorkload
        }
        // Iterate over the sub-grids and EnqueueWriteMeshBuffer to each sub-grid that runs the program
        for (auto& [device_range, program] : programs_) {
            std::size_t kernel_bin_size = program.get_program_transfer_info().binary_data.size() * sizeof(uint32_t);
            for (std::size_t logical_x = device_range.start_coord.x; logical_x < device_range.end_coord.x;
                 logical_x++) {
                for (std::size_t logical_y = device_range.start_coord.y; logical_y < device_range.end_coord.y;
                     logical_y++) {
                    IDevice* device = mesh_device->get_device(logical_y, logical_x);
                    // Get a view of the allocated buffer that matches the size of the kernel binary
                    // for the sub grid
                    std::shared_ptr<Buffer> mesh_buffer_view = Buffer::create(
                        mesh_device,
                        (*(kernel_bin_buffers_.begin()))->address(),
                        kernel_bin_size,
                        HostMemDeviceCommand::PROGRAM_PAGE_SIZE,
                        BufferType::DRAM,
                        TensorMemoryLayout::INTERLEAVED,
                        std::nullopt,
                        false);
                    std::shared_ptr<Buffer> buffer_view = Buffer::create(
                        device,
                        (*(kernel_bin_buffers_.begin()))->address(),
                        kernel_bin_size,
                        HostMemDeviceCommand::PROGRAM_PAGE_SIZE,
                        BufferType::DRAM,
                        TensorMemoryLayout::INTERLEAVED,
                        std::nullopt,
                        false);
                    EnqueueWriteBuffer(
                        device->command_queue(mesh_cq.id()),
                        buffer_view,
                        program.get_program_transfer_info().binary_data.data(),
                        false);
                    // Assign this memory region to the program. Required when the program
                    // object is used to generate dispatch commands
                    program.set_kernels_bin_buffer(mesh_buffer_view);
                    program.set_program_binary_status(device->id(), ProgramBinaryStatus::InFlight);
                }
            }
        }
        program_binary_status_[mesh_device->id()] = ProgramBinaryStatus::InFlight;
    }
}

ProgramBinaryStatus MeshWorkload::get_program_binary_status(std::size_t mesh_id) const {
    if (program_binary_status_.find(mesh_id) != program_binary_status_.end()) {
        return program_binary_status_.at(mesh_id);
    }
    return ProgramBinaryStatus::NotSent;
}

void MeshWorkload::set_program_binary_status(std::size_t mesh_id, ProgramBinaryStatus status) {
    program_binary_status_[mesh_id] = status;
}

void MeshWorkload::generate_dispatch_commands(MeshCommandQueue& mesh_cq) {
    // Generate Dispatch Commands for each Program in the MeshWorkload.
    // These commands will be updated based on MeshDevice state when the
    // workload is enqueued.
    auto mesh_device = mesh_cq.device();
    for (auto& [device_range, program] : programs_) {
        program.generate_dispatch_commands(mesh_device);
    }
}

bool MeshWorkload::runs_on_noc_multicast_only_cores() {
    // Return true if any program in the MeshWorkload runs on cores
    // that can be multicasted to
    bool ret = false;
    for (auto& [device_range, program] : programs_) {
        ret = ret || (program.runs_on_noc_multicast_only_cores());
    }
    return ret;
}

bool MeshWorkload::runs_on_noc_unicast_only_cores() {
    // Return true if any program in the MeshWorkload runs on cores
    // that can only be unicasted to
    bool ret = false;
    for (auto& [device_range, program] : programs_) {
        ret = ret || (program.runs_on_noc_unicast_only_cores());
    }
    return ret;
}

bool MeshWorkload::kernel_binary_always_stored_in_ringbuffer() {
    // Return true if kernel binaries cannot be placed in a ring buffer for
    // any program in the MeshWorkload
    bool stored_in_ring_buf = true;
    for (auto& [device_range, program] : programs_) {
        stored_in_ring_buf &= program.kernel_binary_always_stored_in_ringbuffer();
    }
    return stored_in_ring_buf;
}

std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& MeshWorkload::get_kernels(
    uint32_t programmable_core_type_index) {
    // Get all kernels across all programs in the MeshWorkload
    if (not kernels_.at(programmable_core_type_index).size()) {
        for (auto& [device_range, program] : programs_) {
            uint32_t device_range_handle = (device_range.start_coord.y << 24) | (device_range.start_coord.x << 16);
            for (const auto& kernel : program.get_kernels(programmable_core_type_index)) {
                KernelHandle handle = (device_range_handle | kernel.first);
                kernels_.at(programmable_core_type_index).insert({handle, kernel.second});
            }
        }
    }
    return kernels_.at(programmable_core_type_index);
}

std::vector<std::shared_ptr<KernelGroup>>& MeshWorkload::get_kernel_groups(uint32_t programmable_core_type_index) {
    // Get all kernel groups across all programs in the MeshWorkload
    if (not kernel_groups_.at(programmable_core_type_index).size()) {
        for (auto& [device_range, program] : programs_) {
            uint32_t device_range_handle = (device_range.start_coord.y << 24) | (device_range.start_coord.x << 16);
            for (auto& kg : program.get_kernel_groups(programmable_core_type_index)) {
                for (auto& optional_kernel_id : kg->kernel_ids) {
                    if (optional_kernel_id.has_value()) {
                        optional_kernel_id = (device_range_handle | optional_kernel_id.value());
                    }
                }
                kernel_groups_.at(programmable_core_type_index).push_back(kg);
            }
        }
    }
    return kernel_groups_.at(programmable_core_type_index);
}

std::vector<Semaphore>& MeshWorkload::semaphores() {
    // Get all semaphores across all programs in the MeshWorkload
    if (not semaphores_.size()) {
        for (auto& [device_range, program] : programs_) {
            semaphores_.insert(semaphores_.end(), program.semaphores().begin(), program.semaphores().end());
        }
    }
    return semaphores_;
}

std::vector<uint32_t> MeshWorkload::get_program_config_sizes() {
    // Get the config sizes for all L1 Program Data Structures
    std::vector<uint32_t> global_program_config_sizes;
    for (auto& program_on_grid : programs_) {
        if (global_program_config_sizes.size()) {
            for (int i = 0; i < global_program_config_sizes.size(); i++) {
                TT_FATAL(
                    global_program_config_sizes[i] == program_on_grid.second.get_program_config_sizes()[i],
                    "Expected config sizes to be identical across all programs in a MeshWorkload.");
            }
        } else {
            global_program_config_sizes = program_on_grid.second.get_program_config_sizes();
        }
    }
    return global_program_config_sizes;
}

std::unordered_set<SubDeviceId> MeshWorkload::determine_sub_device_ids(MeshDevice* mesh_device) {
    // Get the sub device ids for all program across all devices in the Workload
    std::unordered_set<SubDeviceId> sub_devices_;
    for (auto& [device_range, program] : programs_) {
        auto grid_start = device_range.start_coord;
        IDevice* device = mesh_device->get_device(grid_start.y, grid_start.x);
        auto sub_devs_for_program = program.determine_sub_device_ids(mesh_device);
        for (auto& sub_dev : sub_devs_for_program) {
            sub_devices_.insert(sub_dev);
        }
    }
    return sub_devices_;
}

ProgramCommandSequence& MeshWorkload::get_dispatch_cmds_for_program(Program& program) {
    // Get the dispatch commands associated with this program
    return program.get_cached_program_command_sequences().begin()->second;
}

// The functions below are for testing purposes only
void MeshWorkload::set_last_used_command_queue_for_testing(MeshCommandQueue* mesh_cq) {
    last_used_command_queue_ = mesh_cq;
}

MeshCommandQueue* MeshWorkload::get_last_used_command_queue() const { return last_used_command_queue_; }

ProgramConfig& MeshWorkload::get_program_config(uint32_t index) {
    TT_FATAL(
        programs_.size() and is_finalized(),
        "Program Configs can only be queried if a MeshWorkload is populated and finalized.");
    return programs_.begin()->second.get_program_config(index);
}

uint32_t MeshWorkload::get_sem_base_addr(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type) {
    HalProgrammableCoreType programmable_core_type =
        ::tt::tt_metal::detail::hal_programmable_core_type_from_core_type(core_type);
    uint32_t base_addr = program_dispatch::program_base_addr_on_core(*this, mesh_device.get(), programmable_core_type);
    return base_addr + get_program_config(hal.get_programmable_core_type_index(programmable_core_type)).sem_offset;
}

uint32_t MeshWorkload::get_sem_size(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type) {
    uint32_t sem_size = 0;
    uint32_t program_idx = 0;
    IDevice* device = mesh_device->get_device(0);
    for (auto& [device_range, program] : programs_) {
        if (program_idx) {
            TT_ASSERT(sem_size == program.get_sem_size(device, logical_core, core_type));
        } else {
            sem_size = program.get_sem_size(device, logical_core, core_type);
        }
        program_idx++;
    }
    return sem_size;
}

uint32_t MeshWorkload::get_cb_base_addr(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type) {
    HalProgrammableCoreType programmable_core_type =
        ::tt::tt_metal::detail::hal_programmable_core_type_from_core_type(core_type);
    uint32_t base_addr = program_dispatch::program_base_addr_on_core(*this, mesh_device.get(), programmable_core_type);
    return base_addr + get_program_config(hal.get_programmable_core_type_index(programmable_core_type)).cb_offset;
}

uint32_t MeshWorkload::get_cb_size(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type) {
    uint32_t cb_size = 0;
    uint32_t program_idx = 0;
    IDevice* device = mesh_device->get_device(0);
    for (auto& [device_range, program] : programs_) {
        if (program_idx) {
            TT_ASSERT(cb_size == program.get_cb_size(device, logical_core, core_type));
        } else {
            cb_size = program.get_cb_size(device, logical_core, core_type);
        }
        program_idx++;
    }
    return cb_size;
}

}  // namespace tt::tt_metal::distributed
