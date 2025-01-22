// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <command_queue.hpp>
#include <device.hpp>
#include <dprint_server.hpp>
#include <event.hpp>
#include <hardware_command_queue.hpp>
#include <overloaded.hpp>
#include <trace_buffer.hpp>

#include "tt_metal/impl/buffers/dispatch.hpp"
#include "tt_metal/impl/debug/watcher_server.hpp"
#include "tt_metal/impl/program/dispatch.hpp"

namespace tt::tt_metal {
namespace {

// Selects all sub-devices in the sub device stall group if none are specified
tt::stl::Span<const SubDeviceId> select_sub_device_ids(
    IDevice* device, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    if (sub_device_ids.empty()) {
        return device->get_sub_device_stall_group();
    } else {
        for (const auto& sub_device_id : sub_device_ids) {
            TT_FATAL(
                sub_device_id.to_index() < device->num_sub_devices(),
                "Invalid sub-device id specified {}",
                sub_device_id.to_index());
        }
        return sub_device_ids;
    }
}

Buffer& get_buffer_object(const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer) {
    return std::visit(
        tt::stl::overloaded{
            [](const std::shared_ptr<Buffer>& b) -> Buffer& { return *b; },
            [](const std::reference_wrapper<Buffer>& b) -> Buffer& { return b.get(); }},
        buffer);
}

}  // namespace

HWCommandQueue::HWCommandQueue(IDevice* device, uint32_t id, NOC noc_index) :
    manager(device->sysmem_manager()), completion_queue_thread{} {
    ZoneScopedN("CommandQueue_constructor");
    this->device = device;
    this->id = id;
    this->noc_index = noc_index;
    this->num_entries_in_completion_q = 0;
    this->num_completed_completion_q_reads = 0;

    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
    this->size_B = tt::Cluster::instance().get_host_channel_size(mmio_device_id, channel) / device->num_hw_cqs();
    if (tt::Cluster::instance().is_galaxy_cluster()) {
        // Galaxy puts 4 devices per host channel until umd can provide one channel per device.
        this->size_B = this->size_B / 4;
    }

    CoreCoord enqueue_program_dispatch_core;
    CoreType core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    if (this->device->num_hw_cqs() == 1 or core_type == CoreType::WORKER) {
        // dispatch_s exists with this configuration. Workers write to dispatch_s
        enqueue_program_dispatch_core = dispatch_core_manager::instance().dispatcher_s_core(device->id(), channel, id);
    } else {
        if (device->is_mmio_capable()) {
            enqueue_program_dispatch_core =
                dispatch_core_manager::instance().dispatcher_core(device->id(), channel, id);
        } else {
            enqueue_program_dispatch_core =
                dispatch_core_manager::instance().dispatcher_d_core(device->id(), channel, id);
        }
    }
    this->virtual_enqueue_program_dispatch_core =
        device->virtual_core_from_logical_core(enqueue_program_dispatch_core, core_type);

    tt_cxy_pair completion_q_writer_location =
        dispatch_core_manager::instance().completion_queue_writer_core(device->id(), channel, this->id);

    this->completion_queue_writer_core = CoreCoord(completion_q_writer_location.x, completion_q_writer_location.y);

    this->exit_condition = false;
    std::thread completion_queue_thread = std::thread(&HWCommandQueue::read_completion_queue, this);
    this->completion_queue_thread = std::move(completion_queue_thread);
    // Set the affinity of the completion queue reader.
    set_device_thread_affinity(this->completion_queue_thread, device->get_completion_queue_reader_core());

    for (uint32_t i = 0; i < dispatch_constants::DISPATCH_MESSAGE_ENTRIES; i++) {
        this->expected_num_workers_completed[i] = 0;
    }
    reset_config_buffer_mgr(dispatch_constants::DISPATCH_MESSAGE_ENTRIES);
}

uint32_t HWCommandQueue::get_id() const { return this->id; }

std::optional<uint32_t> HWCommandQueue::get_tid() const { return this->tid; }

SystemMemoryManager& HWCommandQueue::sysmem_manager() { return this->manager; }

void HWCommandQueue::set_num_worker_sems_on_dispatch(uint32_t num_worker_sems) {
    // Not needed for regular dispatch kernel
    if (!this->device->dispatch_s_enabled()) {
        return;
    }
    uint32_t cmd_sequence_sizeB = hal.get_alignment(HalMemType::HOST);
    void* cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->id);
    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);
    command_sequence.add_dispatch_set_num_worker_sems(num_worker_sems, DispatcherSelect::DISPATCH_SLAVE);
    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->id);
    this->manager.fetch_queue_reserve_back(this->id);
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->id);
}

void HWCommandQueue::set_go_signal_noc_data_on_dispatch(const vector_memcpy_aligned<uint32_t>& go_signal_noc_data) {
    uint32_t pci_alignment = hal.get_alignment(HalMemType::HOST);
    uint32_t cmd_sequence_sizeB = align(
        sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd) + go_signal_noc_data.size() * sizeof(uint32_t), pci_alignment);
    void* cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->id);
    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);
    DispatcherSelect dispatcher_for_go_signal =
        this->device->dispatch_s_enabled() ? DispatcherSelect::DISPATCH_SLAVE : DispatcherSelect::DISPATCH_MASTER;
    command_sequence.add_dispatch_set_go_signal_noc_data(go_signal_noc_data, dispatcher_for_go_signal);
    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->id);
    this->manager.fetch_queue_reserve_back(this->id);
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->id);
}

uint32_t HWCommandQueue::get_expected_num_workers_completed_for_sub_device(uint32_t sub_device_index) const {
    TT_FATAL(
        sub_device_index < dispatch_constants::DISPATCH_MESSAGE_ENTRIES,
        "Expected sub_device_index to be less than dispatch_constants::DISPATCH_MESSAGE_ENTRIES");
    return this->expected_num_workers_completed[sub_device_index];
}

void HWCommandQueue::set_expected_num_workers_completed_for_sub_device(
    uint32_t sub_device_index, uint32_t num_workers) {
    TT_FATAL(
        sub_device_index < dispatch_constants::DISPATCH_MESSAGE_ENTRIES,
        "Expected sub_device_index to be less than dispatch_constants::DISPATCH_MESSAGE_ENTRIES");
    this->expected_num_workers_completed[sub_device_index] = num_workers;
}

void HWCommandQueue::reset_worker_dispatch_state_on_device(bool reset_launch_msg_state) {
    auto num_sub_devices = device->num_sub_devices();
    uint32_t go_signals_cmd_size = 0;
    if (reset_launch_msg_state) {
        uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
        go_signals_cmd_size = align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd), pcie_alignment) * num_sub_devices;
    }
    uint32_t cmd_sequence_sizeB =
        reset_launch_msg_state * this->device->dispatch_s_enabled() *
            hal.get_alignment(
                HalMemType::HOST) +  // dispatch_d -> dispatch_s sem update (send only if dispatch_s is running)
        go_signals_cmd_size +        // go signal cmd
        (hal.get_alignment(
             HalMemType::HOST) +  // wait to ensure that reset go signal was processed (dispatch_d)
                                  // when dispatch_s and dispatch_d are running on 2 cores, workers update dispatch_s.
                                  // dispatch_s is responsible for resetting worker count and giving dispatch_d the
                                  // latest worker state. This is encapsulated in the dispatch_s wait command (only to
                                  // be sent when dispatch is distributed on 2 cores)
         this->device->distributed_dispatcher() * hal.get_alignment(HalMemType::HOST)) *
            num_sub_devices;
    void* cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->id);
    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);
    bool clear_count = true;
    DispatcherSelect dispatcher_for_go_signal = DispatcherSelect::DISPATCH_MASTER;
    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    uint32_t dispatch_message_base_addr =
        dispatch_constants::get(dispatch_core_type)
            .get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE);
    if (reset_launch_msg_state) {
        if (device->dispatch_s_enabled()) {
            uint16_t index_bitmask = 0;
            for (uint32_t i = 0; i < num_sub_devices; ++i) {
                index_bitmask |= 1 << i;
            }
            command_sequence.add_notify_dispatch_s_go_signal_cmd(false, index_bitmask);
            dispatcher_for_go_signal = DispatcherSelect::DISPATCH_SLAVE;
        }
        go_msg_t reset_launch_message_read_ptr_go_signal;
        reset_launch_message_read_ptr_go_signal.signal = RUN_MSG_RESET_READ_PTR;
        reset_launch_message_read_ptr_go_signal.master_x = (uint8_t)this->virtual_enqueue_program_dispatch_core.x;
        reset_launch_message_read_ptr_go_signal.master_y = (uint8_t)this->virtual_enqueue_program_dispatch_core.y;
        for (uint32_t i = 0; i < num_sub_devices; ++i) {
            reset_launch_message_read_ptr_go_signal.dispatch_message_offset =
                (uint8_t)dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(i);
            uint32_t dispatch_message_addr =
                dispatch_message_base_addr + dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(i);
            // Wait to ensure that all kernels have completed. Then send the reset_rd_ptr go_signal.
            command_sequence.add_dispatch_go_signal_mcast(
                expected_num_workers_completed[i],
                *reinterpret_cast<uint32_t*>(&reset_launch_message_read_ptr_go_signal),
                dispatch_message_addr,
                device->num_noc_mcast_txns({i}),
                device->num_noc_unicast_txns({i}),
                device->noc_data_start_index({i}),
                dispatcher_for_go_signal);
            expected_num_workers_completed[i] += device->num_worker_cores(HalProgrammableCoreType::TENSIX, {i});
            expected_num_workers_completed[i] += device->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, {i});
        }
    }
    // Wait to ensure that all workers have reset their read_ptr. dispatch_d will stall until all workers have completed
    // this step, before sending kernel config data to workers or notifying dispatch_s that its safe to send the
    // go_signal. Clear the dispatch <--> worker semaphore, since trace starts at 0.
    for (uint32_t i = 0; i < num_sub_devices; ++i) {
        uint32_t dispatch_message_addr =
            dispatch_message_base_addr + dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(i);
        if (device->distributed_dispatcher()) {
            command_sequence.add_dispatch_wait(
                false, dispatch_message_addr, expected_num_workers_completed[i], clear_count, false, true, 1);
        }
        command_sequence.add_dispatch_wait(
            false, dispatch_message_addr, expected_num_workers_completed[i], clear_count);
    }
    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->id);
    this->manager.fetch_queue_reserve_back(this->id);
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->id);

    if (clear_count) {
        std::fill(expected_num_workers_completed.begin(), expected_num_workers_completed.begin() + num_sub_devices, 0);
    }
}

void HWCommandQueue::reset_worker_state(
    bool reset_launch_msg_state, uint32_t num_sub_devices, const vector_memcpy_aligned<uint32_t>& go_signal_noc_data) {
    TT_FATAL(!this->manager.get_bypass_mode(), "Cannot reset worker state during trace capture");
    // TODO: This could be further optimized by combining all of these into a single prefetch entry
    // Currently each one will be pushed into its own prefetch entry
    this->reset_worker_dispatch_state_on_device(reset_launch_msg_state);
    this->set_num_worker_sems_on_dispatch(num_sub_devices);
    this->set_go_signal_noc_data_on_dispatch(go_signal_noc_data);
    this->reset_config_buffer_mgr(num_sub_devices);
    if (reset_launch_msg_state) {
        this->manager.reset_worker_launch_message_buffer_state(num_sub_devices);
    }
}

HWCommandQueue::~HWCommandQueue() {
    ZoneScopedN("HWCommandQueue_destructor");
    if (this->exit_condition) {
        this->completion_queue_thread.join();  // We errored out already prior
    } else {
        TT_ASSERT(
            this->issued_completion_q_reads.empty(),
            "There should be no reads in flight after closing our completion queue thread");
        TT_ASSERT(
            this->num_entries_in_completion_q == this->num_completed_completion_q_reads,
            "There shouldn't be any commands in flight after closing our completion queue thread. Num uncompleted "
            "commands: {}",
            this->num_entries_in_completion_q - this->num_completed_completion_q_reads);
        this->set_exit_condition();
        this->completion_queue_thread.join();
    }
}

void HWCommandQueue::increment_num_entries_in_completion_q() {
    // Increment num_entries_in_completion_q and inform reader thread
    // that there is work in the completion queue to process
    this->num_entries_in_completion_q++;
    {
        std::lock_guard lock(this->reader_thread_cv_mutex);
        this->reader_thread_cv.notify_one();
    }
}

void HWCommandQueue::set_exit_condition() {
    this->exit_condition = true;
    {
        std::lock_guard lock(this->reader_thread_cv_mutex);
        this->reader_thread_cv.notify_one();
    }
}

template <typename T>
void HWCommandQueue::enqueue_command(T& command, bool blocking, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    command.process();
    if (blocking) {
        this->finish(sub_device_ids);
    }
}

void HWCommandQueue::enqueue_read_buffer(
    std::shared_ptr<Buffer>& buffer,
    void* dst,
    const BufferRegion& region,
    bool blocking,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    this->enqueue_read_buffer(*buffer, dst, region, blocking, sub_device_ids);
}

// Read buffer command is enqueued in the issue region and device writes requested buffer data into the completion
// region
void HWCommandQueue::enqueue_read_buffer(
    Buffer& buffer,
    void* dst,
    const BufferRegion& region,
    bool blocking,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    ZoneScopedN("HWCommandQueue_read_buffer");
    TT_FATAL(!this->manager.get_bypass_mode(), "Enqueue Read Buffer cannot be used with tracing");
    sub_device_ids = select_sub_device_ids(this->device, sub_device_ids);

    if (is_sharded(buffer.buffer_layout())) {
        // Forward data from each core to the completion queue.
        // Then have the completion queue reader thread copy this data to user space.
        auto dispatch_params = buffer_dispatch::initialize_sharded_buf_read_dispatch_params(
            buffer, this->id, this->expected_num_workers_completed);
        auto cores = buffer_dispatch::get_cores_for_sharded_buffer(
            dispatch_params.width_split, dispatch_params.buffer_page_mapping, buffer);
        for (uint32_t core_id = 0; core_id < buffer.num_cores(); ++core_id) {
            buffer_dispatch::copy_sharded_buffer_from_core_to_completion_queue(
                core_id,
                buffer,
                dispatch_params,
                sub_device_ids,
                cores[core_id],
                dispatch_core_manager::instance().get_dispatch_core_type(device->id()));
            if (dispatch_params.pages_per_txn > 0) {
                this->issued_completion_q_reads.push(
                    buffer_dispatch::generate_sharded_buffer_read_descriptor(dst, dispatch_params, buffer));
                this->increment_num_entries_in_completion_q();
            }
        }
    } else {
        // Forward data from device to the completion queue.
        // Then have the completion queue reader thread copy this data to user space.
        auto dispatch_params = buffer_dispatch::initialize_interleaved_buf_read_dispatch_params(
            buffer, this->id, this->expected_num_workers_completed, region);
        buffer_dispatch::copy_interleaved_buffer_to_completion_queue(
            dispatch_params,
            buffer,
            sub_device_ids,
            dispatch_core_manager::instance().get_dispatch_core_type(device->id()));
        if (dispatch_params.pages_per_txn > 0) {
            this->issued_completion_q_reads.push(
                buffer_dispatch::generate_interleaved_buffer_read_descriptor(dst, dispatch_params, buffer));
            this->increment_num_entries_in_completion_q();
        }
    }
    if (blocking) {
        this->finish(sub_device_ids);
    }
}

void HWCommandQueue::enqueue_write_buffer(
    const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
    HostDataType src,
    const BufferRegion& region,
    bool blocking,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    // Top level API to accept different variants for buffer and src
    // For shared pointer variants, object lifetime is guaranteed at least till the end of this function
    auto* data = std::visit(
        tt::stl::overloaded{
            [](const void* raw_data) -> const void* { return raw_data; },
            [](const auto& data) -> const void* { return data->data(); }},
        src);
    Buffer& buffer_obj = get_buffer_object(buffer);
    this->enqueue_write_buffer(buffer_obj, data, region, blocking, sub_device_ids);
}

CoreType HWCommandQueue::get_dispatch_core_type() {
    return dispatch_core_manager::instance().get_dispatch_core_type(device->id());
}

void HWCommandQueue::enqueue_write_buffer(
    Buffer& buffer,
    const void* src,
    const BufferRegion& region,
    bool blocking,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    ZoneScopedN("HWCommandQueue_write_buffer");
    TT_FATAL(!this->manager.get_bypass_mode(), "Enqueue Write Buffer cannot be used with tracing");

    sub_device_ids = select_sub_device_ids(this->device, sub_device_ids);
    auto dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());

    buffer_dispatch::write_to_device_buffer(
        src, buffer, region, this->id, this->expected_num_workers_completed, dispatch_core_type, sub_device_ids);

    if (blocking) {
        this->finish(sub_device_ids);
    }
}

void HWCommandQueue::enqueue_program(Program& program, bool blocking) {
    ZoneScopedN("HWCommandQueue_enqueue_program");
    std::vector<SubDeviceId> sub_device_ids = {program.determine_sub_device_ids(device)};
    TT_FATAL(sub_device_ids.size() == 1, "Programs must be executed on a single sub-device");
    // Finalize Program: Compute relative offsets for data structures (semaphores, kernel binaries, etc) in L1
    program_dispatch::finalize_program_offsets(program, device);

    if (program.get_program_binary_status(device->id()) == ProgramBinaryStatus::NotSent) {
        // Write program binaries to device if it hasn't previously been cached
        program.allocate_kernel_bin_buf_on_device(device);
        if (program.get_program_transfer_info().binary_data.size()) {
            const BufferRegion buffer_region(0, program.get_kernels_buffer(device)->size());
            this->enqueue_write_buffer(
                *program.get_kernels_buffer(device),
                program.get_program_transfer_info().binary_data.data(),
                buffer_region,
                false);
        }
        program.set_program_binary_status(device->id(), ProgramBinaryStatus::InFlight);
    }
    // Lower the program to device: Generate dispatch commands.
    // Values in these commands will get updated based on kernel config ring
    // buffer state at runtime.
    program.generate_dispatch_commands(device);
    program.set_last_used_command_queue_for_testing(this);

#ifdef DEBUG
    if (tt::llrt::RunTimeOptions::get_instance().get_validate_kernel_binaries()) {
        TT_FATAL(!this->manager.get_bypass_mode(), "Tracing cannot be used while validating program binaries");
        if (const auto buffer = program.get_kernels_buffer(device)) {
            std::vector<uint32_t> read_data(buffer->page_size() * buffer->num_pages() / sizeof(uint32_t));
            const BufferRegion region(0, buffer->size());
            this->enqueue_read_buffer(*buffer, read_data.data(), region, true);
            TT_FATAL(
                program.get_program_transfer_info().binary_data == read_data,
                "Binary for program to be executed is corrupted. Another program likely corrupted this binary");
        }
    }
#endif
    auto sub_device_id = sub_device_ids[0];
    auto sub_device_index = sub_device_id.to_index();

    // Snapshot of expected workers from previous programs, used for dispatch_wait cmd generation.
    uint32_t expected_workers_completed = this->manager.get_bypass_mode()
                                              ? this->trace_ctx->descriptors[sub_device_id].num_completion_worker_cores
                                              : this->expected_num_workers_completed[sub_device_index];
    if (this->manager.get_bypass_mode()) {
        if (program.runs_on_noc_multicast_only_cores()) {
            this->trace_ctx->descriptors[sub_device_id].num_traced_programs_needing_go_signal_multicast++;
            this->trace_ctx->descriptors[sub_device_id].num_completion_worker_cores +=
                device->num_worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id);
        }
        if (program.runs_on_noc_unicast_only_cores()) {
            this->trace_ctx->descriptors[sub_device_id].num_traced_programs_needing_go_signal_unicast++;
            this->trace_ctx->descriptors[sub_device_id].num_completion_worker_cores +=
                device->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id);
        }
    } else {
        if (program.runs_on_noc_multicast_only_cores()) {
            this->expected_num_workers_completed[sub_device_index] +=
                device->num_worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id);
        }
        if (program.runs_on_noc_unicast_only_cores()) {
            this->expected_num_workers_completed[sub_device_index] +=
                device->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id);
        }
    }

    auto& worker_launch_message_buffer_state =
        this->manager.get_worker_launch_message_buffer_state()[sub_device_id.to_index()];
    auto command = EnqueueProgramCommand(
        this->id,
        this->device,
        this->noc_index,
        program,
        this->virtual_enqueue_program_dispatch_core,
        this->manager,
        this->get_config_buffer_mgr(sub_device_index),
        expected_workers_completed,
        // The assembled program command will encode the location of the launch messages in the ring buffer
        worker_launch_message_buffer_state.get_mcast_wptr(),
        worker_launch_message_buffer_state.get_unicast_wptr(),
        sub_device_id);
    // Update wptrs for tensix and eth launch message in the device class
    if (program.runs_on_noc_multicast_only_cores()) {
        worker_launch_message_buffer_state.inc_mcast_wptr(1);
    }
    if (program.runs_on_noc_unicast_only_cores()) {
        worker_launch_message_buffer_state.inc_unicast_wptr(1);
    }
    this->enqueue_command(command, blocking, sub_device_ids);

#ifdef DEBUG
    if (tt::llrt::RunTimeOptions::get_instance().get_validate_kernel_binaries()) {
        TT_FATAL(!this->manager.get_bypass_mode(), "Tracing cannot be used while validating program binaries");
        if (const auto buffer = program.get_kernels_buffer(device)) {
            std::vector<uint32_t> read_data(buffer->page_size() * buffer->num_pages() / sizeof(uint32_t));
            const BufferRegion region(0, buffer->size());
            this->enqueue_read_buffer(*buffer, read_data.data(), region, true);
            TT_FATAL(
                program.get_program_transfer_info().binary_data == read_data,
                "Binary for program that executed is corrupted. This program likely corrupted its own binary.");
        }
    }
#endif

    log_trace(
        tt::LogMetal,
        "Created EnqueueProgramCommand (active_cores: {} bypass_mode: {} expected_workers_completed: {})",
        program.get_program_transfer_info().num_active_cores,
        this->manager.get_bypass_mode(),
        expected_workers_completed);
}

void HWCommandQueue::enqueue_record_event(
    const std::shared_ptr<Event>& event, bool clear_count, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    ZoneScopedN("HWCommandQueue_enqueue_record_event");

    TT_FATAL(!this->manager.get_bypass_mode(), "Enqueue Record Event cannot be used with tracing");

    // Populate event struct for caller. When async queues are enabled, this is in child thread, so consumers
    // of the event must wait for it to be ready (ie. populated) here. Set ready flag last. This couldn't be
    // in main thread otherwise event_id selection would get out of order due to main/worker thread timing.
    event->cq_id = this->id;
    event->event_id = this->manager.get_next_event(this->id);
    event->device = this->device;
    event->ready = true;  // what does this mean???

    sub_device_ids = select_sub_device_ids(this->device, sub_device_ids);

    auto command = EnqueueRecordEventCommand(
        this->id,
        this->device,
        this->noc_index,
        this->manager,
        event->event_id,
        this->expected_num_workers_completed,
        sub_device_ids,
        clear_count,
        true);
    this->enqueue_command(command, false, sub_device_ids);

    if (clear_count) {
        for (const auto& id : sub_device_ids) {
            this->expected_num_workers_completed[id.to_index()] = 0;
        }
    }
    this->issued_completion_q_reads.push(std::make_shared<detail::CompletionReaderVariant>(
        std::in_place_type<detail::ReadEventDescriptor>, event->event_id));
    this->increment_num_entries_in_completion_q();
}

void HWCommandQueue::enqueue_wait_for_event(const std::shared_ptr<Event>& sync_event, bool clear_count) {
    ZoneScopedN("HWCommandQueue_enqueue_wait_for_event");

    auto command = EnqueueWaitForEventCommand(this->id, this->device, this->manager, *sync_event, clear_count);
    this->enqueue_command(command, false, {});

    if (clear_count) {
        this->manager.reset_event_id(this->id);
    }
}

void HWCommandQueue::enqueue_trace(const uint32_t trace_id, bool blocking) {
    ZoneScopedN("HWCommandQueue_enqueue_trace");

    auto trace_inst = this->device->get_trace(trace_id);
    auto command = EnqueueTraceCommand(
        this->id,
        this->device,
        this->manager,
        trace_inst->desc,
        *trace_inst->buffer,
        this->expected_num_workers_completed,
        this->noc_index,
        this->virtual_enqueue_program_dispatch_core);

    this->enqueue_command(command, false, {});

    for (const auto& [id, desc] : trace_inst->desc->descriptors) {
        auto index = id.to_index();
        // Increment the expected worker cores counter due to trace programs completion
        this->expected_num_workers_completed[index] += desc.num_completion_worker_cores;
        // After trace runs, the rdptr on each worker will be incremented by the number of programs in the trace
        // Update the wptr on host to match state. If the trace doesn't execute on a
        // class of worker (unicast or multicast), it doesn't reset or modify the
        // state for those workers.
        auto& worker_launch_message_buffer_state = this->manager.get_worker_launch_message_buffer_state()[index];
        if (desc.num_traced_programs_needing_go_signal_multicast) {
            worker_launch_message_buffer_state.set_mcast_wptr(desc.num_traced_programs_needing_go_signal_multicast);
        }
        if (desc.num_traced_programs_needing_go_signal_unicast) {
            worker_launch_message_buffer_state.set_unicast_wptr(desc.num_traced_programs_needing_go_signal_unicast);
        }
        // The config buffer manager is unaware of what memory is used inside the trace, so mark all memory as used so
        // that it will force a stall and avoid stomping on in-use state.
        // TODO(jbauman): Reuse old state from the trace.
        this->config_buffer_mgr[index].mark_completely_full(this->expected_num_workers_completed[index]);
    }
    if (blocking) {
        this->finish(trace_inst->desc->sub_device_ids);
    }
}

void HWCommandQueue::read_completion_queue() {
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
    while (true) {
        {
            std::unique_lock<std::mutex> lock(this->reader_thread_cv_mutex);
            this->reader_thread_cv.wait(lock, [this] {
                return this->num_entries_in_completion_q > this->num_completed_completion_q_reads or
                       this->exit_condition;
            });
        }
        if (this->num_entries_in_completion_q > this->num_completed_completion_q_reads) {
            ZoneScopedN("CompletionQueueReader");
            uint32_t num_events_to_read = this->num_entries_in_completion_q - this->num_completed_completion_q_reads;
            for (uint32_t i = 0; i < num_events_to_read; i++) {
                ZoneScopedN("CompletionQueuePopulated");
                auto read_descriptor = *(this->issued_completion_q_reads.pop());
                {
                    ZoneScopedN("CompletionQueueWait");
                    this->manager.completion_queue_wait_front(
                        this->id, this->exit_condition);  // CQ DISPATCHER IS NOT HANDSHAKING WITH HOST RN
                }
                if (this->exit_condition) {  // Early exit
                    return;
                }

                std::visit(
                    [&](auto&& read_descriptor) {
                        using T = std::decay_t<decltype(read_descriptor)>;
                        if constexpr (std::is_same_v<T, detail::ReadBufferDescriptor>) {
                            ZoneScopedN("CompletionQueueReadData");
                            buffer_dispatch::copy_completion_queue_data_into_user_space(
                                read_descriptor,
                                mmio_device_id,
                                channel,
                                this->id,
                                this->manager,
                                this->exit_condition);
                        } else if constexpr (std::is_same_v<T, detail::ReadEventDescriptor>) {
                            ZoneScopedN("CompletionQueueReadEvent");
                            uint32_t read_ptr = this->manager.get_completion_queue_read_ptr(this->id);
                            thread_local static std::vector<uint32_t> dispatch_cmd_and_event(
                                (sizeof(CQDispatchCmd) + dispatch_constants::EVENT_PADDED_SIZE) / sizeof(uint32_t));
                            tt::Cluster::instance().read_sysmem(
                                dispatch_cmd_and_event.data(),
                                sizeof(CQDispatchCmd) + dispatch_constants::EVENT_PADDED_SIZE,
                                read_ptr,
                                mmio_device_id,
                                channel);
                            uint32_t event_completed = dispatch_cmd_and_event[sizeof(CQDispatchCmd) / sizeof(uint32_t)];

                            TT_ASSERT(
                                event_completed == read_descriptor.event_id,
                                "Event Order Issue: expected to read back completion signal for event {} but got {}!",
                                read_descriptor.event_id,
                                event_completed);
                            this->manager.completion_queue_pop_front(1, this->id);
                            this->manager.set_last_completed_event(this->id, read_descriptor.get_global_event_id());
                            log_trace(
                                LogAlways,
                                "Completion queue popped event {} (global: {})",
                                event_completed,
                                read_descriptor.get_global_event_id());
                        }
                    },
                    read_descriptor);
            }
            this->num_completed_completion_q_reads += num_events_to_read;
            {
                std::unique_lock<std::mutex> lock(this->reads_processed_cv_mutex);
                this->reads_processed_cv.notify_one();
            }
        } else if (this->exit_condition) {
            return;
        }
    }
}

void HWCommandQueue::finish(tt::stl::Span<const SubDeviceId> sub_device_ids) {
    ZoneScopedN("HWCommandQueue_finish");
    tt::log_debug(tt::LogDispatch, "Finish for command queue {}", this->id);
    std::shared_ptr<Event> event = std::make_shared<Event>();
    this->enqueue_record_event(event, false, sub_device_ids);
    if (tt::llrt::RunTimeOptions::get_instance().get_test_mode_enabled()) {
        while (this->num_entries_in_completion_q > this->num_completed_completion_q_reads) {
            if (DPrintServerHangDetected()) {
                // DPrint Server hang. Mark state and early exit. Assert in main thread.
                this->dprint_server_hang = true;
                this->set_exit_condition();
                return;
            } else if (tt::watcher_server_killed_due_to_error()) {
                // Illegal NOC txn killed watcher. Mark state and early exit. Assert in main thread.
                this->illegal_noc_txn_hang = true;
                this->set_exit_condition();
                return;
            }
        }
    } else {
        std::unique_lock<std::mutex> lock(this->reads_processed_cv_mutex);
        this->reads_processed_cv.wait(
            lock, [this] { return this->num_entries_in_completion_q == this->num_completed_completion_q_reads; });
    }
}

volatile bool HWCommandQueue::is_dprint_server_hung() { return dprint_server_hang; }

volatile bool HWCommandQueue::is_noc_hung() { return illegal_noc_txn_hang; }

void HWCommandQueue::record_begin(const uint32_t tid, std::shared_ptr<detail::TraceDescriptor> ctx) {
    auto num_sub_devices = this->device->num_sub_devices();
    // Record the original value of expected_num_workers_completed, and reset it to 0.
    std::copy(
        this->expected_num_workers_completed.begin(),
        this->expected_num_workers_completed.begin() + num_sub_devices,
        this->expected_num_workers_completed_reset.begin());
    std::fill(
        this->expected_num_workers_completed.begin(),
        this->expected_num_workers_completed.begin() + num_sub_devices,
        0);
    // Record commands using bypass mode
    this->tid = tid;
    this->trace_ctx = std::move(ctx);
    // Record original value of launch msg buffer
    auto& worker_launch_message_buffer_state = this->manager.get_worker_launch_message_buffer_state();
    std::copy(
        worker_launch_message_buffer_state.begin(),
        worker_launch_message_buffer_state.begin() + num_sub_devices,
        this->worker_launch_message_buffer_state_reset.begin());
    for (uint32_t i = 0; i < num_sub_devices; ++i) {
        // Set launch msg wptr to 0. Every time trace runs on device, it will ensure that the workers
        // reset their rptr to be in sync with device.
        worker_launch_message_buffer_state[i].reset();
    }
    this->manager.set_bypass_mode(true, true);  // start
    // Record original value of config buffer manager
    std::copy(
        this->config_buffer_mgr.begin(),
        this->config_buffer_mgr.begin() + num_sub_devices,
        this->config_buffer_mgr_reset.begin());
    for (uint32_t i = 0; i < num_sub_devices; ++i) {
        // Sync values in the trace need to match up with the counter starting at 0 again.
        this->config_buffer_mgr[i].mark_completely_full(this->expected_num_workers_completed[i]);
    }
}

void HWCommandQueue::record_end() {
    auto& trace_data = this->trace_ctx->data;
    trace_data = std::move(this->manager.get_bypass_data());
    // Add command to terminate the trace buffer
    DeviceCommand command_sequence(hal.get_alignment(HalMemType::HOST));
    command_sequence.add_prefetch_exec_buf_end();
    for (int i = 0; i < command_sequence.size_bytes() / sizeof(uint32_t); i++) {
        trace_data.push_back(((uint32_t*)command_sequence.data())[i]);
    }
    // Reset the expected workers, launch msg buffer state, and config buffer mgr to their original value,
    // so device can run programs after a trace was captured. This is needed since trace capture modifies the state on
    // host, even though device doesn't run any programs.
    auto num_sub_devices = this->device->num_sub_devices();
    std::copy(
        this->expected_num_workers_completed_reset.begin(),
        this->expected_num_workers_completed_reset.begin() + num_sub_devices,
        this->expected_num_workers_completed.begin());
    std::copy(
        this->worker_launch_message_buffer_state_reset.begin(),
        this->worker_launch_message_buffer_state_reset.begin() + num_sub_devices,
        this->manager.get_worker_launch_message_buffer_state().begin());
    std::copy(
        this->config_buffer_mgr_reset.begin(),
        this->config_buffer_mgr_reset.begin() + num_sub_devices,
        this->config_buffer_mgr.begin());

    // Copy the desc keys into a separate vector. When enqueuing traces, we sometimes need to pass sub-device ids
    // separately
    this->trace_ctx->sub_device_ids.reserve(this->trace_ctx->descriptors.size());
    for (const auto& [id, _] : this->trace_ctx->descriptors) {
        auto index = id.to_index();
        this->trace_ctx->sub_device_ids.push_back(id);
    }
    this->tid = std::nullopt;
    this->trace_ctx = nullptr;
    this->manager.set_bypass_mode(false, true);  // stop
}

void HWCommandQueue::terminate() {
    ZoneScopedN("HWCommandQueue_terminate");
    TT_FATAL(!this->manager.get_bypass_mode(), "Terminate cannot be used with tracing");
    tt::log_debug(tt::LogDispatch, "Terminating dispatch kernels for command queue {}", this->id);
    auto command = EnqueueTerminateCommand(this->id, this->device, this->manager);
    this->enqueue_command(command, false, {});
}

WorkerConfigBufferMgr& HWCommandQueue::get_config_buffer_mgr(uint32_t index) { return config_buffer_mgr[index]; }

void HWCommandQueue::reset_config_buffer_mgr(const uint32_t num_entries) {
    for (uint32_t i = 0; i < num_entries; ++i) {
        this->config_buffer_mgr[i] = WorkerConfigBufferMgr();
        program_dispatch::initialize_worker_config_buf_mgr(this->config_buffer_mgr[i]);
    }
}

}  // namespace tt::tt_metal
