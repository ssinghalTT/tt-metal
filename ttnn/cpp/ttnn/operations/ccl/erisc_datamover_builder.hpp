// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/distributed/types.hpp"
#include "umd/device/types/cluster_descriptor_types.h"
#include "cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_types.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_packet_header.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/program_impl.hpp>
#include <tt-metalium/hal_exp.hpp>

#include <vector>
#include <unordered_map>
#include <optional>

namespace ttnn {
namespace ccl {


struct FabricEriscDatamoverConfig {
    static constexpr std::size_t field_size = 16;
    static constexpr std::size_t buffer_alignment = 32;
    static_assert(((buffer_alignment - 1) & buffer_alignment) == 0);

    // Global
    static constexpr std::size_t eth_channel_sync_size = 16;
    std::size_t handshake_addr = tt::tt_metal::experimental::hal::get_erisc_l1_unreserved_base()/* + 1024*/;
    std::size_t edm_channel_ack_addr = handshake_addr + eth_channel_sync_size;
    std::size_t termination_signal_address =
        edm_channel_ack_addr + (2 * eth_channel_sync_size);  // pad extra bytes to match old EDM so handshake logic will still work

    // ----------- Sender Channel 0
    std::size_t sender_channel_0_buffer_index_address = termination_signal_address + field_size;
    std::size_t sender_channel_0_worker_connection_info_address =
        sender_channel_0_buffer_index_address + field_size;
    std::size_t sender_channel_0_local_flow_control_semaphore_address =
        sender_channel_0_worker_connection_info_address + field_size;
    std::size_t sender_channel_0_producer_terminate_connection_address =
        sender_channel_0_local_flow_control_semaphore_address + field_size;
    // persistent mode field
    std::size_t sender_channel_0_connection_semaphore_address =
        sender_channel_0_producer_terminate_connection_address + field_size;
    // persistent mode field
    std::size_t sender_channel_0_buffer_index_semaphore_address =
        sender_channel_0_connection_semaphore_address + field_size;

    static_assert(field_size >= sizeof(tt::fabric::EDMChannelWorkerLocationInfo));

    // ----------- Sender Channel 1
    std::size_t sender_channel_1_buffer_index_address =
        sender_channel_0_buffer_index_semaphore_address + field_size;
    std::size_t sender_channel_1_worker_connection_info_address =
        sender_channel_1_buffer_index_address + field_size;
    std::size_t sender_channel_1_local_flow_control_semaphore_address =
        sender_channel_1_worker_connection_info_address + field_size;
    std::size_t sender_channel_1_producer_terminate_connection_address =
        sender_channel_1_local_flow_control_semaphore_address + field_size;
    // persistent mode field
    std::size_t sender_channel_1_connection_semaphore_address =
        sender_channel_1_producer_terminate_connection_address + field_size;
    // persistent mode field
    std::size_t sender_channel_1_buffer_index_semaphore_address =
        sender_channel_1_connection_semaphore_address + field_size;

    // ----------- Receiver Channel
    std::size_t receiver_channel_local_buffer_index_address =
        sender_channel_1_buffer_index_semaphore_address + field_size;
    // persistent mode field
    std::size_t receiver_channel_downstream_flow_control_semaphore_address =
        receiver_channel_local_buffer_index_address + field_size;

    // Channel Allocations
    std::size_t max_l1_loading_size = tt::tt_metal::experimental::hal::get_erisc_l1_unreserved_size() + tt::tt_metal::experimental::hal::get_erisc_l1_unreserved_base();
    std::size_t buffer_region_start =
        (receiver_channel_downstream_flow_control_semaphore_address + field_size + buffer_alignment) & ~(buffer_alignment - 1); // Align
    std::size_t available_channel_buffering_space =
        max_l1_loading_size - buffer_region_start;

    FabricEriscDatamoverConfig(
        std::size_t channel_buffer_size_bytes, std::size_t sender_ratio_size, std::size_t receiver_ratio_size);

    std::size_t channel_buffer_size_bytes = 0;
    std::size_t channel_buffer_size_bytes_with_channel_sync = 0;
    std::size_t sender_0_channel_size_bytes = 0;
    std::size_t sender_0_num_buffers = 0;
    std::size_t sender_1_channel_size_bytes = 0;
    std::size_t sender_1_num_buffers = 0;
    std::size_t receiver_channel_size_bytes = 0;
    std::size_t receiver_num_buffers = 0;

    std::size_t sender_0_channel_base_address = 0;
    std::size_t sender_1_channel_base_address = 0;
    std::size_t receiver_channel_base_address = 0;
};

struct SenderWorkerAdapterSpec {
    size_t edm_noc_x = 0;
    size_t edm_noc_y = 0;
    size_t edm_buffer_base_addr = 0;
    size_t num_buffers_per_channel = 0;
    size_t edm_l1_sem_addr = 0;
    size_t edm_connection_handshake_addr = 0;
    size_t edm_worker_location_info_addr = 0;  // The EDM's location for `EDMChannelWorkerLocationInfo`
    size_t buffer_size_bytes = 0;
    size_t buffer_index_semaphore_id = 0; // the semaphore ID on the EDM, not the worker
    bool persistent_fabric = false;
};


struct edm_termination_info_t {
    uint32_t distance = 0;
    uint32_t edm_noc_x = 0;
    uint32_t edm_noc_y = 0;
    uint32_t termination_addr = 0;
};

void get_runtime_args_for_edm_termination_infos(std::vector<edm_termination_info_t> const& edm_termination_infos, std::vector<uint32_t>& args_out);
void append_worker_to_fabric_edm_sender_rt_args(
    const SenderWorkerAdapterSpec& connection,
    size_t sender_worker_flow_control_semaphore_id,
    size_t sender_worker_teardown_semaphore_id,
    size_t sender_worker_buffer_index_semaphore_id,
    std::vector<uint32_t>& args_out);
size_t log_worker_to_fabric_edm_sender_rt_args(std::vector<uint32_t> const& args, size_t starting_arg_idx = 0);

class FabricEriscDatamoverBuilder {
   public:
       static constexpr size_t default_firmware_context_switch_interval = 200000;

       FabricEriscDatamoverBuilder(
           const CoreCoord& my_eth_core_logical,
           size_t my_noc_x,
           size_t my_noc_y,
           size_t my_chip_id,
           size_t peer_chip_id,

           std::optional<size_t> receiver_channel_downstream_flow_control_semaphore_id,
           std::optional<size_t> receiver_channel_downstream_teardown_semaphore_id,
           size_t sender_channel_0_flow_control_semaphore_id,
           size_t sender_channel_1_flow_control_semaphore_id,
           size_t sender_channel_0_connection_semaphore_id,
           size_t sender_channel_1_connection_semaphore_id,
           size_t sender_channel_0_buffer_index_semaphore_id,
           size_t sender_channel_1_buffer_index_semaphore_id,

           const FabricEriscDatamoverConfig& config,
           bool enable_persistent_mode,
           bool build_in_worker_connection_mode = false);

       static FabricEriscDatamoverBuilder build(
           tt::tt_metal::IDevice* device,
           tt::tt_metal::Program& program,
           const CoreCoord& ethernet_core,
           chip_id_t local_chip_id,
           chip_id_t peer_chip_id,
           const FabricEriscDatamoverConfig& config,
           bool enable_persistent_mode,
           bool build_in_worker_connection_mode = false);

       [[nodiscard]] SenderWorkerAdapterSpec build_connection_to_worker_channel() const;
       [[nodiscard]] SenderWorkerAdapterSpec build_connection_to_fabric_channel() const;

       [[nodiscard]] std::vector<uint32_t> get_compile_time_args() const;

       [[nodiscard]] std::vector<uint32_t> get_runtime_args() const;

       void connect_to_downstream_edm(const FabricEriscDatamoverBuilder& downstream_edm);

       void dump_to_log() const {
           // TODO
       }

    void teardown_from_host(IDevice*d, tt::fabric::TerminationSignal termination_signal = tt::fabric::TerminationSignal::GRACEFULLY_TERMINATE) const;

    void set_firmware_context_switch_interval(size_t interval);

    //    protected:
    friend class EdmLineFabricOpInterface;
    CoreCoord my_eth_core_logical;
    size_t my_noc_x = 0;
    size_t my_noc_y = 0;

    FabricEriscDatamoverConfig config;

    size_t my_chip_id = 0;
    size_t peer_chip_id = 0;
    size_t handshake_address = 0;
    size_t channel_buffer_size = 0;

    size_t sender_0_num_buffers = 0;
    size_t sender_1_num_buffers = 0;
    size_t receiver_num_buffers = 0;

    size_t local_sender_channel_0_buffer_address = 0;
    size_t local_sender_channel_0_connection_info_addr = 0;
    size_t local_sender_channel_1_buffer_address = 0;
    size_t local_sender_channel_1_connection_info_addr = 0;
    size_t local_receiver_channel_buffer_address = 0;

    size_t termination_signal_ptr = 0;

    // Semaphore IDs
    // this is the receiver channel's local sem for flow controlling with downstream fabric sender
    std::optional<size_t> receiver_channel_downstream_flow_control_semaphore_id;
    std::optional<size_t> receiver_channel_downstream_teardown_semaphore_id;
    size_t sender_channel_0_flow_control_semaphore_id = 0;
    size_t sender_channel_1_flow_control_semaphore_id = 0;
    size_t sender_channel_0_connection_semaphore_id = 0;
    size_t sender_channel_1_connection_semaphore_id = 0;
    size_t sender_channel_0_buffer_index_semaphore_id = 0;
    size_t sender_channel_1_buffer_index_semaphore_id = 0;
    size_t receiver_channel_local_buffer_index_address = 0;

    std::optional<size_t> downstream_edm_noc_x;
    std::optional<size_t> downstream_edm_noc_y;
    std::optional<size_t> downstream_edm_buffer_base_address;
    std::optional<size_t> downstream_edm_semaphore_address;
    std::optional<size_t> downstream_edm_worker_registration_address;
    std::optional<size_t> downstream_edm_worker_location_info_address;
    std::optional<size_t> downstream_sender_channel_buffer_index_semaphore_id;
    bool enable_persistent_mode = false;
    bool build_in_worker_connection_mode = false;
    size_t firmware_context_switch_interval = default_firmware_context_switch_interval;
};



class EdmLineFabricOpInterface {
   public:
    enum Direction {
        // Ascending chips in the sequence
        FORWARD,

        // Descending chips in the sequence
        BACKWARD,
    };


    //   The constructor will assemble/connect the line across the specified device sequence, for all available links.
    EdmLineFabricOpInterface (std::vector<IDevice*> const& device_sequence, std::vector<Program*> const& program_sequence, bool enable_persistent_mode, std::optional<size_t> desired_num_links = std::nullopt, bool build_in_worker_connection_mode = false);

    // Invocable per chip if we want to collectively build the fabric by building this separately per chip
    // (and implicitly building the fabric that way)
    EdmLineFabricOpInterface (IDevice* local_device, std::optional<IDevice*> forward_device, std::optional<IDevice*> backward_device,  Program* program, bool enable_persistent_mode, std::optional<size_t> desired_num_links, bool build_in_worker_connection_mode = false);

    static EdmLineFabricOpInterface build_program_builder_worker_connection_fabric(std::vector<IDevice*> const& device_sequence, std::vector<Program*> const& program_sequence, bool enable_persistent_mode, std::optional<size_t> desired_num_links = std::nullopt);
    static EdmLineFabricOpInterface build_program_builder_worker_connection_fabric(IDevice* local_device, std::optional<IDevice*> forward_device, std::optional<IDevice*> backward_device,  Program* program, bool enable_persistent_mode, std::optional<size_t> desired_num_links = std::nullopt);

    // Will create a connection adapter for a worker which can be used to pass args to the worker kernel talking to the
    // corresponding fabric endpoint. This interface will guarantee unique connections only so requesting more unique connections
    // than available will result in an error.
    SenderWorkerAdapterSpec uniquely_connect_worker(tt::tt_metal::IDevice* device, Direction direction);

    // builds the ethernet kernels for all EDMs in the "fabric"
    void build_kernels() const;

    // Generates a list of target cores (for now assumed from chip 0 in the line) from farthest
    // to nearest for the sake of sending teardown/termination signals on workload completion.
    // Returns: A list of termination infos which can be passed to a terminate kernel
    // Note there is currently a small bug in that with multiple links, we don't currently know
    // who will be sending the termination signals (and which link(s) they are connected to)
    // and so a termination signal may be sent to our link first before the other eth core links
    // on the chip so multi-link isn't officially supported yet
    std::vector<edm_termination_info_t> generate_ordered_termination_info_farthest_to_nearest() const;

    // Generates a list of termination infos for the local chip's EDMs
    std::vector<edm_termination_info_t> generate_local_chip_fabric_termination_infos(IDevice*device) const;

    // Accessors
    size_t get_num_links() const { return num_links; }

    size_t get_device_count() const { return device_sequence.size(); }

    size_t get_index_of_device(IDevice*device) const {
        for (size_t i = 0; i < device_sequence.size(); i++) {
            if (device_sequence[i] == device) {
                return i;
            }
        }
        TT_THROW("Device {} not found in device sequence of line fabric", device->id());
        return -1;
    }

    size_t get_edm_buffer_size_bytes() const { return buffer_size_bytes; }

    void teardown_from_host(tt::fabric::TerminationSignal termination_signal = tt::fabric::TerminationSignal::GRACEFULLY_TERMINATE) const;

    static void launch_mesh_fabric(MeshDevice *mesh_device);
    static void teardown_edm_fabric(MeshDevice *mesh_device);

    void set_firmware_context_switch_interval(size_t interval);

    // Device ID -> EDM Builders
    std::unordered_map<size_t, std::vector<FabricEriscDatamoverBuilder>> edm_builders_forward_direction;
    std::unordered_map<size_t, std::vector<FabricEriscDatamoverBuilder>> edm_builders_backward_direction;
   private:

    // Device ID -> link index
    std::unordered_map<size_t, size_t> next_forward_direction_edm_available;
    std::unordered_map<size_t, size_t> next_backward_direction_edm_available;

    std::vector<IDevice*> device_sequence;
    std::vector<Program*> programs;

    size_t num_links;
    size_t buffer_size_bytes;
    size_t firmware_context_switch_interval = FabricEriscDatamoverBuilder::default_firmware_context_switch_interval;
};

void initialize_edm_fabric(distributed::MeshDevice* mesh_device);
void teardown_edm_fabric(distributed::MeshDevice* mesh_device);

};  // namespace ccl
};  // namespace ttnn
