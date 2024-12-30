// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dispatch/kernel_config/fd_kernel.hpp"
#include "tt_metal/impl/device/device.hpp"

#define DISPATCH_MAX_UPSTREAM 4
#define DISPATCH_MAX_DOWNSTREAM 4

namespace tt::tt_metal::dispatch {

typedef struct {
    int id;
    chip_id_t device_id;                          // Device that this kernel is located on
    chip_id_t servicing_device_id;                // Remote device that this kernel services, used for kernels on MMIO
    uint8_t cq_id;                                // CQ this kernel implements
    DispatchWorkerType kernel_type;               // Type of dispatch kernel this is
    int upstream_ids[DISPATCH_MAX_UPSTREAM];      // Upstream dispatch kernels
    int downstream_ids[DISPATCH_MAX_DOWNSTREAM];  // Downstream dispatch kernels
    NOC my_noc;                                   // NOC this kernel uses to dispatch kernels
    NOC upstream_noc;                             // NOC used to communicate upstream
    NOC downstream_noc;                           // NOC used to communicate downstream
} dispatch_kernel_node_t;

// Create the graph given the nodes
template <typename Generator = DefaultFDKernelGenerator>
std::vector<FDKernel*> connect_fd_graph_edges(std::vector<dispatch_kernel_node_t>& nodes) {
    std::vector<FDKernel*> node_id_to_kernel;

    // Read the input table, create configs for each node
    for (const auto& node : nodes) {
        TT_ASSERT(node_id_to_kernel.size() == node.id);
        node_id_to_kernel.push_back(Generator().Generate(
            node.id,
            node.device_id,
            node.servicing_device_id,
            node.cq_id,
            {node.my_noc, node.upstream_noc, node.downstream_noc},
            static_cast<uint32_t>(node.kernel_type)));
    }

    // Connect the graph with upstream/downstream kernels
    for (const auto& node : nodes) {
        for (int idx = 0; idx < DISPATCH_MAX_UPSTREAM; idx++) {
            if (node.upstream_ids[idx] >= 0) {
                node_id_to_kernel.at(node.id)->AddUpstreamKernel(node_id_to_kernel.at(node.upstream_ids[idx]));
            }
        }
        for (int idx = 0; idx < DISPATCH_MAX_DOWNSTREAM; idx++) {
            if (node.downstream_ids[idx] >= 0) {
                node_id_to_kernel.at(node.id)->AddDownstreamKernel(node_id_to_kernel.at(node.downstream_ids[idx]));
            }
        }
    }

    return node_id_to_kernel;
}

// Create FD kernels for all given device ids. Creates all objects, but need to call create_and_compile_cq_program() use
// a created Device to fill out the settings.
std::vector<FDKernel*> populate_fd_kernels(const std::set<chip_id_t>& device_ids, uint32_t num_hw_cqs);

// Fill out all settings for FD kernels on the given device, and add them to a Program and return it.
std::unique_ptr<tt::tt_metal::Program> create_and_compile_cq_program(
    tt::tt_metal::Device* device, std::vector<FDKernel*>& node_id_to_kernel);

// Performa additional configuration (writing to specific L1 addresses, etc.) for FD kernels on this device.
void configure_dispatch_cores(tt::tt_metal::Device* device, std::vector<FDKernel*>& node_to_to_kernel);

};  // namespace tt::tt_metal::dispatch
