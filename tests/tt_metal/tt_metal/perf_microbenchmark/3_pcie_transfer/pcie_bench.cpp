// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <future>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/hal_exp.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/rtoptions.hpp>
#include <tt-metalium/memcpy.hpp>
#include <tt-metalium/helpers.hpp>
#include "bfloat16.hpp"
#include "buffer.hpp"
#include "core_coord.hpp"
#include "data_types.hpp"
#include "hal.hpp"
#include "host_api.hpp"
#include "kernel_types.hpp"
#include "umd/device/types/cluster_descriptor_types.h"
#include <benchmark/benchmark.h>
#include <cstdint>

using namespace tt::tt_metal;
using namespace tt::tt_metal::dispatch;  // _KB, _MB, _GB
using namespace std::chrono_literals;    // s

// Uses low level APIs to benchmark Pcie transfer.
// Fast dispatch needs to be disabled because this benchmark will write into hugepage.
// For better benchmark outputs, run it with TT_METAL_LOGGER_LEVEL=FATAL
class MemCpyPcieBench : public benchmark::Fixture {
private:
    static constexpr std::string_view k_PcieBenchKernel =
        "tests/tt_metal/tt_metal/perf_microbenchmark/3_pcie_transfer/kernels/pcie_bench.cpp";

    struct PcieTransferResults {
        // Host Results
        std::chrono::duration<double> host_seconds;
        std::chrono::duration<double> program_wait_seconds;
        int64_t host_wr_bytes;

        // Device Results
        std::vector<int64_t> dev_cycles;
        std::vector<int64_t> dev_rd_bytes;
        std::vector<int64_t> dev_wr_bytes;
    };

    // Mini Mem Map
    struct DeviceAddresses {
        uint32_t cycles;
        uint32_t rd_bytes;
        uint32_t wr_bytes;
        uint32_t unreserved;
    };

    // Get pointer to the host hugepage
    void* GetHostHugePage(uint32_t base_offset) const {
        const auto dut_id = this->device->id();  // device under test
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(dut_id);
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(dut_id);
        return (void*)((uint64_t)(tt::Cluster::instance().host_dma_address(0, mmio_device_id, channel)) + base_offset);
    }

    // Get size of the host hugepage
    uint32_t GetHostHugePageSize() const {
        const auto dut_id = this->device->id();
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(dut_id);
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(dut_id);
        return tt::Cluster::instance().get_host_channel_size(mmio_device_id, channel);
    }

    std::vector<uint32_t> GenSrcData(uint32_t num_bytes, uint32_t host_alignment) const {
        std::vector<uint32_t> vec = create_random_vector_of_bfloat16(
            num_bytes + 2 * host_alignment, 1234, std::chrono::system_clock::now().time_since_epoch().count());

        return vec;
    }

    std::vector<int64_t> GetWordsFromDevice(const CoreRange& core, uint32_t addr) {
        std::vector<int64_t> data;
        for (int xi = core.start_coord.x; xi <= core.end_coord.x; xi++) {
            for (int yi = core.start_coord.y; yi <= core.end_coord.y; yi++) {
                std::vector<uint32_t> single_data;
                detail::ReadFromDeviceL1(device, CoreCoord{xi, yi}, addr, sizeof(uint32_t), single_data);
                data.push_back(single_data[0]);
            }
        }
        return data;
    }

    DeviceAddresses GetDevAddrMap() const {
        const auto l1_alignment = hal.get_alignment(HalMemType::L1);
        DeviceAddresses addrs;
        addrs.cycles = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::UNRESERVED);
        addrs.rd_bytes = align_addr(addrs.cycles + sizeof(uint32_t), l1_alignment);
        addrs.wr_bytes = align_addr(addrs.rd_bytes + sizeof(uint32_t), l1_alignment);
        addrs.unreserved = align_addr(addrs.wr_bytes + sizeof(uint32_t), l1_alignment);
        return addrs;
    }

    std::chrono::duration<double> HostWriteHP(
        void* hugepage_base,
        uint32_t hugepage_size,
        const std::vector<uint32_t>& src_data,
        size_t total_size,
        size_t page_size) {
        uint64_t hugepage_addr = reinterpret_cast<uint64_t>(hugepage_base);
        uint64_t hugepage_end = hugepage_addr + hugepage_size;
        uint64_t src_addr = align_addr(reinterpret_cast<uint64_t>(src_data.data()), sizeof(__m256i));
        size_t num_pages;
        if (!page_size) {
            num_pages = 1;
            page_size = total_size;
        } else {
            num_pages = total_size / page_size;
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_pages; i++) {
            memcpy_to_device<false /*fence*/>((void*)(hugepage_addr), (void*)(src_addr), page_size);

            hugepage_addr += page_size;
            src_addr += page_size;

            // This may exceed the maximum hugepage
            if (hugepage_addr >= hugepage_end) {
                hugepage_addr = reinterpret_cast<uint64_t>(hugepage_base);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    }

    void ConfigureReaderKernels(
        Program& program,
        const DeviceAddresses& dev_addrs,
        const CoreRange& reader_range,
        uint32_t total_size,
        uint32_t page_size,
        uint32_t pcie_size,
        uint32_t pcie_offset = 0) const {
        if (!page_size) {
            page_size = total_size;
        }
        [[maybe_unused]] KernelHandle read_kernel = CreateKernel(
            program,
            std::string{k_PcieBenchKernel},
            reader_range,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC_0,
                .compile_args =
                    {
                        dev_addrs.unreserved,  // my_rd_dst_addr
                        pcie_offset,  // pcie_rd_base. From the device's perspective the pcie base is 0. device bar is
                                      // mapped to hugepage.
                        pcie_size,    // pcie_rd_size
                        page_size,    // pcie_rd_transfer_size
                        dev_addrs.rd_bytes,  // my_bytes_rd_addr

                        0,  // my_wr_src_addr
                        0,  // pcie_wr_base
                        0,  // pcie_wr_size
                        0,  // pcie_wr_transfer_size
                        0,  // my_bytes_wr_addr

                        total_size,        // total_bytes
                        dev_addrs.cycles,  // cycles
                    },
                .defines = {},
            });
    }

public:
    // Device under test
    IDevice* device;

    MemCpyPcieBench() : benchmark::Fixture{} {
        UseManualTime();
        // Iterations(10);
    }

    void SetUp(benchmark::State& state) override {
        const chip_id_t target_device_id = 0;
        this->device = CreateDevice(target_device_id, 1);
        if (!this->device->is_mmio_capable()) {
            state.SkipWithMessage("MemCpyPcieBench can only be run on a MMIO capable device");
        }

        if (this->device->using_fast_dispatch()) {
            state.SkipWithMessage(
                "MemCpyPcieBench can only be run with slow dispatch enabled. It conflicts with fast dispatch because "
                "it needs to read/write into HugePages");
        }
    }

    void TearDown(const benchmark::State& state) override {
        tt::DevicePool::instance().close_device(this->device->id());
    }

    void BenchmarkDefineFImpl(benchmark::State& state) {
        const auto total_size = state.range(0);
        const auto page_size = state.range(1);
        const auto num_readers = state.range(2);

        int64_t program_wait_time = 0;
        int64_t host_bytes = 0;
        int64_t dev_cycles = 0;  // All cores
        int64_t dev_bytes = 0;   // All cores

        for (auto _ : state) {
            PcieTransferResults result = this->BenchPcieTransfer(total_size, page_size, num_readers);

            state.SetIterationTime(result.host_seconds.count());

            host_bytes += result.host_wr_bytes;
            program_wait_time += result.program_wait_seconds.count() * 1000;
            dev_cycles += std::reduce(result.dev_cycles.begin(), result.dev_cycles.end());
            dev_bytes += std::reduce(result.dev_rd_bytes.begin(), result.dev_rd_bytes.end());
            dev_bytes += std::reduce(result.dev_wr_bytes.begin(), result.dev_wr_bytes.end());
        }

        state.SetBytesProcessed(host_bytes);
        state.counters["program_wait_time_ms"] = program_wait_time;
        state.counters["host_bytes"] = host_bytes;
        state.counters["dev_cycles"] = dev_cycles;
        state.counters["dev_bytes"] = dev_bytes;
    }

    PcieTransferResults BenchPcieTransfer(
        const uint32_t total_size, const uint32_t page_size, const uint32_t num_readers) {
        auto src_data = this->GenSrcData(total_size, sizeof(__m256i));
        const auto pages = total_size / page_size;
        const auto pcie_size = this->GetHostHugePageSize();
        const auto pcie_base = this->GetHostHugePage(0);
        const auto pcie_end = reinterpret_cast<uint64_t>(pcie_base) + pcie_size;

        uint32_t bytes_transferred = 0;
        uint32_t pcie_rd_offset = 0;
        uint32_t pcie_wr_offset = 0;

        // Device
        CoreCoord reader_start{0, 0};
        CoreCoord reader_end{reader_start.x, reader_start.y + num_readers - 1};
        CoreRange reader_range{reader_start, reader_end};
        const auto dev_addrs = this->GetDevAddrMap();

        auto program = Program();
        if (reader_range.size()) {
            ConfigureReaderKernels(program, dev_addrs, reader_range, total_size, page_size, pcie_size);
        }

        // IO caused by wait_until_cores_done should not affect the results much
        std::future<std::chrono::duration<double>> device_elapsed_seconds = std::async([&]() {
            auto launch_start = std::chrono::high_resolution_clock::now();
            detail::LaunchProgram(this->device, program, true);
            auto launch_end = std::chrono::high_resolution_clock::now();
            return std::chrono::duration_cast<std::chrono::duration<double>>(launch_start - launch_end);
        });

        auto hp_duration = HostWriteHP(pcie_base, pcie_size, src_data, total_size, page_size);
        auto program_duration = device_elapsed_seconds.get();

        return {
            .host_seconds = hp_duration,
            .program_wait_seconds = program_duration,
            .host_wr_bytes = total_size,  // bytes_transferred,
            .dev_cycles = this->GetWordsFromDevice(reader_range, dev_addrs.cycles),
            .dev_rd_bytes = this->GetWordsFromDevice(reader_range, dev_addrs.rd_bytes),
            .dev_wr_bytes = {},
        };
    }
};

BENCHMARK_DEFINE_F(MemCpyPcieBench, BM_HostWrite_DeviceRead)(benchmark::State& state) { BenchmarkDefineFImpl(state); }

BENCHMARK_REGISTER_F(MemCpyPcieBench, BM_HostWrite_DeviceRead)
    ->ArgsProduct({
        {512_MB, 1_GB},        // Total
        {8_KB, 16_KB, 32_KB},  // Page Size
        {1, 4},                // Number of Reader kernels
        {0},                   // Number of Writer kernels
        {0},                   // Split Kernels
    });

BENCHMARK_MAIN();
