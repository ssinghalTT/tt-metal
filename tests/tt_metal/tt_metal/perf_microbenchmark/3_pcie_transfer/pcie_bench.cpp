// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <future>
#include <numeric>
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
#include "program_impl.hpp"
#include "tt_cluster.hpp"
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
public:
    static constexpr std::string_view k_PcieBenchKernel =
        "tests/tt_metal/tt_metal/perf_microbenchmark/3_pcie_transfer/kernels/pcie_bench.cpp";

    struct PcieTransferResults {
        std::chrono::duration<double> host_hugepage_writing_duration;
        int64_t host_hugepage_bytes_processed;

        std::chrono::duration<double> host_wait_for_kernels_duration;

        std::chrono::duration<double> kernel_duration;
        int64_t kernel_bytes_rd;
        int64_t kernel_bytes_wr;
    };

    // Mini Mem Map
    struct DeviceAddresses {
        uint32_t cycles;
        uint32_t rd_bytes;
        uint32_t unreserved;
    };

    // Device under test
    IDevice* device;

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
        addrs.unreserved = align_addr(addrs.rd_bytes + sizeof(uint32_t), l1_alignment);
        return addrs;
    }

    template <bool repeating_src_vector>
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

            if constexpr (!repeating_src_vector) {
                src_addr += page_size;
            }

            // This may exceed the maximum hugepage
            if (hugepage_addr >= hugepage_end) {
                hugepage_addr = reinterpret_cast<uint64_t>(hugepage_base);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    }

    std::optional<CoreRange> ConfigureReaderKernels(
        Program& program,
        const DeviceAddresses& dev_addrs,
        uint32_t start_y,
        uint32_t num_readers,
        uint32_t total_size,
        uint32_t page_size,
        uint32_t pcie_size,
        uint32_t pcie_offset = 0) const {
        if (!page_size) {
            page_size = total_size;
        }
        if (!num_readers) {
            return {};
        }

        const auto grid_size = device->logical_grid_size();
        const auto max_x = grid_size.x;
        const auto max_y = grid_size.y;

        // Number readers either less than one row
        // or a multiple of the rows
        CoreCoord start_coord{0, start_y};
        CoreCoord end_coord;
        if (num_readers <= max_x) {
            end_coord.x = start_coord.x + num_readers - 1;
            end_coord.y = start_coord.y;
        } else {
            const auto number_of_rows = num_readers / max_x;
            const auto last_row_width = (num_readers % max_x) ? num_readers % max_x : max_x;
            end_coord.x = start_coord.x + last_row_width - 1;
            end_coord.y = number_of_rows - 1;
        }
        CoreRange core_range{start_coord, end_coord};

        [[maybe_unused]] KernelHandle read_kernel = CreateKernel(
            program,
            std::string{k_PcieBenchKernel},
            core_range,
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

        return core_range;
    }

    MemCpyPcieBench() : benchmark::Fixture{} {
        UseManualTime();
        Iterations(3);
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

    template <bool caching_src_vector = false>
    PcieTransferResults HostHP_Perf_ReaderKernelsImpl(benchmark::State& state) {
        const auto total_size = state.range(0);
        const auto page_size = state.range(1);
        const auto pages = total_size / page_size;
        const auto num_readers = state.range(2);
        auto src_data = this->GenSrcData(total_size, sizeof(__m256i));
        const auto pcie_size = this->GetHostHugePageSize();
        const auto pcie_base = this->GetHostHugePage(0);
        const auto pcie_end = reinterpret_cast<uint64_t>(pcie_base) + pcie_size;
        const auto dev_addrs = this->GetDevAddrMap();
        PcieTransferResults results;

        auto program = Program();
        auto configured_readers =
            ConfigureReaderKernels(program, dev_addrs, 0, num_readers, total_size, page_size, pcie_size);

        // IO caused by wait_until_cores_done should not affect the results much
        std::future<std::chrono::duration<double>> kernel_waiting_duration = std::async([&]() {
            auto launch_start = std::chrono::high_resolution_clock::now();
            detail::LaunchProgram(this->device, program, true);
            auto launch_end = std::chrono::high_resolution_clock::now();
            return std::chrono::duration_cast<std::chrono::duration<double>>(launch_start - launch_end);
        });

        auto hp_duration = HostWriteHP<caching_src_vector>(pcie_base, pcie_size, src_data, total_size, page_size);

        if (configured_readers.has_value()) {
            results.host_wait_for_kernels_duration = kernel_waiting_duration.get();

            auto dev_cycles = this->GetWordsFromDevice(configured_readers.value(), dev_addrs.cycles);
            auto dev_bytes_read = this->GetWordsFromDevice(configured_readers.value(), dev_addrs.rd_bytes);
            auto dev_clk = tt::Cluster::instance().get_device_aiclk(device->id()) * 1e6;  // Hz

            double all_cores_cycles = std::reduce(dev_cycles.begin(), dev_cycles.end());
            double all_cores_bytes_read = std::reduce(dev_bytes_read.begin(), dev_bytes_read.end());
            std::chrono::duration<double> kernel_duration{all_cores_cycles / dev_clk};

            results.kernel_duration = kernel_duration;
            results.kernel_bytes_rd = all_cores_bytes_read;
        }

        results.host_hugepage_writing_duration = hp_duration;
        results.host_hugepage_bytes_processed = total_size;

        return results;
    }
};

//
// BM_HostHP_N_Readers
// - Host copying various sizes of data to hugepage using a cached vector
// Benchmark host copying various sizes of data to hugepage with cached vector and no reader kernels
// Reports Host B/s
//
BENCHMARK_DEFINE_F(MemCpyPcieBench, BM_HostHP_N_Readers)(benchmark::State& state) {
    const auto total_size = state.range(0);
    const auto cached_vector = static_cast<bool>(state.range(3));
    double total_device_time = 0;
    double total_device_bytes = 0;
    double total_iteration_time = 0;
    for (auto _ : state) {
        PcieTransferResults res;
        if (cached_vector) {
            res = this->HostHP_Perf_ReaderKernelsImpl<true>(state);
        } else {
            res = this->HostHP_Perf_ReaderKernelsImpl<false>(state);
        }
        state.SetIterationTime(res.host_hugepage_writing_duration.count());
        total_device_time += res.kernel_duration.count();
        total_device_bytes += res.kernel_bytes_rd;
        total_iteration_time += res.host_hugepage_writing_duration.count();
    }

    if (!total_device_time) {
        // No division by 0
        total_device_time = 1;
    }

    state.SetBytesProcessed(total_size * state.iterations());
    state.counters["device_bandwidth"] = benchmark::Counter(
        (total_device_bytes / total_device_time) *
            total_iteration_time,  // Multiply by total_iteration_time to negate kIsRate
        benchmark::Counter::kIsRate,
        benchmark::Counter::kIs1024);
}

BENCHMARK_REGISTER_F(MemCpyPcieBench, BM_HostHP_N_Readers)
    ->Name("Host_Write_HP_N_Readers_Cached_Vector")
    ->ArgsProduct({
        /*Total Size*/ {512_MB, 1_GB},
        /*Page Size*/ {32_KB},
        /*N Reader Kernels*/ {0, 1, 4, 8, 16},
        /*Cached Vector*/ {1},
    });

BENCHMARK_REGISTER_F(MemCpyPcieBench, BM_HostHP_N_Readers)
    ->Name("Host_Write_HP_N_Readers_Uncached_Vector")
    ->ArgsProduct({
        /*Total Size*/ {512_MB, 1_GB},
        /*Page Size*/ {32_KB},
        /*N Reader Kernels*/ {0, 1, 4, 8, 16},
        /*Cached Vector*/ {0},
    });

BENCHMARK_MAIN();
