#include <cstdint>
#include "dataflow_api.h"
#include "hostdevcommon/kernel_structs.h"

using namespace tt;

void kernel_main() {
    constexpr uint32_t oneTile = 1;

    uint8_t src0_cb_index = tt::CBIndex::c_0;
    uint8_t src1_cb_index = tt::CBIndex::c_1;
    uint8_t src2_cb_index = tt::CBIndex::c_2;
    uint32_t src0Addr = get_arg_val<uint32_t>(0);
    uint32_t src1Addr = get_arg_val<uint32_t>(1);
    uint32_t src2Addr = get_arg_val<uint32_t>(2);
    uint32_t batch = get_arg_val<uint32_t>(3);
    uint32_t tileId = get_arg_val<uint32_t>(4);
    const uint32_t single_tile_size_bytes = get_tile_size(src0_cb_index);
    const DataFormat data_format = get_dataformat(src0_cb_index);
    const InterleavedAddrGenFast<true> s0 = {
        .bank_base_address = src0Addr, .page_size = single_tile_size_bytes, .data_format = data_format};
    const InterleavedAddrGenFast<true> s1 = {
        .bank_base_address = src1Addr, .page_size = single_tile_size_bytes, .data_format = data_format};
    const InterleavedAddrGenFast<true> s2 = {
        .bank_base_address = src2Addr, .page_size = single_tile_size_bytes, .data_format = data_format};
    for (uint32_t i = 0; i < batch; i++) {
        cb_reserve_back(src0_cb_index, oneTile);
        cb_reserve_back(src1_cb_index, oneTile);
        cb_reserve_back(src2_cb_index, oneTile);

        uint32_t l1_write_addr_in0 = get_write_ptr(src0_cb_index);
        uint32_t l1_write_addr_in1 = get_write_ptr(src1_cb_index);
        uint32_t l1_write_addr_in2 = get_write_ptr(src2_cb_index);

        noc_async_read_tile(tileId + i, s0, l1_write_addr_in0);
        noc_async_read_barrier();
        l1_write_addr_in0 += single_tile_size_bytes;
        cb_push_back(src0_cb_index, oneTile);

        noc_async_read_tile(tileId + i, s1, l1_write_addr_in1);
        noc_async_read_barrier();
        l1_write_addr_in1 += single_tile_size_bytes;
        cb_push_back(src1_cb_index, oneTile);

        noc_async_read_tile(tileId + i, s2, l1_write_addr_in2);
        noc_async_read_barrier();
        l1_write_addr_in2 += single_tile_size_bytes;
        cb_push_back(src2_cb_index, oneTile);
    }
}
