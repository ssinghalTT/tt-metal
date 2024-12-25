#include "dataflow_api.h"
#include "hostdevcommon/kernel_structs.h"

void kernel_main() {
    uint8_t src0_cb_index = tt::CBIndex::c_0;
    uint8_t src1_cb_index = tt::CBIndex::c_1;
    uint8_t src2_cb_index = tt::CBIndex::c_2;
    uint32_t src0Addr = get_arg_val<uint32_t>(0);
    uint32_t src1Addr = get_arg_val<uint32_t>(1);
    uint32_t src2Addr = get_arg_val<uint32_t>(2);

    uint32_t num_shards_per_core = get_arg_val<uint32_t>(3);
    uint32_t num_tiles_per_shard = get_arg_val<uint32_t>(4);
    uint32_t num_shards_written = get_arg_val<uint32_t>(5);

    const uint32_t single_tile_size_bytes = get_tile_size(src0_cb_index);

    const InterleavedAddrGenFast<true> a = {
        .bank_base_address = src0Addr,
        .page_size = single_tile_size_bytes,
        .data_format = DataFormat::Float16_b,
    };
    const InterleavedAddrGenFast<true> b = {
        .bank_base_address = src1Addr,
        .page_size = single_tile_size_bytes,
        .data_format = DataFormat::Float16_b,
    };
    const InterleavedAddrGenFast<true> c = {
        .bank_base_address = src2Addr,
        .page_size = single_tile_size_bytes,
        .data_format = DataFormat::Float16_b,
    };

    for (uint32_t i = 0; i < num_shards_per_core; i++) {
        uint32_t start_id = (i + num_shards_written) * num_tiles_per_shard;
        uint32_t end_id = start_id + num_tiles_per_shard;
        for (uint32_t j = start_id; j < end_id; j++) {
            cb_reserve_back(src0_cb_index, 1);
            cb_reserve_back(src1_cb_index, 1);
            cb_reserve_back(src2_cb_index, 1);

            uint32_t l1_write_addr_in0 = get_write_ptr(src0_cb_index);
            uint32_t l1_write_addr_in1 = get_write_ptr(src1_cb_index);
            uint32_t l1_write_addr_in2 = get_write_ptr(src2_cb_index);

            noc_async_read_tile(j, a, l1_write_addr_in0);
            noc_async_read_tile(j, b, l1_write_addr_in1);
            noc_async_read_tile(j, c, l1_write_addr_in2);

            noc_async_read_barrier();

            cb_push_back(src0_cb_index, 1);
            cb_push_back(src1_cb_index, 1);
            cb_push_back(src2_cb_index, 1);
        }
    }
}
