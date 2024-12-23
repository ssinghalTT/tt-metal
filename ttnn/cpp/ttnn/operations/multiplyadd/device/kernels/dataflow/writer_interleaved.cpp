#include "dataflow_api.h"
#include "hostdevcommon/kernel_structs.h"

void kernel_main() {
    uint8_t dst1_cb_index = tt::CBIndex::c_4;
    uint32_t dstAddr = get_arg_val<uint32_t>(0);
    uint32_t batch = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    uint32_t end_id = start_id + batch;
    const uint32_t single_tile_size_bytes = get_tile_size(dst1_cb_index);

    const InterleavedAddrGenFast<true> c = {
        .bank_base_address = dstAddr,
        .page_size = single_tile_size_bytes,
        .data_format = DataFormat::Float16_b,
    };

    for (uint32_t i = start_id; i < end_id; i++) {
        cb_wait_front(dst1_cb_index, 1);
        uint32_t l1_read_addr = get_read_ptr(dst1_cb_index);
        noc_async_write_tile(i, c, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(dst1_cb_index, 1);
    }
}
