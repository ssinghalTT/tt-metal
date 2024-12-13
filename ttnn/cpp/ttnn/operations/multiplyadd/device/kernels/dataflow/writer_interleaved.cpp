#include <cstdint>
#include "dataflow_api.h"
#include "hostdevcommon/kernel_structs.h"

using namespace tt;

void kernel_main() {
    constexpr uint32_t oneTile = 1;

    uint8_t dst1_cb_index = tt::CBIndex::c_4;
    uint32_t dstAddr = get_arg_val<uint32_t>(0);
    uint32_t batch = get_arg_val<uint32_t>(1);
    uint32_t tileId = get_arg_val<uint32_t>(2);
    const uint32_t single_tile_size_bytes = get_tile_size(dst1_cb_index);
    const DataFormat data_format = get_dataformat(dst1_cb_index);
    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = dstAddr, .page_size = single_tile_size_bytes, .data_format = data_format};

    for (uint32_t i = 0; i < batch; i++) {
        cb_wait_front(dst1_cb_index, oneTile);
        uint32_t l1_read_addr = get_read_ptr(dst1_cb_index);
        noc_async_write_tile(tileId + i, s, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(dst1_cb_index, oneTile);
    }
}
