#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compile_time_args.h"
#include "hostdevcommon/kernel_structs.h"

namespace NAMESPACE {
void MAIN {
    uint8_t src0_cb_index = tt::CBIndex::c_0;
    uint8_t src1_cb_index = tt::CBIndex::c_1;
    uint8_t src2_cb_index = tt::CBIndex::c_2;
    uint8_t dst0_cb_index = tt::CBIndex::c_3;
    uint8_t dst1_cb_index = tt::CBIndex::c_4;
    uint32_t num_shards_per_core = get_arg_val<uint32_t>(0);
    uint32_t num_tiles_per_shard = get_arg_val<uint32_t>(1);

    for (uint32_t i = 0; i < num_shards_per_core; i++) {
        for (uint32_t j = 0; j < num_tiles_per_shard; j++) {
            acquire_dst();
            cb_wait_front(src0_cb_index, 1);
            cb_wait_front(src1_cb_index, 1);
            binary_op_init_common(src0_cb_index, src1_cb_index, dst0_cb_index);
            mul_tiles_init();
            mul_tiles(src0_cb_index, src1_cb_index, 0, 0, 0);

            cb_reserve_back(dst0_cb_index, 1);
            pack_tile(0, dst0_cb_index);
            cb_push_back(dst0_cb_index, 1);

            cb_pop_front(src0_cb_index, 1);
            cb_pop_front(src1_cb_index, 1);
            release_dst();

            acquire_dst();
            cb_wait_front(dst0_cb_index, 1);
            cb_wait_front(src2_cb_index, 1);
            binary_op_init_common(dst0_cb_index, src2_cb_index, dst1_cb_index);
            add_tiles_init();
            add_tiles(dst0_cb_index, src2_cb_index, 0, 0, 0);

            cb_reserve_back(dst1_cb_index, 1);
            pack_tile(0, dst1_cb_index);
            cb_push_back(dst1_cb_index, 1);

            cb_pop_front(dst0_cb_index, 1);
            cb_pop_front(src2_cb_index, 1);
            release_dst();
        }
    }
};
}  // namespace NAMESPACE
