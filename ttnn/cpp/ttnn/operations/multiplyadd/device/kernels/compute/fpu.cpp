#include <cstdint>
#include "compile_time_args.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "hostdevcommon/kernel_structs.h"
// #include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    // DPRINT_MATH(DPRINT << "FPU" << ENDL());
    constexpr uint32_t dst_reg = 0;

    uint8_t src0_cb_index = tt::CBIndex::c_0;
    uint8_t src1_cb_index = tt::CBIndex::c_1;
    uint8_t src2_cb_index = tt::CBIndex::c_2;
    uint8_t dst0_cb_index = tt::CBIndex::c_3;
    uint8_t dst1_cb_index = tt::CBIndex::c_4;

    uint32_t batch = get_arg_val<uint32_t>(0);

    // DPRINT_MATH(DPRINT << "FPU BATCH " << batch << ENDL());
    for (uint32_t i = 0; i < batch; i++) {
        acquire_dst();
        cb_wait_front(src0_cb_index, 1);
        cb_wait_front(src1_cb_index, 1);
        // DPRINT << i << ENDL() << TSLICE(src0_cb_index, 0, SliceRange::hw0_32_16()) << ENDL();
        // DPRINT << i << ENDL() << TSLICE(src1_cb_index, 0, SliceRange::hw0_32_16()) << ENDL();
        binary_op_init_common(src0_cb_index, src1_cb_index, dst0_cb_index);
        mul_tiles_init();
        mul_tiles(src0_cb_index, src1_cb_index, 0, 0, dst_reg);

        cb_reserve_back(dst0_cb_index, 1);
        pack_tile(dst_reg, dst0_cb_index);
        // DPRINT << i << ENDL() << TSLICE(dst1_cb_index, 0, SliceRange::hw0_32_16()) << ENDL();
        cb_push_back(dst0_cb_index, 1);

        cb_pop_front(src0_cb_index, 1);
        cb_pop_front(src1_cb_index, 1);
        release_dst();

        acquire_dst();
        cb_wait_front(dst0_cb_index, 1);
        cb_wait_front(src2_cb_index, 1);
        binary_op_init_common(dst0_cb_index, src2_cb_index, dst1_cb_index);
        add_tiles_init();
        add_tiles(dst0_cb_index, src2_cb_index, 0, 0, dst_reg);

        cb_reserve_back(dst1_cb_index, 1);
        pack_tile(dst_reg, dst1_cb_index);
        // DPRINT << i << ENDL() << TSLICE(dst1_cb_index, 0, SliceRange::hw0_32_16()) << ENDL();
        cb_push_back(dst1_cb_index, 1);

        cb_pop_front(dst0_cb_index, 1);
        cb_pop_front(src2_cb_index, 1);
        release_dst();
    }
    // DPRINT_MATH(DPRINT << "END COMPUTE" << ENDL());
    // DPRINT_PACK(DPRINT << "END PACK" << ENDL());
};
}  // namespace NAMESPACE
