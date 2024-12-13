#include <cstdint>

namespace NAMESPACE {
void MAIN {
    constexpr int onetile = 1;

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t src1_cb_index = tt::CBIndex::c_1;
    uint32_t src2_cb_index = tt::CBIndex::c_2;
    uint32_t dst0_cb_index = tt::CBIndex::c_3;
    uint32_t dst1_cb_index = tt::CBIndex::c_4;

    uint32_t batch = get_compile_time_arg_val(0);

    ckernel::mul_tiles_init(src0_cb_index, src1_cb_index);
    ckernel::add_tiles_init(dst0_cb_index, src10_cb_index);
    for (uint32_t i = 0; i < batch; i++) {
        ckernel::tile_regs_acquire();
        cb_wait_front(src0_cb_index, oneTile);
        cb_wait_front(src1_cb_index, oneTile);

        ckernel::mul_tiles(src0_cb_index, src1_cb_index, i, i, 0);

        ckernel::tile_regs_commit();

        pack_tile(0, dst0_cb_index, i);

        ckernel::cb_pop_front(src0_cb_index, oneTile);
        ckernel::cb_pop_front(src1_cb_index, oneTile);

        cb_push_back(dst0_cb_index, oneTile);

        ckernel::tile_regs_acquire();
        cb_wait_front(dst0_cb_index, oneTile);
        cb_wait_front(src2_cb_index, oneTile);

        ckernel::add_tiles(dst0_cb_index, src2_cb_index, i, i, 0);
        ckernel::tile_regs_commit();

        pack_tile(0, dst1_cb_index, i);

        ckernel::cb_pop_front(dst0_cb_index, oneTile);
        ckernel::cb_pop_front(src2_cb_index, oneTile);

        ckernel::tile_regs_release();
    }
    cb_push_back(dst1_cb_index, batch);
}
}  // namespace NAMESPACE
