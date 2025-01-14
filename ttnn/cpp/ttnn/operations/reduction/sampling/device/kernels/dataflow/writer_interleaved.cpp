#include "utils/bfloat16.h"
#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);

    uint32_t arg_id = 0;
    constexpr bool dst_is_dram = (bool)get_compile_time_arg_val(arg_id++);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(arg_id++);
    constexpr uint32_t cb_id_mask = get_compile_time_arg_val(arg_id++);
    constexpr uint32_t scale_cb_index = get_compile_time_arg_val(arg_id++);
    constexpr uint32_t packed_identity_scalar = get_compile_time_arg_val(arg_id++);
    constexpr uint32_t output_final_indices_rm_cb_index = get_compile_time_arg_val(arg_id++);
    constexpr uint32_t output_local_values_rm_cb_index = get_compile_time_arg_val(arg_id++);
    constexpr uint32_t output_local_indices_rm_cb_index = get_compile_time_arg_val(arg_id++);
    constexpr uint32_t values_stick_size = get_compile_time_arg_val(arg_id++);
    constexpr uint32_t im_indices_stick_size = get_compile_time_arg_val(arg_id++);
    constexpr uint32_t final_indices_stick_size = get_compile_time_arg_val(arg_id++);
    constexpr uint32_t out_stick_size = get_compile_time_arg_val(arg_id++);

    constexpr uint32_t k = get_arg_val<uint32_t>(1);              // get_compile_time_arg_val(arg_id++);
    uint32_t core_id = get_arg_val<uint32_t>(2);                  // get_compile_time_arg_val(arg_id++);
    constexpr uint32_t ids_per_batch = get_arg_val<uint32_t>(3);  // get_compile_time_arg_val(arg_id++);
    constexpr uint32_t ids_per_batch_final = get_compile_time_arg_val(arg_id++);
    constexpr uint32_t rand = get_compile_time_arg_val(arg_id++);

    // Reduce ops need to multiply by a scalar. We always want to multiply by 1.0f
    generate_reduce_scaler(scale_cb_index, packed_identity_scalar);

    // generate the top-k mask
    generate_mask<cb_id_mask, 1, ids_per_batch / 32>(1, k);

    // wait for compute kernel
    cb_wait_front(output_final_indices_rm_cb_index, final_indices_stick_size);
    cb_wait_front(output_local_values_rm_cb_index, values_stick_size);
    cb_wait_front(output_local_indices_rm_cb_index, im_indices_stick_size);

    // Use cb as L1 scratch memory
    uint32_t cb_local_values_addr = get_write_ptr(output_local_values_rm_cb_index);
    volatile tt_l1_ptr uint16_t* local_values = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_local_values_addr);

    uint32_t cb_local_indices_addr = get_write_ptr(output_local_indices_rm_cb_index);
    volatile tt_l1_ptr uint16_t* local_indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_local_indices_addr);

    uint32_t cb_final_indices_addr = get_write_ptr(output_final_indices_rm_cb_index);
    volatile tt_l1_ptr uint16_t* final_indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_final_indices_addr);

    uint32_t out_addr = get_write_ptr(cb_id_out);
    volatile tt_l1_ptr uint32_t* index_out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_addr);

    uint32_t start_id_local = core_id * ids_per_batch;
    uint32_t end_id_local = start_id_local + k;

    uint32_t start_id_final = core_id * ids_per_batch_final;
    uint32_t end_id_final = start_id_final + ids_per_batch_final;

    uint32_t cum_sum = 0;
    index_out[core_id] = final_indices[start_id_final + local_indices[end_id_local - 1]];
    // Sample from the top-k values
    for (uint32_t i = start_id_local; i < end_id_local; ++i) {
        // cum sum of local values
        cum_sum = bfloat16_add(cum_sum, local_values[i]);
        if (bfloat16_greater(cum_sum, rand)) {
            index_out[core_id] = final_indices[start_id_final + local_indices[i]];
            break;
        }
    }

    const InterleavedAddrGen<dst_is_dram> s_out = {.bank_base_address = dst_addr, .page_size = out_stick_size};
    uint64_t dst_noc_addr = get_noc_addr(0, s_out);
    noc_async_write(out_addr, dst_noc_addr + core_id * 4, out_stick_size);
    noc_async_write_barrier();
}
