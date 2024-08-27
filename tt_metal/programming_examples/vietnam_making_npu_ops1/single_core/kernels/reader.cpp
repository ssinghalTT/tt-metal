
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {

   DPRINT << "Hello, World! I'm reader kernel" << ENDL();
   DPRINT << "Reader noc_idx is " << uint32_t(noc_index) << ENDL();

   DPRINT << "noc coords 0 : " << (uint)my_x[0] << "," << (uint)my_y[0]
          << "\tnoc coords 1 : " << (uint)my_x[1] << "," << (uint)my_y[1]
          << ENDL();

  uint32_t arg = 0;
  const auto device_buffer0_addr = get_arg_val<uint32_t>(arg++);
  const auto cb0_id = get_arg_val<uint32_t>(arg++);
  const auto num_tiles = get_arg_val<uint32_t>(arg++);

  constexpr bool device_buffer0_is_dram = get_compile_time_arg_val(0) == 1;

  const uint32_t cb0_page_size = get_tile_size(cb0_id);
  const auto cb0_data_format = get_dataformat(cb0_id);
  const InterleavedAddrGenFast<device_buffer0_is_dram> input_addrg = {
      .bank_base_address = device_buffer0_addr,
      .page_size = cb0_page_size,
      .data_format = cb0_data_format};

  for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    // TODO: read a single tile within a for loop
    cb_reserve_back(cb0_id, 1/* TODO */);
    const auto cb0_l1_addr = get_write_ptr(cb0_id/* TODO */);
    noc_async_read_tile(tile_idx, input_addrg, cb0_l1_addr, 0/* TODO */);
    noc_async_read_barrier();
    cb_push_back(cb0_id, 1/* TODO */);
  }

  DPRINT << "READER END" << ENDL();
}
