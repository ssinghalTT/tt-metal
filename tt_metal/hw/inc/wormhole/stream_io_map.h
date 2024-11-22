// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _STREAM_IO_MAP_
#define _STREAM_IO_MAP_

#include <stdint.h>

#include "risc_attribs.h"

#define   STREAM_RD_RESP_RECEIVED   0
#define   STREAM_NONPOSTED_WR_REQ_SENT   1
#define   STREAM_NONPOSTED_WR_ACK_RECEIVED   2
#define   STREAM_NONPOSTED_ATOMIC_RESP_RECEIVED   3
#define   STREAM_POSTED_WR_REQ_SENT   4


// TODO: in ll-buda we can probably just start at stream 0 and not at stream 8?
/*
   Kernel operand mapping scheme:
   - ID 0-7 (inputs, unpacker-only) => streams 8-15
   - ID 8-15 (params, unpacker-only) => streams 16-23
   - ID 16-23 (outputs, packer-only) => streams 24-31
   - ID 24-31 (intermediates, packer/unpacker) => streams 32-39
*/
const uint32_t OPERAND_START_STREAM = 8;
const uint32_t OPERAND_BRISC_NCRISC_SYNC_STREAM = 0;

// Indexed with operand = kernel operand ID (0-31) per the table above
// Used for tile push/pop operations.
inline __attribute__((always_inline)) uint32_t get_operand_stream_id(int operand) {
  return OPERAND_START_STREAM + operand;
}

inline __attribute__((always_inline)) uint32_t get_stream_reg_index(uint32_t rirsc_id, uint32_t noc, uint32_t index) {
  return uint32_t(rirsc_id << 4) + uint32_t(noc << 3) + index;
}

// Pointers to stream scratch registers (implemented using don't-care functional registers) that are used for CB synchronization

inline __attribute__((always_inline)) volatile uint32_t* get_cb_tiles_received_ptr(int operand) {
  return (volatile uint32_t*)(uintptr_t)(STREAM_REG_ADDR(get_operand_stream_id(operand), STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX));
}

inline __attribute__((always_inline)) volatile uint32_t* get_cb_tiles_acked_ptr(int operand) {
  return (volatile uint32_t*)(uintptr_t)(STREAM_REG_ADDR(get_operand_stream_id(operand), STREAM_REMOTE_DEST_BUF_START_REG_INDEX));
}

// noc_reads_num_issued
inline __attribute__((always_inline)) uint32_t get_noc_reads_num_issued(uint32_t rirsc_id, uint32_t noc) {
  return NOC_STREAM_READ_REG(OPERAND_BRISC_NCRISC_SYNC_STREAM, get_stream_reg_index(rirsc_id, noc, STREAM_RD_RESP_RECEIVED));
}

inline __attribute__((always_inline)) void inc_noc_reads_num_issued(uint32_t rirsc_id, uint32_t noc) {
  volatile uint32_t val = NOC_STREAM_READ_REG(OPERAND_BRISC_NCRISC_SYNC_STREAM, get_stream_reg_index(rirsc_id, noc, STREAM_RD_RESP_RECEIVED)) + 1;
  NOC_STREAM_WRITE_REG(OPERAND_BRISC_NCRISC_SYNC_STREAM, get_stream_reg_index(rirsc_id, noc, STREAM_RD_RESP_RECEIVED), val);
}

inline __attribute__((always_inline)) void set_noc_reads_num_issued(uint32_t rirsc_id, uint32_t noc, uint32_t val) {
  NOC_STREAM_WRITE_REG(OPERAND_BRISC_NCRISC_SYNC_STREAM, get_stream_reg_index(rirsc_id, noc, STREAM_RD_RESP_RECEIVED), val);
}

// noc_nonposted_writes_num_issued
inline __attribute__((always_inline)) uint32_t get_noc_nonposted_writes_num_issued(uint32_t rirsc_id, uint32_t noc) {
  return NOC_STREAM_READ_REG(OPERAND_BRISC_NCRISC_SYNC_STREAM, get_stream_reg_index(rirsc_id, noc, STREAM_POSTED_WR_REQ_SENT));
}

inline __attribute__((always_inline)) void inc_noc_nonposted_writes_num_issued(uint32_t rirsc_id, uint32_t noc, uint32_t inc = 1) {
  volatile uint32_t val = NOC_STREAM_READ_REG(OPERAND_BRISC_NCRISC_SYNC_STREAM, get_stream_reg_index(rirsc_id, noc, STREAM_POSTED_WR_REQ_SENT)) + inc;
  NOC_STREAM_WRITE_REG(OPERAND_BRISC_NCRISC_SYNC_STREAM, get_stream_reg_index(rirsc_id, noc, STREAM_POSTED_WR_REQ_SENT), val);
}

inline __attribute__((always_inline)) void set_noc_nonposted_writes_num_issued(uint32_t rirsc_id, uint32_t noc, uint32_t val) {
  NOC_STREAM_WRITE_REG(OPERAND_BRISC_NCRISC_SYNC_STREAM, get_stream_reg_index(rirsc_id, noc, STREAM_POSTED_WR_REQ_SENT), val);
}

// noc_nonposted_writes_acked
inline __attribute__((always_inline)) uint32_t get_noc_nonposted_writes_acked(uint32_t rirsc_id, uint32_t noc) {
  return NOC_STREAM_READ_REG(OPERAND_BRISC_NCRISC_SYNC_STREAM, get_stream_reg_index(rirsc_id, noc, STREAM_NONPOSTED_WR_ACK_RECEIVED));
}

inline __attribute__((always_inline)) void inc_noc_nonposted_writes_acked(uint32_t rirsc_id, uint32_t noc) {
  volatile uint32_t val = NOC_STREAM_READ_REG(OPERAND_BRISC_NCRISC_SYNC_STREAM, get_stream_reg_index(rirsc_id, noc, STREAM_NONPOSTED_WR_ACK_RECEIVED)) + 1;
  NOC_STREAM_WRITE_REG(OPERAND_BRISC_NCRISC_SYNC_STREAM, get_stream_reg_index(rirsc_id, noc, STREAM_NONPOSTED_WR_ACK_RECEIVED), val);
}

inline __attribute__((always_inline)) void set_noc_nonposted_writes_acked(uint32_t rirsc_id, uint32_t noc, uint32_t val) {
  NOC_STREAM_WRITE_REG(OPERAND_BRISC_NCRISC_SYNC_STREAM, get_stream_reg_index(rirsc_id, noc, STREAM_NONPOSTED_WR_ACK_RECEIVED), val);
}

// noc_nonposted_atomics_acked
inline __attribute__((always_inline)) uint32_t get_noc_nonposted_atomics_acked(uint32_t rirsc_id, uint32_t noc) {
  return NOC_STREAM_READ_REG(OPERAND_BRISC_NCRISC_SYNC_STREAM, get_stream_reg_index(rirsc_id, noc, STREAM_NONPOSTED_ATOMIC_RESP_RECEIVED));
}

inline __attribute__((always_inline)) void inc_noc_nonposted_atomics_acked(uint32_t rirsc_id, uint32_t noc) {
  volatile uint32_t val = NOC_STREAM_READ_REG(OPERAND_BRISC_NCRISC_SYNC_STREAM, get_stream_reg_index(rirsc_id, noc, STREAM_NONPOSTED_ATOMIC_RESP_RECEIVED)) + 1;
  NOC_STREAM_WRITE_REG(OPERAND_BRISC_NCRISC_SYNC_STREAM, get_stream_reg_index(rirsc_id, noc, STREAM_NONPOSTED_ATOMIC_RESP_RECEIVED), val);
}

inline __attribute__((always_inline)) void set_noc_nonposted_atomics_acked(uint32_t rirsc_id, uint32_t noc, uint32_t val) {
  NOC_STREAM_WRITE_REG(OPERAND_BRISC_NCRISC_SYNC_STREAM, get_stream_reg_index(rirsc_id, noc, STREAM_NONPOSTED_ATOMIC_RESP_RECEIVED), val);
}

// noc_posted_writes_num_issued
inline __attribute__((always_inline)) uint32_t get_noc_posted_writes_num_issued(uint32_t rirsc_id, uint32_t noc) {
  return NOC_STREAM_READ_REG(OPERAND_BRISC_NCRISC_SYNC_STREAM, get_stream_reg_index(rirsc_id, noc, STREAM_POSTED_WR_REQ_SENT));
}

inline __attribute__((always_inline)) void inc_noc_posted_writes_num_issued(uint32_t rirsc_id, uint32_t noc) {
  volatile uint32_t val = NOC_STREAM_READ_REG(OPERAND_BRISC_NCRISC_SYNC_STREAM, get_stream_reg_index(rirsc_id, noc, STREAM_POSTED_WR_REQ_SENT)) + 1;
  NOC_STREAM_WRITE_REG(OPERAND_BRISC_NCRISC_SYNC_STREAM, get_stream_reg_index(rirsc_id, noc, STREAM_POSTED_WR_REQ_SENT), val);
}

inline __attribute__((always_inline)) void set_noc_posted_writes_num_issued(uint32_t rirsc_id, uint32_t noc, uint32_t val) {
  NOC_STREAM_WRITE_REG(OPERAND_BRISC_NCRISC_SYNC_STREAM, get_stream_reg_index(rirsc_id, noc, STREAM_POSTED_WR_REQ_SENT), val);
}

inline __attribute__((always_inline)) volatile uint32_t* get_cq_finish_ptr() {
    return (volatile uint32_t*)(uintptr_t)(STREAM_REG_ADDR(
        get_operand_stream_id(0), STREAM_REMOTE_DEST_BUF_START_REG_INDEX));
}

inline __attribute__((always_inline)) volatile uint32_t* get_sync_register_ptr() {
  return (volatile uint32_t*)(uintptr_t)(STREAM_REG_ADDR(0, STREAM_PHASE_AUTO_CFG_PTR_REG_INDEX));
}
#endif
