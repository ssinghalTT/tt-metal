# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
import numpy as np
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.ccl.test_reduce_scatter_async import (
    create_and_load_sub_device_manager_with_fabric_interface,
    teardown_fabric_interface,
)
from tests.tt_eager.python_api_testing.unit_testing.misc.test_speculative_flash_decode import (
    nearest_n,
    nearest_pow_2,
    num_to_corerange,
    fa_rand,
    get_speculative_flash_decode_expected,
    prepare_test_config_and_data,
)


def create_multi_device_tensors(
    input_tensors, mesh_device, mem_config, worker_sub_device_id=None, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
):
    sub_device_ids = [worker_sub_device_id] if worker_sub_device_id is not None else []
    tt_tensors = []
    for i, t in enumerate(input_tensors):
        tt_tensors.append(
            ttnn.Tensor(t, dtype).to(layout).to(mesh_device.get_devices()[i], mem_config, sub_device_ids=sub_device_ids)
        )
    tensor_mesh = ttnn.aggregate_as_tensor(tt_tensors)
    return tensor_mesh


def read_multi_device_tensor(tt_tensor, worker_sub_device_id=None):
    sub_device_ids = [worker_sub_device_id] if worker_sub_device_id is not None else []
    tensors = []
    for i, t in enumerate(ttnn.get_device_tensors(tt_tensor)):
        t = t.cpu(sub_device_ids=sub_device_ids).to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        tensors.append(t)
    return tensors


def get_speculative_flash_decode_tt_ccl(
    tt_Q,
    tt_K,
    tt_V,
    tt_priority_tensors,
    tt_gathered_priority_tensors,
    sfd_semaphore_handles,
    mesh_device,
    worker_sub_device_id,
    start_indices,
    nh,
    lambda_,
    scale,
    program_config,
    compute_kernel_config,
    memory_config,
    causal=True,
    cur_pos_tensor=False,
    sharded_out=False,
    height_sharded_memcfg=None,
):
    """
    Wrapper function for speculative flash decode tensor operations.
    Returns tuple of (gt, spec, spec_lp_distance, lp_norm_x) tensors.
    """
    # find the sender device idx based on priority tensor value
    p_tensors = read_multi_device_tensor(tt_priority_tensors, worker_sub_device_id)
    min_p_device0 = torch.min(p_tensors[0].squeeze())
    min_p_device1 = torch.min(p_tensors[1].squeeze())
    sender_idx = 0 if min_p_device0 > min_p_device1 else 1
    receiver_idx = 1 - sender_idx
    logger.info(f"tt_sender_idx: {sender_idx}")

    if causal:
        if cur_pos_tensor:
            start_indices_multidevice = [torch.tensor(start_indices), torch.tensor(start_indices)]
            start_indices_tt = create_multi_device_tensors(
                start_indices_multidevice,
                mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
                worker_sub_device_id,
                ttnn.ROW_MAJOR_LAYOUT,
                ttnn.int32,
            )
            outputs = ttnn.experimental.speculative_scaled_dot_product_attention_decode(
                tt_Q,
                tt_K,
                tt_V,
                lambda_=lambda_,
                cur_pos_tensor=start_indices_tt,
                scale=scale,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                memory_config=height_sharded_memcfg if sharded_out else memory_config,
                priority_tensor=tt_priority_tensors,
                other_priority_tensor=tt_gathered_priority_tensors,
                ccl_enabled=True,
                multi_device_global_semaphore=sfd_semaphore_handles,
            )
        else:
            outputs = ttnn.experimental.speculative_scaled_dot_product_attention_decode(
                tt_Q,
                tt_K,
                tt_V,
                lambda_=lambda_,
                cur_pos=start_indices,
                scale=scale,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                memory_config=height_sharded_memcfg if sharded_out else memory_config,
                priority_tensor=tt_priority_tensors,
                other_priority_tensor=tt_gathered_priority_tensors,
                ccl_enabled=True,
                multi_device_global_semaphore=sfd_semaphore_handles,
            )
    else:
        raise NotImplementedError("Non-causal not implemented")

    tt_back_gt_md, tt_back_spec_md, tt_back_spec_lp_distance_md, tt_back_lp_norm_x_md = outputs

    for device in mesh_device.get_devices():
        ttnn.synchronize_device(device, sub_device_ids=[worker_sub_device_id])

    # read multi-device tensors and assign output to the sender device
    tt_back_gt = read_multi_device_tensor(tt_back_gt_md, worker_sub_device_id)[sender_idx]
    tt_back_spec = read_multi_device_tensor(tt_back_spec_md, worker_sub_device_id)[sender_idx]
    tt_back_spec_lp_distance = read_multi_device_tensor(tt_back_spec_lp_distance_md, worker_sub_device_id)[sender_idx]
    tt_back_lp_norm_x = read_multi_device_tensor(tt_back_lp_norm_x_md, worker_sub_device_id)[sender_idx]
    tt_back_gt_receiver = read_multi_device_tensor(tt_back_gt_md, worker_sub_device_id)[receiver_idx]

    # slice to correct number of heads
    tt_back_gt = tt_back_gt[:, :, :nh, :]
    tt_back_spec = tt_back_spec[:, :, :nh, :]
    tt_back_gt_receiver = tt_back_gt_receiver[:, :, :nh, :]

    # uncomment this assert after enable ccl in kernel
    # # assert tt_back_spec on the sender device is the same as tt_back_gt on the receiver device
    # out_pass, out_pcc = comp_pcc(tt_back_spec, tt_back_gt_receiver, 0.99)
    # logger.debug(f"spec tt sender vs gt receiver: {out_pcc}")
    # assert out_pass

    return tt_back_gt, tt_back_spec, tt_back_spec_lp_distance, tt_back_lp_norm_x


def run_speculative_flash_decode_ccl_impl(
    mesh_device,
    b,
    nh,
    nkv,
    s,
    d,
    dtype,
    grid_size,
    q_dtype=ttnn.bfloat16,
    cur_pos_tensor=False,
    k_chunk_size=128,
    lambda_=0.2,
    sharded_in=False,
    sharded_out=False,
    causal=True,
    enable_async=False,
):
    ############################################################
    # Setup and Defines
    ############################################################
    num_devices = 2
    # Use Async mode based on test input config
    mesh_device.enable_async(enable_async)
    if enable_async:
        logger.info(f"Using Async Mode for All Gather Op Dispatch")
    enable_persistent_fabric = True
    create_persistent_fabric = True
    teardown_persistent_fabric = True
    ############################################################

    ############################################################
    ### Persistent fabric and ccl setup ###
    ############################################################
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    if create_persistent_fabric:
        mesh_sub_device_manager_id = create_and_load_sub_device_manager_with_fabric_interface(
            mesh_device, [worker_sub_device], 0, 0, enable_persistent_fabric
        )
    ############################################################

    ############################################################
    ### Warmup all gather ccl ###
    ############################################################
    logger.info(f"Performing warmup all gather ccl")
    output_shape = [1, 1, 64, 128]
    dim = 2
    layout = ttnn.TILE_LAYOUT
    logger.info(f"Output shape: {output_shape}")
    logger.info(f"dim: {dim}")
    mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)

    output_tensor = torch.rand(output_shape).bfloat16()

    input_tensors = torch.chunk(output_tensor, num_devices, dim)
    input_tensor_mesh = create_multi_device_tensors(
        input_tensors, mesh_device, mem_config, worker_sub_device_id, layout, ttnn.bfloat16
    )

    tt_out_tensor = ttnn.experimental.all_gather_async(
        input_tensor_mesh,
        dim,
        num_links=1,
        memory_config=mem_config,
        topology=ttnn.Topology.Ring,
        subdevice_id=worker_sub_device_id,
        enable_persistent_fabric_mode=enable_persistent_fabric,
    )

    logger.info(f"Waiting for op")
    for device in mesh_device.get_devices():
        ttnn.synchronize_device(device, sub_device_ids=[worker_sub_device_id])
    logger.info(f"Done iteration")

    for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
        tt_output_tensor = t.cpu(sub_device_ids=[worker_sub_device_id]).to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        logger.info(f"Checking for device {t.device().id()}")
        eq, output = comp_equal(tt_output_tensor, output_tensor)
        assert eq, f"{i} FAILED: {output}"
    logger.info(f"Done warmup all gather ccl")
    ############################################################

    logger.info(f"Performing speculative flash decode ccl")
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y - 1:
        pytest.skip(
            f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size} with last column dedicated for ccl"
        )

    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    torch.manual_seed(1234)

    num_parallel_cores = grid_size[0] * grid_size[1] // b
    if num_parallel_cores == 1:
        min_pcc = 0.90
    else:
        min_pcc = 0.99
        if q_dtype == ttnn.bfloat8_b:
            min_pcc = 0.98
        min_pcc = 0.91 if dtype == ttnn.bfloat4_b else min_pcc

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    dram_memcfg = ttnn.DRAM_MEMORY_CONFIG
    shard_grid = ttnn.CoreRangeSet({num_to_corerange(b)})
    shard_spec = ttnn.ShardSpec(shard_grid, (padded_num_heads, d), ttnn.ShardOrientation.ROW_MAJOR, False)
    height_sharded_memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    Ks = [fa_rand(b, nkv, s, d) for _ in range(num_devices)]
    Vs = [fa_rand(b, nkv, s, d) for _ in range(num_devices)]

    tt_K = create_multi_device_tensors(Ks, mesh_device, dram_memcfg, worker_sub_device_id, ttnn.TILE_LAYOUT, dtype)
    tt_V = create_multi_device_tensors(Vs, mesh_device, dram_memcfg, worker_sub_device_id, ttnn.TILE_LAYOUT, dtype)

    scale = d**-0.5
    min_start_idx = 2 * k_chunk_size
    max_start_idx = min_start_idx

    # create global semaphore handles for speculative flash decode
    sfd_semaphore_handles = ttnn.create_global_semaphore(
        mesh_device, ccl_sub_device_crs, 0, sub_device_ids=[worker_sub_device_id]
    )
    addrs = ttnn.get_global_semaphore_address(sfd_semaphore_handles)
    logger.info(f"semaphore handle addresses: {addrs}")
    # assert all addresses are the same
    # assert len(set(addrs)) == 1 # uncomment this after new changes from global semaphore is pulled
    assert addrs[0] == addrs[1]

    while max_start_idx < s:
        # Set start indices if not provided or in multi-iteration mode
        start_indices = (
            np.linspace(min_start_idx, max_start_idx, b, dtype=np.int32).tolist() if b > 1 else [max_start_idx]
        )
        # create a random alternating priority tensor [0,1] or [1,0]
        priority_tensors = (
            [torch.zeros(1, 1, b, 1), torch.ones(1, 1, b, 1)]
            if max_start_idx % 2 == 0
            else [torch.ones(1, 1, b, 1), torch.zeros(1, 1, b, 1)]
        )

        ##########################################
        #### Prepare test config and data
        ##########################################
        program_config, padded_layer_len, attn_mask, Q = prepare_test_config_and_data(
            b, nh, s, d, grid_size, padded_num_heads, k_chunk_size, max_start_idx, start_indices, causal
        )

        ##########################################
        #### TT Calculation ####
        ##########################################
        Qs = [Q[:, :, :nh] for _ in range(num_devices)]
        q_mem_config = height_sharded_memcfg if sharded_in else dram_memcfg
        tt_Q = create_multi_device_tensors(
            Qs, mesh_device, q_mem_config, worker_sub_device_id, ttnn.TILE_LAYOUT, q_dtype
        )
        gathered_priority_tensors = priority_tensors[
            ::-1
        ]  # priority tensor value of its pair device, same shape as priority_tensors
        tt_priority_tensors = create_multi_device_tensors(
            priority_tensors, mesh_device, dram_memcfg, worker_sub_device_id, ttnn.ROW_MAJOR_LAYOUT, ttnn.int32
        )
        tt_gathered_priority_tensors = create_multi_device_tensors(
            gathered_priority_tensors, mesh_device, dram_memcfg, worker_sub_device_id, ttnn.ROW_MAJOR_LAYOUT, ttnn.int32
        )

        tt_back_gt, tt_back_spec, tt_back_spec_lp_distance, tt_back_lp_norm_x = get_speculative_flash_decode_tt_ccl(
            tt_Q,
            tt_K,
            tt_V,
            tt_priority_tensors,
            tt_gathered_priority_tensors,
            sfd_semaphore_handles,
            mesh_device,
            worker_sub_device_id,
            start_indices,
            nh,
            lambda_,
            scale,
            program_config,
            compute_kernel_config,
            dram_memcfg,
            cur_pos_tensor=cur_pos_tensor,
            sharded_out=sharded_out,
            height_sharded_memcfg=height_sharded_memcfg,
        )

        ##########################################
        #### Expected Calculation ####
        ##########################################
        min_p_device0 = torch.min(priority_tensors[0].squeeze())
        min_p_device1 = torch.min(priority_tensors[1].squeeze())
        sender_idx = 0 if min_p_device0 > min_p_device1 else 1
        logger.info(f"sender_idx: {sender_idx}")
        K = Ks[sender_idx]
        V = Vs[sender_idx]
        expected_gt, expected_spec, lp_distance, lp_norm_x = get_speculative_flash_decode_expected(
            Q,
            K,
            V,
            attn_mask,
            start_indices,
            nh,
            nkv,
            k_chunk_size,  # speculative chunk size
            scale,
            padded_layer_len,
            lambda_,
        )
        passing = torch.all(lp_distance < lambda_ * lp_norm_x)
        priority_tensors[sender_idx][0, 0, :, 0] = (lp_distance <= lambda_ * lp_norm_x).to(torch.int32) * 2
        logger.debug(f"gt speculation passing: {passing}")

        ##########################################
        #### Comparison ####
        ##########################################

        out_pass, out_pcc = comp_pcc(expected_gt, tt_back_gt, min_pcc)
        logger.debug(f"gt tt vs pytorch: {out_pcc}")
        assert out_pass

        out_pass, out_pcc = comp_pcc(expected_spec, tt_back_spec, min_pcc)
        logger.debug(f"spec tt vs pytorch: {out_pcc}")
        assert out_pass

        min_frac_tol = 0.25 if torch.all(lp_distance.squeeze() > 2) else 0.5
        out_pass = torch.allclose(
            lp_distance.squeeze(),
            tt_back_spec_lp_distance.to(torch.float32).squeeze() ** (0.5),
            rtol=min_frac_tol,
            atol=1e-1,
        )
        logger.debug(
            f"lp distance output tt vs pytorch: {lp_distance.squeeze()}, {tt_back_spec_lp_distance.to(torch.float32).squeeze()**(0.5)}"
        )
        assert out_pass

        min_frac_tol = 0.25 if torch.all(lp_norm_x.squeeze() > 2) else 0.5
        out_pass = torch.allclose(
            lp_norm_x.squeeze(), tt_back_lp_norm_x.to(torch.float32).squeeze() ** (0.5), rtol=min_frac_tol, atol=1e-1
        )
        logger.debug(
            f"lp norm output tt vs pytorch: {lp_norm_x.squeeze()}, {tt_back_lp_norm_x.to(torch.float32).squeeze()**(0.5)}"
        )
        assert out_pass

        max_start_idx += 71 if max_start_idx < 4096 else 3001

    logger.info(f"Done speculative flash decode ccl")

    ############################################################
    ### Teardown persistent fabric ###
    ############################################################
    if enable_persistent_fabric and teardown_persistent_fabric:
        teardown_fabric_interface(mesh_device)
    ############################################################


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat8_b, ttnn.bfloat16],
    ],
    ids=[
        "kv_bfp8",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size, cur_pos_tensor",
    ([1, 32, 8, 16 * 8192, 128, (8, 4), True],),  # llama 3.1 8b
)
@pytest.mark.parametrize(
    "k_chunk_size",
    [
        128,
    ],
)
@pytest.mark.parametrize("enable_async", [False])
def test_speculative_flash_decode_ccl(
    t3k_mesh_device,
    b,
    nh,
    nkv,
    s,
    d,
    dtype,
    grid_size,
    q_dtype,
    cur_pos_tensor,
    k_chunk_size,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_speculative_flash_decode_ccl_impl(
        t3k_mesh_device,
        b,
        nh,
        nkv,
        s,
        d,
        dtype,
        grid_size,
        q_dtype,
        cur_pos_tensor,
        k_chunk_size,
        sharded_in=False,
        sharded_out=False,
        enable_async=enable_async,
    )
