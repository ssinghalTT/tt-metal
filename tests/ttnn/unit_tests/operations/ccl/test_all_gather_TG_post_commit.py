# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull
from ttnn import ShardTensor2dMesh, ConcatMesh2dToTensor
from tests.ttnn.unit_tests.operations.ccl.test_ccl_common import (
    create_and_load_sub_device_manager_with_fabric_interface,
    teardown_fabric_interface,
    create_global_semaphore_with_same_address,
)


def report_mismatches(golden, actual, max_printable=None):
    printed = 0
    for w in range(golden.shape[0]):
        for z in range(golden.shape[1]):
            for y in range(0, golden.shape[2], 32):
                for x in range(0, golden.shape[3], 32):
                    print_it = (max_printable is None or printed < max_printable) and golden[w, z, y, x] != actual[
                        w, z, y, x
                    ]
                    if print_it:
                        printed += 1
                        print(
                            f"output mismatch for tensor at [{w}, {z}, {y}, {x}]: expected {golden[w, z, y, x]} != actual {actual[w, z, y, x]}"
                        )


def print_tile_corners_of_tensor(t):
    for w in range(t.shape[0]):
        for z in range(t.shape[1]):
            str = ""
            for x in range(0, t.shape[3], 32):
                str += f"{x:<5} "[:5]
            print(f"     {str}")
            for y in range(0, t.shape[2], 32):
                str_vals = f"y={y:<3} "[:5]
                for x in range(0, t.shape[3], 32):
                    yy = 0
                    xx = 0
                    val = int(t[w, z, y + yy, x + xx].item())
                    str_vals += f"{val:<5} "[:5]
                print(f"{str_vals}")


def run_with_trace(
    mesh_device,
    all_gather_topology,
    input_tensor,
    dim,
    num_links,
    cluster_axis,
    output_mem_config,
    n_worker=None,
    n_buffer=None,
    num_iter=20,
):
    # Compile Run
    logger.info("Compiling model")
    tt_out_tensor = ttnn.all_gather(
        input_tensor,
        dim=dim,
        cluster_axis=cluster_axis,
        mesh_device=mesh_device,
        num_links=num_links,
        memory_config=output_mem_config,
        topology=all_gather_topology,
    )
    for d in mesh_device.get_devices():
        ttnn.synchronize_device(d)

    # Capture trace
    logger.info("Capturing trace")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(num_iter):
        tt_out_tensor = ttnn.all_gather(
            input_tensor,
            dim=dim,
            cluster_axis=cluster_axis,
            mesh_device=mesh_device,
            num_links=num_links,
            memory_config=output_mem_config,
            topology=all_gather_topology,
        )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    for d in mesh_device.get_devices():
        ttnn.synchronize_device(d)

    # Run the op
    logger.info("Starting Trace perf test...")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    for d in mesh_device.get_devices():
        ttnn.synchronize_device(d)

    return tt_out_tensor


def run_line_all_gather_on_TG_with_mesh_tensor_along_rows(
    mesh_device,
    num_devices_per_line,
    per_chip_output_shape,
    tensor_memory_layout,
    dim,
    num_links,
    input_dtype,
    layout,
    buffer_type: ttnn.BufferType,
    use_program_cache,
    function_level_defaults,
    enable_async,
    input_shard_spec: ttnn.ShardSpec = None,
    num_all_gather_instances: int = 1,
    num_iters: int = 1,
    cluster_axis: int = 0,
    tile=(32, 32),
    trace_mode=False,
    debug=False,
    # New all-gather-async and persistent fabric params
    use_all_gather_async=False,
    enable_persistent_fabric=False,
    create_persistent_fabric=False,
    teardown_persistent_fabric=False,
):
    if create_persistent_fabric:
        assert use_all_gather_async
        assert enable_persistent_fabric
    if teardown_persistent_fabric:
        assert use_all_gather_async
        assert enable_persistent_fabric
    if not use_all_gather_async:
        assert not create_persistent_fabric
        assert not teardown_persistent_fabric
        assert not enable_persistent_fabric

    mesh_device.enable_async(enable_async)

    input_shape_per_chip = list(per_chip_output_shape)
    input_shape_per_chip[dim] //= num_devices_per_line
    tensor_height_per_all_gather = per_chip_output_shape[-2]

    full_mesh_input_shape = list(per_chip_output_shape)
    ## The `all_gather_instances_concat_dim` is the dimension we will split the cluster spanning tensor along in order to split it
    ## off into per-all-gather tensors
    all_gather_instances_concat_dim = 1 if dim == 0 else 0
    full_mesh_input_shape[all_gather_instances_concat_dim] *= num_all_gather_instances
    logger.info(
        f"per_chip_output_shape: {full_mesh_input_shape}, dim: {dim}, all_gather_instances_concat_dim: {all_gather_instances_concat_dim}, num_devices_per_line: {num_devices_per_line}"
    )

    all_gather_instances_goldens = []
    full_input_tensor_unfractured = torch.rand(full_mesh_input_shape, dtype=torch.bfloat16)

    input_mem_config = ttnn.MemoryConfig(tensor_memory_layout, buffer_type=buffer_type, shard_spec=input_shard_spec)
    shard_dims = (dim, all_gather_instances_concat_dim) if cluster_axis == 0 else (all_gather_instances_concat_dim, dim)
    concat_dims = shard_dims

    mesh_shape = (
        (num_devices_per_line, num_all_gather_instances)
        if cluster_axis == 0
        else (num_all_gather_instances, num_devices_per_line)
    )

    output_shard_spec = None
    if input_shard_spec is not None:
        output_shard_shape = list(input_shard_spec.shape)
        if dim == 3:
            output_shard_shape[1] *= num_devices_per_line
        else:
            output_shard_shape[0] *= num_devices_per_line
        output_shard_spec = ttnn.ShardSpec(
            input_shard_spec.grid,
            output_shard_shape,
            input_shard_spec.orientation,
            False,
        )
    output_mem_config = ttnn.MemoryConfig(tensor_memory_layout, buffer_type=buffer_type, shard_spec=output_shard_spec)
    ttnn_tensor = ttnn.from_torch(
        full_input_tensor_unfractured,
        tile=ttnn.Tile(tile),
        dtype=input_dtype,
        device=mesh_device,
        layout=layout,
        memory_config=input_mem_config,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=shard_dims),
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device)

    sub_device_stall_group = []
    if use_all_gather_async:
        compute_grid_size = mesh_device.compute_with_storage_grid_size()
        ccl_sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )
        worker_sub_device = ttnn.SubDevice(
            [
                ccl_sub_device_crs,
            ]
        )
        worker_sub_device_id = ttnn.SubDeviceId(0)
        sub_device_stall_group = [worker_sub_device_id]
        if create_persistent_fabric:
            logger.info("Create persistent fabric interface")
            mesh_sub_device_manager_id = create_and_load_sub_device_manager_with_fabric_interface(
                mesh_device, [worker_sub_device], 0, 0, enable_persistent_fabric
            )
            logger.info("Done Create persistent fabric interface")
            mesh_device.set_sub_device_stall_group(sub_device_stall_group)

        # create global semaphore handles
        ccl_semaphore_handles = create_global_semaphore_with_same_address(mesh_device, ccl_sub_device_crs, 0)

    # ttnn.visualize_mesh_device(mesh_device, tensor=ttnn_tensor)
    if trace_mode:
        ttnn_tensor_out = run_with_trace(
            input_tensor=ttnn_tensor,
            dim=dim,
            cluster_axis=cluster_axis,
            mesh_device=mesh_device,
            num_links=num_links,
            output_mem_config=output_mem_config,
            all_gather_topology=ttnn.Topology.Linear,
            num_iter=num_iters,
        )
    else:
        for _ in range(num_iters):
            if use_all_gather_async:
                logger.info("Running all-gather async")
                ttnn_tensor_out = ttnn.experimental.all_gather_async(
                    ttnn_tensor,
                    dim,
                    cluster_axis=cluster_axis,
                    mesh_device=mesh_device,
                    topology=ttnn.Topology.Linear,
                    multi_device_global_semaphore=ccl_semaphore_handles,
                    num_links=num_links,
                    memory_config=output_mem_config,
                    subdevice_id=worker_sub_device_id,
                    enable_persistent_fabric_mode=enable_persistent_fabric,
                )
            else:
                ttnn_tensor_out = ttnn.all_gather(
                    ttnn_tensor,
                    dim=dim,
                    cluster_axis=cluster_axis,
                    mesh_device=mesh_device,
                    num_links=num_links,
                    memory_config=output_mem_config,
                    topology=ttnn.Topology.Linear,
                )

        if enable_persistent_fabric:
            logger.info(f"Waiting for op")
            ttnn.synchronize_devices(mesh_device, sub_device_ids=sub_device_stall_group)
            logger.info(f"Done iteration")

    if enable_persistent_fabric and teardown_persistent_fabric:
        logger.info("Tearing down persistent fabric interface")
        mesh_device.reset_sub_device_stall_group()
        teardown_fabric_interface(mesh_device)
        logger.info("Done tearing down persistent fabric interface")

    # ttnn.visualize_mesh_device(mesh_device, tensor=ttnn_tensor_out)
    tt_output_tensor = ttnn.to_torch(
        ttnn_tensor_out, mesh_composer=ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=concat_dims)
    )
    output_tensors_list = torch.chunk(tt_output_tensor, num_all_gather_instances, dim=all_gather_instances_concat_dim)
    output_golden = torch.zeros(tt_output_tensor.shape)

    # Repeat the input tensor to represent the fact that the full concatenated input tensor lives across every
    # device in the line
    repeat_factor = [1] * len(output_golden.shape)
    repeat_factor[dim] = num_devices_per_line
    output_golden[:, :, :, :] = full_input_tensor_unfractured.repeat(repeat_factor)

    eq = True
    if input_dtype == ttnn.bfloat16:
        eq, output = comp_equal(tt_output_tensor, output_golden)
        if not eq and debug is True:
            logger.error(f"found mismatches")
            report_mismatches(tt_output_tensor, output_golden, 100)
            print_tile_corners_of_tensor(tt_output_tensor)
    else:
        eq, output = comp_pcc(tt_output_tensor, output_golden)
    if not eq:
        logger.error(f"output mismatch for tensor: {output}")

    assert eq, f"FAILED: {output}"


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, per_chip_output_shape, dim, layout",
    [
        (4, 3, [4, 1, 32, 1280], 0, ttnn.TILE_LAYOUT),
        (4, 3, [1, 1, 32, 16384 * 4], 3, ttnn.TILE_LAYOUT),
        (4, 3, [1, 4, 32, 2304], 1, ttnn.TILE_LAYOUT),
        (4, 3, [1, 4, 32, 4096], 1, ttnn.TILE_LAYOUT),
        (4, 3, [1, 4, 32, 6656], 1, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "buffer_type",
    [
        ttnn.BufferType.DRAM,
        ttnn.BufferType.L1,
    ],
)
@pytest.mark.parametrize("replication_factor", [8])  # 1, 8])
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
def test_line_all_gather_on_TG_rows_post_commit(
    mesh_device,
    num_devices,
    per_chip_output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    buffer_type,
    use_program_cache,
    function_level_defaults,
    enable_async,
    replication_factor,
    num_iters=1,
):
    if len(mesh_device.get_devices()) != 32:
        pytest.skip("Not TG!")
    run_line_all_gather_on_TG_with_mesh_tensor_along_rows(
        mesh_device,
        num_devices,
        per_chip_output_shape,
        ttnn.TensorMemoryLayout.INTERLEAVED,
        dim,
        num_links,
        input_dtype,
        layout,
        buffer_type,
        use_program_cache,
        function_level_defaults,
        enable_async=enable_async,
        num_iters=num_iters,
        num_all_gather_instances=replication_factor,
        cluster_axis=1,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, per_chip_output_shape, dim, layout",
    [
        (8, 4, [1, 8, 32, 1280], 1, ttnn.TILE_LAYOUT),
        (8, 4, [8, 1, 32, 1280], 0, ttnn.TILE_LAYOUT),
        (8, 4, [1, 8, 32, 2048], 1, ttnn.TILE_LAYOUT),
        (8, 4, [1, 8, 32, 2304], 1, ttnn.TILE_LAYOUT),
        (8, 4, [1, 8, 32, 4096], 1, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "buffer_type",
    [
        ttnn.BufferType.DRAM,
    ],
)
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("replication_factor", [4])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
def test_line_all_gather_on_TG_cols_post_commit(
    mesh_device,
    num_devices,
    per_chip_output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    buffer_type,
    use_program_cache,
    function_level_defaults,
    enable_async,
    replication_factor,
    num_iters=1,
):
    if len(mesh_device.get_devices()) != 32:
        pytest.skip("Not TG!")
    run_line_all_gather_on_TG_with_mesh_tensor_along_rows(
        mesh_device,
        num_devices,
        per_chip_output_shape,
        ttnn.TensorMemoryLayout.INTERLEAVED,
        dim,
        num_links,
        input_dtype,
        layout,
        buffer_type,
        use_program_cache,
        function_level_defaults,
        enable_async=enable_async,
        num_iters=num_iters,
        num_all_gather_instances=replication_factor,
        cluster_axis=0,
    )
