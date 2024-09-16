# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time
import ttnn
import pytest

from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.functional_unet.tt.model_preprocessing import (
    create_unet_input_tensors,
    create_unet_model_parameters,
)
from models.experimental.functional_unet.tt import unet_shallow_torch
from models.experimental.functional_unet.tt import unet_shallow_ttnn
from models.experimental.functional_unet.tests.common import (
    check_pcc_conv,
    is_n300_with_eth_dispatch_cores,
    is_t3k_with_eth_dispatch_cores,
)

from models.perf.perf_utils import prep_perf_report
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.utility_functions import (
    profiler,
    skip_for_grayskull,
)


@skip_for_grayskull("UNet not currently supported on GS")
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 68864, "trace_region_size": 423936}], indirect=True)
@pytest.mark.parametrize(
    "batch, groups, iterations",
    ((2, 1, 16),),
)
def test_unet_trace(
    batch: int,
    groups: int,
    iterations: int,
    device,
    use_program_cache,
    reset_seeds,
):
    torch_input, ttnn_input = create_unet_input_tensors(device, batch, groups, pad_input=True)

    model = unet_shallow_torch.UNet.from_random_weights(groups=1)
    torch_output_tensor = model(torch_input)

    parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=device)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device)

    input_tensor = ttnn.allocate_tensor_on_device(
        ttnn_input.shape, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    logger.info(f"Compiling model with warmup run")
    ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=0)
    output_tensor = ttnn_model(input_tensor).cpu()

    logger.info(f"Capturing trace")
    ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=0)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    output_tensor = ttnn_model(input_tensor)
    ttnn.end_trace_capture(device, tid, cq_id=0)

    logger.info(f"Running trace for {iterations} iterations...")

    outputs = []
    start = time.time()
    for _ in range(iterations):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        outputs.append(output_tensor.cpu(blocking=False))
    ttnn.synchronize_device(device)
    end = time.time()
    logger.info(f"PERF={iterations * batch / (end-start) : .2f} fps")

    logger.info(f"Running sanity check against reference model output")
    B, C, H, W = torch_output_tensor.shape
    ttnn_tensor = ttnn.to_torch(outputs[-1]).reshape(B, H, W, -1)[:, :, :, :C].permute(0, 3, 1, 2)
    assert_with_pcc(torch_output_tensor, ttnn_tensor, 0.986)


@pytest.mark.skip()
@skip_for_grayskull("UNet not currently supported on GS")
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 68864, "trace_region_size": 423936}], indirect=True)
@pytest.mark.parametrize(
    "batch, groups, iterations",
    ((2, 1, 16),),
)
def test_unet_trace_2cq(
    batch: int,
    groups: int,
    iterations: int,
    device,
    use_program_cache,
    reset_seeds,
):
    torch_input, ttnn_input = create_unet_input_tensors(device, batch, groups, pad_input=True)

    model = unet_shallow_torch.UNet.from_random_weights(groups=1)
    torch_output_tensor = model(torch_input)

    parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=device)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device)

    input_tensor = ttnn.allocate_tensor_on_device(
        ttnn_input.shape, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    logger.info(f"Compiling model with warmup run")
    ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=0)
    output_tensor = ttnn_model(input_tensor).cpu()

    logger.info(f"Capturing trace")
    ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=0)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    output_tensor = ttnn_model(input_tensor)
    ttnn.end_trace_capture(device, tid, cq_id=0)

    logger.info(f"Running trace for {iterations} iterations...")

    outputs = []
    start = time.time()
    for _ in range(iterations):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        outputs.append(output_tensor.cpu(blocking=False))
    ttnn.synchronize_device(device)
    end = time.time()
    logger.info(f"PERF={iterations * batch / (end-start) : .2f} fps")
