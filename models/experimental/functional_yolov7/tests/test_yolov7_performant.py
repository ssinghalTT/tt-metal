# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from models.utility_functions import run_for_wormhole_b0
from models.experimental.functional_yolov7.tests.yolov7_performant import (
    run_yolov7_inference,
    run_yolov7_trace_inference,
    run_yolov7_trace_2cqs_inference,
)


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
def test_run_yolov7_inference(device, use_program_cache, batch_size, act_dtype, weight_dtype, model_location_generator):
    run_yolov7_inference(device, batch_size, act_dtype, weight_dtype, model_location_generator)


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 1843200}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
def test_run_yolov7_trace_inference(
    device,
    use_program_cache,
    batch_size,
    act_dtype,
    weight_dtype,
    enable_async_mode,
    model_location_generator,
):
    run_yolov7_trace_inference(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 3686400, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
def test_run_7_trace_2cqs_inference(
    device,
    use_program_cache,
    batch_size,
    act_dtype,
    weight_dtype,
    enable_async_mode,
    model_location_generator,
):
    run_yolov7_trace_2cqs_inference(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator,
    )
