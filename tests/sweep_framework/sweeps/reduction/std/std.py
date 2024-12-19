# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial
import numpy as np

import torch
import random
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes, sanitize_shape_rm
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "xfail": {
        "input_shape": gen_shapes([1, 1, 1, 1], [2, 6, 128, 128], [1, 1, 32, 32], 32)
        + gen_shapes([1, 1, 1, 1], [2, 6, 254, 129], [1, 1, 20, 33], 33)
        + gen_shapes([1, 1, 1, 1], [2, 7, 255, 130], [1, 1, 21, 34], 34)
        + gen_shapes([1, 1, 1], [2, 6, 254], [1, 1, 32], 8)
        + gen_shapes([1, 1, 1], [4, 12, 255], [1, 1, 32], 16)
        + gen_shapes([1, 1, 1], [8, 18, 256], [1, 1, 32], 32)
        + gen_shapes([1, 1], [2, 6], [1, 1], 2)
        + gen_shapes([1, 1], [3, 7], [1, 1], 2)
        + gen_shapes([1, 1], [4, 8], [1, 1], 2)
        + gen_shapes([1], [32], [1], 4)
        + gen_shapes([1], [33], [1], 5)
        + gen_shapes([1], [34], [1], 6),
        "dim": [
            0,
            1,
            2,
            3,
            None,
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 3],
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
            [0, 1, 2, 3],
        ],
        "keepdim": [True, False],
        "input_a_dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["input_layout"] == ttnn.ROW_MAJOR_LAYOUT and not (
        test_vector["input_a_dtype"] == ttnn.float32 or test_vector["input_a_dtype"] == ttnn.bfloat16
    ):
        return True, "Row major is only supported for fp32 & fp16"
    if not test_vector["keepdim"]:
        return True, "keepdim = false is not supported"

    device = ttnn.open_device(device_id=0)
    if test_vector["input_a_dtype"] == ttnn.float32 and ttnn.device.is_grayskull(device):
        return True, "Dest Fp32 mode is not supported for arch grayskull"
    ttnn.close_device(device)
    del device

    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    dim,
    keepdim,
    input_a_dtype,
    input_layout,
    input_a_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    if input_layout == ttnn.ROW_MAJOR_LAYOUT:
        input_shape = sanitize_shape_rm(input_shape)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)

    golden_function = ttnn.get_golden_function(ttnn.std)
    torch_output_tensor = golden_function(torch_input_tensor_a, dim=dim, keepdim=keepdim)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.std(input_tensor_a, dim=dim, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)

    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
