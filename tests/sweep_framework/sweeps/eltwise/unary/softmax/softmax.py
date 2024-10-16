# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial
import tracy

import torch
import random
import ttnn
from tests.sweep_framework.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)

# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "softmax_test_suite_2048": {
        "input_shape": gen_shapes([1, 1, 64, 64], [1, 1, 1024, 1024], [1, 1, 64, 64]),
        # "input_shape" : gen_shapes([1, 1, 10, 10], [1, 1, 1000, 1000], [1, 1, 100, 100]),
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


def tracy_testing(input_shape, input_a_dtype, input_a_layout, input_a_memory_config, output_memory_config, device):
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float16), input_a_dtype
    )(input_shape)
    torch_output_tensor = torch.softmax(torch_input_tensor_a, -1)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    result = ttnn.softmax(input_tensor_a, dim=-1, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(result)


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a device_mesh_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    profiler = tracy.Profiler()
    profiler.enable()

    tracy_testing(input_shape, input_a_dtype, input_a_layout, input_a_memory_config, output_memory_config, device)
    ttnn.DumpDeviceProfiler(device)
    ttnn.synchronize_device(device)
    profiler.disable()
    return [(True, "OK"), 0]
