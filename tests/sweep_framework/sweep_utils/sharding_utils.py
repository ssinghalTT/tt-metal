# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import random
import ttnn
import math

from tests.sweep_framework.sweep_utils.utils import get_device_grid_size
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import (
    gen_func_with_cast_tt,
    _gen_reshape_args_from_volume,
    _get_factors,
)


Y, X = get_device_grid_size()


def gen_unary_sharded_spec(
    num_shapes,
    num_core_samples,
    shard_orientation,
    sharding_strategy,
    max_tensor_size_per_core=480 * 480,
):
    assert sharding_strategy in ["BLOCK", "WIDTH", "HEIGHT", "TENSOR_HW"]

    assert shard_orientation in ["COL_MAJOR", "ROW_MAJOR"]

    for i in range(num_core_samples):
        y = random.randint(1, Y)
        x = random.randint(1, X)
        max_tensor_size = y * x * max_tensor_size_per_core
        for j in range(num_shapes):
            for rank in [2, 3, 4]:
                if sharding_strategy == "TENSOR_HW":
                    min_tensor_height = 32
                    min_tensor_width = 32
                    mul_height = random.randint(1, 10)
                    mul_width = random.randint(1, 10)
                    tensor_height = min_tensor_height * mul_height
                    tensor_width = min_tensor_width * mul_width
                    input_shape = [tensor_height, tensor_width]
                    if rank != 2:
                        rest_volume = random.randint(1, max_tensor_size // (tensor_height * tensor_width))
                        rest_dims = random.choice(_gen_reshape_args_from_volume(rest_volume, step=1, out_dims=rank - 2))
                        rest_dims = list(rest_dims["reshape_dims"])

                elif sharding_strategy == "BLOCK":
                    min_pre_sharded_height = 32 * y
                    min_pre_sharded_width = 32 * x

                    mul_1 = random.randint(1, y * 2)
                    mul_2 = random.randint(1, x * 2)

                    if shard_orientation == "ROW_MAJOR":
                        pre_sharded_width = mul_1 * min_pre_sharded_width
                        pre_sharded_height = mul_2 * min_pre_sharded_height
                    else:
                        pre_sharded_width = mul_1 * min_pre_sharded_height
                        pre_sharded_height = mul_2 * min_pre_sharded_width

                    input_shape = random.choice(
                        _gen_reshape_args_from_volume(pre_sharded_height, step=1, out_dims=rank - 1)
                    )
                    input_shape = list(input_shape["reshape_dims"])
                    input_shape.append(pre_sharded_width)

                elif sharding_strategy == "HEIGHT":
                    min_pre_sharded_height = 32 * y * x
                    min_pre_sharded_width = 32

                    mul_1 = random.randint(1, 16)

                    pre_sharded_width = mul_1 * min_pre_sharded_width
                    pre_sharded_height = random.randrange(
                        min_pre_sharded_height, max_tensor_size // pre_sharded_width + 1, 32 * y * x
                    )

                    input_shape = random.choice(
                        _gen_reshape_args_from_volume(pre_sharded_height, step=1, out_dims=rank - 1)
                    )
                    input_shape = list(input_shape["reshape_dims"])
                    input_shape.append(pre_sharded_width)

                else:
                    min_pre_sharded_height = 32
                    min_pre_sharded_width = 32 * y * x

                    mul_1 = random.randint(1, 16)

                    pre_sharded_height = mul_1 * min_pre_sharded_height
                    pre_sharded_width = random.randrange(
                        min_pre_sharded_width, max_tensor_size // pre_sharded_height + 1, 32 * y * x
                    )

                    input_shape = random.choice(
                        _gen_reshape_args_from_volume(pre_sharded_height, step=1, out_dims=rank - 1)
                    )
                    input_shape = list(input_shape["reshape_dims"])
                    input_shape.append(pre_sharded_width)

                yield {
                    "input_shape": input_shape,
                    "core_grid_size": (y, x),
                    "sharding_strategy": sharding_strategy,
                    "shard_orientation": shard_orientation,
                }


def parse_sharding_spec(input_spec):
    input_shape = input_spec["input_shape"]
    sharding_strategy = input_spec["sharding_strategy"]
    shard_orientation = input_spec["shard_orientation"]
    core_grid_size = input_spec["core_grid_size"]

    assert sharding_strategy in ["HEIGHT", "WIDTH", "BLOCK", "TENSOR_HW"]

    tensor_hw_as_shard_shape = False

    if sharding_strategy == "HEIGHT":
        sharding_strategy = ttnn.ShardStrategy.HEIGHT
    elif sharding_strategy == "WIDTH":
        sharding_strategy = ttnn.ShardStrategy.WIDTH
    elif sharding_strategy == "BLOCK":
        sharding_strategy = ttnn.ShardStrategy.BLOCK
    else:
        sharding_strategy = ttnn.ShardStrategy.BLOCK
        tensor_hw_as_shard_shape = True

    if shard_orientation == "COL_MAJOR":
        shard_orientation = ttnn.ShardOrientation.COL_MAJOR
    else:
        shard_orientation = ttnn.ShardOrientation.ROW_MAJOR

    return input_shape, core_grid_size, shard_orientation, sharding_strategy, tensor_hw_as_shard_shape
