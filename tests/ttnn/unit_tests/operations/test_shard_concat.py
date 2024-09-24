# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest

import ttnn
from models.utility_functions import (
    skip_for_grayskull,
)

from tests.ttnn.utils_for_testing import check_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("should_fail", [True])
def test_concat(device, should_fail):
    torch_input = torch.randn(1, 1, 400, 256, dtype=torch.bfloat16)
    goldern_concat = torch.cat((torch_input, torch_input), dim=3)

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_input = ttnn.to_memory_config(
        ttnn_input,
        memory_config=ttnn.create_sharded_memory_config(
            [32, 256],
            core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (7, 0)), ttnn.CoreRange((0, 1), (4, 1))}),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        ),
    )
    output_sharded_memory_config = ttnn.create_sharded_memory_config(
        [32, 512],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (7, 0)), ttnn.CoreRange((0, 1), (4, 1))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )

    if should_fail:
        ttnn_concat_result = ttnn.concat([ttnn_input, ttnn_input], dim=3, memory_config=output_sharded_memory_config)
    else:
        ttnn_concat_result = ttnn.concat([ttnn_input, ttnn_input], dim=3)
        ttnn_concat_result = ttnn.to_memory_config(ttnn_concat_result, output_sharded_memory_config)
    ttnn_output_as_torch = ttnn.to_torch(ttnn_concat_result)

    pcc = 0.99
    passing, pcc_msg = check_with_pcc(goldern_concat, ttnn_output_as_torch, pcc=pcc)
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing
