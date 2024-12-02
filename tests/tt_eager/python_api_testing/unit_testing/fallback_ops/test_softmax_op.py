# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import tt_lib.fallback_ops as fallback_ops
from models.utility_functions import (
    comp_allclose_and_pcc,
    comp_pcc,
)
from loguru import logger
import pytest


@pytest.mark.parametrize(
    "input_shape",
    [
        torch.Size([1, 3, 6, 4]),
        torch.Size([2, 35, 9, 6]),
        torch.Size([1, 2, 64, 32]),
        torch.Size([5, 10, 23, 32]),
    ],
)
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
@pytest.mark.parametrize("on_device", [True, False])
def test_softmax_fallback(input_shape, dim, on_device, device):
    torch.manual_seed(1234)

    x = torch.randn(input_shape).bfloat16().float()
    pt_out = torch.nn.functional.softmax(x, dim)

    # Test on host RM
    t0 = ttnn.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    if on_device:
        t0 = t0.to(device)

    t1 = fallback_ops.softmax(t0, dim)

    output = t1.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    comp_pass, _ = comp_pcc(pt_out, output, 0.9999)
    _, comp_out = comp_allclose_and_pcc(pt_out, output)
    logger.debug(comp_out)
    assert comp_pass
