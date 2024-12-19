# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("w", [1, 4, 8, 32])
def test_plus_one(device, w):
    device.enable_async(True)
    torch_input_tensor = torch.randint(32000, (w,))
    torch_output_tensor = torch_input_tensor + 1

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.int32, device=device)
    ttnn.plus_one(input_tensor)
    output_tensor = ttnn.to_torch(input_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    device.enable_async(False)


@pytest.mark.parametrize("w", [1, 4, 8, 32])
@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
def test_plus_one_subdevice(device, w):
    device.enable_async(True)
    torch_input_tensor = torch.randint(32000, (w,))
    torch_output_tensor = torch_input_tensor + 1
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.int32, device=device)
    ttnn.plus_one(
        input_tensor, sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(1, 1))])
    )
    output_tensor = ttnn.to_torch(input_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    device.enable_async(False)
