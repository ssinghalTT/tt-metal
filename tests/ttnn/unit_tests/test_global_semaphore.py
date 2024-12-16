# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


def run_global_semaphore(device):
    tensix_cores0 = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(3, 3),
            ),
        }
    )
    tensix_cores1 = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(4, 4),
                ttnn.CoreCoord(4, 4),
            ),
        }
    )
    global_sem0 = ttnn.create_global_semaphore(device, tensix_cores0, 1)
    global_sem1 = ttnn.create_global_semaphore(device, tensix_cores1, 2)

    assert ttnn.get_global_semaphore_address(global_sem0) != ttnn.get_global_semaphore_address(global_sem1)

    ttnn.reset_global_semaphore_value(global_sem0, 3)


@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
def test_global_semaphore(device, enable_async_mode):
    run_global_semaphore(device)


@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
def test_global_semaphore_mesh(mesh_device, enable_async_mode):
    run_global_semaphore(mesh_device)
