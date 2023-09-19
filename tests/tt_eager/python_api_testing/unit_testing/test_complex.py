# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys

import torch

import tt_lib as ttl
from models.utility_functions import print_diff_argmax
import pytest
from loguru import logger

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
    comp_equal,
)
from tests.tt_eager.python_api_testing.sweep_tests.common import (
    is_wormhole_b0,
    skip_for_wormhole_b0,
)
from functools import partial


def is_DRAM(memcfg: ttl.tensor.MemoryConfig):
    return memcfg.buffer_type == ttl.tensor.BufferType.DRAM


class Complex:
    def __init__(self, input_shape: torch.Size):
        val = 1 + torch.arange(0, input_shape.numel()).reshape(input_shape).bfloat16()
        self._cplx = (
            val[:, :, :, input_shape[-1] // 2]
            + val[:, :, :, input_shape[-1] // 2 :] * 1j
        )

    def reset(self, val: torch.Tensor):
        self._cplx = val

    @property
    def real(self):
        return self._cplx.real

    @property
    def imag(self):
        return self._cplx.imag

    @property
    def metal(self):
        return torch.cat([self.real, self.imag], -1)

    ## operations
    def abs(self):
        return (self.real**2 + self.imag**2).sqrt()

    def conj(self) -> "Complex":
        self._cplx = self._cplx.conj()
        return self

    def recip(self):
        self._cplx = 1.0 / self._cplx
        return self

    def add(self, that: "Complex"):
        self._cplx += that._cplx
        return self

    def sub(self, that: "Complex"):
        self._cplx -= that._cplx
        return self

    def __mul__(self, scale):
        self._cplx *= scale
        return self

    def mul(self, that: "Complex"):
        self._cplx *= that._cplx
        return self

    def div(self, that: "Complex"):
        self._cplx /= that._cplx
        return self


@pytest.mark.parametrize(
    "memcfg",
    (
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttl.tensor.DataType.BFLOAT16,)))
def test_level1_real(memcfg, dtype, device, function_level_defaults):
    if is_wormhole_b0() and is_DRAM(memcfg):
        pytest.skip("[REAL]: skip for DRAM inputs on WH B0")
    input_shape = torch.Size([1, 1, 32, 64])
    # check real
    x = Complex(input_shape)
    xtt = (
        ttl.tensor.Tensor(x.metal, dtype)
        .to(ttl.tensor.Layout.ROW_MAJOR)
        .to(device, memcfg)
    )
    tt_dev = ttl.tensor.real(xtt, memcfg)
    tt_dev = tt_dev.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    tt_cpu = x.real
    passing, output = comp_equal(tt_cpu, tt_dev)
    logger.info(output)
    assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttl.tensor.DataType.BFLOAT16,)))
@pytest.mark.parametrize("layout", ((ttl.tensor.Layout.ROW_MAJOR,)))
def test_level1_imag(memcfg, dtype, device, function_level_defaults, layout):
    if is_wormhole_b0() and is_DRAM(memcfg):
        pytest.skip("[IMAG]: skip with DRAM inputs on WH B0")
    input_shape = torch.Size([1, 1, 32, 64])
    # check imag
    x = Complex(input_shape)
    tt_cpu = x.imag
    xtt = ttl.tensor.Tensor(x.metal, dtype).to(layout).to(device, memcfg)
    tt_dev = ttl.tensor.imag(xtt)
    tt_dev = tt_dev.cpu().to(layout).to_torch()
    passing, output = comp_equal(tt_cpu, tt_dev)
    logger.info(output)
    assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttl.tensor.DataType.BFLOAT16,)))
@pytest.mark.parametrize("layout", ((ttl.tensor.Layout.ROW_MAJOR,)))
def test_level1_abs(memcfg, dtype, device, function_level_defaults, layout):
    input_shape = torch.Size([1, 1, 32, 64])
    # check abs
    x = Complex(input_shape)
    xtt = ttl.tensor.Tensor(x.metal, dtype).to(layout).to(device, memcfg)
    tt_dev = ttl.tensor.complex_abs(xtt, memcfg)
    tt_dev = tt_dev.cpu().to(layout).to_torch()
    tt_cpu = x.abs()
    if is_wormhole_b0() and is_DRAM(memcfg):
        pytest.skip("[ABS] skip for wormhole b0  and DRAM")
        passing, output = comp_pcc(tt_cpu, tt_dev, pcc=0.70)
    else:
        passing, output = comp_pcc(tt_cpu, tt_dev)
    logger.info(output)
    assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttl.tensor.DataType.BFLOAT16,)))
def test_level1_conj(memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([1, 1, 32, 64])
    # check abs
    x = Complex(input_shape)
    xtt = (
        ttl.tensor.Tensor(x.metal, dtype)
        .to(ttl.tensor.Layout.ROW_MAJOR)
        .to(device, memcfg)
    )
    tt_dev = ttl.tensor.conj(xtt, memcfg)
    tt_dev = tt_dev.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    tt_cpu = x.conj().metal
    if is_wormhole_b0():
        pytest.skip("[CONJ] skip for wormhole b0  since we see a PCC of -0.7 so odd!")
        passing, output = comp_pcc(tt_cpu, tt_dev, pcc=0.8)
    else:
        passing, output = comp_pcc(tt_cpu, tt_dev)
    logger.info(output)
    assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttl.tensor.DataType.BFLOAT16,)))
def test_level1_add(memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([1, 1, 32, 64])
    # check abs
    x = Complex(input_shape)
    y = Complex(input_shape) * -0.5

    xtt = (
        ttl.tensor.Tensor(x.metal, dtype)
        .to(ttl.tensor.Layout.ROW_MAJOR)
        .to(device, memcfg)
    )
    ytt = (
        ttl.tensor.Tensor(y.metal, dtype)
        .to(ttl.tensor.Layout.ROW_MAJOR)
        .to(device, memcfg)
    )

    tt_dev = ttl.tensor.add(xtt, ytt)
    tt_dev = tt_dev.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    tt_cpu = x.add(y).metal

    if is_wormhole_b0():
        pytest.skip("[ADD]: skip assert on WH B0")

    passing, output = comp_pcc(tt_cpu, tt_dev)
    logger.info(output)
    assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttl.tensor.DataType.BFLOAT16,)))
def test_level1_sub(memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([1, 1, 32, 64])
    # check abs
    x = Complex(input_shape)
    y = Complex(input_shape) * 0.5

    xtt = (
        ttl.tensor.Tensor(x.metal, dtype)
        .to(ttl.tensor.Layout.ROW_MAJOR)
        .to(device, memcfg)
    )
    ytt = (
        ttl.tensor.Tensor(y.metal, dtype)
        .to(ttl.tensor.Layout.ROW_MAJOR)
        .to(device, memcfg)
    )

    tt_dev = ttl.tensor.sub(xtt, ytt)
    tt_dev = tt_dev.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    tt_cpu = x.sub(y).metal

    if is_wormhole_b0():
        pytest.skip("[SUB]: skip assert on WH B0")

    passing, output = comp_pcc(tt_cpu, tt_dev)
    logger.info(output)
    assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttl.tensor.DataType.BFLOAT16,)))
def test_level1_mul(memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([1, 1, 32, 64])
    # check abs
    x = Complex(input_shape)
    y = Complex(input_shape) * 0.75

    xtt = (
        ttl.tensor.Tensor(x.metal, dtype)
        .to(ttl.tensor.Layout.ROW_MAJOR)
        .to(device, memcfg)
    )
    ytt = (
        ttl.tensor.Tensor(y.metal, dtype)
        .to(ttl.tensor.Layout.ROW_MAJOR)
        .to(device, memcfg)
    )

    tt_dev = ttl.tensor.complex_mul(xtt, ytt, memcfg)
    tt_dev = tt_dev.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    tt_cpu = x.mul(y).metal

    if is_wormhole_b0():
        pytest.skip("[MUL] test not asserted for WH B0")
    passing, output = comp_pcc(tt_cpu, tt_dev, pcc=0.96)
    logger.info(output)
    assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttl.tensor.DataType.BFLOAT16,)))
def test_level1_div(memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([1, 1, 32, 64])
    # check abs
    x = Complex(input_shape)
    y = x.div(x)

    xtt = (
        ttl.tensor.Tensor(x.metal, dtype)
        .to(ttl.tensor.Layout.ROW_MAJOR)
        .to(device, memcfg)
    )
    ytt = (
        ttl.tensor.Tensor(y.metal, dtype)
        .to(ttl.tensor.Layout.ROW_MAJOR)
        .to(device, memcfg)
    )

    tt_dev = ttl.tensor.complex_div(ytt, xtt, memcfg)
    tt_dev = tt_dev.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    tt_cpu = y.div(x).metal

    if is_wormhole_b0():
        pytest.skip("[DIV]: skip assertion for this test on WH B0")

    passing, output = comp_pcc(tt_cpu, tt_dev, pcc=0.96)
    logger.info(output)
    assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttl.tensor.DataType.BFLOAT16,)))
def test_level1_recip(memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([1, 1, 32, 64])
    # check abs
    x = Complex(input_shape)
    x = x.div(x * 0.5)
    xtt = (
        ttl.tensor.Tensor(x.metal, dtype)
        .to(ttl.tensor.Layout.ROW_MAJOR)
        .to(device, memcfg)
    )

    tt_dev = ttl.tensor.complex_recip(xtt, memcfg)
    tt_dev = tt_dev.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    tt_cpu = x.recip().metal

    if is_wormhole_b0():
        pytest.skip("[RECIP]: skip assertion for this test on WH B0")

    passing, output = comp_pcc(tt_cpu, tt_dev, pcc=0.96)
    logger.info(output)
    assert passing
