# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import math
from models.utility_functions import is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "act_shape",  ## NCHW
    (
        (
            [1, 48, 128, 128],
            [1, 96, 32, 32],
        )
    ),
)
@pytest.mark.parametrize(
    "kernel_size",
    ((2, 2),),
)
@pytest.mark.parametrize(
    "padding",
    ((0, 0),),
)
@pytest.mark.parametrize(
    "stride",
    ((2, 2),),
)
@pytest.mark.parametrize("dilation", ((1, 1),))  ## default
@pytest.mark.parametrize(
    "nblocks",
    (1,),
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_run_max_pool(
    act_shape,
    kernel_size,
    padding,
    stride,
    dilation,
    nblocks,
    device,
    dtype,
):
    in_n, in_c, in_h, in_w = act_shape
    kernel_h, kernel_w = kernel_size
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    if 2 * pad_h > kernel_h or 2 * pad_w > kernel_w:
        pytest.skip("Invalid case")

    if (kernel_h == 3 and pad_h != 1) or (kernel_h == 2 and pad_h != 0):
        pytest.skip("kernel size and padding combination not supported")

    out_h = math.floor((in_h + 2 * pad_h - (dilation_h * kernel_h - 1) - 1) / stride_h) + 1
    out_w = math.floor((in_w + 2 * pad_w - (dilation_w * kernel_w - 1) - 1) / stride_w) + 1
    if out_w % nblocks != 0:
        pytest.skip(f"Unsupported case when out_w ({out_w}) % nblocks ({nblocks}) != 0")

    if in_c % 16 != 0:
        pytest.skip("Current maxpool writer needs nchannels to be multiple of 16!")

    if in_c == 16 and dtype == ttnn.bfloat8_b and in_n * in_h * in_w > 600000:
        pytest.skip("This case runs out of memory on Grayskull")

    if in_n >= 16 and in_c >= 64 and dtype == ttnn.bfloat8_b and is_wormhole_b0():
        pytest.skip("This case runs out of memory on Wormhole b0")

    if (
        is_wormhole_b0()
        and act_shape == [16, 64, 112, 112]
        and kernel_size == (3, 3)
        and padding == (1, 1)
        and stride == (2, 2)
        and dilation == (1, 1)
        and dtype == ttnn.bfloat16
    ):
        pytest.skip("Issue #6992: Statically allocated circular buffers in program clash with L1 buffers on core range")

    torch.manual_seed(0)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=500, threshold=10000, edgeitems=32)

    ## construct the tensor in NCHW shape
    act = torch.randn(act_shape, dtype=torch.bfloat16)
    # act = torch.zeros(act_shape, dtype=torch.bfloat16)
    # act = torch.ones(act_shape, dtype=torch.bfloat16)
    # act = torch.arange(0, volume(act_shape), dtype=torch.bfloat16).reshape(act_shape)
    # for n in range(act_shape[0]):
    #     for c in range(act_shape[1]):
    #         for h in range(act_shape[2]):
    #             for w in range(act_shape[3]):
    #                 act[n, c, h, w] = 1 + n + h + w + c # + torch.rand(1) * 0.15
    # torch.save(act, "act.pt")
    # act = torch.load("act.pt")

    ## this op expects input tensor as { N, 1, H * W, C }, so rearrange and reshape tensor
    ## but before that, make sure in_c is multiple of tile width
    act_shape = (in_n, 1, in_h * in_w, in_c)
    act_permuted = torch.permute(act, (0, 2, 3, 1))

    act_reshaped = act_permuted.reshape(act_shape)

    reader_patterns_cache = {}
    max_pool = ttnn.MaxPool2d(
        kernel_size=(kernel_h, kernel_w),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dilation_h, dilation_w),
        dtype=dtype,
        device=device,
        batch_size=in_n,
        input_height=in_h,
        input_width=in_w,
        reader_patterns_cache=reader_patterns_cache,
    )

    if dtype == ttnn.bfloat8_b:
        if (in_h * in_w) % 32 != 0:
            pytest.skip("For BFP8_B datatype, input height * width should be multiple of 32")
        ttact = ttnn.from_torch(act_reshaped, dtype, layout=ttnn.TILE_LAYOUT)
    else:
        ttact = ttnn.from_torch(act_reshaped, dtype)
    ttact_d = max_pool.copy_input_to_device(ttact)

    out_d = max_pool(ttact_d)
    out_padded = max_pool.copy_output_from_device(out_d)

    # clear the cache maps
    reader_patterns_cache.clear()

    out_pytorch_padded = ttnn.to_torch(out_padded)
    out_pytorch = out_pytorch_padded[:, :, :, :in_c]

    ## reference
    golden_pytorch = torch.nn.MaxPool2d(
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        return_indices=False,
        ceil_mode=False,
    )(act)

    ## test for equivalance
    golden_shape = golden_pytorch.shape
    out_pytorch = out_pytorch.reshape(golden_shape[0], golden_shape[2], golden_shape[3], golden_shape[1])
    out_pytorch = torch.permute(out_pytorch, (0, 3, 1, 2))  ## N, C, H, W

    assert_with_pcc(out_pytorch, golden_pytorch)


"""
E       RuntimeError: TT_FATAL @ ../ttnn/cpp/ttnn/operations/pool/maxpool/device/max_pool_program_factory.cpp:30: is_pow2
E       info:
E       Row size (nchannels * bytes = 192) should be power of 2 (false).
E       backtrace:
E        --- tt::tt_metal::MaxPool::validate(std::__1::vector<tt::tt_metal::Tensor, std::__1::allocator<tt::tt_metal::Tensor>> const&) const
"""
