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


def run_conv_concat(
    device,
    input,
    torch_input,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    use_1d_systolic_array,
    config_override,
    dilation=1,
    use_shallow_conv_variant=False,
    transpose_mcast=True,
    enable_auto_formatting=False,
    padded_input_channels=None,
    fp32_accum=False,
    packer_l1_acc=False,
    output_layout=ttnn.TILE_LAYOUT,
    deallocate_activation=False,
    debug=False,
    groups=1,
    has_bias=True,
):
    torch.manual_seed(0)
    conv_input_shape = [batch_size, input_channels, input_height, input_width]
    conv_weight_shape = [output_channels, input_channels // groups, filter_height, filter_width]
    conv_bias_shape = [1, 1, 1, output_channels]
    torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()

    torch_bias_tensor = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float() if has_bias else None
    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input,
        torch_weight_tensor,
        bias=torch_bias_tensor.reshape(-1) if has_bias else None,
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dilation, dilation),
        groups=groups,
    )

    output_shape_nhwc = [
        torch_out_golden_tensor.shape[0],
        torch_out_golden_tensor.shape[2],
        torch_out_golden_tensor.shape[3],
        torch_out_golden_tensor.shape[1],
    ]

    reader_patterns_cache = {}

    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
    )
    tt_bias_tensor = None
    if has_bias:
        tt_bias_tensor = ttnn.from_torch(
            torch_bias_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
        )

    tt_input_tensor = input
    conv_config = ttnn.Conv2dConfig(
        dtype=activations_dtype,
        weights_dtype=weights_dtype,
        math_fidelity=math_fidelity,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        input_channels_alignment=(
            16 if use_shallow_conv_variant or (input_channels == 16 and input_height == 115) else 32
        ),
        deallocate_activation=deallocate_activation,
        fp32_dest_acc_enabled=fp32_accum,
        packer_l1_accum_enabled=packer_l1_acc,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
    )
    if config_override and "act_block_h" in config_override:
        conv_config.act_block_h_override = config_override["act_block_h"]
        print("Setting Act Block H to ", conv_config.act_block_h_override)
    if config_override and "num_cores_nhw" in config_override:
        if config_override["num_cores_nhw"] == 98:
            conv_config.core_grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (11, 7)), ttnn.CoreRange((0, 8), (1, 8))})
            conv_config.override_sharding_config = True
            print("Setting num_cores_nhw to 98")

    [tt_output_tensor_on_device, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        bias_tensor=tt_bias_tensor,
        kernel_size=(filter_height, filter_width),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dilation, dilation),
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        conv_config=conv_config,
        conv_op_cache=reader_patterns_cache,
        debug=debug,
        groups=groups,
    )

    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)

    # torch_output_tensor is in row major layout and NHWC shape
    # NHWC to NCHW
    torch_output_tensor = torch_output_tensor.reshape(batch_size, out_height, out_width, torch_output_tensor.shape[-1])
    torch_output_tensor = torch_output_tensor[:, :, :, :output_channels]

    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))
    reader_patterns_cache.clear()

    if not fp32_accum:
        pcc = 0.99
    elif math_fidelity == ttnn.MathFidelity.LoFi and activations_dtype == ttnn.bfloat8_b:
        pcc = 0.9969
    else:
        pcc = 0.998
    passing, pcc_msg = check_with_pcc(torch_output_tensor, torch_out_golden_tensor, pcc=pcc)
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_channels, output_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, groups, use_1d_systolic_array, config_override, use_shallow_conv_variant",
    ((1, 512, 512, 20, 20, 1, 1, 1, 1, 0, 0, 1, True, None, False),),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
def test_concat_conv_yolov4(
    device,
    use_program_cache,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    use_1d_systolic_array,
    config_override,
    use_shallow_conv_variant,
    groups,
    output_layout,
):
    torch_input = torch.randn(1, 256, 20, 20, dtype=torch.bfloat16).float()
    concat_out = torch.concat((torch_input, torch_input), dim=1)
    torch_input_nhwc = torch.permute(torch_input, (0, 2, 3, 1))
    torch_input_nhwc = torch.reshape(torch_input_nhwc, (1, 1, 400, 256))
    ttnn_input = ttnn.from_torch(torch_input_nhwc, dtype=ttnn.bfloat16, device=device)

    output_sharded_memory_config = ttnn.create_sharded_memory_config(
        [32, 512],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (7, 0)), ttnn.CoreRange((0, 1), (4, 1))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn.concat([ttnn_input, ttnn_input], dim=3, memory_config=output_sharded_memory_config)

    run_conv_concat(
        device,
        output_tensor,
        concat_out,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        use_1d_systolic_array,
        config_override,
        use_shallow_conv_variant=False,
        groups=1,
        padded_input_channels=16 if input_channels == 3 else None,
        output_layout=ttnn.TILE_LAYOUT,
    )
