# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


def preprocess_conv2d_weights(weight, mesh_mapper, mesh_device):
    # weight = weight.T.contiguous()
    weight = ttnn.from_torch(
        weight,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=mesh_mapper,
    )
    return weight


def preprocess_conv2d_bias(bias, mesh_mapper, mesh_device):
    bias = bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    bias = ttnn.from_torch(
        bias, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mesh_mapper, device=mesh_device
    )
    return bias


def preprocess_linear_weights(weight, mesh_mapper, mesh_device):
    weight = weight.T.contiguous()
    weight = ttnn.from_torch(
        weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh_mapper, device=mesh_device
    )
    return weight


def preprocess_linear_bias(bias, mesh_mapper, mesh_device):
    bias = ttnn.from_torch(
        bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh_mapper, device=mesh_device
    )
    return bias


def custom_preprocessor(state_dict, mesh_mapper, mesh_device):
    parameters = {}
    for name, parameter in state_dict.items():
        if "features." in name and "weight" in name:
            parameters[name] = preprocess_conv2d_weights(parameter, mesh_mapper, mesh_device)
        if "features." in name and "bias" in name:
            parameters[name] = preprocess_conv2d_bias(parameter, mesh_mapper, mesh_device)

        if "classifier." in name and "weight" in name:
            parameters[name] = preprocess_linear_weights(parameter, mesh_mapper, mesh_device)
        if "classifier." in name and "bias" in name:
            parameters[name] = preprocess_linear_bias(parameter, mesh_mapper, mesh_device)

    return parameters
