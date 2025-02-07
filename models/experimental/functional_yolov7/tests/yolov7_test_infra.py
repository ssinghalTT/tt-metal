# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import os
import pytest
import torch
import torchvision
from tests.ttnn.utils_for_testing import assert_with_pcc
import ttnn
from models.experimental.functional_yolov7.reference.model import Yolov7_model
from models.experimental.functional_yolov7.ttnn.tt_yolov7 import ttnn_yolov7 as TtYolov7_model
from tests.ttnn.integration_tests.yolov7.test_ttnn_yolov7 import create_custom_preprocessor
from models.experimental.functional_yolov7.reference.yolov7_utils import download_yolov7_weights
from ttnn.model_preprocessing import preprocess_model_parameters


from models.utility_functions import (
    is_wormhole_b0,
    is_grayskull,
    divup,
)


class Yolov7TestInfra:
    def __init__(
        self,
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator=None,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.batch_size = batch_size
        self.act_dtype = act_dtype
        self.weight_dtype = weight_dtype
        self.model_location_generator = model_location_generator

        torch_model = Yolov7_model()

        new_state_dict = {}
        keys = [name for name, parameter in torch_model.state_dict().items()]
        ds_state_dict = {k: v for k, v in torch_model.state_dict().items()}
        values = [parameter for name, parameter in ds_state_dict.items()]
        for i in range(len(keys)):
            new_state_dict[keys[i]] = values[i]
        torch_model.load_state_dict(new_state_dict)
        torch_model.eval()

        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(None), device=None
        )
        weights_path = download_yolov7_weights("tests/ttnn/integration_tests/yolov7/yolov7.pt")

        self.load_weights(torch_model, weights_path)

        grid = [torch.randn(1)] * 3
        nx_ny = [80, 40, 20]
        grid_tensors = []
        for i in range(3):
            yv, xv = torch.meshgrid([torch.arange(nx_ny[i]), torch.arange(nx_ny[i])])
            grid_tensors.append(torch.stack((xv, yv), 2).view((1, 1, nx_ny[i], nx_ny[i], 2)).float())

        self.ttnn_yolov7_model = TtYolov7_model(device, parameters, grid_tensors)

        input_shape = (1, 640, 640, 3)
        torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
        self.input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)
        self.torch_input_tensor = torch_input_tensor.permute(0, 3, 1, 2)
        self.torch_output_tensor = torch_model(self.torch_input_tensor)

    def load_weights(self, model, weights_path):
        ckpt = torch.load(weights_path, map_location="cpu")
        state_dict = ckpt["model"].float().state_dict()
        model.load_state_dict(state_dict, strict=False)

    def run(self):
        self.output_tensor = self.ttnn_yolov7_model(self.input_tensor)

    def setup_l1_sharded_input(self, device, torch_input_tensor=None):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            exit("Unsupported device")
        num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()
        # torch tensor
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor

        n, c, h, w = torch_input_tensor.shape
        # sharded mem config for fold input
        num_cores = core_grid.x * core_grid.y
        shard_h = (n * w * h + num_cores - 1) // num_cores
        grid_size = core_grid
        grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, 64), ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardMode.PHYSICAL)
        # input_mem_config = ttnn.MemoryConfig(
        #     ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
        # )
        input_mem_config = ttnn.DRAM_MEMORY_CONFIG

        torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)
        torch_input_tensor = torch_input_tensor.reshape(1, 1, h * w * n, c)
        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        # tt_inputs_host = ttnn.pad(tt_inputs_host, [1, 1, n * h * w, 16], [0, 0, 0, 0], 0)
        return tt_inputs_host, input_mem_config

    def setup_dram_sharded_input(self, device, torch_input_tensor=None, mesh_mapper=None, mesh_composer=None):
        tt_inputs_host, input_mem_config = self.setup_l1_sharded_input(device)
        dram_grid_size = device.dram_grid_size()
        dram_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
            ),
            [
                divup(tt_inputs_host.volume() // tt_inputs_host.shape[-1], (dram_grid_size.x * dram_grid_size.y)),
                16,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded_mem_config_DRAM = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        )

        return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config

    def validate(self, output_tensor=None):
        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        output_tensor = ttnn.to_torch(self.output_tensor[0])
        output_tensor = output_tensor.reshape(1, 40, 40, 255)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

        valid_pcc = 0.985
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor[0], output_tensor, pcc=valid_pcc)

        logger.info(
            f"Yolov7 batch_size={self.batch_size}, act_dtype={self.act_dtype}, weight_dtype={self.weight_dtype}, PCC={self.pcc_message}"
        )


def create_test_infra(
    device,
    batch_size,
    act_dtype,
    weight_dtype,
    model_location_generator=None,
):
    return Yolov7TestInfra(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator,
    )
