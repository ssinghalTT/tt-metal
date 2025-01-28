# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import torch
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.llama3_subdevices.tt.llama_ccl import tt_all_reduce


class LMHead(LightweightModule):
    def __init__(
        self,
        args,
        mesh_device,
        dtype,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        max_columns_per_device=128256 // 4,  # larger values per device lead to OOM or hangs
        tt_ccl=None,
    ):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.vocab_size = args.vocab_size
        self.padded_vocab_size = args.padded_vocab_size
        self.num_devices = args.num_devices
        self.tt_ccl = tt_ccl

        size_per_device = self.vocab_size // self.num_devices

        if args.is_galaxy:
            size_per_device = self.padded_vocab_size // self.num_devices
        num_splits = math.ceil(size_per_device / max_columns_per_device)

        split_sizes = [min(size_per_device, max_columns_per_device)] * (num_splits - 1)
        split_sizes.append(size_per_device - sum(split_sizes))  # remaining columns

        # Split the output weights
        torch_output_weights = state_dict[f"{state_dict_prefix}output.weight"].permute(1, 0)

        self.output_weights = []
        if args.is_galaxy:
            num_splits = 4
            cache_file_name = (
                None if args.dummy_weights else weight_cache_path / f"output_lm_head_{num_splits}_split_shard_0"
            )
            padded_lm_head = torch.zeros(1, 1, args.dim, self.padded_vocab_size)
            padded_lm_head[:, :, :, : self.vocab_size] = torch_output_weights

            if args.is_70b:
                memory_config = ttnn.DRAM_MEMORY_CONFIG
            else:
                memory_config = (
                    ttnn.DRAM_MEMORY_CONFIG
                    if args.dim == 2048
                    else args.create_dram_sharded_mem_config(k=args.dim // 4, n=self.padded_vocab_size // 8)
                )
            for i in range(num_splits):
                index = i * self.padded_vocab_size // num_splits
                self.output_weights.append(  # (2k, 16k) 128* 1024
                    ttnn.as_tensor(
                        padded_lm_head[..., index : index + self.padded_vocab_size // 4],
                        device=mesh_device,
                        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, 2), mesh_shape=args.cluster_shape),
                        layout=ttnn.TILE_LAYOUT,
                        dtype=dtype,
                        memory_config=memory_config,
                        # cache_file_name=cache_file_name,
                    )
                )
        else:
            for i, split_size in enumerate(split_sizes):
                cache_file_name = (
                    None if args.dummy_weights else weight_cache_path / f"output_lm_head_{num_splits}_split_shard_{i}"
                )

                # Create a list to store the split tensors for each device
                device_splits = []
                for device in range(self.num_devices):
                    start = device * size_per_device + sum(split_sizes[:i])
                    end = start + split_size
                    device_splits.append(torch_output_weights[:, start:end])

                # Concatenate the splits from all devices
                combined_split = torch.cat(device_splits, dim=-1)

                memory_config = args.create_dram_sharded_mem_config(
                    k=args.dim, n=combined_split.shape[-1] // self.num_devices
                )
                self.output_weights.append(
                    ttnn.as_tensor(
                        combined_split,
                        device=mesh_device,
                        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
                        layout=ttnn.TILE_LAYOUT,
                        dtype=dtype,
                        memory_config=memory_config,
                        cache_file_name=cache_file_name,
                    )
                )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
            dst_full_sync_en=True,
        )

        self.output_memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
        if args.is_galaxy:
            # self.program_configs = [
            #     None
            #     if args.dim == 2048
            #     else args.dram_matmul_config(
            #         args.tile_padded_batch_rows,  # (8k, 128k) -> (2k, 16k)
            #         args.dim // 4,
            #         16 * 1024,
            #         args.lm_head_core_grid.num_cores,
            #     )
            # ]

            self.program_configs = [args.model_config["LM_HEAD_TG_RING_PROGCFG"]] * num_splits

            if args.is_70b:
                self.output_memory_config = args.model_config["LM_HEAD_OUT_RING_MEMCFG"]
        else:
            self.program_configs = [
                args.dram_matmul_config(
                    args.tile_padded_batch_rows,
                    args.dim,
                    split_size,
                    args.lm_head_core_grid.num_cores,
                )
                for split_size in split_sizes
            ]

    def forward_on_host(self, x: ttnn.Tensor):
        x_torch = ttnn.to_torch(
            x,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                self.mesh_device,
                dims=(0, 3),
                mesh_shape=(8, 4),
            ),
        )  # [8, 1, 32, 2048 * 4]
        x_torch = x_torch[:1]

        weight_torch = ttnn.to_torch(
            self.output_weights[0],
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device=self.mesh_device, dims=(3, 2), mesh_shape=list(self.mesh_device.shape)
            ),
        )

        output_torch = torch.matmul(x_torch.float(), weight_torch.float())

        output = ttnn.as_tensor(
            output_torch,
            dtype=ttnn.bfloat8_b,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device=self.mesh_device, dims=(3, None), mesh_shape=list(self.mesh_device.shape)
            ),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        return output

    def forward(self, x: ttnn.Tensor):
        # workaround for OOM issue
        # return self.forward_on_host(x)

        # ttnn.device.dump_device_memory_state(self.mesh_device.get_device(self.mesh_device.get_device_ids()[0]), prefix="")

        outputs = []
        for weight, pc in zip(self.output_weights, self.program_configs):
            weight_l1 = ttnn.to_memory_config(weight, self.args.model_config["LM_HEAD_RING_MEMCFG"])
            output = ttnn.linear(
                x,
                weight_l1,
                compute_kernel_config=self.compute_kernel_config,
                program_config=pc,
                memory_config=self.output_memory_config,
                dtype=ttnn.bfloat8_b,
            )
            outputs.append(ttnn.sharded_to_interleaved(output, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            weight_l1.deallocate(True)
            output.deallocate(True)

        outputs_reduced = []
        for output in outputs:
            output_reduced = self.tt_ccl.line_all_reduce(
                output, cluster_axis=1, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            outputs_reduced.append(output_reduced)

        return outputs_reduced
