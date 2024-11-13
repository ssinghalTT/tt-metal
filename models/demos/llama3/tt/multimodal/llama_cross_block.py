# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional
import torch

import ttnn
from models.utility_functions import (
    nearest_32,
)
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.demos.llama3.tt.multimodal.llama_cross_attention import TtLlamaCrossAttention
from models.demos.llama3.tt.llama_mlp import TtLlamaMLP
from models.demos.llama3.tt.distributed_norm import DistributedNorm
import os


class TtLlamaCrossAttentionTransformerBlock(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        configuration,
        no_ffn=False,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.num_devices = configuration.num_devices
        self.n_heads = configuration.n_heads
        self.n_kv_heads = configuration.n_kv_heads
        self.hidden_size = configuration.dim
        self.head_dim = self.hidden_size // self.n_heads
        self.model_config = configuration.get_model_config()

        assert not no_ffn, "No FFN not supported"

        self.attention = TtLlamaCrossAttention(
            mesh_device,
            state_dict,
            state_dict_prefix=f"{state_dict_prefix}attention.",
            weight_cache_path=weight_cache_path,
            dtype=dtype,
            configuration=configuration,
            dim=self.hidden_size,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            norm_eps=configuration.norm_eps,
        )

        self.attention_norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=self.hidden_size,
                state_dict=state_dict,
                state_dict_prefix=state_dict_prefix,
                weight_cache_path=None if configuration.dummy_weights else weight_cache_path,
                weight_key="attention_norm",
                is_distributed=configuration.is_distributed_norm,
                sharded_program_config=self.model_config["XATTN_SHARDED_NORM_ATTN_PRGM_CFG"],
                sharded_output_config=self.model_config["DECODE_VISION_INPUT_MEMCFG"],
            ),
            configuration,
        )

        self.gate_attn = ttnn.as_tensor(
            state_dict[f"{state_dict_prefix}gate_attn"].unsqueeze(0).expand(1, self.hidden_size),
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.feed_forward = TtLlamaMLP(
            mesh_device=mesh_device,
            args=configuration,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=-1,
            dtype=dtype,
            model_config=self.model_config,
            state_dict_prefix=f"{state_dict_prefix}feed_forward",
        )

        self.ffn_norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=self.hidden_size,
                state_dict=state_dict,
                state_dict_prefix=state_dict_prefix,
                weight_cache_path=None if configuration.dummy_weights else weight_cache_path,
                weight_key="ffn_norm",
                is_distributed=configuration.is_distributed_norm,
                sharded_program_config=self.model_config["SHARDED_NORM_MLP_PRGM_CFG"],
                sharded_output_config=self.model_config["SHARDED_MLP_INPUT_MEMCFG"],
            ),
            configuration,
        )

        self.gate_ffwd = ttnn.as_tensor(
            state_dict[f"{state_dict_prefix}gate_ffwd"].unsqueeze(0).expand(1, self.hidden_size),
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def compute_xattn_kv_cache(self, xattn_tokens, xattn_cache, user_id):
        return self.attention.compute_xattn_kv_cache(xattn_tokens, xattn_cache, user_id)

    def forward(
        self,
        x_11SH,
        xattn_mask,
        # Broadcast ops broken so pass in two copies of same mask, different shapes
        full_text_row_masked_out_mask_11SD,
        full_text_row_masked_out_mask_1NSH,
        xattn_cache,
        mode,
        user_id=0,
    ):
        attn_out = self.attention(
            x_11SH=self.attention_norm(x_11SH, mode=mode),
            xattn_mask=xattn_mask,
            xattn_cache=xattn_cache,
            full_text_row_masked_out_mask_1NSH=full_text_row_masked_out_mask_1NSH,
            mode=mode,
            user_id=user_id,
        )

        attn_out = ttnn.mul(attn_out, ttnn.tanh(self.gate_attn))

        res = ttnn.add(x_11SH, attn_out)
        mlp_out = self.feed_forward(self.ffn_norm(res, mode=mode), mode=mode)
        if mode == "prefill":
            # Making the assumption that you never mask decode rows
            mlp_out = ttnn.mul(mlp_out, full_text_row_masked_out_mask_11SD)
        mlp_out = ttnn.mul(mlp_out, ttnn.tanh(self.gate_ffwd))
        out = ttnn.add(res, mlp_out)
        return out
