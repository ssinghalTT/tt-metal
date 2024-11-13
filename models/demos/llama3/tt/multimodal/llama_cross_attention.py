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


class TtLlamaCrossAttention(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        configuration,
        dim,
        head_dim,
        n_heads,
        n_kv_heads,
        norm_eps,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.num_devices = configuration.num_devices

        self.dim = dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.norm_eps = norm_eps

        self.n_local_heads = self.n_heads // configuration.num_devices
        self.n_local_kv_heads = self.n_kv_heads // configuration.num_devices

        self.dtype = dtype

        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2
        self.compute_kernel_config_hifi4 = configuration.compute_kernel_config_hifi4
        self.compute_kernel_config_sdpa = configuration.compute_kernel_config_sdpa

        self.configuration = configuration

        self.model_config = configuration.get_model_config()
        self.is_multichip = configuration.is_multichip

        if configuration.dummy_weights or (weight_cache_path is None):
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / (f"{state_dict_prefix}.{name}")

        wq_str = f"{state_dict_prefix}wq.weight"
        wk_str = f"{state_dict_prefix}wk.weight"
        wv_str = f"{state_dict_prefix}wv.weight"
        wo_str = f"{state_dict_prefix}wo.weight"

        # when splitting the devices, we need to make sure that the number of heads is divisible by the number of devices
        assert self.n_heads % configuration.num_devices == 0
        assert self.n_kv_heads % configuration.num_devices == 0

        # TODO DRAM Shard the weights (see llama3 text)
        wq_mem_config = configuration.create_dram_sharded_mem_config(
            configuration.dim, configuration.dim // configuration.num_devices
        )

        self.wq = ttnn.as_tensor(
            self.state_dict[wq_str].transpose(-2, -1),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
            dtype=self.dtype,
            memory_config=wq_mem_config,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=cache_name("wq_dram_sharded"),
        )

        self.wk = ttnn.as_tensor(
            self.state_dict[wk_str].transpose(-2, -1),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=cache_name("wk_sharded"),
        )

        self.wv = ttnn.as_tensor(
            self.state_dict[wv_str].transpose(-2, -1),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=cache_name("wv_sharded"),
        )

        wo_mem_config = configuration.create_dram_sharded_mem_config(
            configuration.dim // configuration.num_devices, configuration.dim
        )
        self.wo = ttnn.as_tensor(
            self.state_dict[wo_str].transpose(-2, -1),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-2),
            memory_config=wo_mem_config,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=cache_name("wo_dram_sharded"),
        )

        self.scale = self.head_dim**-0.5

        self.q_norm = RMSNorm(
            device=mesh_device,
            dim=self.head_dim,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}",
            weight_cache_path=None if configuration.dummy_weights else weight_cache_path,
            weight_key="q_norm",
            eps=self.norm_eps,
        )

        self.k_norm = RMSNorm(
            device=mesh_device,
            dim=self.head_dim,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}",
            weight_cache_path=None if configuration.dummy_weights else weight_cache_path,
            weight_key="k_norm",
            eps=self.norm_eps,
        )

    def compute_xattn_kv_cache(self, xattn_tokens, xattn_cache, user_id):
        # Always runs with batch=1
        B, seqlen_y = xattn_tokens.shape[1], xattn_tokens.shape[2]
        assert B == 1, "Batch size must be 1"
        MAX_MM_SEQ_LEN = self.configuration.VISION_MAX_MM_SEQ
        if seqlen_y > MAX_MM_SEQ_LEN:
            xattn_tokens = ttnn.reshape(xattn_tokens, [1, B * seqlen_y // MAX_MM_SEQ_LEN, MAX_MM_SEQ_LEN, -1])

        xk = ttnn.linear(
            xattn_tokens,
            self.wk,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            program_config=self.model_config["VISION_XATTN_KV_PROGCFG"](seqlen_y, MAX_MM_SEQ_LEN),
        )
        xv = ttnn.linear(
            xattn_tokens,
            self.wv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            program_config=self.model_config["VISION_XATTN_KV_PROGCFG"](seqlen_y, MAX_MM_SEQ_LEN),
        )
        if seqlen_y > MAX_MM_SEQ_LEN:
            xk = ttnn.reshape(xk, [1, B, seqlen_y, -1])
            xv = ttnn.reshape(xv, [1, B, seqlen_y, -1])

        if self.n_local_kv_heads == 1:
            # Only a simple reshape required, no need to split
            xk = ttnn.reshape(xk, [B, 1, seqlen_y, -1])
            xv = ttnn.reshape(xv, [B, 1, seqlen_y, -1])
        else:
            # 1, B, S, D -> B, NH, S, DH
            xk, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                xk,
                xk,
                num_heads=self.n_local_kv_heads,
                num_kv_heads=self.n_local_kv_heads // 2,
                transpose_k_heads=False,
            )
            xv, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                xv,
                xv,
                num_heads=self.n_local_kv_heads,
                num_kv_heads=self.n_local_kv_heads // 2,
                transpose_k_heads=False,
            )
            # def create_heads(x):
            #     x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
            #     x = ttnn.reshape(x, [B, seqlen_y, self.n_local_kv_heads, self.head_dim])
            #     x = ttnn.transpose(x, 1, 2)
            #     x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
            #     return x

            # xk = create_heads(xk)
            # xv = create_heads(xv)

        xk = self.k_norm(xk, mode="decode")

        # NOTE: Doing repeat in xattn_cache generation to avoid massive overhead in forward
        xk = ttnn.repeat_interleave(xk, self.n_local_heads // self.n_local_kv_heads, dim=1)
        xv = ttnn.repeat_interleave(xv, self.n_local_heads // self.n_local_kv_heads, dim=1)

        k_cache, v_cache = xattn_cache

        # Work around fill_cache memory constraint by making these sharded
        k_fill = ttnn.interleaved_to_sharded(xk, self.model_config["XATTN_KV_PREFILL_MEM_CFG"](seqlen_y))
        v_fill = ttnn.interleaved_to_sharded(xv, self.model_config["XATTN_KV_PREFILL_MEM_CFG"](seqlen_y))

        ttnn.fill_cache(k_cache, k_fill, user_id)
        ttnn.fill_cache(v_cache, v_fill, user_id)

        return xattn_cache

        ### Below is how I would like to implement TMs, but it results in poor PCC
        xk = ttnn.to_layout(xk, layout=ttnn.ROW_MAJOR_LAYOUT)
        xv = ttnn.to_layout(xv, layout=ttnn.ROW_MAJOR_LAYOUT)

        xk = xk.reshape(bsz, seqlen_y, self.n_local_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen_y, self.n_local_kv_heads, self.head_dim)

        xk = ttnn.transpose(xk, 1, 2)
        xv = ttnn.transpose(xv, 1, 2)

        xk = ttnn.to_layout(xk, layout=ttnn.TILE_LAYOUT)
        xv = ttnn.to_layout(xv, layout=ttnn.TILE_LAYOUT)

        # PREFERRED METHOD
        # xk = xk.reshape(bsz, seqlen_y, self.n_local_kv_heads, self.head_dim)
        # xv = xv.reshape(bsz, seqlen_y, self.n_local_kv_heads, self.head_dim)
        # xk, xv = [ttnn.transpose(tensor, 1, 2) for tensor in (xk, xv)] # HANG!
        return [xk, xv]

    def forward_decode(self, x_11SH, xattn_mask, full_text_row_masked_out_mask_1NSH, xattn_cache):
        batch = xattn_cache[0].shape[0]
        xq = ttnn.linear(
            x_11SH,
            self.wq,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            program_config=self.model_config["DECODE_VISION_XATTN_Q_PROGCFG"],
        )

        xq = ttnn.sharded_to_interleaved(xq, ttnn.DRAM_MEMORY_CONFIG)

        # # Below is how we want to reshape. It results in poor PCC
        # # 1, B, D -> B, 1, NH, DH -> B, NH, 1, DH
        # xq = ttnn.to_layout(xq, layout=ttnn.ROW_MAJOR_LAYOUT)
        # # Tell shape about padding
        # xq = ttnn.reshape(
        #     xq,
        #     shape=ttnn.Shape(
        #         [1, 1, batch, xq.shape[-1]],
        #         [1, 1, xq.shape[-2], xq.shape[-1]],
        #     ),
        # )
        # xq = ttnn.reshape(xq, (1, batch, self.n_local_heads, self.head_dim))
        # xq = ttnn.to_layout(xq, layout=ttnn.TILE_LAYOUT)

        xq, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            xq, xq, num_heads=self.n_local_heads, num_kv_heads=self.n_local_heads // 2, transpose_k_heads=False
        )
        xq = ttnn.to_layout(xq, layout=ttnn.ROW_MAJOR_LAYOUT)
        xq = ttnn.slice(xq, (0, 0, 0, 0), (xq.shape[0], xq.shape[1], batch, xq.shape[3]))
        xq = ttnn.transpose(xq, 1, 2)
        xq = ttnn.to_layout(xq, layout=ttnn.TILE_LAYOUT)

        xq = self.q_norm(xq, mode="decode")

        xk, xv = xattn_cache
        cache_seq_len = xk.shape[-2]

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=32,
            k_chunk_size=128,
            exp_approx_mode=False,
        )

        # TODO: Can I get rid of the KV repeat_interleave?

        output = ttnn.transformer.scaled_dot_product_attention_decode(
            xq,
            xk,
            xv,
            is_causal=False,
            attn_mask=xattn_mask,
            scale=self.scale,
            program_config=program_config,
            compute_kernel_config=self.compute_kernel_config_sdpa,
        )

        # WARNING: this broadcast is also broken, must broadcast on host
        output = ttnn.mul(output, full_text_row_masked_out_mask_1NSH)

        # This is how we should be reshaping
        # output = ttnn.to_layout(output, layout=ttnn.ROW_MAJOR_LAYOUT)
        # output = ttnn.reshape(output, (1, 1, batch, self.n_local_heads * self.head_dim))
        # output = ttnn.to_layout(output, layout=ttnn.TILE_LAYOUT)

        # output = ttnn.to_layout(output, layout=ttnn.ROW_MAJOR_LAYOUT)
        # output = ttnn.transpose(output, 1, 2)  # 1, B, NH, DH -> 1, NH, B, DH
        # output = ttnn.slice(output, (0, 0, 0, 0), (1, self.n_local_heads, batch, self.head_dim))
        # output = ttnn.to_layout(output, layout=ttnn.TILE_LAYOUT)
        # output = ttnn.experimental.nlp_concat_heads(output)
        # output = ttnn.interleaved_to_sharded(output, self.model_config["DECODE_VISION_XATTN_OUTPUT_MEMCFG"])

        output = ttnn.interleaved_to_sharded(output, self.model_config["DECODE_VISION_SDPA_MEMCFG"])
        output = ttnn.experimental.nlp_concat_heads_decode(output, num_heads=self.n_local_heads)
        output = ttnn.reshard(output, self.model_config["DECODE_VISION_XATTN_OUTPUT_MEMCFG"])

        output = ttnn.matmul(
            output,
            self.wo,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.model_config["DECODE_VISION_XATTN_DENSE_PROGCFG"](batch),
        )

        output = ttnn.sharded_to_interleaved(output, ttnn.DRAM_MEMORY_CONFIG)

        # All reduce
        if self.is_multichip:
            dense_out_reduced = ttnn.reduce_scatter(
                output,
                scatter_dim=3,
                math_op=ttnn.ReduceType.Sum,
                num_links=1,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            return dense_out_reduced
        else:
            return output

    def forward_prefill(self, x_11SH, xattn_mask, full_text_row_masked_out_mask_1NSH, xattn_cache, user_id):
        seq_len = x_11SH.shape[-2]
        # B, S, D
        assert seq_len % 32 == 0 and seq_len > 0, "Seqlen must be divisible by 32"

        if seq_len > 1024:
            x_11SH = ttnn.reshape(x_11SH, [1, seq_len // 1024, 1024, -1])

        xq = ttnn.linear(
            x_11SH,
            self.wq,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            program_config=self.model_config["VISION_XATTN_Q_PROGCFG"](seq_len),
        )
        if seq_len > 1024:
            xq = ttnn.reshape(xq, [1, 1, seq_len, -1])

        ttnn.deallocate(x_11SH)

        xq, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            xq, xq, num_heads=self.n_local_heads, num_kv_heads=self.n_local_heads // 2, transpose_k_heads=False
        )

        xq = self.q_norm(xq, mode="prefill")

        k_cache, v_cache = xattn_cache
        cache_seq_len = k_cache.shape[-2]

        k_cache_user = ttnn.slice(
            k_cache, (user_id, 0, 0, 0), (user_id + 1, k_cache.shape[1], k_cache.shape[2], k_cache.shape[3])
        )

        scores = ttnn.matmul(
            xq,
            ttnn.transpose(k_cache_user, -1, -2),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            program_config=self.model_config["VISION_XATTN_SCORE_PROGCFG"](seq_len, cache_seq_len),
        )

        scores = ttnn.multiply(scores, self.scale)
        # WARNING: This add is buggy if xattn_mask has to be broadcasted to n_local_heads. Workaround is to broadcast on host side
        scores = ttnn.add(scores, xattn_mask)
        scores = ttnn.softmax(scores, dim=-1, numeric_stable=True)

        v_cache_user = ttnn.slice(
            v_cache, (user_id, 0, 0, 0), (user_id + 1, v_cache.shape[1], v_cache.shape[2], v_cache.shape[3])
        )
        output = ttnn.matmul(
            scores,
            v_cache_user,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            program_config=self.model_config["VISION_XATTN_OUTPUT_PROGCFG"](seq_len, cache_seq_len),
        )

        # WARNING: this broadcast is also broken, must broadcast on host
        output = ttnn.mul(output, full_text_row_masked_out_mask_1NSH)

        output = ttnn.experimental.nlp_concat_heads(output)
        if seq_len > 1024:
            output = ttnn.reshape(output, [1, seq_len // 1024, 1024, -1])

        output = ttnn.matmul(
            output,
            self.wo,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.model_config["VISION_XATTN_DENSE_PROGCFG"](seq_len),
        )
        if seq_len > 1024:
            output = ttnn.reshape(output, [1, 1, seq_len, -1])

        # Reduce-scatter
        if self.is_multichip:  # TODO use_fused_all_gather_matmul
            dense_out_reduced = ttnn.reduce_scatter(
                output,
                scatter_dim=3,
                math_op=ttnn.ReduceType.Sum,
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            return dense_out_reduced
        else:
            return output

    def forward(self, x_11SH, xattn_mask, full_text_row_masked_out_mask_1NSH, xattn_cache, mode, user_id=0):
        if mode == "prefill":
            return self.forward_prefill(
                x_11SH, xattn_mask, full_text_row_masked_out_mask_1NSH, xattn_cache, user_id=user_id
            )
        else:
            return self.forward_decode(x_11SH, xattn_mask, full_text_row_masked_out_mask_1NSH, xattn_cache)
