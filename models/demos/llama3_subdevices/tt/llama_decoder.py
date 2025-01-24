# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
from models.demos.llama3_subdevices.tt.llama_attention import TtLlamaAttention
from models.demos.llama3_subdevices.tt.llama_mlp import TtLlamaMLP
from models.common.rmsnorm import RMSNorm
from models.common.lightweightmodule import LightweightModule
from models.demos.llama3_subdevices.tt.distributed_norm import DistributedNorm
import torch.nn.functional as F


class TtTransformerBlock(LightweightModule):
    def __init__(
        self,
        args,
        mesh_device,
        dtype,
        state_dict,
        layer_num,
        weight_cache_path,
        transformation_mats,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        prefetcher_setup=None,
        tt_ccl=None,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device

        self.args = args
        self.hidden_size = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = args.max_seq_len
        self.dim = args.dim
        self.max_batch_size = args.max_batch_size
        self.n_kv_heads = args.n_kv_heads
        self.current = 0
        self.model_config = args.get_model_config()

        self.layer_num = layer_num

        self.prefetcher_setup = prefetcher_setup
        self.tt_ccl = tt_ccl

        self.attention = TtLlamaAttention(
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=args,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
            prefetcher_setup=prefetcher_setup,
            tt_ccl=tt_ccl,
        )
        self.feed_forward = TtLlamaMLP(
            mesh_device=mesh_device,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            model_config=self.model_config,
            prefetcher_setup=prefetcher_setup,
            tt_ccl=tt_ccl,
        )
        self.attention_norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="attention_norm",
                is_distributed=self.args.is_distributed_norm,
                sharded_program_config=self.model_config["SHARDED_NORM_ATTN_PRGM_CFG"],
                sharded_output_config=self.model_config["SHARDED_ATTN_INPUT_MEMCFG"],
            ),
            args,
            TG=args.is_galaxy,
            tt_ccl=tt_ccl,
        )
        self.ff_norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="ffn_norm",
                is_distributed=self.args.is_distributed_norm,
                sharded_program_config=self.model_config["SHARDED_NORM_MLP_PRGM_CFG"],
                sharded_output_config=self.model_config["SHARDED_MLP_INPUT_MEMCFG"],
            ),
            args,
            TG=args.is_galaxy,
            tt_ccl=tt_ccl,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
    ) -> ttnn.Tensor:
        TG = self.args.is_galaxy
        # x is fractured across devices and interleaved in DRAM (for prefill) and sharded in L1 (for decode)
        skip_mem_cfg = self.model_config["DECODE_RESIDUAL_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
        assert (
            x.memory_config() == skip_mem_cfg
        ), f"decoder input memcfg mismatch: {x.memory_config()} != {skip_mem_cfg}"
        # Norms take fractured inputs and output replicated across devices
        attn_in = self.attention_norm(x, mode)
        # Attention takes replicated inputs and produces fractured outputs

        # pad attn input
        attn_in = ttnn.to_memory_config(attn_in, self.model_config["SHARDED_ATTN_INPUT_RING_MEMCFG"])
        attn_out = self.attention.forward(
            attn_in,
            current_pos,
            rot_mats,
            user_id,
            mode,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            kv_cache=kv_cache,
        )

        # Here x and attn_out are both fractured across devices
        h = ttnn.add(x, attn_out, memory_config=skip_mem_cfg)
        ttnn.deallocate(attn_out)
        if mode == "prefill":
            x.deallocate(True)

        # Norms take fractured inputs and output replicated across devices

        ff_in = self.ff_norm(h, mode)
        # if TG and mode == "decode":
        #     ff_in = ttnn.to_memory_config(ff_in, memory_config=self.model_config["MLP_ACT_MEMCFG"])

        # MLP takes replicated inputs and produces fractured outputs
        ff_in = ttnn.to_memory_config(ff_in, self.model_config["SHARDED_FF12_RING_MEMCFG"])
        ff_out = self.feed_forward.forward(ff_in, mode)

        out = ttnn.add(h, ff_out, memory_config=skip_mem_cfg)
        return out  # fractured across devices
