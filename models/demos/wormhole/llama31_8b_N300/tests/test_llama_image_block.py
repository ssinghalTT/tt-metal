# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
import importlib

llama_reference_mod = importlib.import_module(
    "models.demos.t3000.llama2_70b.reference.llama-models.models.llama3.reference_impl.multimodal.model"
)
from models.demos.wormhole.llama31_8b_N300.tt.llama_image_block import TtLlamaImageTransformerBlock
from models.demos.wormhole.llama31_8b_N300.tt.model_config import TtModelArgs
from models.demos.wormhole.llama31_8b_N300.tt.llama_common import (
    prepare_inputs_ttnn_prefill,
)
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "seq_len",
    (5120,),
)
@pytest.mark.parametrize(
    "gated",
    (True, False),
)
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (2, 4), "TG": (8, 4)}.get(os.environ.get("FAKE_DEVICE"), None)],
    indirect=True,
)
def test_llama_block_inference(seq_len, mesh_device, gated, use_program_cache, reset_seeds):
    dtype = ttnn.bfloat16
    pcc = 0.99

    model_args = TtModelArgs(mesh_device)
    state_dict = torch.load(model_args.consolidated_weights_path, map_location=torch.device("cpu"))

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    if gated:
        first_layer_prefix = "vision_model.vision_encoder.global_transformer.resblocks.0."
    else:
        first_layer_prefix = "vision_model.vision_encoder.transformer.resblocks.31."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    dim = model_args.vision_dim
    heads = model_args.vision_attn_n_heads
    reference_model = llama_reference_mod.ImageTransformerBlock(
        d_model=dim, n_head=heads, mlp_ratio=model_args.vision_mlp_ratio, gated=gated
    )
    reference_model.load_state_dict(partial_state_dict)

    batch = 1

    all_tests_pass = True

    tt_model = TtLlamaImageTransformerBlock(
        mesh_device,
        state_dict,
        state_dict_prefix=first_layer_prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        configuration=model_args,
        gated=gated,
    )

    # pt_block_input = (torch.rand(batch, seq_len, dim) * 2) - 1
    pt_block_input = torch.load("/home/cglagovich/tt-metal/layer_31_intermediate.pt")
    # pt_block_input = pt_block_input[..., :seq_len, :].bfloat16().float()
    pt_block_input = pt_block_input.bfloat16().float()
    pt_block_input = torch.nn.functional.pad(pt_block_input, (0, 0, 0, seq_len - pt_block_input.shape[-2]))
    mask = torch.load("/home/cglagovich/tt-metal/mask.pt")
    # mask = mask[..., :seq_len, :seq_len]
    mask = torch.nn.functional.pad(mask, (0, seq_len - mask.shape[-1], 0, seq_len - mask.shape[-2]), value=-1e9)
    tt_block_input = pt_block_input.clone()
    block_input = prepare_inputs_ttnn_prefill(
        tt_block_input,
        mesh_device,
    )

    # mask = torch.bernoulli(
    #     torch.full(
    #         (
    #             batch,
    #             seq_len,
    #             seq_len,
    #         ),
    #         0.25,
    #     )
    # )
    # mask = mask.unsqueeze(1)
    # mask = mask * -1e9

    tt_mask = ttnn.from_torch(
        mask,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_out = tt_model(block_input, mask=tt_mask)
    tt_output_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))[:, 0, :, :].view(
        batch, seq_len, -1
    )  # [ batch, seq, hidden_dim]

    reference_output = reference_model(pt_block_input, mask=mask)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info(f"Llama_Attention Passed!")
    else:
        logger.warning(f"Llama_Attention Failed!")
        all_tests_pass = False

    if all_tests_pass:
        logger.info("Llama Attention output Passed!")
    else:
        logger.warning("Llama Attention output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
