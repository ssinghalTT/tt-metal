# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Optional
from loguru import logger

from PIL import Image as PIL_Image
from termcolor import cprint

import pytest
import os
import ttnn

import llama_models.llama3.reference_impl.generation as llama_reference_generation
from llama_models.llama3.api.tokenizer import Tokenizer
from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.datatypes import ImageMedia, UserMessage

from pkg_resources import resource_filename

IMG_PATH = Path(resource_filename("llama_models", "scripts/resources/"))

from models.demos.llama3.tt.multimodal.vision_generator import LlamaVision
from models.demos.llama3.demo.simple_vision_demo import create_multimodal_model


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "target",
    ("tt", "cpu"),
)
def test_llama_multimodal_demo_chat(
    mesh_device,
    target,
    temperature: float = 0.5,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 1,
    max_gen_len: Optional[int] = 200,
    model_parallel_size: Optional[int] = None,
):
    ckpt_dir = os.environ["LLAMA_DIR"]
    tokenizer_path = str(Path(ckpt_dir) / "tokenizer.model")

    logger.info(f"Creating reference model from checkpoint in '{ckpt_dir}'")
    if target == "cpu":
        generator = llama_reference_generation.Llama.build(
            ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            model_parallel_size=model_parallel_size,
        )
    else:
        logger.info(f"Creating TT model on {len(mesh_device.get_devices())} devices")
        mesh_device.enable_program_cache()
        mesh_device.enable_async(True)
        model_args, model = create_multimodal_model(mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len)
        tokenizer = Tokenizer(model_path=tokenizer_path)
        formatter = ChatFormat(tokenizer)
        generator = LlamaVision(model, model_args, mesh_device, tokenizer=tokenizer, formatter=formatter)

    # image understanding
    dialogs = []
    with open(IMG_PATH / "dog.jpg", "rb") as f:
        img = PIL_Image.open(f).convert("RGB")

    dialogs = [
        [
            UserMessage(
                content=[
                    ImageMedia(image=img),
                    "Describe this image in two sentences",
                ],
            )
        ],
    ]

    print(f"Running text completion on {target}")
    for dialog in dialogs:
        result = generator.chat_completion(
            dialog,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for msg in dialog:
            print(f"{msg.role.capitalize()}: {msg.content}\n")

        out_message = result.generation
        print(f"> {out_message.role.capitalize()}: {out_message.content}")
        for t in out_message.tool_calls:
            print(f"  Tool call: {t.tool_name} ({t.arguments})")
        print("\n==================================\n")
