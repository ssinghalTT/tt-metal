# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import json
from time import time
from datetime import datetime
from loguru import logger
import os
import ttnn
import math
import pytest
from models.demos.wormhole.llama31_8b.tt.llama_common import (
    get_single_rot_mat,
    get_prefill_rot_mat,
    prepare_inputs_ttnn_prefill,
    get_rot_transformation_mat,
    HostEmbedding,
    encode_prompt_llama_instruct,
)
from models.demos.wormhole.llama31_8b.tt.llama_model import TtTransformer
from models.demos.wormhole.llama31_8b.tt.llama_embedding import TtLlamaEmbedding
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer

from models.perf.benchmarking_utils import BenchmarkProfiler
from models.demos.utils.llm_demo_utils import create_benchmark_data, verify_perf


# load from json, return as a list
def load_inputs(user_input, batch):
    if isinstance(user_input, str):
        with open(user_input, "r") as f:
            user_input = json.load(f)
    assert len(user_input) >= batch, f"Number of users (batch) must be {batch}!"
    in_prompt = []
    for i in range(batch):
        in_prompt.append(user_input[i]["prompt"])
    return in_prompt


def preprocess_inputs_prefill(
    input_prompts,
    tokenizer,
    model_args,
    instruct,
    max_generated_tokens,
    max_prefill_len=32768,
):
    """
    Run tokenizer on inputs, and create embeddings for the first token of each input
    """
    # The maximum KV-cache len supported is 32k. To avoid going out of memory, clip the max prefill length by the maximum number of tokens that will be generated
    if max_prefill_len == 32768:
        max_prefill_len = 32768 - max_generated_tokens

    if instruct:
        # Pre append [INST] and post append [/INST] to the encoded prompts if instruct mode
        encoded_prompts = [encode_prompt_llama_instruct(tokenizer, prompt) for prompt in input_prompts]

    else:
        encoded_prompts = [tokenizer.encode(prompt) for prompt in input_prompts]

    prompt_lens = [len(x) for x in encoded_prompts]
    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)

    # The large input demo we provide contains more tokens than the maximum (32k tokens)
    # To avoid running out of memory, clip to max_prefill_len
    if min_prompt_len > max_prefill_len:
        logger.info(f"Clipping prompts to {max_prefill_len}")
        if instruct:  # When clipping, make sure to add the ` [/INST]` token at the end (4 tokens)
            encoded_prompts = [encod[: max_prefill_len - 4] for encod in encoded_prompts]
            dec_prompts = [tokenizer.decode(encod) + " [/INST]" for encod in encoded_prompts]
            encoded_prompts = [tokenizer.encode(prompt) for prompt in dec_prompts]
        else:
            encoded_prompts = [encod[:max_prefill_len] for encod in encoded_prompts]

        # Update prompt lengths
        prompt_lens = [len(x) for x in encoded_prompts]
        min_prompt_len = min(prompt_lens)
        max_prompt_len = max(prompt_lens)

    assert (
        max_prompt_len <= model_args.max_seq_len
    ), f"Max prompt length {max_prompt_len} exceeds model max seq len {model_args.max_seq_len}"
    assert min_prompt_len > 0, "Minimum prompt length must be greater than 0"
    assert min_prompt_len <= max_prompt_len, f"Minimum prompt length {min_prompt_len} exceeds max len {max_prompt_len}"

    logger.info(f"# of users: {len(encoded_prompts)}")
    input_tokens_prefill = []
    decoding_pos = []
    prefill_lens = []

    # Always prefill the nearest power of 2 for each user. This means that the majority of cases we will prefill more tokens than needed.
    # To avoid issues, we keep track of the decoding position to decode correctly the user's prompt
    for i, encoded in enumerate(encoded_prompts):
        # Prefill size is nearest power of 2
        prefill_seq_len = max(2 ** math.ceil(math.log(len(encoded), 2)), 128)

        # Initial prefill tensors full of pad tokens
        input_tokens_prefill_i = torch.full((1, prefill_seq_len), 0, dtype=torch.int32)
        input_tokens_prefill_i[0, : len(encoded[:])] = torch.tensor(encoded[:]).to(input_tokens_prefill_i)
        input_tokens_prefill.append(input_tokens_prefill_i)

        # Keep the correct decoding position of each user
        decoding_pos.append(len(encoded))
        prefill_lens.append(prefill_seq_len)

    return (
        input_tokens_prefill,
        encoded_prompts,
        decoding_pos,
        prefill_lens,
    )


def run_llama_demo(user_input, batch_size, device, instruct_mode, is_ci_env, num_batches, print_to_file, is_n300):
    # Creat batch output file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_directory = "models/demos/wormhole/llama31_8b/demo/output"
    os.makedirs(output_directory, exist_ok=True)
    os.chmod(output_directory, 0o755)
    output_filename = f"{output_directory}/demo_user_output_{timestamp}.txt"

    # Set Llama flags for CI
    # if is_ci_env and instruct_mode:  # Update paths for instruct mode, otherwise use default paths for general weights
    os.environ["LLAMA_CKPT_DIR"] = "/proj_sw/user_dev/hf_data/llama/Meta-Llama-3.1-8B-Instruct/"
    os.environ["LLAMA_TOKENIZER_PATH"] = "/proj_sw/user_dev/hf_data/llama/Meta-Llama-3.1-8B-Instruct/"
    os.environ["LLAMA_CACHE_PATH"] = "/proj_sw/user_dev/hf_data/llama/Meta-Llama-3.1-8B-Instruct/"
    # This module requires the env paths above for CI runs
    from models.demos.wormhole.llama31_8b.tt.model_config import TtModelArgs

    dtype = ttnn.bfloat8_b

    # We disregard any warmup iteration for profiling, in favour of just measuring compile time on the first iteration
    N_warmup_iter = {"inference_prefill": 0, "inference_decode": 0}

    if len(user_input) == 1:
        input_prompts = user_input * batch_size
    else:
        input_prompts = load_inputs(user_input, batch_size)

    # Generate the batched prompts (rotate the inputs between the users, for each batch)
    # If batch_size == 1, the same prompt is repeated for each batch
    batch_prompts = []
    for i in range(num_batches):
        batch_prompts.append([input_prompts[(j + i) % len(input_prompts)] for j in range(len(input_prompts))])

    # Load model args, weights, and tokenizer
    model_args = TtModelArgs(device, instruct=instruct_mode)
    tokenizer = Tokenizer(model_args.tokenizer_path)

    model_args.n_layers = 32

    logger.info("Loading weights...")
    state_dict = torch.load(model_args.consolidated_weights_path, map_location=torch.device("cpu"))
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            any([f"layers.{i}." in k for i in range(model_args.n_layers)])
            or k in ["tok_embeddings.weight", "norm.weight", "output.weight"]
        )
    }
    logger.info("Loading weights finished!")

    # Load TTNN Llama3.1 model
    logger.info("Loading weights to device...")
    tt_model = TtTransformer(
        args=model_args,
        device=device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layers=list(range(model_args.n_layers)),
    )
    tt_embd = TtLlamaEmbedding(
        device=device,
        args=model_args,
        weight_cache_path=model_args.weight_cache_path(dtype),
        state_dict=state_dict,
        dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
    )
    embd = HostEmbedding(model_args)
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    logger.info("Finished loading weights to device. Starting inference...")
    max_generated_tokens = 120  # Maximum number of tokens to generate per user
    num_tokens_generated_decode = []
    for batch_idx, input_prompts in enumerate(batch_prompts):
        logger.info(f"Processing batch {batch_idx}")
        # Preprocess initial prompt inputs
        (
            input_tokens_prefill_pt,
            encoded_prompts,
            decoding_pos,
            prefill_lens,
        ) = preprocess_inputs_prefill(
            input_prompts,
            tokenizer,
            model_args,
            instruct_mode,
            max_generated_tokens,
        )
        pt_prefill_input = [embd(input_tokens_prefill_pt[b]).view(1, prefill_lens[b], -1) for b in range(batch_size)]

        # set kv cache to zeros if not first batch, to avoid context leaking
        if batch_idx != 0:
            for layer in tt_model.layers:
                k_cache, v_cache = layer.attention.layer_past
                k_cache = ttnn.mul(k_cache, 0, output_tensor=k_cache)
                v_cache = ttnn.mul(v_cache, 0, output_tensor=v_cache)

        logger.info(f"Starting prefill...")

        head_dim = model_args.dim // model_args.n_heads
        transformation_mat_torch = get_rot_transformation_mat(head_dim)
        transformation_mats = ttnn.as_tensor(
            transformation_mat_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # First user is used for compile time
        num_users_generated_prefill = batch_size - 1 if batch_size > 1 else 1  # First user is used for compile time

        pt_out = []
        for batch_id in range(batch_size):
            prefill_seq_len = prefill_lens[batch_id]
            rot_mats_prefill = get_prefill_rot_mat(
                model_args.head_dim, model_args.max_seq_len, device, seq_len=prefill_seq_len
            )
            if decoding_pos[batch_id] < prefill_seq_len:
                pt_prefill_input[batch_id][
                    :, decoding_pos[batch_id] :, :
                ] = 0  # Zero out the tokens after the prefill length

            prefill_input = prepare_inputs_ttnn_prefill(
                pt_prefill_input[batch_id],
                device,
            )
            tt_out = tt_model(
                prefill_input,
                None,  # Current position
                None,  # Current position for attention
                rot_mats_prefill,
                transformation_mats,
                user_id=batch_id,
                mode="prefill",
                get_last_token=((decoding_pos[batch_id] - 1) // 32) * 32,
            )
            pt_out.append(ttnn.to_torch(tt_out)[0, 0, (decoding_pos[batch_id] - 1) % 32, :])
            ttnn.deallocate(tt_out)
            ttnn.synchronize_device(device)
        logger.info(f"Prefill finished !")

        # Preparing first decode token
        pt_out_batched = torch.stack(pt_out, dim=-2)
        pt_out_batched = torch.argmax(pt_out_batched, dim=-1)
        tt_out_tok = ttnn.from_torch(
            torch.nn.functional.pad(pt_out_batched.unsqueeze(0).unsqueeze(0).unsqueeze(0), (0, 31), "constant", 0),
            # pt_out_batched.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            device=device,
            dtype=ttnn.uint32,
        )

        # Keep track of generated outputs to print out every iteration
        all_outputs = [encoded_prompts[b][:prefill_seq_len] for b in range(batch_size)]
        for user in range(batch_size):
            user_tok = int(pt_out_batched[user].item())
            all_outputs[user].append(user_tok)

        user_done = [False] * batch_size  # Keeps track when a user reaches EoD token

        logger.info("Starting decode...")

        current_rot_mat, rot_matrix = get_single_rot_mat(
            model_args.head_dim,
            device,
            start_pos=decoding_pos[0] - 2,
        )

        # Create events
        op_event = ttnn.create_event(device)
        write_event = ttnn.create_event(device)

        current_pos = ttnn.from_torch(torch.tensor(decoding_pos, dtype=torch.int32), device=device, dtype=ttnn.int32)
        current_pos_attn = ttnn.from_torch(
            torch.tensor(decoding_pos * 8, dtype=torch.int32), device=device, dtype=ttnn.int32
        )

        # Compile
        decode_input = ttnn.unsqueeze_to_4D(tt_embd(tt_out_tok))
        tt_out = tt_model(decode_input, current_pos, current_pos_attn, rot_mat=current_rot_mat)
        tt_out_rm = ttnn.untilize(tt_out, use_multicore=True)
        ttnn.deallocate(tt_out)
        tt_out_tok = ttnn.argmax(tt_out_rm, dim=3, use_multicore=True, output_tensor=tt_out_tok)
        ttnn.deallocate(tt_out_rm)
        new_rot_mat = ttnn.linear(rot_matrix, current_rot_mat)
        current_rot_mat = ttnn.copy(new_rot_mat, current_rot_mat)
        ttnn.plus_one(current_pos)
        ttnn.plus_one(current_pos_attn)

        # Capture Trace
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)

        decode_input = ttnn.unsqueeze_to_4D(tt_embd(tt_out_tok))
        tt_out = tt_model(decode_input, current_pos, current_pos_attn, rot_mat=current_rot_mat)
        tt_out_rm = ttnn.untilize(tt_out, use_multicore=True)
        ttnn.deallocate(tt_out)
        tt_out_tok = ttnn.argmax(tt_out_rm, dim=3, use_multicore=True, output_tensor=tt_out_tok)
        ttnn.deallocate(tt_out_rm)
        new_rot_mat = ttnn.linear(rot_matrix, current_rot_mat)
        current_rot_mat = ttnn.copy(new_rot_mat, current_rot_mat)
        ttnn.plus_one(current_pos)
        ttnn.plus_one(current_pos_attn)

        ttnn.end_trace_capture(device, trace_id, cq_id=0)

        current_pos_reset = ttnn.from_torch(torch.tensor(decoding_pos, dtype=torch.int32), dtype=ttnn.int32)
        current_pos_attn_reset = ttnn.from_torch(torch.tensor(decoding_pos * 8, dtype=torch.int32), dtype=ttnn.int32)
        tt_out_tok_reset = ttnn.from_torch(
            torch.nn.functional.pad(pt_out_batched.unsqueeze(0).unsqueeze(0).unsqueeze(0), (0, 31), "constant", 0),
            dtype=ttnn.uint32,
        )

        ttnn.copy_host_to_device_tensor(current_pos_reset, current_pos)
        ttnn.copy_host_to_device_tensor(current_pos_attn_reset, current_pos_attn)
        ttnn.copy_host_to_device_tensor(tt_out_tok_reset, tt_out_tok)

        # Start decoding
        iteration = 0
        users_decoding = True  # reset to handle next batch

        ttnn.record_event(1, write_event)

        while users_decoding:
            iteration_time_start = time()
            # Execute trace
            ttnn.wait_for_event(0, write_event)
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            ttnn.record_event(0, op_event)

            # Write to host
            ttnn.wait_for_event(1, op_event)
            tt_output_torch = ttnn.to_torch(tt_out_tok.cpu(blocking=False, cq_id=1))[0, 0, 0, :batch_size]
            ttnn.record_event(1, write_event)

            # Save output token to print out later
            for user in range(batch_size):
                user_tok = tt_output_torch[user].tolist()
                if user_tok != 28803 and user_done[user] == False:  # Stop saving the ouput after hitting the EOS token
                    all_outputs[user].append(user_tok)
                else:
                    user_done[user] = True
                    logger.trace(f"[User {user}] Finished decoding at iteration {iteration}")
                    if all(user_done):
                        users_decoding = False

            # Print out generated outputs for each user at the end of every iteration
            iteration_time = time() - iteration_time_start
            tokens_per_second_per_user = 1 / iteration_time

            # Print out generated outputs for each user at the end of every iteration
            if not is_ci_env:
                if len(user_input) == 1:
                    logger.info("[User 0] {}".format("".join(tokenizer.decode(all_outputs[0]))))
                else:
                    for user in range(batch_size):
                        text = "".join(tokenizer.decode(all_outputs[user]))
                        if len(text) > 100:
                            text = "..." + text[-97:]
                        text = text.replace("\n", " ")
                        logger.info("[User {}] {}".format(user, text))

            # Always print perf at every iteration
            logger.info(
                f"Iteration {iteration}: {1000*iteration_time:.0f}ms @ {tokens_per_second_per_user:.1f} tok/s/user ({batch_size*tokens_per_second_per_user:.1f} tok/s throughput)"
            )
            iteration += 1

            # Reset rotation matrix every 100 iterations
            if iteration % 100 == 0:
                current_rot_mat_reset, rot_matrix_reset = get_single_rot_mat(
                    model_args.head_dim,
                    device=None,
                    start_pos=decoding_pos[0] + iteration,
                )
                ttnn.copy_host_to_device_tensor(current_rot_mat_reset, current_rot_mat)

            # Upper limit of generated tokens for each user (to avoid infinite generation in case eos is not seen)
            if iteration >= max_generated_tokens:
                users_decoding = False

            if not users_decoding:
                with open(output_filename, "a") as f:
                    for i, (output, prompt) in enumerate(zip(all_outputs, input_prompts)):
                        text = tokenizer.decode(output)
                        if instruct_mode:
                            split_text = text.split("[/INST]", 1)
                        else:
                            split_text = text.split(prompt, 1)
                        if len(split_text) > 1:
                            text_after_prompt = split_text[1]
                        else:
                            text_after_prompt = text  # If prompt is not found, use the whole text
                        if print_to_file:
                            f.write(
                                f"\nbatch: {batch_idx} user: {i}\nprompt: {prompt} \noutput:\n{text_after_prompt}\n"
                            )
                        else:
                            logger.info(
                                f"\nbatch: {batch_idx} user: {i}\nprompt: {prompt} \noutput:\n{text_after_prompt}\n"
                            )

        num_tokens_generated_decode.append(
            iteration - 1
        )  # Save the number of tokens generated for each batch (excluding the first token which is used for compile time)

        # Release trace
        ttnn.release_trace(device, trace_id)


@pytest.mark.parametrize(
    "input_prompts, instruct_weights, num_batches",
    [
        ("models/demos/wormhole/llama31_8b/demo/input_data_prefill_128.json", False, 1),
        ("models/demos/wormhole/llama31_8b/demo/input_data_prefill_128.json", False, 3),
        ("models/demos/wormhole/llama31_8b/demo/input_data_questions_prefill_128.json", True, 1),
        ("models/demos/wormhole/llama31_8b/demo/input_data_questions_prefill_128.json", True, 3),
    ],
    ids=[
        "general_weights-1_batch",
        "general_weights-3_batch",
        "instruct_weights-1_batch",
        "instruct_weights-3_batch",
    ],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 7815168, "num_command_queues": 2}], indirect=True)
def test_llama_demo(
    device, use_program_cache, input_prompts, instruct_weights, is_ci_env, is_single_card_n300, num_batches
):
    if is_ci_env and instruct_weights == False:
        pytest.skip("CI demo test only runs instruct weights to reduce CI pipeline load (both are supported)")

    return run_llama_demo(
        user_input=input_prompts,
        batch_size=1,
        device=device,
        instruct_mode=instruct_weights,
        is_ci_env=is_ci_env,
        num_batches=num_batches,
        print_to_file=False,
        is_n300=is_single_card_n300,
    )
