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
import requests
from pathlib import Path
import hashlib

from models.demos.llama3.tt.llama_common import (
    get_prefill_rot_mat,
    get_rot_transformation_mat,
    HostEmbedding,
    encode_prompt_llama_instruct,
)
from models.demos.llama3.tt.llama_model import TtTransformer
from models.demos.llama3.tt.llama_embedding import TtLlamaEmbedding
from models.demos.llama3.tt.llama_rope import TtLlamaRotarySetup
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer

from models.perf.benchmarking_utils import BenchmarkProfiler
from models.demos.utils.llm_demo_utils import create_benchmark_data, verify_perf
from models.demos.llama3.tt.model_config import LlamaOptimizations


def load_and_cache_context(context_url, cache_dir):
    cache_file = cache_dir / hashlib.md5(context_url.encode()).hexdigest()

    if cache_file.exists():
        with open(cache_file, "r") as f:
            context_text = f.read()
        logger.info(f"Loaded context from cache: {context_url}")
    else:
        try:
            response = requests.get(context_url)
            if response.status_code == 200:
                context_text = response.text
                with open(cache_file, "w") as f:
                    f.write(context_text)
                logger.info(f"Downloaded and cached context: {context_url}")
            else:
                logger.warning(f"Failed to fetch context from URL: {context_url}. Status code: {response.status_code}")
                context_text = ""
        except Exception as e:
            logger.error(f"Error fetching context from URL: {context_url}. Error: {str(e)}")
            context_text = ""

    return context_text


# load from json, return as a list
def load_inputs(user_input, batch):
    if isinstance(user_input, str):
        with open(user_input, "r") as f:
            user_input = json.load(f)
    assert len(user_input) >= batch, f"Number of users (batch) must be {batch}!"
    in_prompt = []
    cache_dir = Path("models/demos/llama3/demo/context_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    for i in range(batch):
        prompt = user_input[i]["prompt"]
        if "context" in user_input[i]:
            context_text = load_and_cache_context(user_input[i]["context"], cache_dir)
            prompt = context_text + "\n\n" + prompt
        in_prompt.append(prompt)
    return in_prompt


def preprocess_inputs_prefill(
    input_prompts,
    tokenizer,
    model_args,
    instruct,
    max_generated_tokens,
    max_prefill_len=128 * 1024,
):
    """
    Run tokenizer on inputs, and create embeddings for the first token of each input
    """
    # The maximum KV-cache len supported is 32k. To avoid going out of memory, clip the max prefill length by the maximum number of tokens that will be generated
    if max_prefill_len == 128 * 1024:
        max_prefill_len = 128 * 1024 - max_generated_tokens

    if instruct:
        encoded_prompts = [encode_prompt_llama_instruct(tokenizer, prompt) for prompt in input_prompts]
    else:
        encoded_prompts = [tokenizer.encode(prompt, bos=True, eos=False) for prompt in input_prompts]

    # Print the length of encoded prompts
    logger.info("Encoded prompt lengths:" + ", ".join(str(len(prompt)) for prompt in encoded_prompts))

    prompt_lens = [len(x) for x in encoded_prompts]
    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)

    # The large input demo we provide contains more tokens than the maximum (32k tokens)
    # To avoid running out of memory, clip to max_prefill_len
    if min_prompt_len > max_prefill_len:
        logger.info(f"Clipping prompts to {max_prefill_len}")
        if instruct:  # When clipping, make sure to add the ` 】 token at the end (4 tokens)
            encoded_prompts = [encod[: max_prefill_len - 4] for encod in encoded_prompts]
            dec_prompts = [tokenizer.decode(encod) + " 】" for encod in encoded_prompts]
            encoded_prompts = [tokenizer.encode(prompt, bos=True, eos=False) for prompt in dec_prompts]
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


def run_llama3_demo(
    user_input,
    batch_size,
    single_layer,
    mesh_device,
    instruct_mode,
    is_ci_env,
    num_batches,
    print_to_file,
    optimizations,
):
    # Creat batch output file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_directory = "models/demos/llama3/demo/output"
    os.makedirs(output_directory, exist_ok=True)
    os.chmod(output_directory, 0o755)
    output_filename = f"{output_directory}/demo_user_output_{timestamp}.txt"

    # This module requires the env paths above for CI runs
    from models.demos.llama3.tt.model_config import TtModelArgs

    dtype = ttnn.bfloat8_b
    # Miguel - parametrize this
    paged_attention = True
    batch_size = 32
    assert batch_size <= 32, "Batch size cannot be greater than 32"

    # We disregard any warmup iteration for profiling, in favour of just measuring compile time on the first iteration
    N_warmup_iter = {"inference_prefill": 0, "inference_decode": 0}

    # Start profiler
    logger.info(f"Start profiler")
    profiler = BenchmarkProfiler()
    profiler.start("run")

    logger.info(f"Reading inputs...")
    profiler.start("loading_inputs")
    if len(user_input) == 1:
        input_prompts = user_input * batch_size
    else:
        input_prompts = load_inputs(user_input, batch_size)
    profiler.end("loading_inputs")

    # Generate the batched prompts (rotate the inputs between the users, for each batch)
    # If batch_size == 1, the same prompt is repeated for each batch
    batch_prompts = []
    for i in range(num_batches):
        batch_prompts.append([input_prompts[(j + i) % len(input_prompts)] for j in range(len(input_prompts))])

    # Load model args, weights, and tokenizer
    model_args = TtModelArgs(mesh_device, instruct=instruct_mode, max_batch_size=batch_size, optimizations=optimizations)
    tokenizer = Tokenizer(model_args.tokenizer_path)

    # TODO Miguel: Setup max sequence length depending on the model being used to actually fit on device
    # Reduce max seq len and KV cache seq_len params to speed up the test
    model_args.max_seq_len = 512
    model_args.kv_seq_len = model_args.max_seq_len

    if single_layer:
        model_args.n_layers = 1

    logger.info("Loading weights...")
    profiler.start("weight_loading")
    state_dict = model_args.load_state_dict()
    profiler.end("weight_loading")

    # Setup RoPE transformation matrices
    rope_setup = TtLlamaRotarySetup(
        mesh_device,
        batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.use_scaled_rope,
    )
    transformation_mats_decode = rope_setup.get_trans_mats()

    transformation_mats_prefill_torch = get_rot_transformation_mat(model_args.head_dim)
    transformation_mats_prefill = ttnn.from_torch(
        transformation_mats_prefill_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    transformation_mats = {"decode": transformation_mats_decode, "prefill": transformation_mats_prefill}

    page_table_tt = None
    paged_attention_config = None
    if paged_attention:
        paged_attention_config = model_args.paged_attention_config if paged_attention else None

        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    # Load TTNN Llama3.1 model
    logger.info("Loading weights to device...")
    profiler.start("loading_weights_to_device")
    tt_model = TtTransformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        transformation_mats=transformation_mats,
        paged_attention_config=paged_attention_config,
    )
    tt_embd = TtLlamaEmbedding(
        mesh_device=mesh_device,
        args=model_args,
        weight_cache_path=model_args.weight_cache_path(dtype),
        state_dict=state_dict,
        dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
    )
    embd = HostEmbedding(model_args)
    state_dict_prefix = model_args.get_state_dict_prefix("", None)
    embd.load_state_dict({"emb.weight": state_dict[f"{state_dict_prefix}tok_embeddings.weight"]})
    profiler.end("loading_weights_to_device")
    logger.info("Finished loading weights to device.")

    # TODO Change this back to 100
    max_generated_tokens = 20  # Maximum number of tokens to generate per user
    num_tokens_generated_decode = []

    logger.info("Starting inference...")
    for batch_idx, input_prompts in enumerate(batch_prompts):
        logger.info(f"Processing batch {batch_idx}")
        profiler.start(f"preprocess_prefill_inputs", iteration=batch_idx)
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
        # Prefill embeddings are on host since we need to mask out the tokens after the prefill length after embeddings are computed
        pt_prefill_input = [embd(input_tokens_prefill_pt[b]).view(1, prefill_lens[b], -1) for b in range(batch_size)]
        profiler.end(f"preprocess_prefill_inputs", iteration=batch_idx)

        # set kv cache to zeros if not first batch, to avoid context leaking when doing multiple batches
        if batch_idx != 0:
            for layer in tt_model.layers:
                k_cache, v_cache = layer.attention.layer_past
                k_cache = ttnn.mul(k_cache, 0, output_tensor=k_cache)
                v_cache = ttnn.mul(v_cache, 0, output_tensor=v_cache)

        logger.info(f"Starting prefill...")

        # Do not count the first user for prefill time and instead log it as compile time
        num_users_generated_prefill = batch_size - 1 if batch_size > 1 else 1

        pt_out = []

        profiler.start(f"inference_prefill", iteration=batch_idx)
        for batch_id in range(batch_size):
            prefill_seq_len = prefill_lens[batch_id]
            rot_mats_prefill = get_prefill_rot_mat(
                model_args.head_dim, model_args.max_seq_len, mesh_device, seq_len=prefill_seq_len
            )
            if decoding_pos[batch_id] < prefill_seq_len:
                pt_prefill_input[batch_id][
                    :, decoding_pos[batch_id] :, :
                ] = 0  # Zero out the tokens after the prefill length

            prefill_input = model_args.prepare_inputs_ttnn_prefill(
                pt_prefill_input[batch_id],
            )

            if batch_id == 0:  # First user prefill accounts for compile time
                profiler.start(f"compile_prefill", iteration=batch_idx)

            tt_out = tt_model(
                prefill_input,
                current_pos=None,
                rot_mats=rot_mats_prefill,
                user_id=batch_id,
                mode="prefill",
                page_table=page_table_tt,
                get_last_token=((decoding_pos[batch_id] - 1) // 32) * 32,
            )

            if (
                batch_id == 0
            ):  # First user prefill accounts for compile time (which will be removed from the full prefill inference time)
                profiler.end(f"compile_prefill", iteration=batch_idx)

            # [PROFILER-ONLY] In runs where there is only one user, run the prefill twice to measure compile and inference prefill times
            # Miguel: Uncomment
            # if batch_size == 1:
            #     ttnn.deallocate(tt_out)
            #     tt_out = tt_model(
            #         prefill_input,
            #         current_pos=None,
            #         rot_mats=rot_mats_prefill,
            #         user_id=batch_id,
            #         mode="prefill",
            #         page_table=page_table_tt,
            #         get_last_token=((decoding_pos[batch_id] - 1) // 32) * 32,
            #     )

            pt_out.append(
                ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))[
                    0, 0, (decoding_pos[batch_id] - 1) % 32, :
                ]
            )
            ttnn.deallocate(tt_out)

        # Synchronize devices to ensure the profile captures the correct timing of all devices
        for i in range(model_args.num_devices):
            ttnn.synchronize_device(mesh_device.get_devices()[i])
        profiler.end(f"inference_prefill", iteration=batch_idx)
        logger.info(f"Prefill finished")

        # Preparing first decode token
        profiler.start(f"prepare_first_decode_token_{batch_idx}")
        pt_out_batched = torch.stack(pt_out, dim=-2)
        pt_out_batched = torch.argmax(pt_out_batched, dim=-1)
        # Pad the output tensor to be tile sized
        tt_out_tok = ttnn.from_torch(
            torch.nn.functional.pad(
                pt_out_batched.unsqueeze(0).unsqueeze(0).unsqueeze(0), (0, 32 - len(pt_out_batched)), "constant", 0
            ),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.uint32,
        )
        profiler.end(f"prepare_first_decode_token_{batch_idx}")

        # Keep track of generated outputs to print out every iteration
        all_outputs = [encoded_prompts[b][:prefill_seq_len] for b in range(batch_size)]
        for user in range(batch_size):
            user_tok = int(pt_out_batched[user].item())
            all_outputs[user].append(user_tok)

        user_done = [False] * batch_size  # Keeps track when a user reaches EoD token

        logger.info("Starting decode...")

        # Create events
        profiler.start(f"compile_trace_{batch_idx}")
        op_event = ttnn.create_event(mesh_device)
        write_event = ttnn.create_event(mesh_device)

        # Initial positions
        current_pos = torch.tensor([decoding_pos[b] for b in range(batch_size)])
        current_pos_tensor = ttnn.from_torch(
            current_pos,
            device=mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # Get cos/sin matrices for the current position of each user
        rot_mats = rope_setup.get_rot_mats(current_pos)
        rot_mat_idxs = rope_setup.get_rot_idxs(current_pos)

        # Compile
        logger.info(f"Compiling model trace...")
        decode_input = ttnn.unsqueeze_to_4D(tt_embd(tt_out_tok))
        decode_input = ttnn.to_memory_config(decode_input, tt_model.args.model_config["DECODE_RESIDUAL_MEMCFG"])
        tt_out = tt_model(
            decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )
        if tt_model.args.num_devices > 1:
            tt_out_gathered = ttnn.all_gather(tt_out, dim=3, num_links=1, topology=ttnn.Topology.Linear)
            ttnn.deallocate(tt_out)
        else:
            tt_out_gathered = tt_out
        tt_out_rm = ttnn.untilize(tt_out_gathered, use_multicore=True)
        ttnn.deallocate(tt_out_gathered)
        tt_out_tok = ttnn.argmax(tt_out_rm, dim=3, use_multicore=False, output_tensor=tt_out_tok)
        ttnn.deallocate(tt_out_rm)
        ttnn.plus_one(current_pos_tensor)
        profiler.end(f"compile_trace_{batch_idx}")

        # Capture Trace
        logger.info(f"Capturing model trace...")
        profiler.start(f"capture_trace_{batch_idx}")
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)

        decode_input = ttnn.unsqueeze_to_4D(tt_embd(tt_out_tok))
        decode_input = ttnn.to_memory_config(decode_input, tt_model.args.model_config["DECODE_RESIDUAL_MEMCFG"])
        # TODO Miguel: I think the problem is here, not updating the get rot mats
        # The problem is that the get_rot_mats is using embedding that ends up on the host.
        rot_mats = rope_setup.get_rot_mats(rot_mat_idxs)
        tt_out = tt_model(
            decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )
        if tt_model.args.num_devices > 1:
            tt_out_gathered = ttnn.all_gather(tt_out, dim=3, num_links=1, topology=ttnn.Topology.Linear)
            ttnn.deallocate(tt_out)
        else:
            tt_out_gathered = tt_out
        tt_out_rm = ttnn.untilize(tt_out_gathered, use_multicore=True)
        ttnn.deallocate(tt_out_gathered)
        tt_out_tok = ttnn.argmax(
            tt_out_rm, dim=3, use_multicore=False, output_tensor=tt_out_tok
        )  # TODO Multicore is not compatible with batch > 1
        ttnn.deallocate(tt_out_rm)
        ttnn.plus_one(current_pos_tensor)
        # ttnn.plus_one(rot_mat_idxs)  # TODO <- This won't work since embedding requires uint32 and plus_one only works for int32

        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

        # Reset the decoding position for the proper run of the model
        current_pos_reset = ttnn.from_torch(
            current_pos,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if tt_model.args.num_devices > 1 else None,
        )
        tt_out_tok_reset = ttnn.from_torch(
            torch.nn.functional.pad(
                pt_out_batched.unsqueeze(0).unsqueeze(0).unsqueeze(0), (0, 32 - len(pt_out_batched)), "constant", 0
            ),
            # torch.nn.functional.pad(pt_out_batched.unsqueeze(0).unsqueeze(0).unsqueeze(0), (0, 30), "constant", 0),
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if tt_model.args.num_devices > 1 else None,
        )

        # Reset the current position and output token tensors for the real decode run
        ttnn.copy_host_to_device_tensor(current_pos_reset, current_pos_tensor)
        ttnn.copy_host_to_device_tensor(tt_out_tok_reset, tt_out_tok)
        rot_mat_idxs_reset = rope_setup.get_rot_idxs(current_pos, on_host=True)
        ttnn.copy_host_to_device_tensor(rot_mat_idxs_reset, rot_mat_idxs)

        profiler.end(f"capture_trace_{batch_idx}")

        # Start decoding
        iteration = 0
        users_decoding = True  # reset to handle next batch
        total_decoding_time = 0  # Track total decoding time
        total_tokens_generated = 0  # Track total tokens generated

        logger.info(f"Starting decode loop...")
        profiler.start(f"inference_decode", iteration=batch_idx)

        ttnn.record_event(1, write_event)
        while users_decoding:
            if iteration == 0:  # First iteration also accounts for compile time
                profiler.start(f"compile_decode", iteration=batch_idx)
            iteration_time_start = time()

            # Execute trace
            ttnn.wait_for_event(0, write_event)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            ttnn.record_event(0, op_event)

            # Update current pos and mat idxs on host and send to device
            # TODO This is required for now since we cannot ttnn.plus_one(rot_mat_idxs) while it being uint32.
            # If this tensor is int32, it won't be supported by ttnn.embedding
            current_pos += 1
            rot_mat_idxs_updated = rope_setup.get_rot_idxs(current_pos, on_host=True)
            ttnn.copy_host_to_device_tensor(rot_mat_idxs_updated, rot_mat_idxs)

            # Write to host
            ttnn.wait_for_event(1, op_event)
            tt_output_torch = ttnn.to_torch(
                tt_out_tok.cpu(blocking=True, cq_id=1), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1)
            )[0, 0, 0, :batch_size]
            ttnn.record_event(1, write_event)

            # TODO Miguel Remove
            print("==== ITERATION", iteration, "====")
            # Check input
            input_torch = ttnn.to_torch(decode_input, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
            for i in range(batch_size):
                input_equal = torch.eq(input_torch[:, :, 0, :], input_torch[:, :, i, :]).all()
                if not input_equal:
                    print("Batch", i, "input not equal")

            # Check output
            for i in range(batch_size):
                out_equal = torch.eq(tt_output_torch[0], tt_output_torch[i])
                if not out_equal:
                    print("Batch", i, "output not equal")

            # Check KV cache [Mismatch]
            k_cache = ttnn.to_torch(
                tt_model.layers[0].attention.layer_past[0], mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1)
            )
            v_cache = ttnn.to_torch(
                tt_model.layers[0].attention.layer_past[1], mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1)
            )
            for i in range(batch_size):
                k_equal = torch.eq(k_cache[0, :, :, :], k_cache[i, :, :, :]).all()
                v_equal = torch.eq(v_cache[0, :, :, :], v_cache[i, :, :, :]).all()
                if not k_equal:
                    print("Batch", i, "k_cache not equal")
                    # print(f"PCC = {comp_pcc(k_cache[0,:,:,:], k_cache[i,:,:,:])}")
                if not v_equal:
                    print("Batch", i, "v_cache not equal")
                    # print(f"PCC = {comp_pcc(v_cache[0,:,:,:], v_cache[i,:,:,:])}")

            # Check rot mats [All equal]
            cos_out = ttnn.to_torch(rot_mats[0], mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0, :, :, :]
            sin_out = ttnn.to_torch(rot_mats[1], mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0, :, :, :]

            for i in range(batch_size):
                cos_equal = torch.eq(cos_out[0, :, :], cos_out[i, :, :]).all()
                sin_equal = torch.eq(sin_out[0, :, :], sin_out[i, :, :]).all()
                if not cos_equal:
                    print("Batch", i, "cos not equal")
                if not sin_equal:
                    print("Batch", i, "sin not equal")
            ###########

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

            # Ignore the first iteration for average speed calculation
            if iteration > 0:
                total_decoding_time += iteration_time
                total_tokens_generated += 1

            tokens_per_second_per_user = 1 / iteration_time

            profiler.start(f"log_printing_iter_{iteration}", iteration=batch_idx)
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
            profiler.end(f"log_printing_iter_{iteration}", iteration=batch_idx)

            if iteration == 0:  # First iteration also accounts for compile time
                profiler.end(f"compile_decode", iteration=batch_idx)

            iteration += 1

            # Upper limit of generated tokens for each user (to avoid infinite generation in case eos is not seen)
            if iteration >= max_generated_tokens:
                users_decoding = False

            if not users_decoding:
                profiler.start(f"log_saving_file", iteration=batch_idx)
                for i, (output, prompt) in enumerate(zip(all_outputs, input_prompts)):
                    text = tokenizer.decode(output)
                    if instruct_mode:
                        split_text = text.split("<|start_header_id|>assistant<|end_header_id|>", 1)
                    else:
                        split_text = text.split(prompt, 1)
                    if len(split_text) > 1:
                        text_after_prompt = split_text[1]
                    else:
                        text_after_prompt = text  # If prompt is not found, use the whole text
                    if print_to_file:
                        with open(output_filename, "a") as f:
                            f.write(
                                f"\nbatch: {batch_idx} user: {i}\nprompt: {prompt} \noutput:\n{text_after_prompt}\n"
                            )
                    else:
                        # Strip leading newlines from output when sent to terminal
                        short_prompt = (
                            (prompt[:100] + "\n<long prompt not printed in full>\n" + prompt[-100:])
                            if len(prompt) > 200
                            else prompt
                        )
                        logger.info(
                            f"\nbatch: {batch_idx} user: {i}\nprompt: {short_prompt} \noutput:\n{text_after_prompt.strip()}\n"
                        )
                profiler.end(f"log_saving_file", iteration=batch_idx)

        num_tokens_generated_decode.append(
            total_tokens_generated
        )  # Save the number of tokens generated for each batch (excluding the first token)

        # Release trace
        ttnn.release_trace(mesh_device, trace_id)

        profiler.end(f"inference_decode", iteration=batch_idx)

    # Finish profiling at the end of all batches inference
    profiler.end("run")

    # Prepare profile benchmark metrics for batch 0
    compile_prefill_time = profiler.get_duration("compile_prefill")
    compile_decode_time = profiler.get_duration("compile_decode")
    inference_prefill_time = profiler.get_duration("inference_prefill")
    inference_decode_time = profiler.get_duration("inference_decode")
    log_printing_time = sum(profiler.get_duration(f"log_printing_iter_{i}") for i in range(max_generated_tokens))
    log_saving_file_time = profiler.get_duration(f"log_saving_file")

    # Correct the inference decode time to remove the time spent on compile (1st iteration) and log_printing (at the end of every iteration)
    inference_decode_time = inference_decode_time - compile_decode_time - log_printing_time - log_saving_file_time
    # Correct the inference prefill time to remove the time spent on compile (1st iteration)
    inference_prefill_time = inference_prefill_time - compile_prefill_time
    # Average prefill time for each user
    prefill_time_to_first = inference_prefill_time / num_users_generated_prefill

    measurements = {
        # Required measurements
        "compile_prefill": compile_prefill_time,
        "compile_decode": compile_decode_time,
        "inference_prefill": inference_prefill_time,
        "inference_decode": inference_decode_time,
        "prefill_time_to_token": prefill_time_to_first,
        "prefill_t/s": num_users_generated_prefill / inference_prefill_time * prefill_seq_len,  # tokens/s
        "decode_t/s/u": num_tokens_generated_decode[0] / inference_decode_time,  # tokens/s/u
        "decode_t/s": num_tokens_generated_decode[0] / inference_decode_time * batch_size,  # tokens/s
        # Optional measurements
        "loading_inputs": profiler.get_duration("loading_inputs"),
        "weight_loading": profiler.get_duration("weight_loading"),
        "prepare_first_decode_token": profiler.get_duration("prepare_first_decode_token_0"),
        "preprocess_prefill_inputs": profiler.get_duration("preprocess_prefill_inputs"),
        "loading_weights_to_device": profiler.get_duration("loading_weights_to_device"),
        "compile_trace": profiler.get_duration("compile_trace_0"),  # Only for batch 0
        "capture_trace": profiler.get_duration("capture_trace_0"),  # Only for batch 0
        "Total compile time": compile_prefill_time + compile_decode_time,
        "Full demo runtime": profiler.get_duration("run"),
    }

    # Print some of the perf metrics
    logger.info("")
    logger.info(f"Performance metrics for batch 0")
    logger.info(f"Prefill compile time: {round(measurements['compile_prefill'], 4)}s")
    logger.info(f"Decode compile time: {round(measurements['compile_decode'], 4)}s")
    logger.info(f"Prefill inference time per user: {round(inference_prefill_time/num_users_generated_prefill, 4)}s")
    logger.info(
        f"Total Decode inference time ({max_generated_tokens-1} iterations): {round(measurements['inference_decode'], 4)}s"
    )
    logger.info("")
    logger.info(f"Time to first token: {round(measurements['prefill_time_to_token']* 1000, 2)}ms")
    logger.info(
        f"Average speed: {round(inference_decode_time / num_tokens_generated_decode[0] * 1000, 2)}ms @ {round(measurements['decode_t/s/u'], 2)} tok/s/user ({round(measurements['decode_t/s'], 2)} tok/s throughput)"
    )
    logger.info("")

    supported_models = ["3.2-1B", "3.2-3B", "3.1-8B", "3.2-11B", "3.1-70B"]
    supported_devices = ["N150", "N300", "T3K"]

    # TODO update targets based on the llama3 model and the target device
    llama_model_name = model_args.model_name
    tt_device_name = model_args.device_name

    assert llama_model_name in supported_models, f"Model {llama_model_name} not supported"
    assert tt_device_name in supported_devices, f"Device {tt_device_name} not supported"

    # Set the target times to first token for every combination of device and model
    target_prefill_tok_s = {
        "N150_3.2-1B": 1050,  # TODO Update target
        "N300_3.2-1B": 1050,  # TODO Update target
        "T3K_3.2-1B": 1050,  # TODO Update target
        #
        "N150_3.2-3B": 1050,  # TODO Update target
        "N300_3.2-3B": 1050,  # TODO Update target
        "T3K_3.2-3B": 1050,  # TODO Update target
        #
        "N150_3.1-8B": 1050,
        "N300_3.1-8B": 1050,
        "T3K_3.1-8B": 1050,
        #
        "N150_3.2-11B": 1050,  # TODO Update target
        "N300_3.2-11B": 1050,  # TODO Update target
        "T3K_3.2-11B": 1050,  # TODO Update target
        #
        "N150_3.1-70B": 1050,  # TODO Update target
        "N300_3.1-70B": 1050,  # TODO Update target
        "T3K_3.1-70B": 1050,  # TODO Update target
    }[f"{tt_device_name}_{llama_model_name}"]

    # Set the target decode timesfor every combination of device and model
    target_decode_tok_s_u = {
        "N150_3.2-1B": 160,  # TODO Update target
        "N300_3.2-1B": 250,  # TODO Update target
        "T3K_3.2-1B": 300,  # TODO Update target
        #
        "N150_3.2-3B": 60,  # TODO Update target
        "N300_3.2-3B": 100,  # TODO Update target
        "T3K_3.2-3B": 150,  # TODO Update target
        #
        "N150_3.1-8B": 23,  # TODO Update target
        "N300_3.1-8B": 38,
        "T3K_3.1-8B": 45,
        #
        "N150_3.2-11B": 23,
        "N300_3.2-11B": 38,  # TODO Update target
        "T3K_3.2-11B": 45,  # TODO Update target
        #
        "T3K_3.1-70B": 20,  # TODO Update target
    }[f"{tt_device_name}_{llama_model_name}"]

    target_decode_tok_s = target_decode_tok_s_u * batch_size
    targets = {
        "prefill_t/s": target_prefill_tok_s,
        "decode_t/s": target_decode_tok_s,
        "decode_t/s/u": target_decode_tok_s_u,
    }

    # Save benchmark data for CI dashboard
    # if is_ci_env:
    if True:
        benchmark_data = create_benchmark_data(profiler, measurements, N_warmup_iter, targets)
        benchmark_data.prep_csvs(
            profiler,
            run_type=f"{tt_device_name}-demo",
            ml_model_name=llama_model_name,
            ml_model_type="llm",
            num_layers=model_args.n_layers,
            batch_size=batch_size,
            input_sequence_length=prefill_seq_len,
            output_sequence_length=1,
            # config_params=,
            # precision=,
        )


@pytest.mark.parametrize(
    "input_prompts, instruct_weights, num_batches, single_layer, optimizations",
    [
        ("models/demos/llama3/demo/input_data_prefill_128.json", False, 1, False, LlamaOptimizations.performance),
        ("models/demos/llama3/demo/input_data_prefill_128.json", False, 2, False, LlamaOptimizations.performance),
        (
            "models/demos/llama3/demo/input_data_questions_prefill_128.json",
            True,
            1,
            False,
            LlamaOptimizations.performance,
        ),
        (
            "models/demos/llama3/demo/input_data_questions_prefill_128.json",
            True,
            2,
            False,
            LlamaOptimizations.performance,
        ),
        ("models/demos/llama3/demo/input_data_long.json", True, 1, False, LlamaOptimizations.performance),
        ("models/demos/llama3/demo/input_data_long.json", True, 1, False, LlamaOptimizations.accuracy),
        (
            "models/demos/llama3/demo/input_data_questions_prefill_128.json",
            True,
            1,
            True,
            LlamaOptimizations.performance,
        ),
        ("models/demos/llama3/demo/mayo.json", True, 1, False),
    ],
    ids=[
        "general_weights-1_batch",
        "general_weights-2_batch",
        "instruct_weights-1_batch",
        "instruct_weights-2_batch",
        "instruct_weights-long-performance",
        "instruct_weights-long-accuracy",
        "single_layer",
        "mayo",
    ],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 23887872, "num_command_queues": 2}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_llama_demo(
    mesh_device,
    use_program_cache,
    input_prompts,
    instruct_weights,
    is_ci_env,
    num_batches,
    single_layer,
    optimizations,
    reset_seeds,
):
    if is_ci_env and (instruct_weights == False or "long" in input_prompts or single_layer == True):
        pytest.skip("CI demo test only runs instruct weights to reduce CI pipeline load (both are supported)")

    mesh_device.enable_async(True)

    return run_llama3_demo(
        user_input=input_prompts,
        single_layer=single_layer,
        mesh_device=mesh_device,
        instruct_mode=instruct_weights,
        is_ci_env=is_ci_env,
        num_batches=num_batches,
        print_to_file=False,
        optimizations=optimizations,
    )
