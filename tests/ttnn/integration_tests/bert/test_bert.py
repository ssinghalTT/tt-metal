# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import torch.nn.functional as F
import transformers


import ttnn

from models.experimental.functional_bert.tt.ttnn_functional_bert import (
    ttnn_bert_for_question_answering,
)

from models.experimental.functional_bert.tt.ttnn_optimized_functional_bert import (
    ttnn_optimized_bert_for_question_answering,
)
from models.experimental.functional_bert.reference.torch_functional_bert import (
    torch_bert_for_question_answering,
)
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_bias,
    preprocess_linear_weight,
)

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0


def ttnn_bert_preprocess_inputs(
    input_ids,
    token_type_ids,
    attention_mask,
    **kwargs,
):
    input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32)
    input_ids = ttnn.to_device(input_ids, kwargs["device"], memory_config=ttnn.L1_MEMORY_CONFIG)

    token_type_ids = ttnn.from_torch(token_type_ids, dtype=ttnn.uint32)
    token_type_ids = ttnn.to_device(token_type_ids, kwargs["device"], memory_config=ttnn.L1_MEMORY_CONFIG)

    if attention_mask is not None:
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        attention_mask = F.pad(attention_mask, (0, 0, 0, 31, 0, 0, 0, kwargs["batch_size"] - 1))
        attention_mask = ttnn.from_torch(attention_mask, dtype=ttnn.bfloat16)
        attention_mask = ttnn.to_layout(attention_mask, ttnn.TILE_LAYOUT)
        attention_mask = ttnn.to_device(attention_mask, kwargs["device"], memory_config=ttnn.L1_MEMORY_CONFIG)

    return input_ids, token_type_ids, attention_mask


def convert_to_ttnn(torch_model, full_name):
    return True


def custom_preprocessor(torch_model, name):
    parameters = {}
    if isinstance(torch_model, transformers.models.bert.modeling_bert.BertSelfAttention):
        qkv_weight = torch.cat(
            [
                torch_model.query.weight,
                torch_model.key.weight,
                torch_model.value.weight,
            ],
            dim=0,
        )
        qkv_bias = torch.cat(
            [torch_model.query.bias, torch_model.key.bias, torch_model.value.bias],
            dim=0,
        )

        parameters = {"query_key_value": {}}
        parameters["query_key_value"]["weight"] = preprocess_linear_weight(qkv_weight, dtype=ttnn.bfloat16)
        parameters["query_key_value"]["bias"] = preprocess_linear_bias(qkv_bias, dtype=ttnn.bfloat16)
    return parameters


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [384])
@pytest.mark.parametrize("use_optimized_version", [True, False])
def test_bert(device, use_program_cache, model_name, batch_size, sequence_size, use_optimized_version):
    torch.manual_seed(1234)

    config = transformers.BertConfig.from_pretrained(model_name)

    # TODO(arakhmati): re-enable the line below once the issue with ttnn.embedding is fixed
    # torch_bert_input = torch.randint(0, config.config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_bert_input = torch.randint(0, 1, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = torch.zeros(1, sequence_size) if use_optimized_version else None

    torch_parameters = preprocess_model_parameters(
        f"torch-{model_name}",
        initialize_model=lambda: transformers.BertForQuestionAnswering.from_pretrained(
            model_name, torchscript=False
        ).eval(),
        convert_to_ttnn=lambda *_: False,
    )

    torch_output = torch_bert_for_question_answering(
        torch_bert_input,
        torch_token_type_ids,
        torch_attention_mask,
        parameters=torch_parameters,
        num_heads=config.num_attention_heads,
    )

    # Run TT Model
    parameters = preprocess_model_parameters(
        "ttnn_" + ("optimized_" if use_optimized_version else "") + model_name,
        initialize_model=lambda: transformers.BertForQuestionAnswering.from_pretrained(
            model_name, torchscript=False
        ).eval(),
        convert_to_ttnn=convert_to_ttnn,
        custom_preprocessor=custom_preprocessor if use_optimized_version else None,
        device=device,
    )

    bert_for_question_answering = (
        ttnn_optimized_bert_for_question_answering if use_optimized_version else ttnn_bert_for_question_answering
    )

    ttnn_bert_inputs = ttnn_bert_preprocess_inputs(
        torch_bert_input,
        torch_token_type_ids,
        torch_attention_mask,
        device=device,
        batch_size=batch_size,
    )

    tt_output = bert_for_question_answering(
        *ttnn_bert_inputs,
        parameters=parameters,
        num_heads=config.num_attention_heads,
    )
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.from_device(tt_output)

    tt_output = ttnn.to_torch(tt_output)
    tt_output = tt_output[..., :2]

    assert_with_pcc(torch_output, tt_output, 0.9999)
