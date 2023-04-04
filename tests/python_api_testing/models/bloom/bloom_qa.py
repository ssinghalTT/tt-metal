from abc import abstractmethod
import torch
from transformers import BloomForQuestionAnswering
import math
import torch.nn as nn
from torch.nn import functional as F

from pymetal import ttmetal as ttm
from python_api_testing.models.bert.embeddings import PytorchEmbeddings
from python_api_testing.models.bert.mha import TtMultiHeadAttentionModel
from python_api_testing.models.bert.ffn import TtFeedForwardModel
from python_api_testing.models.bert.bert_encoder import TtBertEncoder
from python_api_testing.fused_ops.linear import Linear as ttLinear
from python_api_testing.fused_ops.softmax import softmax as tt_softmax
from python_api_testing.fused_ops.layernorm import Layernorm as ttLayernorm

from utility_functions import pad_activation, pad_weight, tilize_to_list, untilize, print_diff_argmax
from utility_functions import enable_binary_cache, enable_compile_cache, get_compile_cache_enabled, get_binary_cache_enabled
import numpy as np
from typing import Optional, Tuple, Union

def dropout_add(x, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out

def tt_dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> ttm.tensor.Tensor:


    tt_res = tilize_to_list(pad_activation(residual))
    tt_res = ttm.tensor.Tensor(tt_res, [1,1,64,64], ttm.tensor.DataType.BFLOAT16,  ttm.tensor.Layout.TILE, device)

    out = F.dropout(x, p=prob, training=training)
    tt_out = tilize_to_list(pad_activation(out))
    tt_out = ttm.tensor.Tensor(tt_out, [1,1,64,64], ttm.tensor.DataType.BFLOAT16,  ttm.tensor.Layout.TILE, device)

    total = ttm.tensor.add(tt_res, tt_out)

    return total


def tensor_torch2tt(torch_tensor):
    tt_tensor = tilize_to_list(pad_activation(torch_tensor))
    if len(torch_tensor.shape)==4:
        tt_tensor = ttm.tensor.Tensor(tt_tensor, torch_tensor.shape, ttm.tensor.DataType.BFLOAT16,  ttm.tensor.Layout.TILE, device)
    else:
        s1 = 1
        s2 = torch_tensor.shape[0]
        s3 = torch_tensor.shape[1]
        s4 = torch_tensor.shape[2]
        tt_tensor = ttm.tensor.Tensor(tt_tensor, [s1, s2, s3, s4], ttm.tensor.DataType.BFLOAT16,  ttm.tensor.Layout.TILE, device)


    return tt_tensor

def torch2tt_tensor(tt_tensor, pytorch_shape):
    if(len(pytorch_shape)==4):
        tt_out_host = tt_tensor.to(host)
        tt_out = untilize(torch.Tensor(tt_out_host.data()).reshape(*pytorch_shape))
        return tt_out
    else:
        s1 = 1
        s2 = pytorch_out.shape[0]
        s3 = pytorch_out.shape[1]
        s4 = pytorch_out.shape[2]
        out_shape = [s1, s2, s3, s4]
        tt_out_host = tt_tensor.to(host)
        tt_out = untilize(torch.Tensor(tt_out_host.data()).reshape(*out_shape))
        return tt_out


def tt_const_tensor(value, shape):
    if (len(shape)==4):
        number_tensor = torch.full(shape, value)
        tt_number_tensor = tilize_to_list(number_tensor)
        tt_number_tensor = ttm.tensor.Tensor(tt_number_tensor, number_tensor.shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

        return tt_number_tensor
    else:
        s1 = 1
        s2 = shape[0]
        s3 = shape[1]
        s4 = shape[2]
        number_tensor = torch.full([s1, s2, s3, s4], value)
        tt_number_tensor = tilize_to_list(number_tensor)
        tt_number_tensor = ttm.tensor.Tensor(tt_number_tensor, number_tensor.shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

        return tt_number_tensor


def tt_baddbmm(input, batch1, batch2, beta=1, alpha=1, out=None) -> ttm.tensor.Tensor:
    tt_batch1 = tensor_torch2tt(batch1)
    tt_batch2 = tensor_torch2tt(batch2)
    tt_input = tensor_torch2tt(input)
    tt_beta = tt_const_tensor(beta, input.shape)
    tt_alpha = tt_const_tensor(alpha, input.shape)

    res1 = ttm.tensor.mul(tt_beta, tt_input)
    res2 = ttm.tensor.matmul(tt_batch1, tt_batch2)
    res3 = ttm.tensor.mul(tt_alpha, res2)
    res4 = ttm.tensor.add(res1, res3)

    return res4



def tt_merge_heads(tt_x):

    num_heads = 32
    head_dim = 1024 // num_heads

    batch_size_and_num_heads, seq_length, _ = x.shape
    batch_size = batch_size_and_num_heads // num_heads

    reshaped = ttm.tensor.reshape(tt_x, batch_size, num_heads, seq_length, head_dim)
    p_reshaped = torch.Tensor(reshaped.to(host).data()).reshape(reshaped.shape())
    p_reshaped = torch.Tensor(x).reshape(batch_size, num_heads, seq_length, head_dim)

    # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
    p_permuted = p_reshaped.permute(0, 2, 1, 3)

    permuted = ttm.tensor.Tensor(tilize_to_list(p_permuted), [batch_size, num_heads, seq_length, head_dim], ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    third = num_heads*head_dim

    reshaped_2 = ttm.tensor.reshape(permuted, 1, batch_size, seq_length, num_heads*head_dim)

    res_reshaped_2 = tensor_torch2tt(reshaped_2)

    return res_reshaped_2

def split_heads(fused_qkv: torch.Tensor, num_heads, head_dim) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split the last dimension into (num_heads, head_dim) without making any copies, results share same memory
    storage as `fused_qkv`

    Args:
        fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

    Returns:
        query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
        value: [batch_size, seq_length, num_heads, head_dim]
    """
    batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
    fused_qkv = fused_qkv.view(batch_size, seq_length, num_heads, 3, head_dim)
    return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]

def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
    """
    Merge heads together over the last dimenstion

    Args:
        x: (`torch.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]

    Returns:
        torch.tensor: [batch_size, seq_length, num_heads * head_dim]
    """
    # What we want to achieve is:
    # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
    batch_size_and_num_heads, seq_length, _ = x.shape
    batch_size = batch_size_and_num_heads // self.num_heads

    # First view     to decompose the batch size
    # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
    x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)

    # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
    x = x.permute(0, 2, 1, 3)

    # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
    return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

class BloomAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden_size = 64
        self.num_heads = 8
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = 0.0
        self.inv_norm_factor = 0.0

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0

        self.query_key_value = torch.nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.dense = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_dropout = torch.nn.Dropout(0.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = split_heads(fused_qkv, self.num_heads, self.head_dim)

        batch_size, q_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        key_layer = key_layer.permute(0, 2, 3, 1).reshape(batch_size * self.num_heads, self.head_dim, q_length)
        value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=2)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        _, _, kv_length = key_layer.shape

        # [batch_size * num_heads, q_length, kv_length]
        # we use `torch.Tensor.baddbmm` instead of `torch.baddbmm` as the latter isn't supported by TorchScript v1.11
        matmul_result = alibi.baddbmm(
            batch1=query_layer,
            batch2=key_layer,
            beta=self.beta,
            alpha=self.inv_norm_factor,
        )

        # change view to [batch_size, num_heads, q_length, kv_length]
        attention_scores = matmul_result.view(batch_size, self.num_heads, q_length, kv_length)

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
        input_dtype = attention_scores.dtype
        # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
        if input_dtype == torch.float16:
            attention_scores = attention_scores.to(torch.float)
        attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
        attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(input_dtype)

        # [batch_size, num_heads, q_length, kv_length]
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # change view [batch_size x num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, kv_length)

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = torch.bmm(attention_probs_reshaped, value_layer)

        # change view [batch_size, num_heads, q_length, head_dim]
        context_layer = self.merge_heads(context_layer)

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)

        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs

class TtBloomAttention(torch.nn.Module):
    def __init__(self, sd, device):
        super().__init__()

        self.hidden_size = 64
        self.num_heads = 8
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = 0.0
        self.inv_norm_factor = 0.0


        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        weight_q = tilize_to_list(pad_weight(sd[f"transformer.h.0.self_attention.query_key_value.weight"]))
        bias_q= tilize_to_list(pad_weight(sd[f"transformer.h.0.self_attention.query_key_value.bias"]))

        weight_d = tilize_to_list(pad_weight(sd[f"transformer.h.0.self_attention.dense.weight"]))
        bias_d = tilize_to_list(pad_weight(sd[f"transformer.h.0.self_attention.dense.bias"]))

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0

        self.query_key_value = ttLinear(self.hidden_size, 3 * self.hidden_size, weight_q, bias_q, device)

        self.dense = ttLinear(self.hidden_size, self.hidden_size, weight_d, bias_d, device)
        self.attention_dropout = torch.nn.Dropout(0.0)


    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):

        pb = BloomAttention()

        s1 = hidden_states.shape[0]
        s2 = hidden_states.shape[1]
        s3 = hidden_states.shape[2]
        s4 = hidden_states.shape[3]

        tt_hidden_states = tilize_to_list(pad_activation(hidden_states))
        tt_hidden_states = ttm.tensor.Tensor(tt_hidden_states, [s1, s2, s3, s4], ttm.tensor.DataType.BFLOAT16,  ttm.tensor.Layout.TILE, device)

        tt_fused_qkv = self.query_key_value(tt_hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

        f_shapes = tt_fused_qkv.shape()
        fused_qkv = torch.Tensor(tt_fused_qkv.to(host).data()).reshape([f_shapes[1], f_shapes[2], f_shapes[3]])

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = split_heads(fused_qkv, self.num_heads, self.head_dim)

        batch_size, q_length, _, _ = query_layer.shape

        #p_reshaped_query_layer = torch.Tensor(fused_qkv).reshape(1, batch_size, seq * self.num_heads,  q_length, self.head_dim)
        #query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        s1 = query_layer.shape[0]
        s2 = query_layer.shape[1]
        s3 = query_layer.shape[2]
        s4 = query_layer.shape[3]
        print("SHAPE")
        print(s1)
        print(s2)
        print(s3)
        print(s4)

        tt_query_layer = tilize_to_list(pad_activation(query_layer))
        tt_query_layer = ttm.tensor.Tensor(tt_query_layer, [s1, s2, s3, s4], ttm.tensor.DataType.BFLOAT16,  ttm.tensor.Layout.TILE, device)

        tt_transposed_query_layer = ttm.tensor.transpose(tt_query_layer)
        tt_reshaped_query_layer = ttm.tensor.reshape(tt_transposed_query_layer, 1, batch_size * self.num_heads, q_length, self.head_dim)

        #key_layer = key_layer.permute(0, 2, 3, 1).reshape(batch_size * self.num_heads, self.head_dim, q_length)
        key_layer = key_layer.permute(0, 2, 3, 1)
        s1 = key_layer.shape[0]
        s2 = key_layer.shape[1]
        s3 = key_layer.shape[2]
        s4 = key_layer.shape[3]

        tt_key_layer = tilize_to_list(pad_activation(key_layer))
        tt_key_layer = ttm.tensor.Tensor(tt_key_layer, [s1, s2, s3, s4], ttm.tensor.DataType.BFLOAT16,  ttm.tensor.Layout.TILE, device)
        tt_reshaped_key_layer = ttm.tensor.reshape(tt_key_query_layer, 1, batch_size * self.num_heads, self.head_dim, q_length)

        #value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        s1 = value_layer.shape[0]
        s2 = value_layer.shape[1]
        s3 = value_layer.shape[2]
        s4 = value_layer.shape[3]

        tt_value_layer = tilize_to_list(pad_activation(value_layer))
        tt_value_layer = ttm.tensor.Tensor(tt_value_layer, [s1, s2, s3, s4], ttm.tensor.DataType.BFLOAT16,  ttm.tensor.Layout.TILE, device)

        tt_transposed_value_layer = ttm.tensor.transpose(tt_value_layer)
        tt_reshaped_value_layer = ttm.tensor.reshape(tt_transposed_value_layer, 1, batch_size * self.num_heads, q_length, self.head_dim)

        # [batch_size * num_heads, q_length, kv_length]
        # we use `torch.Tensor.baddbmm` instead of `torch.baddbmm` as the latter isn't supported by TorchScript v1.11
        p_reshaped_query_layer = torch.Tensor(tt_reshaped_query_layer.to(host).data()).reshape(p_reshaped_query_layer.shape())
        p_reshaped_query_layer = torch.Tensor(p_reshaped_query_layer).reshape(1, batch_size * self.num_heads,  q_length, self.head_dim)

        p_reshaped_key_layer = torch.Tensor(tt_reshaped_key_layer.to(host).data()).reshape(p_reshaped_key_layer.shape())
        p_reshaped_key_layer = torch.Tensor(p_reshaped_key_layer).reshape(1, batch_size * self.num_heads, self.head_dim, q_length)

        matmul_result = tt_baddbmm(alibi, batch1=p_reshaped_query_layer, batch2=p_reshaped_key_layer, beta=self.beta, alpha=self.inv_norm_facto)

        # change view to [batch_size, num_heads, q_length, kv_length]
        tt_attention_scores = ttm.tensor.reshape(tt_matmul_result, 1, batch_size, self.num_heads, q_length, kv_length)
        p_attention_scores = torch2tt_tensor(tt_attention_scores, tt_attention_scores.shape())

        attention_scores = p_attention_scores.to(torch.float)

        attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)


        tt_attn_weights = tensor_torch2tt(attn_weights)

        tt_attention_probs = tt_softmax.softmax(tt_attn_weights)

        #TO BE DONE
        # [batch_size, num_heads, q_length, kv_length]
        #attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            tt_head_mask =  tensor_torch2tt(head_mask)
            tt_attention_probs = ttm.mul(tt_attention_probs, head_mask)

        # change view [batch_size x num_heads, q_length, kv_length]
        tt_attention_probs_reshaped = ttm.tensor.reshape(tt_attention_probs, 1, batch_size * self.num_heads, q_length, kv_length)

        # matmul: [batch_size * num_heads, q_length, head_dim]
        tt_context_layer = ttm.tensor.matmul(tt_attention_probs_reshaped, value_layer)

        # change view [batch_size, num_heads, q_length, head_dim]
        merged_context_layer = tt_merge_heads(tt_context_layer)

        output_tensor = self.dense(merged_context_layer)

        output_tensor = tt_dropout_add(output_tensor, residual, self.hidden_dropout, False)

        outputs = ttm.tensor.add(output_tensor, attention_probs)

        return outputs

        return tt_reshaped_matmul_result


def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out

def tt_dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> ttm.tensor.Tensor:


    tt_res = tilize_to_list(pad_activation(residual))
    tt_res = ttm.tensor.Tensor(tt_res, [1,1,64,64], ttm.tensor.DataType.BFLOAT16,  ttm.tensor.Layout.TILE, device)

    out = F.dropout(x, p=prob, training=training)
    tt_out = tilize_to_list(pad_activation(out))
    tt_out = ttm.tensor.Tensor(tt_out, [1,1,64,64], ttm.tensor.DataType.BFLOAT16,  ttm.tensor.Layout.TILE, device)

    total = ttm.tensor.add(tt_res, tt_out)

    return total

def bloom_gelu_forward(x: torch.Tensor) -> torch.Tensor:
    """
    Custom bias GELU function. Adapted from Megatron-DeepSpeed code. Here we use a simple implementation (inference) to
    make the model jitable.

    Args:
        x (`torch.tensor`, *required*):
            input hidden states
    """
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

def tt_bloom_gelu_forward(x, d1, d2, d3, d4):
    z = x

    k1 = torch.full((d1, d2, d3, d4), 0.5)
    k1 = tilize_to_list(k1)
    k1_dev = ttm.tensor.Tensor(k1, [d1, d2, d3, d4], ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    k2 = torch.full((d1, d2, d3, d4), 0.044715)
    k2 = tilize_to_list(k2)
    k2_dev = ttm.tensor.Tensor(k2, [d1, d2, d3, d4], ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    k3 = torch.full((d1, d2, d3, d4), 0.79788456)
    k3 = tilize_to_list(k3)
    k3_dev = ttm.tensor.Tensor(k3, [d1, d2, d3, d4], ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    #0.5*x
    factor1 = ttm.tensor.mul(k1_dev, z) # exp(z)
    #x*x
    pow2 = ttm.tensor.mul(z, z)
    #(x + 0.044715 * torch.pow(x, 3)))
    #torch.pow(x, 3))
    pow3 = ttm.tensor.mul(pow2, z)
    factor3 = ttm.tensor.mul(k2_dev, pow3)
    #(x + 0.044715 * torch.pow(x, 3)))
    factor3 = ttm.tensor.add(factor3, z)

    sumtanh = ttm.tensor.mul(k3_dev, factor3)

    tanh = ttm.tensor.tanh(sumtanh)

    k4 = torch.full((d1, d2, d3, d4), 1)
    k4 = tilize_to_list(k4)
    k4_dev = ttm.tensor.Tensor(k4, [d1, d2, d3, d4], ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    total = ttm.tensor.add(k4_dev, tanh)

    output = ttm.tensor.mul(factor1, total)

    return output



class ttBloomMLP(torch.nn.Module):
    def __init__(self, sd):
        super().__init__()
        hidden_size = 64

        self.pretraining_tp = 1
        self.slow_but_exact = False

        tt_weight_mlp_h4h = tilize_to_list(pad_weight(sd[f"transformer.h.0.mlp.dense_h_to_4h.bias"]))
        tt_bias_mlp_h4h = tilize_to_list(pad_weight(sd[f"transformer.h.0.mlp.dense_h_to_4h.bias"]))

        self.aux_dense_h_to_4h = torch.nn.Linear(hidden_size, 4 * hidden_size)
        self.aux_dense_4h_to_h = torch.nn.Linear(4 * hidden_size, hidden_size)

        self.dense_h_to_4h = ttLinear(hidden_size, 4 * hidden_size, tt_weight_mlp_h4h, tt_bias_mlp_h4h, device)

        self.gelu_impl = tt_bloom_gelu_forward

        tt_weight_mlp_4hh = tilize_to_list(pad_weight(sd[f"transformer.h.0.mlp.dense_4h_to_h.bias"]))
        tt_bias_mlp_4hh = tilize_to_list(pad_weight(sd[f"transformer.h.0.mlp.dense_4h_to_h.bias"]))

        self.dense_4h_to_h = ttLinear(4*hidden_size, hidden_size, tt_weight_mlp_4hh, tt_bias_mlp_4hh, device)

        self.hidden_dropout = 0.0
        self.training = False

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:

        tt_hidden_states_input = tilize_to_list(pad_activation(hidden_states))
        tt_hs = ttm.tensor.Tensor(tt_hidden_states_input, hidden_states.shape, ttm.tensor.DataType.BFLOAT16,  ttm.tensor.Layout.TILE, device)

        tt_h4h = self.dense_h_to_4h(tt_hs)

        s1 = hidden_states.shape[0]
        s2 = hidden_states.shape[1]
        s3 = hidden_states.shape[2]
        s4 = hidden_states.shape[3]

        tt_hidden_states = self.gelu_impl(tt_h4h, s1, s2, s3, s4)

        tt_intermediate_output = self.dense_4h_to_h(tt_hidden_states)

        tt_res_temp = tilize_to_list(residual)

        tt_res = ttm.tensor.Tensor(tt_res_temp, tt_intermediate_output.shape(), ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

        res = tt_res.to(host).data()

        tt_got_back_res = torch.Tensor(res).reshape(tt_intermediate_output.shape())
        tt_got_back_res = untilize(tt_got_back_res)

        intermediate_output =tt_intermediate_output.to(host).data()

        tt_got_back_intermediate_output = torch.Tensor(intermediate_output).reshape((1,1,64,64))
        tt_got_back_intermediate_output = untilize(tt_got_back_intermediate_output)

        output = tt_dropout_add(tt_got_back_intermediate_output, tt_got_back_res, self.hidden_dropout, self.training)

        return output



class GeLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return bloom_gelu_forward(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input = ctx.saved_tensors
        tmp = bloom_gelu_back(grad_output, input)
        return tmp

class BloomGelu(torch.nn.Module):
    """
    BloomBiasGelu wrapper function that make use of the simple function on inference mode to make the model
    torchscriptable and use the autograd function in training mode to get the accurate results of the gradients Partly
    copied from Megatron-DeepSpeed code and adapted for our needs

    See here why autograd functions are not torchscriptable: https://github.com/pytorch/pytorch/issues/22329
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return GeLUFunction.apply(x)
        else:
            return bloom_gelu_forward(x)

class BloomMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 64

        self.pretraining_tp = False
        self.slow_but_exact = False
        self.dense_h_to_4h = torch.nn.Linear(hidden_size, 4 * hidden_size)
        self.gelu_impl = BloomGelu()
        self.dense_4h_to_h = torch.nn.Linear(4 * hidden_size, hidden_size)
        self.hidden_dropout = 0.0

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))

        intermediate_output = self.dense_4h_to_h(hidden_states)

        output = dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)

        return output


class TtBloomBlock(nn.Module):
    def __init__(self, sd, device):
        super().__init__()
        hidden_size = 64

        self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.num_heads = 8
        self.self_attention = TtBloomAttention(self, sd, device)


        tt_beta = tilize_to_list(pad_weight(sd[f"h.0.input_layernorm.bias"]))
        tt_gamma = tilize_to_list(pad_weight(sd[f"h.0.input_layernorm.weight"]))

        self.post_attention_layernorm = ttLayernorm(tt_gamma, tt_beta, 1e-05, hidden_size, hidden_size, device, 1)

        self.mlp = ttBloomMLP(sd)

        self.apply_residual_connection_post_layernorm = False
        self.hidden_dropout = 0.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        # hidden_states: [batch_size, seq_length, hidden_size]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Layer norm post the self attention.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # Self attention.
        attn_outputs = self.self_attention(
            layernorm_output,
            residual,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs
        p_attn_outputs = torch2tt_tensor(attention_output, attention_output.shape())

        outputs = p_attn_outputs[1:]
        tt_outputs = tensor_torch2tt(outputs)
        outputs_1 = tensor_torch2tt(outputs[1:])

        layernorm_output = self.post_attention_layernorm(attention_output)

        # Get residual
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output

        # MLP.
        output = self.mlp(layernorm_output, residual)

        if use_cache:
            tt_outputs = ttm.tensor.add(output, tt_outputs)
        else:
            tt_outputs = ttm.tensor.add(output, output_1)

        return tt_outputs  # hidden_states, present, attentions



def _make_causal_mask(
    input_ids_shape: torch.Size, device: torch.device, past_key_values_length: int
) -> torch.BoolTensor:
    """
    Make causal mask used for self-attention.
    """
    batch_size, target_length = input_ids_shape
    mask = torch.empty((target_length, target_length + past_key_values_length), dtype=torch.bool, device=device)
    # ONNX doesn't support `torch.Tensor.triu` properly, thus we use this workaround
    seq_ids = torch.arange(target_length, device=device)
    mask[:, past_key_values_length:] = seq_ids[:, None] < seq_ids[None, :]

    if past_key_values_length > 0:
        mask[:, :past_key_values_length] = False

    expanded_mask = mask[None, None, :, :].expand(batch_size, 1, target_length, target_length + past_key_values_length)
    return expanded_mask


def _expand_mask(mask: torch.Tensor, tgt_length: int) -> torch.BoolTensor:
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length

    expanded_mask = ~(mask[:, None, None, :].to(torch.bool))
    return expanded_mask.expand(batch_size, 1, tgt_length, src_length)


def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.

    Args:
    Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
        attention_mask (`torch.Tensor`):
            Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        num_heads (`int`, *required*):
            number of heads
        dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            dtype of the output tensor
    """
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
    )
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)



class TtBloomModel():
    def __init__(self, config: BloomConfig):
        super().__init__(config)

        self.embed_dim = 64
        self.num_heads = 8
        self.vocab_size = 250880
        self.num_hidden_layers = 2

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)

        tt_beta_first = tilize_to_list(pad_weight(sd[f"h.0.input_layernorm.bias"]))
        tt_gamma_first = tilize_to_list(pad_weight(sd[f"h.0.input_layernorm.weight"]))

        self.word_embeddings_layernorm = ttLayernorm(tt_gamma_first, tt_beta_first, 1e-05, hidden_size, hidden_size, device, 1)


        # Transformer blocks
        self.h = nn.ModuleList([BloomBlock() for _ in range(self.num_hidden_layers)])

        # Final Layer Norm
        tt_beta_final = tilize_to_list(pad_weight(sd[f"h.23.post_attention_layernorm.bias"]))
        tt_gamma_final = tilize_to_list(pad_weight(sd[f"h.23.post_attention_layernorm.weight"]))

        self.word_embeddings_layernorm = ttLayernorm(tt_gamma_final, tt_beta_final, 1e-05, hidden_size, hidden_size, device, 1)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


    def get_input_embeddings(self):
        return self.word_embeddings

    def _prepare_attn_mask(
        self, attention_mask: torch.Tensor, input_shape: Tuple[int, int], past_key_values_length: int
    ) -> torch.BoolTensor:
        # create causal mask
        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        combined_attention_mask = None
        device = attention_mask.device
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, device=device, past_key_values_length=past_key_values_length
            )

        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
        )

        return combined_attention_mask

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        tt_inputs_embeds = tensor_torch2tt(inputs_embeds)

        tt_hidden_states = self.word_embeddings_layernorm(tt_inputs_embeds)
        hidden_states = torch2tt_tensor(tt_hidden_states, tt_hidden_states.shape())

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)


        alibi = build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)

        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)


                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Add last hidden state

        tt_input_hidden_states = tensor_torch2tt(hidden_states)

        tt_hidden_states_last = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class BloomModel(BloomPreTrainedModel):
    def __init__(self, config: BloomConfig):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        self.word_embeddings_layernorm = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Transformer blocks
        self.h = nn.ModuleList([BloomBlock(config) for _ in range(config.num_hidden_layers)])

        # Final Layer Norm
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
    return self.word_embeddings

    def _prepare_attn_mask(
        self, attention_mask: torch.Tensor, input_shape: Tuple[int, int], past_key_values_length: int
    ) -> torch.BoolTensor:
        # create causal mask
        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        combined_attention_mask = None
        device = attention_mask.device
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, device=device, past_key_values_length=past_key_values_length
            )

        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
        )

        return combined_attention_mask

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        alibi = build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)

        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache=use_cache, output_attentions=output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    alibi,
                    causal_mask,
                    layer_past,
                    head_mask[i],
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class TtBloomForQuestionAnswering(BloomPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = TtBloomModel()

        tt_bias_qa = tilize_to_list(pad_weight(sd[f"qa_outputs.bias"]))
        tt_weight_qa = tilize_to_list(pad_weight(sd[f"qa_outputs.weight"]))

        self.qa_outputs = ttLinear(64, 2, weight_qa, bias_qa, device)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        tt_logits = self.qa_outputs(sequence_output)
        logits = torch2tt_tensor(tt_logits, tt_logits.shape())

        #Misses toch.split and squeeze
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()


        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BloomForQuestionAnswering(BloomPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.transformer = BloomModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)




        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def run_bloom_qa_inference():
    hugging_bloom_reference_model = BloomForQuestionAnswering.from_pretrained("bigscience/bloom-560m", torchscript=False)

    tbloom = TtBloomForQuestionAnswering(hugging_bloom_reference_model.state_dict(), device)

    # Prepare input
    torch.manual_seed(0)


    # To be completed

    pytorch_out = pbloom.forward(hidden_states, residual, alibi, attention_mask)

    assert np.allclose(pytorch_out.detach().numpy(), tt_out.numpy(), 1e-5, 0.17)

if __name__ == "__main__":
    # Initialize the device
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()
    run_bloom_block_inference()
    ttm.device.CloseDevice(device)
