# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ... import initialization as init
from ...activations import ACT2FN, get_activation
from ...cache_utils import Cache, DynamicCache, EncoderDecoderCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseAutoencoderOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
    CausalLMAutoencoderOutputWithCrossAttentions,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...pytorch_utils import Conv1D
from ...utils import (
    ModelOutput,
    auto_docstring,
    logging,
)
from .configuration_latent_gpt2 import LatentGPT2Config


logger = logging.get_logger(__name__)


def eager_attention_forward(module, query, key, value, attention_mask, **kwargs):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

    # Layer-wise attention scaling
    if module.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(module.layer_idx + 1)

    if not module.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = module.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

    if attention_mask is not None:
        # Apply the attention mask
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = module.attn_dropout(attn_weights)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)

    return attn_output, attn_weights


class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.config = config
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = not is_cross_attention

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with torch.autocast(query.device.type, enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: Optional[tuple[torch.FloatTensor]],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> tuple[Union[torch.Tensor, tuple[torch.Tensor]], ...]:
        is_cross_attention = encoder_hidden_states is not None
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    # after the first generated id, we can subsequently re-use all key/value_layer from cache
                    curr_past_key_values = past_key_values.cross_attention_cache
                else:
                    curr_past_key_values = past_key_values.self_attention_cache
            else:
                curr_past_key_values = past_key_values

        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            query_states = self.q_attn(hidden_states)
            attention_mask = encoder_attention_mask

            # Try to get key/value states from cache if possible
            if past_key_values is not None and is_updated:
                key_states = curr_past_key_values.layers[self.layer_idx].keys
                value_states = curr_past_key_values.layers[self.layer_idx].values
            else:
                key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
                shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
                key_states = key_states.view(shape_kv).transpose(1, 2)
                value_states = value_states.view(shape_kv).transpose(1, 2)
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
            shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        if (past_key_values is not None and not is_cross_attention) or (
            past_key_values is not None and is_cross_attention and not is_updated
        ):
            # save all key/value_layer to cache to be re-used for fast auto-regressive generation
            cache_position = cache_position if not is_cross_attention else None
            key_states, value_states = curr_past_key_values.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )
            # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
            if is_cross_attention:
                past_key_values.is_updated[self.layer_idx] = True

        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if using_eager and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                **kwargs,
            )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config=config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config=config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[tuple[torch.FloatTensor]],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Union[tuple[torch.Tensor], Optional[tuple[torch.Tensor, tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, self_attn_weights = self.attn(
            hidden_states,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_output, cross_attn_weights = self.crossattention(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            # residual connection
            hidden_states = residual + cross_attn_output

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
            if encoder_hidden_states is not None:
                outputs += (cross_attn_weights,)

        return outputs


# Copied from transformers.models.xlm.modeling_xlm.XLMSequenceSummary with XLM->GPT2
class GPT2SequenceSummary(nn.Module):
    r"""
    Compute a single vector summary of a sequence hidden states.

    Args:
        config ([`LatentGPT2Config`]):
            The config used by the model. Relevant arguments in the config class of the model are (refer to the actual
            config class of your model for the default values it uses):

            - **summary_type** (`str`) -- The method to use to make this summary. Accepted values are:

                - `"last"` -- Take the last token hidden state (like XLNet)
                - `"first"` -- Take the first token hidden state (like Bert)
                - `"mean"` -- Take the mean of all tokens hidden states
                - `"cls_index"` -- Supply a Tensor of classification token position (GPT/GPT-2)
                - `"attn"` -- Not implemented now, use multi-head attention

            - **summary_use_proj** (`bool`) -- Add a projection after the vector extraction.
            - **summary_proj_to_labels** (`bool`) -- If `True`, the projection outputs to `config.num_labels` classes
              (otherwise to `config.hidden_size`).
            - **summary_activation** (`Optional[str]`) -- Set to `"tanh"` to add a tanh activation to the output,
              another string or `None` will add no activation.
            - **summary_first_dropout** (`float`) -- Optional dropout probability before the projection and activation.
            - **summary_last_dropout** (`float`)-- Optional dropout probability after the projection and activation.
    """

    def __init__(self, config: LatentGPT2Config):
        super().__init__()

        self.summary_type = getattr(config, "summary_type", "last")
        if self.summary_type == "attn":
            # We should use a standard multi-head attention module with absolute positional embedding for that.
            # Cf. https://github.com/zihangdai/xlnet/blob/master/modeling.py#L253-L276
            # We can probably just use the multi-head attention module of PyTorch >=1.1.0
            raise NotImplementedError

        self.summary = nn.Identity()
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)

        activation_string = getattr(config, "summary_activation", None)
        self.activation: Callable = get_activation(activation_string) if activation_string else nn.Identity()

        self.first_dropout = nn.Identity()
        if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.last_dropout = nn.Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

    def forward(
        self, hidden_states: torch.FloatTensor, cls_index: Optional[torch.LongTensor] = None
    ) -> torch.FloatTensor:
        """
        Compute a single vector summary of a sequence hidden states.

        Args:
            hidden_states (`torch.FloatTensor` of shape `[batch_size, seq_len, hidden_size]`):
                The hidden states of the last layer.
            cls_index (`torch.LongTensor` of shape `[batch_size]` or `[batch_size, ...]` where ... are optional leading dimensions of `hidden_states`, *optional*):
                Used if `summary_type == "cls_index"` and takes the last token of the sequence as classification token.

        Returns:
            `torch.FloatTensor`: The summary of the sequence hidden states.
        """
        if self.summary_type == "last":
            output = hidden_states[:, -1]
        elif self.summary_type == "first":
            output = hidden_states[:, 0]
        elif self.summary_type == "mean":
            output = hidden_states.mean(dim=1)
        elif self.summary_type == "cls_index":
            if cls_index is None:
                cls_index = torch.full_like(
                    hidden_states[..., :1, :],
                    hidden_states.shape[-2] - 1,
                    dtype=torch.long,
                )
            else:
                cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                cls_index = cls_index.expand((-1,) * (cls_index.dim() - 1) + (hidden_states.size(-1),))
            # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
            output = hidden_states.gather(-2, cls_index).squeeze(-2)  # shape (bsz, XX, hidden_size)
        elif self.summary_type == "attn":
            raise NotImplementedError

        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output


@auto_docstring
class GPT2PreTrainedModel(PreTrainedModel):
    config: LatentGPT2Config
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPT2Block"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True

    _can_compile_fullgraph = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            # Here we need the check explicitly, as we slice the weight in the `zeros_` call, so it looses the flag
            if module.padding_idx is not None and not getattr(module.weight, "_is_hf_initialized", False):
                init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            init.zeros_(module.bias)
            init.ones_(module.weight)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        if isinstance(module, PreTrainedModel):
            for name, p in module.named_parameters():
                if name == "c_proj.weight":
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    init.normal_(p, mean=0.0, std=self.config.initializer_range / math.sqrt(2 * self.config.n_layer))


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for outputs of models predicting if two sentences are consecutive or not.
    """
)
class GPT2DoubleHeadsModelOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss.
    mc_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `mc_labels` is provided):
        Multiple choice classification loss.
    logits (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    mc_logits (`torch.FloatTensor` of shape `(batch_size, num_choices)`):
        Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    """

    loss: Optional[torch.FloatTensor] = None
    mc_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    mc_logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


@auto_docstring(custom_intro="The base GPT2 model transformer outputting raw hidden-states.")
class GPT2ModelBase(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # based on pattern from src/transformers/models/whisper/modeling_whisper.py::WhisperDecoder
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache(config=self.config)

            if self.config.add_cross_attention and not isinstance(past_key_values, EncoderDecoderCache):
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache(config=self.config))

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)

        # Attention mask.
        # ._update_causal_mask() and ._prepare_4d_causal_attention_mask_with_cache_position() copied from LlamaModel
        if attention_mask is not None and attention_mask.ndim < 4:
            attention_mask = attention_mask.view(batch_size, -1)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif self._attn_implementation != "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                past_key_values if not (self.gradient_checkpointing and self.training) else None,
                cache_position,
                causal_mask,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )

            hidden_states = outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[2],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        past_key_values = past_key_values if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, past_key_values, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

@auto_docstring(
    custom_intro="""
    The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """
)
class GPT2LMHeadModel(GPT2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2ModelBase(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        labels (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Flatten the tokens
            loss = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

@auto_docstring(custom_intro="Base class for language encoding with fixed-size window aggregation.")
class LanguageEncoderBase:
    def __init__(
        self,
        window_size: int,
        padding_token: Optional[int] = None,
        padding_embed: Optional[torch.Tensor] = None,
        wte: Optional[torch.nn.Embedding] = None,
    ):
        self.__batch_size: int = None
        self.__seq_len: int = None
        self.__segment_num: int = None
        
        self.__window_size: int = window_size  
        self.__padding_token = padding_token
        self.__padding_embed = padding_embed
        self.__wte: torch.nn.Embedding = wte
        
    @property
    def batch_size(self):
        return self.__batch_size

    @property
    def seq_len(self):
        return self.__seq_len
    
    @property
    def segment_num(self):
        return self.__segment_num
    
    @property
    def window_size(self):
        return self.__window_size
    
    def __get_padding_embed(self) -> torch.FloatTensor:
        if self.__padding_embed is None:
            if self.__wte is None:
                raise ValueError("To get padding_embed, either provide padding_embed or wte")
            self.__padding_embed = self.__wte(self.__padding_token)
        return self.__padding_embed
    def __get_padding_token(self) -> torch.LongTensor:
        if self.__padding_token is None:
            raise ValueError("padding_token is not set")
        return self.__padding_token
    def __pad(
        self,
        sequence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pad the given sequence to be a multiple of window_size.
        
        If sequence is None, return None.
        
        For 2D sequence (batch_size, seq_len), pad with padding_token.
        For 3D sequence (batch_size, seq_len, embedding_size), pad with padding_embed.
        
        Raises:
            NotImplementedError: If the input dimension is not supported.
        
        Returns:
            torch.Tensor: Padded sequence. Shape is (batch_size, seq_len + pad) if sequence is 2D, otherwise (batch_size, seq_len + pad, embedding_size).
        """
        if sequence is None:
            return sequence
        batch_size: int = sequence.shape[0]
        seq_len: int = sequence.shape[1]
        if sequence.dim() == 2:
            # input_ids shape: batch_size x seq_len
            pad_len: int = (self.__window_size - (seq_len % self.__window_size)) % self.__window_size
            pad = torch.ones((batch_size, pad_len), dtype=sequence.dtype, device=sequence.device) * self.__get_padding_token()
            padded_sequence = torch.cat((sequence, pad), 1)  # input_ids shape: batch_size x seq_len + pad
        elif sequence.dim() == 3:
            # inputs_embeds shape: batch_size x seq_len x embedding size
            pad_len: int = (self.__window_size - (seq_len % self.__window_size)) % self.__window_size
            pad = torch.ones((batch_size, pad_len, sequence.shape[2]), dtype=sequence.dtype, device=sequence.device) * self.__get_padding_embed()
            padded_sequence = torch.cat((sequence, pad), 1)  # input_ids shape: batch_size x seq_len + pad
        else:
            raise NotImplementedError(f"Unsupported input dimension: {sequence.dim()} with shape {sequence.shape}")
        return padded_sequence
    def agg_sequence(
        self,
        sequence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aggregate a sequence into fixed-size windows.

        Args:
            sequence (Optional[torch.Tensor]): The sequence to aggregate. For 2D sequence (batch_size, seq_len), pad with padding_token.
                For 3D sequence (batch_size, seq_len, embedding_size), pad with padding_embed.

        Returns:
            torch.Tensor: The aggregated sequence. For 2D sequence (batch_size, seq_len), it would return (batch_size * self.__segment_num, self.__window_size).
                For 3D sequence (batch_size * self.__segment_num, self.__window_size, embedding_size), where self.__segment_num = (seq_len + pad) // self.__window_size
        """
        if sequence is None:
            return sequence
        self.__batch_size: int = sequence.shape[0]
        self.__seq_len: int = sequence.shape[1]
        padded_sequence = self.__pad(sequence=sequence)
        self.__segment_num: int = padded_sequence.shape[1] // self.__window_size
        return padded_sequence.view(self.__batch_size * self.__segment_num, self.__window_size, *padded_sequence.shape[2:])
    def split_sequence(
        self,
        sequence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Split a sequence into fixed-size windows.

        Args:
            sequence (Optional[torch.Tensor]): The sequence to split, with size (batch_size * self.__segment_num, self.__window_size or 1, embedding_size)

        Returns:
            torch.Tensor: The split sequence into shape (batch_size, self.__segment_num * (self.__window_size or 1), embedding_size)
        """
        if sequence is None:
            return sequence
        return sequence.view(self.__batch_size, self.__segment_num * sequence.shape[1], *sequence.shape[2:])
    
@auto_docstring(custom_intro="Base class for language decoding with fixed-size window disaggregation.")
class LanguageDecoderBase:
    def __init__(
        self,
        window_size: int,
    ):
        self.__batch_size: int = None
        self.__seq_len: int = None
        self.__segment_num: int = None
        
        self.__window_size: int = window_size  
        
    @property
    def batch_size(self):
        return self.__batch_size

    @property
    def seq_len(self):
        return self.__seq_len
    
    @property
    def segment_num(self):
        return self.__segment_num
    
    @property
    def window_size(self):
        return self.__window_size
    
    def agg_sequence(
        self,
        sequence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aggregate a sequence into single element.

        Args:
            sequence (Optional[torch.Tensor]): The sequence to aggregate. For 2D sequence (batch_size, seq_len), pad with padding_token.
                For 3D sequence (batch_size, seq_len, embedding_size), pad with padding_embed.

        Returns:
            torch.Tensor: The aggregated sequence. For 2D sequence (batch_size, seq_len), it would return (batch_size * self.__segment_num, 1).
                For 3D sequence (batch_size * self.__segment_num, 1, embedding_size), where self.__segment_num = seq_len
        """
        if sequence is None:
            return sequence
        self.__batch_size: int = sequence.shape[0]
        self.__seq_len: int = sequence.shape[1]
        self.__segment_num: int = sequence.shape[1]
        return sequence.view(self.__batch_size * self.__segment_num, 1, *sequence.shape[2:])
    def split_sequence(
        self,
        sequence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Split a sequence back to original.

        Args:
            sequence (Optional[torch.Tensor]): The sequence to split, with size (batch_size * self.__segment_num, 1, embedding_size), where self.__segment_num = seq_len

        Returns:
            torch.Tensor: The split sequence into shape (batch_size, self.__segment_num, embedding_size), where self.__segment_num = seq_len
        """
        if sequence is None:
            return sequence
        return sequence.view(self.__batch_size, self.__segment_num, *sequence.shape[2:])
    def flatten_multi_heads_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Project decoder hidden states to logits using multi-head mechanism.
        
        Args:
            logits: list of tensors of shape (batch_size, self.__segment_num, vocab_size) with length self.__window_size
        
        Returns:
            flattened_logits: Shape (batch_size, self.__segment_num * self.__window_size, vocab_size)
        """
        flatten_seq_shape = (self.__batch_size, self.__segment_num * self.__window_size, *logits[0].shape[2:])
        # Stack to create tensor with shape: (batch_size, self.__segment_num,  self.__window_size, vocab_size)
        return torch.stack(logits, dim=2).view(flatten_seq_shape)

@dataclass
class PreprocessOutput:
    input_ids: torch.LongTensor
    inputs_embeds: torch.FloatTensor

@auto_docstring(
    custom_intro="Language encoder model based on GPT-2 for encoding text into latent representations.",
    custom_args="window_size (`int`): The window size for aggregating input sequences into segments.",
)
class LanguageEncoder(GPT2ModelBase):
    def __init__(self, config: LatentGPT2Config):
        super().__init__(config)
        self.ae_base = LanguageEncoderBase(
            window_size = config.window_size,
            padding_token = config.pad_token_id,
            # padding_token = self.config.eos_token_id,
            padding_embed = None,
            wte=self.wte,
        )
        # Initialize weights and apply final processing
        self.post_init()
    def __pre_process_inputs(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> PreprocessOutput:
        """
        Pre-processes and segments the inputs for the language encoder.

        Args:
            input_ids (Optional[torch.LongTensor]): The input ids.
            inputs_embeds (Optional[torch.FloatTensor]): The input embeddings.

        Returns:
            A tuple containing the pre-processed input ids and embeddings.
        """
        if input_ids is not None:
            input_ids = self.ae_base.agg_sequence(sequence=input_ids)
        if inputs_embeds is not None:
            inputs_embeds = self.ae_base.agg_sequence(sequence=inputs_embeds)
        return PreprocessOutput(input_ids=input_ids, inputs_embeds=inputs_embeds)
    def __post_process_outputs(
        self,
        outputs: BaseModelOutputWithPastAndCrossAttentions,
    ) -> BaseAutoencoderOutputWithPastAndCrossAttentions:
        """
        Post-processes and segment the outputs of the language encoder.

        Args:
            outputs: The output of the language encoder.

        Returns:
            The processed output of the language encoder.
        """
        if outputs is None:
            return outputs
        # Only keep the last hidden_state of hidden_state of the last layer
        logits_to_keep: int = 1
        slice_indices = slice(-logits_to_keep, None, None)
        return BaseAutoencoderOutputWithPastAndCrossAttentions(
            last_tail_hidden_state=self.ae_base.split_sequence(sequence=outputs.last_hidden_state[:, slice_indices, :]),
            last_hidden_state=self.ae_base.split_sequence(sequence=outputs.last_hidden_state),
            past_key_values=outputs.past_key_values,
            hidden_states=tuple(self.ae_base.split_sequence(sequence=hidden_state) for hidden_state in outputs.hidden_states) if outputs.hidden_states is not None else None,
            # TODO: Handle attentions and cross_attentions reshaping
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def init_weight_from_pretrained(self, pretrained_model: GPT2ModelBase):
        """
        Initializes the language encoder from a pre-trained model.

        Args:
            pretrained_model: The pre-trained model to use.

        Returns:
            A new instance of the class.
        """
        self.wte = copy.deepcopy(pretrained_model.wte)
        self.wpe = copy.deepcopy(pretrained_model.wpe)
        self.drop = copy.deepcopy(pretrained_model.drop)
        self.h = copy.deepcopy(pretrained_model.h[:self.config.num_hidden_layers_encoder])
        self.ln_f = copy.deepcopy(pretrained_model.ln_f)
        self.gradient_checkpointing = pretrained_model.gradient_checkpointing
        return self
    
    # @classmethod
    # def build_from_pretrained(
    #     cls,
    #     pretrained_model: GPT2ModelBase,
    #     window_size: int,
    #     num_hidden_layers: int,
    # ):
    #     """
    #     Builds a new instance of the LanguageEncoder from a pre-trained model.

    #     Args:
    #         cls: The class to instantiate.
    #         pretrained: The pre-trained model to use.
    #         window_size: The window size for the language encoder.

    #     Returns:
    #         A new instance of the class.
    #     """
    #     config: LatentGPT2Config = copy.deepcopy(pretrained_model.config)
    #     config.num_hidden_layers = num_hidden_layers
    #     encoder: LanguageEncoder = cls(config=config, window_size=window_size)

    #     encoder.wte = copy.deepcopy(pretrained_model.wte)
    #     encoder.wpe = copy.deepcopy(pretrained_model.wpe)
    #     encoder.drop = copy.deepcopy(pretrained_model.drop)
    #     encoder.h = copy.deepcopy(pretrained_model.h[:num_hidden_layers])
    #     encoder.ln_f = copy.deepcopy(pretrained_model.ln_f)
    #     encoder.gradient_checkpointing = pretrained_model.gradient_checkpointing

    #     return encoder
        
    @auto_docstring(
        custom_args="return_segment (`bool`, *optional*, defaults to `True`): Whether to return segmented outputs.",
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_segment: bool = True,
        **kwargs,
    ) -> Union[tuple, BaseAutoencoderOutputWithPastAndCrossAttentions]:
        pre_process_res: PreprocessOutput = self.__pre_process_inputs(input_ids=input_ids, inputs_embeds=inputs_embeds)
        output = super().forward(
            input_ids=pre_process_res.input_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=pre_process_res.inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        return self.__post_process_outputs(outputs=output)
        
class LanguageEncoderLMHead(GPT2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}

    def __init__(self, config: LatentGPT2Config):
        super().__init__(config)
        self.transformer = LanguageEncoder(config=config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        
    def init_weight_from_pretrained(self, pretrained_model: GPT2LMHeadModel):
        """
        Initializes the language encoder from a pre-trained model.

        Args:
            pretrained_model: The pre-trained model to use.

        Returns:
            A new instance of the class.
        """
        self.transformer.init_weight_from_pretrained(pretrained_model=pretrained_model.transformer)
        self.lm_head = copy.deepcopy(pretrained_model.lm_head)
        return self
        
    # @classmethod
    # def build_from_pretrained(
    #     cls,
    #     pretrained_model: GPT2LMHeadModel,
    #     window_size: int,
    #     num_hidden_layers: int,
    # ):
    #     """
    #     Builds a new instance of the LanguageDecoder from a pre-trained model.

    #     Args:
    #         cls: The class to instantiate.
    #         pretrained: The pre-trained model to use.
    #         window_size: The window size for the language decoder.

    #     Returns:
    #         A new instance of the class.
    #     """
    #     config: LatentGPT2Config = copy.deepcopy(pretrained_model.config)
    #     config.num_hidden_layers = num_hidden_layers
    #     encoder: LanguageEncoder = LanguageEncoder.build_from_pretrained(
    #         pretrained_model=pretrained_model.transformer,
    #         window_size=window_size,
    #         num_hidden_layers=num_hidden_layers)

    #     encoder_lm_head = cls(config=config, window_size=window_size)
    #     encoder_lm_head.transformer = encoder
    #     encoder_lm_head.lm_head = copy.deepcopy(pretrained_model.lm_head)

    #     return encoder_lm_head
        
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[tuple, CausalLMAutoencoderOutputWithCrossAttentions]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        labels (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        # Only use the last hidden_state of the last layer as latent
        latents = transformer_outputs.last_tail_hidden_state

        # Project the latents to logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(latents[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Flatten the tokens
            loss = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMAutoencoderOutputWithCrossAttentions(
            last_tail_hidden_state=transformer_outputs.last_tail_hidden_state,
            last_hidden_state=transformer_outputs.last_hidden_state,
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
            latents=transformer_outputs.last_tail_hidden_state,
        )


class LatentAR(GPT2ModelBase):
    def __init__(self, config: LatentGPT2Config):
        super().__init__(config)
        self.ae_base = LanguageEncoderBase(
            window_size = config.window_size,
            padding_token = config.pad_token_id,
            # padding_token = self.config.eos_token_id,
            padding_embed = None,
            wte=self.wte,
        )
        # Initialize weights and apply final processing
        self.post_init()
    def __pre_process_inputs(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> PreprocessOutput:
        """
        Pre-processes and segments the inputs for the language encoder.

        Args:
            input_ids (Optional[torch.LongTensor]): The input ids.
            inputs_embeds (Optional[torch.FloatTensor]): The input embeddings.

        Returns:
            A tuple containing the pre-processed input ids and embeddings.
        """
        # if input_ids is not None:
        #     input_ids = self.ae_base.agg_sequence(sequence=input_ids)
        # if inputs_embeds is not None:
        #     inputs_embeds = self.ae_base.agg_sequence(sequence=inputs_embeds)
        # return PreprocessOutput(input_ids=input_ids, inputs_embeds=inputs_embeds)

        # No need to aggregated again
        return PreprocessOutput(input_ids=input_ids, inputs_embeds=inputs_embeds)


    def __post_process_outputs(
        self,
        outputs: BaseModelOutputWithPastAndCrossAttentions,
    ) -> BaseAutoencoderOutputWithPastAndCrossAttentions:
        """
        Post-processes and segment the outputs of the language encoder.

        Args:
            outputs: The output of the language encoder.

        Returns:
            The processed output of the language encoder.
        """
        if outputs is None:
            return outputs
        return BaseAutoencoderOutputWithPastAndCrossAttentions(
            last_tail_hidden_state=outputs.last_hidden_state),
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=tuple(hidden_state for hidden_state in outputs.hidden_states) if outputs.hidden_states is not None else None,
            # TODO: Handle attentions and cross_attentions reshaping
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def init_weight_from_pretrained(self, pretrained_model: GPT2ModelBase):
        """
        Initializes the language encoder from a pre-trained model.

        Args:
            pretrained_model: The pre-trained model to use.

        Returns:
            A new instance of the class.
        """
        self.wte = copy.deepcopy(pretrained_model.wte)
        self.wpe = copy.deepcopy(pretrained_model.wpe)
        self.drop = copy.deepcopy(pretrained_model.drop)
        self.h = copy.deepcopy(pretrained_model.h[self.config.num_hidden_layers_encoder:-self.config.num_hidden_layers_encoder])
        self.ln_f = copy.deepcopy(pretrained_model.ln_f)
        self.gradient_checkpointing = pretrained_model.gradient_checkpointing
        return self
    
    # @classmethod
    # def build_from_pretrained(
    #     cls,
    #     pretrained_model: GPT2ModelBase,
    #     window_size: int,
    #     num_hidden_layers: int,
    # ):
    #     """
    #     Builds a new instance of the LanguageEncoder from a pre-trained model.

    #     Args:
    #         cls: The class to instantiate.
    #         pretrained: The pre-trained model to use.
    #         window_size: The window size for the language encoder.

    #     Returns:
    #         A new instance of the class.
    #     """
    #     config: LatentGPT2Config = copy.deepcopy(pretrained_model.config)
    #     config.num_hidden_layers = num_hidden_layers
    #     encoder: LanguageEncoder = cls(config=config, window_size=window_size)

    #     encoder.wte = copy.deepcopy(pretrained_model.wte)
    #     encoder.wpe = copy.deepcopy(pretrained_model.wpe)
    #     encoder.drop = copy.deepcopy(pretrained_model.drop)
    #     encoder.h = copy.deepcopy(pretrained_model.h[:num_hidden_layers])
    #     encoder.ln_f = copy.deepcopy(pretrained_model.ln_f)
    #     encoder.gradient_checkpointing = pretrained_model.gradient_checkpointing

    #     return encoder
        
    @auto_docstring(
        custom_args="return_segment (`bool`, *optional*, defaults to `True`): Whether to return segmented outputs.",
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_segment: bool = True,
        **kwargs,
    ) -> Union[tuple, BaseAutoencoderOutputWithPastAndCrossAttentions]:
        pre_process_res: PreprocessOutput = self.__pre_process_inputs(input_ids=input_ids, inputs_embeds=inputs_embeds)
        output = super().forward(
            input_ids=None,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=pre_process_res.inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        return self.__post_process_outputs(outputs=output)


@auto_docstring(
    custom_intro="Language decoder model based on GPT-2 for decoding latent representations back to text.",
    custom_args="window_size (`int`): The window size for disaggregating latent representations back to sequences.",
)
class LanguageDecoder(GPT2ModelBase):
    def __init__(self, config: LatentGPT2Config):
        super().__init__(config)
        self.ae_base = LanguageDecoderBase(
            window_size = config.window_size,
        )
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        
    def __pre_process_inputs(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> PreprocessOutput:
        """
        Pre-processes and segments the inputs for the language decoder.

        Args:
            input_ids (Optional[torch.LongTensor]): The input ids.
            inputs_embeds (Optional[torch.FloatTensor]): The input embeddings.

        Returns:
            A PreprocessOutput containing the pre-processed input ids and embeddings.
        """
        if input_ids is not None:
            input_ids = self.ae_base.agg_sequence(sequence=input_ids)
        if inputs_embeds is not None:
            inputs_embeds = self.ae_base.agg_sequence(sequence=inputs_embeds)
        return PreprocessOutput(input_ids=input_ids, inputs_embeds=inputs_embeds)
    
    def __post_process_outputs(
        self,
        outputs: CausalLMAutoencoderOutputWithCrossAttentions,
    ):
        """
        Post-processes and segment the outputs of the language decoder.

        Args:
            outputs: The output of the language decoder.

        Returns:
            The processed output of the language decoder.
        """
        if outputs is None:
            return outputs
        # The input last_hidden_state have already the last hidden_states of the last layer, no need to slice
        return BaseAutoencoderOutputWithPastAndCrossAttentions(
            last_tail_hidden_state=self.ae_base.split_sequence(sequence=outputs.last_hidden_state),
            last_hidden_state=self.ae_base.split_sequence(sequence=outputs.last_hidden_state),
            past_key_values=outputs.past_key_values,
            hidden_states=tuple(self.ae_base.split_sequence(sequence=hidden_state) for hidden_state in outputs.hidden_states) if outputs.hidden_states is not None else None,
            # TODO: Handle attentions and cross_attentions reshaping
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def init_weight_from_pretrained(self, pretrained_model: GPT2ModelBase):
        """
        Initializes the language decoder from a pre-trained model.
        
        Args:
            pretrained_model: The pre-trained model to use.
        
        Returns:
            A new instance of the class.
        """
        self.wte = copy.deepcopy(pretrained_model.wte)
        self.wpe = copy.deepcopy(pretrained_model.wpe)
        self.drop = copy.deepcopy(pretrained_model.drop)
        self.h = copy.deepcopy(pretrained_model.h[-self.config.num_hidden_layers_decoder:])
        self.ln_f = copy.deepcopy(pretrained_model.ln_f)
        self.gradient_checkpointing = pretrained_model.gradient_checkpointing

        # Re-index blocks to match the new layer positions (0 to num_hidden_layers_decoder-1)
        for new_idx, block in enumerate(self.h):
            block.attn.layer_idx = new_idx
            if hasattr(block, 'crossattention'):
                block.crossattention.layer_idx = new_idx
        return self
        
    # @classmethod
    # def build_from_pretrained(
    #     cls,
    #     pretrained_model: GPT2ModelBase,
    #     window_size: int,
    #     num_hidden_layers: int,
    # ):
    #     """
    #     Builds a new instance of the LanguageDecoder from a pre-trained model.

    #     Args:
    #         cls: The class to instantiate.
    #         pretrained: The pre-trained model to use.
    #         window_size: The window size for the language decoder.

    #     Returns:
    #         A new instance of the class.
    #     """
    #     config: LatentGPT2Config = copy.deepcopy(pretrained_model.config)
    #     config.num_hidden_layers = num_hidden_layers
    #     decoder: LanguageDecoder = cls(config=config, window_size=window_size)

    #     decoder.wte = copy.deepcopy(pretrained_model.wte)
    #     decoder.wpe = copy.deepcopy(pretrained_model.wpe)
    #     decoder.drop = copy.deepcopy(pretrained_model.drop)
    #     decoder.h = copy.deepcopy(pretrained_model.h[num_hidden_layers:])
    #     decoder.ln_f = copy.deepcopy(pretrained_model.ln_f)
    #     decoder.gradient_checkpointing = pretrained_model.gradient_checkpointing

    #     # Re-index blocks to match the new layer positions (0 to num_hidden_layers-1)
    #     for new_idx, block in enumerate(decoder.h):
    #         block.attn.layer_idx = new_idx
    #         if hasattr(block, 'crossattention'):
    #             block.crossattention.layer_idx = new_idx

    #     return decoder
        
    # @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_segment: bool = True,
        **kwargs,
    ) -> Union[tuple, CausalLMAutoencoderOutputWithCrossAttentions]:
        pre_process_res: PreprocessOutput = self.__pre_process_inputs(input_ids=input_ids, inputs_embeds=inputs_embeds)
        output = super().forward(
            input_ids=pre_process_res.input_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=pre_process_res.inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        return self.__post_process_outputs(outputs=output)

class LanguageDecoderLMHead(GPT2PreTrainedModel, GenerationMixin):
    # _tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}

    def __init__(self, config: LatentGPT2Config):
        super().__init__(config)
        self.transformer = LanguageDecoder(config=config)
        
        # Single head projection mechanism
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Multi-head projection mechanism using window_size heads
        self.multi_lm_head = nn.ModuleList([
            nn.Linear(config.n_embd, config.vocab_size, bias=False) 
            for _ in range(config.window_size)
        ])

        # Initialize weights and apply final processing
        self.post_init()
        
    def init_weight_from_pretrained(self, pretrained_model: GPT2LMHeadModel):
        """
        Initializes the language encoder from a pre-trained model.

        Args:
            pretrained_model: The pre-trained model to use.

        Returns:
            A new instance of the class.
        """
        self.transformer.init_weight_from_pretrained(pretrained_model=pretrained_model.transformer)
        for i in range(len(self.multi_lm_head)):
            self.multi_lm_head[i] = copy.deepcopy(pretrained_model.lm_head)
        return self

    # @classmethod
    # def build_from_pretrained(
    #     cls,
    #     pretrained_model: GPT2LMHeadModel,
    #     window_size: int,
    #     num_hidden_layers: int,
    # ):
    #     """
    #     Builds a new instance of the LanguageDecoder from a pre-trained model.

    #     Args:
    #         cls: The class to instantiate.
    #         pretrained: The pre-trained model to use.
    #         window_size: The window size for the language decoder.

    #     Returns:
    #         A new instance of the class.
    #     """
    #     config: LatentGPT2Config = copy.deepcopy(pretrained_model.config)
    #     config.num_hidden_layers = num_hidden_layers
    #     decoder: LanguageDecoder = LanguageDecoder.build_from_pretrained(
    #         pretrained_model=pretrained_model.transformer,
    #         window_size=window_size,
    #         num_hidden_layers=num_hidden_layers)

    #     decoder_lm_head = cls(config=config, window_size=window_size)
    #     decoder_lm_head.transformer = decoder
    #     for i in range(len(decoder_lm_head.multi_lm_head)):
    #         decoder_lm_head.multi_lm_head[i] = copy.deepcopy(pretrained_model.lm_head)

    #     return decoder_lm_head
    
    def _project_with_multi_heads(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Project decoder hidden states to logits using multi-head mechanism.
        
        Args:
            latents: Shape (batch_size, sequence_length, hidden_size)
        
        Returns:
            logits: Shape (batch_size, sequence_length * config.window_size, vocab_size)
        """
        multi_head_logits: List[torch.Tensor] = []
        for i in range(self.config.window_size):
            multi_head_logits.append(self.multi_lm_head[i](latents))
        return self.transformer.ae_base.flatten_multi_heads_logits(logits=multi_head_logits)
        
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[tuple, CausalLMAutoencoderOutputWithCrossAttentions]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        labels (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        latents = transformer_outputs.last_tail_hidden_state

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # logits = self.lm_head(latents[:, slice_indices, :])
        logits = self._project_with_multi_heads(latents[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Flatten the tokens
            loss = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMAutoencoderOutputWithCrossAttentions(
            last_tail_hidden_state=transformer_outputs.last_tail_hidden_state,
            last_hidden_state=transformer_outputs.last_hidden_state,
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
            latents=transformer_outputs.last_tail_hidden_state,
        )

class LanguageAutoencoder(GPT2PreTrainedModel, GenerationMixin):
    def __init__(self, config: LatentGPT2Config):
        super().__init__(config)
        # self.window_size: int = window_size
        self.encoder: LanguageEncoderLMHead = LanguageEncoderLMHead(config=config)
        self.decoder: LanguageDecoderLMHead = LanguageDecoderLMHead(config=config)
        
        # Initialize weights and apply final processing
        self.post_init()

        self.ltar: LatentAR = LatentAR(config=config)

        self.fm: FlowMatchingModel = FlowMatchingModel(
            hidden_dim = config.n_embd,
            semantic_dim = config.n_embd,
            time_embed_dim = 256,
            num_layers = 4,
            num_heads = 8,
            dropout = 0.1
        )

    def init_weight_from_pretrained(self, pretrained_model: GPT2LMHeadModel):
        """
        Initializes the language encoder from a pre-trained model.

        Args:
            pretrained_model: The pre-trained model to use.

        Returns:
            A new instance of the class.
        """
        self.encoder.init_weight_from_pretrained(pretrained_model=pretrained_model)
        self.decoder.init_weight_from_pretrained(pretrained_model=pretrained_model)

        self.ltar.init_weight_from_pretrained(pretrained_model=pretrained_model)

        return self

    # @classmethod
    # def build_from_pretrained(
    #     cls,
    #     pretrained_model: GPT2LMHeadModel,
    #     window_size: int,
    #     num_hidden_layers_encoder: int,
    #     num_hidden_layers_decoder: int,
    # ):
    #     """
    #     Builds a new instance of the LanguageDecoder from a pre-trained model.

    #     Args:
    #         cls: The class to instantiate.
    #         pretrained: The pre-trained model to use.
    #         window_size: The window size for the language decoder.

    #     Returns:
    #         A new instance of the class.
    #     """
    #     config: LatentGPT2Config = copy.deepcopy(pretrained_model.config)
    #     config.num_hidden_layers_encoder = num_hidden_layers_encoder
    #     config.num_hidden_layers_decoder = num_hidden_layers_decoder

    #     autoencoder: LanguageAutoencoder = LanguageAutoencoder(config=config, window_size=window_size)
    #     autoencoder.encoder = LanguageEncoderLMHead.build_from_pretrained(
    #         pretrained_model=pretrained_model,
    #         window_size=window_size,
    #         num_hidden_layers=num_hidden_layers_encoder
    #     )
    #     autoencoder.decoder = LanguageDecoderLMHead.build_from_pretrained(
    #         pretrained_model=pretrained_model,
    #         window_size=window_size,
    #         num_hidden_layers=num_hidden_layers_decoder
    #     )
    #     return autoencoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
<<<<<<< HEAD
        use_latent_ar: bool = False,
=======
        use_latent_ar: Optional[bool] = None,
>>>>>>> f4f8d21e5f (Add AR and FM support)
        **kwargs,
    ) -> Union[tuple, CausalLMAutoencoderOutputWithCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_output = self.encoder(
            input_ids=input_ids,
            # past_key_values=past_key_values,
            # attention_mask=attention_mask,
            # cache_position=cache_position,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            # inputs_embeds=inputs_embeds,
            # encoder_hidden_states=encoder_hidden_states,
            # encoder_attention_mask=encoder_attention_mask,
            past_key_values=None,
            attention_mask=None,
            cache_position=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )


        if use_latent_ar:
            ltar_output = self.ltar(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                cache_position=cache_position,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=encoder_output.last_tail_hidden_state,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
            )

            # hidden_size = ltar_output.last_tail_hidden_state.size(-1)
            fm_output = self.fm.sample(ltar_output.last_tail_hidden_state, num_steps=100)
            # fm_output = fm_output.reshape(ltar_output.last_tail_hidden_state.size())


        # hidden_states = encoder_output[0]
        decoder_output = self.decoder(
            input_ids=None,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            # past_key_values=None,
            # attention_mask=None,
            # cache_position=None,
            # token_type_ids=None,
            # position_ids=None,
            inputs_embeds=fm_output if use_latent_ar else encoder_output.last_tail_hidden_state,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            # encoder_hidden_states=None,
            # encoder_attention_mask=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # logits = self.lm_head(decoder_output.hidden_states[:, slice_indices, :])
        logits = decoder_output.logits[:, slice_indices, :]

        loss = None
        if labels is not None:
            # Flatten the tokens
            loss = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        # When use_latent_ar=True, return tuple of (ar_outputs, ltar_output, encoder_output)
        # for the trainer to compute AR loss and flow matching loss
        if use_latent_ar:
            ar_outputs = CausalLMAutoencoderOutputWithCrossAttentions(
                last_tail_hidden_state=decoder_output.last_tail_hidden_state,
                last_hidden_state=decoder_output.last_hidden_state,
                loss=loss,
                logits=logits,
                past_key_values=decoder_output.past_key_values,
                hidden_states=decoder_output.hidden_states,
                attentions=decoder_output.attentions,
                cross_attentions=decoder_output.cross_attentions,
            )
            # ltar_output is the latent AR prediction (placeholder: using encoder_output for now)
            # encoder_output contains the actual encoder latents
            return (ar_outputs, encoder_output, encoder_output)

        if not return_dict:
            output = (logits,) + decoder_output[1:]
            if use_latent_ar:
                output = output + (ltar_output, encoder_output)
            return ((loss,) + output) if loss is not None else output

        if return_dict:

            res = CausalLMAutoencoderOutputWithCrossAttentions(
                last_tail_hidden_state=decoder_output.last_tail_hidden_state,
                last_hidden_state=decoder_output.last_hidden_state,
                loss=loss,
                logits=logits,
                past_key_values=decoder_output.past_key_values,
                hidden_states=decoder_output.hidden_states,
                attentions=decoder_output.attentions,
                cross_attentions=decoder_output.cross_attentions,
            )
            if use_latent_ar:
                return (res, ltar_output, encoder_output)
            return res

# @auto_docstring(
#     custom_intro="""
#         The GPT2 Model transformer with a language modeling and a multiple-choice classification head on top e.g. for
#     RocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the
#     input embeddings, the classification head takes as input the input of a specified classification token index in the
#     input sequence).
#     """
# )
# class GPT2DoubleHeadsModel(GPT2PreTrainedModel, GenerationMixin):
#     _tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}

#     def __init__(self, config):
#         super().__init__(config)
#         config.num_labels = 1
#         self.transformer = GPT2ModelBase(config)
#         self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
#         self.multiple_choice_head = GPT2SequenceSummary(config)

#         # Initialize weights and apply final processing
#         self.post_init()

#     @auto_docstring
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Cache] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         token_type_ids: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         mc_token_ids: Optional[torch.LongTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         mc_labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         **kwargs,
#     ) -> Union[tuple, GPT2DoubleHeadsModelOutput]:
#         r"""
#         input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
#             `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
#             `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
#             sequence tokens in the vocabulary.

#             If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
#             `input_ids`.

#             Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
#             [`PreTrainedTokenizer.__call__`] for details.

#             [What are input IDs?](../glossary#input-ids)
#         mc_token_ids (`torch.LongTensor` of shape `(batch_size, num_choices)`, *optional*, default to index of the last token of the input):
#             Index of the classification token in each input sequence. Selected in the range `[0, input_ids.size(-1) -
#             1]`.
#         labels (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
#             Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
#             `labels = input_ids`. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`. All labels set to
#             `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size - 1]`
#         mc_labels (`torch.LongTensor` of shape `(batch_size)`, *optional*):
#             Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
#             where *num_choices* is the size of the second dimension of the input tensors. (see *input_ids* above)

#         Example:

#         ```python
#         >>> import torch
#         >>> from transformers import AutoTokenizer, GPT2DoubleHeadsModel

#         >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
#         >>> model = GPT2DoubleHeadsModel.from_pretrained("openai-community/gpt2")

#         >>> # Add a [CLS] to the vocabulary (we should train it also!)
#         >>> num_added_tokens = tokenizer.add_special_tokens({"cls_token": "[CLS]"})
#         >>> # Update the model embeddings with the new vocabulary size
#         >>> embedding_layer = model.resize_token_embeddings(len(tokenizer))

#         >>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
#         >>> encoded_choices = [tokenizer.encode(s) for s in choices]
#         >>> cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

#         >>> input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
#         >>> mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

#         >>> outputs = model(input_ids, mc_token_ids=mc_token_ids)
#         >>> lm_logits = outputs.logits
#         >>> mc_logits = outputs.mc_logits
#         ```"""
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         transformer_outputs = self.transformer(
#             input_ids,
#             past_key_values=past_key_values,
#             cache_position=cache_position,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         hidden_states = transformer_outputs[0]

#         lm_logits = self.lm_head(hidden_states)
#         mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

#         mc_loss = None
#         if mc_labels is not None:
#             loss_fct = CrossEntropyLoss()
#             mc_loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
#         lm_loss = None
#         if labels is not None:
#             labels = labels.to(lm_logits.device)
#             shift_logits = lm_logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             loss_fct = CrossEntropyLoss()
#             lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

#         if not return_dict:
#             output = (lm_logits, mc_logits) + transformer_outputs[1:]
#             if mc_loss is not None:
#                 output = (mc_loss,) + output
#             return ((lm_loss,) + output) if lm_loss is not None else output

#         return GPT2DoubleHeadsModelOutput(
#             loss=lm_loss,
#             mc_loss=mc_loss,
#             logits=lm_logits,
#             mc_logits=mc_logits,
#             past_key_values=transformer_outputs.past_key_values,
#             hidden_states=transformer_outputs.hidden_states,
#             attentions=transformer_outputs.attentions,
#         )


# @auto_docstring(
#     custom_intro="""
#     The GPT2 Model transformer with a sequence classification head on top (linear layer).

#     [`GPT2ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
#     (e.g. GPT-1) do.

#     Since it does classification on the last token, it requires to know the position of the last token. If a
#     `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
#     no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
#     padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
#     each row of the batch).
#     """
# )
# class GPT2ForSequenceClassification(GPT2PreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.transformer = GPT2ModelBase(config)
#         self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()

#     @auto_docstring
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Cache] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         token_type_ids: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         **kwargs,
#     ) -> Union[tuple, SequenceClassifierOutputWithPast]:
#         r"""
#         input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
#             `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
#             `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
#             sequence tokens in the vocabulary.

#             If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
#             `input_ids`.

#             Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
#             [`PreTrainedTokenizer.__call__`] for details.

#             [What are input IDs?](../glossary#input-ids)
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         transformer_outputs = self.transformer(
#             input_ids,
#             past_key_values=past_key_values,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         hidden_states = transformer_outputs[0]
#         logits = self.score(hidden_states)

#         if input_ids is not None:
#             batch_size, sequence_length = input_ids.shape[:2]
#         else:
#             batch_size, sequence_length = inputs_embeds.shape[:2]

#         if self.config.pad_token_id is None and batch_size != 1:
#             raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
#         if self.config.pad_token_id is None:
#             last_non_pad_token = -1
#         elif input_ids is not None:
#             # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
#             non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
#             token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
#             last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
#         else:
#             last_non_pad_token = -1
#             logger.warning_once(
#                 f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
#                 "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
#             )

#         pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

#         loss = None
#         if labels is not None:
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"

#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss()
#                 if self.num_labels == 1:
#                     loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(pooled_logits, labels)
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(pooled_logits, labels)
#         if not return_dict:
#             output = (pooled_logits,) + transformer_outputs[1:]
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutputWithPast(
#             loss=loss,
#             logits=pooled_logits,
#             past_key_values=transformer_outputs.past_key_values,
#             hidden_states=transformer_outputs.hidden_states,
#             attentions=transformer_outputs.attentions,
#         )


# @auto_docstring
# class GPT2ForTokenClassification(GPT2PreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels

#         self.transformer = GPT2ModelBase(config)
#         if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
#             classifier_dropout = config.classifier_dropout
#         elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
#             classifier_dropout = config.hidden_dropout
#         else:
#             classifier_dropout = 0.1
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)

#         # Initialize weights and apply final processing
#         self.post_init()

#     @auto_docstring
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Cache] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         token_type_ids: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         **kwargs,
#     ) -> Union[tuple, TokenClassifierOutput]:
#         r"""
#         input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
#             `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
#             `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
#             sequence tokens in the vocabulary.

#             If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
#             `input_ids`.

#             Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
#             [`PreTrainedTokenizer.__call__`] for details.

#             [What are input IDs?](../glossary#input-ids)
#         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         transformer_outputs = self.transformer(
#             input_ids,
#             past_key_values=past_key_values,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         hidden_states = transformer_outputs[0]
#         hidden_states = self.dropout(hidden_states)
#         logits = self.classifier(hidden_states)

#         loss = None
#         if labels is not None:
#             labels = labels.to(logits.device)
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

#         if not return_dict:
#             output = (logits,) + transformer_outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return TokenClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=transformer_outputs.hidden_states,
#             attentions=transformer_outputs.attentions,
#         )


# @auto_docstring
# class GPT2ForQuestionAnswering(GPT2PreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.transformer = GPT2ModelBase(config)
#         self.qa_outputs = nn.Linear(config.hidden_size, 2)

#         # Initialize weights and apply final processing
#         self.post_init()

#     @auto_docstring
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         token_type_ids: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         start_positions: Optional[torch.LongTensor] = None,
#         end_positions: Optional[torch.LongTensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         **kwargs,
#     ) -> Union[tuple, QuestionAnsweringModelOutput]:
#         r"""
#         input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
#             `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
#             `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
#             sequence tokens in the vocabulary.

#             If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
#             `input_ids`.

#             Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
#             [`PreTrainedTokenizer.__call__`] for details.

#             [What are input IDs?](../glossary#input-ids)
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.transformer(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         sequence_output = outputs[0]

#         logits = self.qa_outputs(sequence_output)
#         start_logits, end_logits = logits.split(1, dim=-1)
#         start_logits = start_logits.squeeze(-1).contiguous()
#         end_logits = end_logits.squeeze(-1).contiguous()

#         total_loss = None
#         if start_positions is not None and end_positions is not None:
#             # If we are on multi-GPU, split add a dimension
#             if len(start_positions.size()) > 1:
#                 start_positions = start_positions.squeeze(-1).to(start_logits.device)
#             if len(end_positions.size()) > 1:
#                 end_positions = end_positions.squeeze(-1).to(end_logits.device)
#             # sometimes the start/end positions are outside our model inputs, we ignore these terms
#             ignored_index = start_logits.size(1)
#             start_positions = start_positions.clamp(0, ignored_index)
#             end_positions = end_positions.clamp(0, ignored_index)

#             loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
#             start_loss = loss_fct(start_logits, start_positions)
#             end_loss = loss_fct(end_logits, end_positions)
#             total_loss = (start_loss + end_loss) / 2

#         if not return_dict:
#             output = (start_logits, end_logits) + outputs[2:]
#             return ((total_loss,) + output) if total_loss is not None else output

#         return QuestionAnsweringModelOutput(
#             loss=total_loss,
#             start_logits=start_logits,
#             end_logits=end_logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


__all__ = [
    # "GPT2DoubleHeadsModel",
    # "GPT2ForQuestionAnswering",
    # "GPT2ForSequenceClassification",
    # "GPT2ForTokenClassification",
    "GPT2ModelBase",
    # "GPT2PreTrainedModel",
    "LanguageAutoencoder",
    "LanguageDecoderLMHead",
    "LanguageEncoderLMHead",
]
