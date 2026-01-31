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
import warnings
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    PreprocessOutput,
    BaseAutoencoderOutputWithPastAndCrossAttentions,
    CausalLMAutoencoderOutputWithCrossAttentions,
)
from ...utils import (
    auto_docstring,
    logging,
)
from .configuration_latent_gpt2 import LatentGPT2Config
from .modeling_gpt2 import GPT2ModelBase, GPT2PreTrainedModel, GPT2LMHeadModel, GPT2Block


logger = logging.get_logger(__name__)

@auto_docstring(
    custom_intro="Base encoder model for language encoding with configurable number of layers.",
)
class LanguageEncoderBase(GPT2ModelBase):
    def __init__(self, config: LatentGPT2Config):
        """
        Initializes the LanguageEncoderBase model.

        Args:
            config (`LatentGPT2Config`):
                The model configuration object.

        """
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers_encoder)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation

        # Initialize weights and apply final processing
        self.post_init()


@auto_docstring(
    custom_intro="Base decoder model for language decoding with configurable number of layers.",
)
class LanguageDecoderBase(GPT2ModelBase):
    def __init__(self, config: LatentGPT2Config):
        """
        Initializes the LanguageDecoderBase model.

        Args:
            config (`LatentGPT2Config`):
                The model configuration object.
        
        Initializes the weights of the model and applies final processing.
        If the latent dimension matches the hidden dimension, sets the weight of `wte_latent` to the identity matrix.
        """
        super().__init__(config)

        self.embed_dim = config.hidden_size

        # Project from latent_dim to hidden_size (instead of token embedding)
        self.wte_latent = nn.Linear(config.latent_dim, self.embed_dim, bias=False)
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers_decoder)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation

        # Initialize weights and apply final processing
        self.post_init()

        # Initialize wte_latent as identity matrix if dimensions match
        if config.latent_dim == self.embed_dim:
            with torch.no_grad():
                self.wte_latent.weight.copy_(torch.eye(self.embed_dim))

class SequenceWindowUtilsBase:
    """
    Base class for sequence windowing utilities with shared properties and padding methods.
    
    Key Concept - segment_num:
    ===========================
    segment_num represents the number of segments (windows) created after dividing a sequence 
    into fixed-size chunks. It is computed as:
    
    segment_num = ceil(seq_len / window_size)
    
    This means:
    - If seq_len = 100, window_size = 4 → segment_num = ceil(100/4) = 25
    - If seq_len = 99, window_size = 4 → segment_num = ceil(99/4) = 25 (with 1 padding)
    - After padding: padded_seq_len = segment_num * window_size
    
    The sequences are reshaped from:
    - Input: (batch_size, seq_len, ...)
    - After aggregation: (batch_size * segment_num, window_size, ...)
    - After splitting: (batch_size, segment_num * window_size, ...)
    """
    TOKEN_TYPE_PADDING: str = "padding"
    TOKEN_TYPE_MASKING: str = "masking"
    SUPPORTED_TOKEN_TYPES: List[str] = [
        TOKEN_TYPE_PADDING,
        TOKEN_TYPE_MASKING
    ]

    def __init__(
        self,
        window_size: int,
        padding_token: Optional[int] = None,
        padding_embed: Optional[torch.FloatTensor] = None,
        masking_token: Optional[int] = None,
        masking_embed: Optional[torch.FloatTensor] = None,
        wte: Optional[torch.nn.Embedding] = None,
    ):
        """
        Initialize SequenceWindowUtilsBase with the given parameters.

        Parameters:
            window_size (int): Window size for sequence division.
            padding_token (Optional[int]): Token to pad with.
            padding_embed (Optional[torch.FloatTensor]): Embedding to pad with.
            masking_token (Optional[int]): Token to mask with.
            masking_embed (Optional[torch.FloatTensor]): Embedding to mask with.
            wte (Optional[torch.nn.Embedding]): Embedding weights to use for padding.
        """
        
        self._batch_size: int = None
        self._seq_len: int = None
        # segment_num: Number of segments after dividing sequence into fixed-size windows
        # See class docstring for detailed explanation of computation and usage
        self._segment_num: int = None
        self._window_size: int = window_size
        self._padding_token = padding_token
        self._padding_embed = padding_embed
        self._masking_token = masking_token
        self._masking_embed = masking_embed
        self._wte: torch.nn.Embedding = wte

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def seq_len(self):
        return self._seq_len

    @property
    def segment_num(self):
        return self._segment_num

    @property
    def window_size(self):
        return self._window_size

    def _get_padding_embed(self) -> torch.FloatTensor:
        """
        Get the padding token embedding.

        Returns:
            `torch.FloatTensor` of shape `(1, hidden_size)`:
                Padding token embedding tensor.
        """
        if self._padding_embed is None:
            if self._wte is None:
                raise ValueError("To get padding_embed, either provide padding_embed or wte")
            self._padding_embed = self._wte(torch.tensor([self._padding_token]))
        return self._padding_embed

    def _get_padding_token(self) -> torch.LongTensor:
        """
        Get the padding token ID.

        Returns:
            `torch.LongTensor`:
                Padding token ID scalar.
        """
        if self._padding_token is None:
            raise ValueError("padding_token is not set")
        return self._padding_token
    
    def _get_masking_embed(self) -> torch.FloatTensor:
        """
        Get the masking token embedding.

        Returns:
            `torch.FloatTensor` of shape `(1, hidden_size)`:
                Masking token embedding tensor.
        """
        if self._masking_embed is None:
            if self._wte is None:
                raise ValueError("To get masking_embed, either provide masking_embed or wte")
            self._masking_embed = self._wte(torch.tensor([self._masking_token]))
        return self._masking_embed

    def _get_masking_token(self) -> torch.LongTensor:
        """
        Get the masking token ID.

        Returns:
            `torch.LongTensor`:
                Masking token ID scalar.
        """
        if self._masking_token is None:
            raise ValueError("masking_token is not set")
        return self._masking_token
    
    def _get_token(self, token_type: str) -> torch.LongTensor:
        """
        Get token ID based on token type.

        Args:
            token_type (`str`):
                Type of token to retrieve. Must be one of `"padding"` or `"masking"`.

        Returns:
            `torch.LongTensor`:
                Token ID scalar for the specified type.
        """
        if token_type == SequenceWindowUtilsBase.TOKEN_TYPE_PADDING:
            return self._get_padding_token()
        elif token_type == SequenceWindowUtilsBase.TOKEN_TYPE_MASKING:
            return self._get_masking_token()
        else:
            raise ValueError(f"token_type, {token_type}, isn't supported, only support {SequenceWindowUtilsBase.SUPPORTED_TOKEN_TYPES}")
        
    def _get_embed(self, token_type: str) -> torch.FloatTensor:
        """
        Get token embedding based on token type.

        Args:
            token_type (`str`):
                Type of token embedding to retrieve. Must be one of `"padding"` or `"masking"`.

        Returns:
            `torch.FloatTensor` of shape `(1, hidden_size)`:
                Token embedding tensor for the specified type.
        """
        if token_type == SequenceWindowUtilsBase.TOKEN_TYPE_PADDING:
            return self._get_padding_embed()
        elif token_type == SequenceWindowUtilsBase.TOKEN_TYPE_MASKING:
            return self._get_masking_embed()
        else:
            raise ValueError(f"token_type, {token_type}, isn't supported, only support {SequenceWindowUtilsBase.SUPPORTED_TOKEN_TYPES}")

    def _pad(self, token_type: str, sequence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pad the given sequence to be a multiple of window_size.

        The padded_seq_len is calculated as the smallest multiple of window_size that is 
        greater than or equal to the input seq_len. Specifically:
        `padded_seq_len = seq_len + pad_len` where 
        `pad_len = (window_size - (seq_len % window_size)) % window_size`

        Args:
            token_type (`str`):
                Type of padding to use. Should be one of `"padding"` or `"masking"`.
            sequence (`torch.Tensor` of shape `(batch_size, seq_len)` or `(batch_size, seq_len, hidden_size)`, *optional*):
                Input sequence to pad. For 2D tensors, pad with token IDs. For 3D tensors, pad with embeddings.

        Returns:
            `torch.Tensor`:
                Padded sequence with shape `(batch_size, padded_seq_len)` for 2D input or 
                `(batch_size, padded_seq_len, hidden_size)` for 3D input, where 
                `padded_seq_len` is the smallest multiple of `window_size` >= `seq_len`.
                Returns `None` if input sequence is `None`.
        """
        if sequence is None:
            return None
        if self._window_size == 0:
            return sequence

        batch_size: int = sequence.shape[0]
        seq_len: int = sequence.shape[1]
        pad_len: int = (self._window_size - (seq_len % self._window_size)) % self._window_size

        if pad_len == 0:
            return sequence

        if sequence.dim() == 2:
            pad = torch.full((batch_size, pad_len), self._get_token(token_type=token_type),
                           dtype=sequence.dtype, device=sequence.device)
        elif sequence.dim() == 3:
            pad = self._get_embed(token_type=token_type).reshape(1, 1, -1).expand(batch_size, pad_len, -1).to(
                dtype=sequence.dtype, device=sequence.device)
        else:
            raise NotImplementedError(f"Unsupported input dimension: {sequence.dim()} with shape {sequence.shape}")
        return torch.cat((sequence, pad), dim=1)

    def split_sequence(self, sequence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Split a windowed sequence back to batch dimensions.

        Args:
            sequence (`torch.Tensor` of shape `(batch_size * segment_num, window_size, ...)`, *optional*):
                Input sequence in windowed format to reshape back to batch dimensions.

        Returns:
            `torch.Tensor`:
                Reshaped sequence with shape `(batch_size, segment_num * window_size, ...)`.
                Returns `None` if input sequence is `None`.
        """
        if sequence is None:
            return None
        # Reshape from (batch_size * segment_num, window_size, ...) back to (batch_size, segment_num * window_size, ...)
        # segment_num was computed as ceil(original_seq_len / window_size) during aggregation
        return sequence.reshape(self._batch_size, self._segment_num * sequence.shape[1], *sequence.shape[2:])
    
    def split_context_target(
        self,
        sequence: torch.Tensor,
        is_padding: bool = True,
    ) -> Tuple[torch.Tensor]:
        """
        Split sequence into context and target parts.

        Args:
            sequence (`torch.Tensor` of shape `(batch_size, seq_len)` or `(batch_size, seq_len, hidden_size)`, *optional*):
                Input sequence to split. The last `window_size` elements become the target,
                and the rest becomes the context.
            is_padding (`bool`, *optional*, defaults to `True`):
                Whether to pad the context to be divisible by window_size.

        Returns:
            `Tuple[torch.Tensor]`:
                - context: Context part with shape `(batch_size, seq_len - window_size)` or 
                  `(batch_size, seq_len - window_size, hidden_size)`. If `is_padding=True`, 
                  may be padded to `(batch_size, padded_len)` or `(batch_size, padded_len, hidden_size)`.
                - target: Target part with shape `(batch_size, window_size)` or 
                  `(batch_size, window_size, hidden_size)`.
        """
        if sequence is None:
            return None
        if sequence.dim() == 2:
            context: torch.Tensor = sequence[..., :-self._window_size].contiguous()
            target: torch.Tensor = sequence[..., -self._window_size:].contiguous()
            if is_padding:
                context: torch.Tensor = self._pad(
                    sequence=context, token_type=SequenceWindowUtilsBase.TOKEN_TYPE_PADDING
                )
        elif sequence.dim() > 2:
            context: torch.Tensor = sequence[..., :-self._window_size, :].contiguous()
            target: torch.Tensor = sequence[..., -self._window_size:, :].contiguous()
            if is_padding:
                context: torch.Tensor = self._pad(
                    sequence=context, token_type=SequenceWindowUtilsBase.TOKEN_TYPE_PADDING
                )
        else:
            raise NotImplementedError(f"Unsupported input dimension: {sequence.dim()} with shape {sequence.shape}")
        return context, target
    
    def split_context_target_then_cat(
        self,
        sequence: torch.Tensor,
        is_padding: bool = True,
        return_context_target: bool = False,
    ) -> Tuple[torch.Tensor]:
        """
        Split sequence into context and target, then concatenate them back.

        This method splits the input sequence and then concatenates the parts back together,
        potentially with padding applied to the context part.

        Args:
            sequence (`torch.Tensor` of shape `(batch_size, seq_len)` or `(batch_size, seq_len, hidden_size)`, *optional*):
                Input sequence to process.
            is_padding (`bool`, *optional*, defaults to `True`):
                Whether to pad the context to be divisible by window_size.
            return_context_target (`bool`, *optional*, defaults to `False`):
                Whether to return the individual context and target parts.

        Returns:
            `torch.Tensor` or `Tuple[torch.Tensor]`:
                If `return_context_target=False`: 
                - joint_seq: Concatenated sequence with shape `(batch_size, context_len + window_size)` 
                  or `(batch_size, context_len + window_size, hidden_size)`.
                If `return_context_target=True`:
                - joint_seq: Concatenated sequence with same shape as above.
                - context: Context part with shape `(batch_size, context_len)` or 
                  `(batch_size, context_len, hidden_size)`.
                - target: Target part with shape `(batch_size, window_size)` or 
                  `(batch_size, window_size, hidden_size)`.
        """
        if sequence is None:
            if return_context_target:
                return None, None, None
            return None
        context, target = self.split_context_target(sequence=sequence, is_padding=is_padding)
        if sequence.dim() == 2:
            joint_seq = torch.cat((context, target), dim=-1)
        elif sequence.dim() > 2:
            joint_seq = torch.cat((context, target), dim=-2)
        else:
            raise NotImplementedError(f"Unsupported input dimension: {sequence.dim()} with shape {sequence.shape}")
        if return_context_target:
            return joint_seq, context, target
        return joint_seq

@auto_docstring(custom_intro="Utility class for language encoding with fixed-size window aggregation.")
class LanguageEncoderUtils(SequenceWindowUtilsBase):
    """Encoder utilities that aggregate sequences into fixed-size windows."""

    def agg_sequence(self, sequence: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """
        Aggregate a sequence into fixed-size windows.

        This method takes an input sequence and reshapes it into fixed-size windows by:
        1. Padding the sequence to make it divisible by window_size
        2. Computing segment_num = ceil(seq_len / window_size)
        3. Reshaping to (batch_size * segment_num, window_size, ...)

        Args:
            sequence (`torch.Tensor` of shape `(batch_size, seq_len)` or `(batch_size, seq_len, embed_dim)`, *optional*):
                The sequence to aggregate into windows. For 2D tensors, represents token IDs.
                For 3D tensors, represents token embeddings.

        Returns:
            `torch.Tensor` of shape `(batch_size * segment_num, window_size)` or `(batch_size * segment_num, window_size, embed_dim)`, *optional*:
                Aggregated sequence reshaped into windowed format where each window becomes a separate batch element.
                Returns `None` if input sequence is `None`.
        """
        if sequence is None:
            return None
        self._batch_size = sequence.shape[0]
        self._seq_len = sequence.shape[1]
        padded_sequence = self._pad(sequence=sequence, token_type=SequenceWindowUtilsBase.TOKEN_TYPE_PADDING)
        # segment_num = ceil(seq_len / window_size) after padding
        # Since padded_sequence is divisible by window_size, this gives the exact segment count
        self._segment_num = padded_sequence.shape[1] // self._window_size
        return padded_sequence.view(self._batch_size * self._segment_num, self._window_size, *padded_sequence.shape[2:])


@auto_docstring(custom_intro="Utility class for language decoding with fixed-size window disaggregation.")
class LanguageDecoderUtils(SequenceWindowUtilsBase):
    """Decoder utilities that handle single-element and window-based sequence operations."""

    def agg_sequence(self, sequence: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """
        Aggregate a sequence into single-element segments.

        This method reshapes each position in the sequence into a separate batch element
        for autoregressive processing. Each token position becomes its own segment of size 1.

        Args:
            sequence (`torch.Tensor` of shape `(batch_size, seq_len)` or `(batch_size, seq_len, ...)`, *optional*):
                The sequence to aggregate into single-element segments.

        Returns:
            `torch.Tensor` of shape `(batch_size * seq_len, 1)` or `(batch_size * seq_len, 1, ...)`, *optional*:
                Aggregated sequence where each original position becomes a separate batch element.
                segment_num = seq_len (each element is its own segment).
                Returns `None` if input sequence is `None`.
        """
        if sequence is None:
            return None
        self._batch_size = sequence.shape[0]
        self._seq_len = sequence.shape[1]
        # For single-element aggregation: segment_num = seq_len (each element is a segment)
        # This means each input position becomes its own segment of size 1
        self._segment_num = sequence.shape[1]
        return sequence.view(self._batch_size * self._segment_num, 1, *sequence.shape[2:])

    def _agg_sequence_by_window(
        self,
        sequence: Optional[torch.Tensor] = None,
        token_type: str = SequenceWindowUtilsBase.TOKEN_TYPE_PADDING,
    ) -> Optional[torch.Tensor]:
        """
        Internal method to aggregate a sequence into window_size-length segments.

        This private method pads the sequence and reshapes it into fixed-size windows,
        similar to LanguageEncoderUtils.agg_sequence but with configurable token types.

        Args:
            sequence (`torch.Tensor` of shape `(batch_size, seq_len)` or `(batch_size, seq_len, ...)`, *optional*):
                The sequence to aggregate into windows.
            token_type (`str`, *optional*, defaults to `"padding"`):
                Token type for padding. Must be one of `"padding"` or `"masking"`.

        Returns:
            `torch.Tensor` of shape `(batch_size * segment_num, window_size)` or `(batch_size * segment_num, window_size, ...)`, *optional*:
                Aggregated sequence reshaped into windowed format.
                segment_num = ceil(seq_len / window_size).
                Returns `None` if input sequence is `None`.
        """
        if sequence is None:
            return None
        self._batch_size = sequence.shape[0]
        self._seq_len = sequence.shape[1]
        padded_sequence = self._pad(sequence=sequence, token_type=token_type)
        # segment_num = ceil(seq_len / window_size) after padding
        # Since padded_sequence is divisible by window_size, this gives the exact segment count
        self._segment_num = padded_sequence.shape[1] // self._window_size
        return padded_sequence.view(self._batch_size * self._segment_num, self._window_size, *padded_sequence.shape[2:])

    def agg_sequence_by_window_size(self, sequence: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """
        Aggregate a sequence into window_size-length segments, padding if necessary.

        This method aggregates the input sequence into fixed-size windows using padding tokens.
        It's a public wrapper around _agg_sequence_by_window with padding token type.

        Args:
            sequence (`torch.Tensor` of shape `(batch_size, seq_len)` or `(batch_size, seq_len, ...)`, *optional*):
                The sequence to aggregate into windows.

        Returns:
            `torch.Tensor` of shape `(batch_size * segment_num, window_size)` or `(batch_size * segment_num, window_size, ...)`, *optional*:
                Aggregated sequence with padding tokens if needed to reach window_size boundaries.
                segment_num = ceil(seq_len / window_size).
                Returns `None` if input sequence is `None`.
        """
        return self._agg_sequence_by_window(sequence, token_type=SequenceWindowUtilsBase.TOKEN_TYPE_PADDING)

    def agg_sequence_mask_diffusion(self, sequence: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """
        Aggregate a masked sequence into window_size-length segments for diffusion decoding.

        Used in mask diffusion models where the decoder receives masked token IDs
        that need to be reshaped to match the windowed structure. Pads with mask
        tokens if sequence length is not divisible by window_size.

        Args:
            sequence (`torch.Tensor` of shape `(batch_size, seq_len)` or `(batch_size, seq_len, ...)`, *optional*):
                The masked token sequence to aggregate.

        Returns:
            `torch.Tensor` of shape `(batch_size * segment_num, window_size)` or `(batch_size * segment_num, window_size, ...)`, *optional*:
                Aggregated sequence with masking tokens used for padding if needed.
                segment_num = ceil(seq_len / window_size).
                Returns `None` if input sequence is `None`.
        """
        return self._agg_sequence_by_window(sequence, token_type=SequenceWindowUtilsBase.TOKEN_TYPE_MASKING)

    def flatten_multi_heads_logits(self, logits: List[torch.Tensor]) -> torch.Tensor:
        """
        Flatten multi-head logits into a single sequence.

        This method combines logits from multiple decoder heads (one per window position) into
        a single contiguous sequence. Used in multi-head language modeling where each position
        in a window has its own dedicated prediction head.

        Args:
            logits (`List[torch.Tensor]`):
                List of logits tensors from multiple heads, each with shape `(batch_size, segment_num, vocab_size)`.
                List length must equal `window_size`.

        Returns:
            `torch.Tensor` of shape `(batch_size, segment_num * window_size, vocab_size)`:
                Flattened logits where predictions from all heads are concatenated along the sequence dimension.
                The order follows: [head_0_predictions, head_1_predictions, ..., head_window_size_predictions].
        """
        # Flatten from multiple heads to single sequence
        # Final shape: (batch_size, segment_num * window_size, vocab_size)
        # where segment_num = ceil(original_seq_len / window_size)
        flatten_seq_shape = (self._batch_size, self._segment_num * self._window_size, *logits[0].shape[2:])
        return torch.stack(logits, dim=2).view(flatten_seq_shape)
    
    def pad(self, sequence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pad the given sequence with padding tokens to be a multiple of window_size.

        The padded_seq_len is calculated as the smallest multiple of window_size that is 
        greater than or equal to the input seq_len. Specifically:
        `padded_seq_len = seq_len + pad_len` where 
        `pad_len = (window_size - (seq_len % window_size)) % window_size`

        Args:
            sequence (`torch.Tensor` of shape `(batch_size, seq_len)` or `(batch_size, seq_len, hidden_size)`, *optional*):
                Input sequence to pad. For 2D tensors, pad with token IDs. For 3D tensors, pad with embeddings.

        Returns:
            `torch.Tensor`:
                Padded sequence with shape `(batch_size, padded_seq_len)` for 2D input or 
                `(batch_size, padded_seq_len, hidden_size)` for 3D input, where 
                `padded_seq_len` is the smallest multiple of `window_size` >= `seq_len`.
                Returns `None` if input sequence is `None`.
        """
        return super()._pad(sequence=sequence, token_type=SequenceWindowUtilsBase.TOKEN_TYPE_PADDING)

@auto_docstring(
    custom_intro="Language encoder model based on GPT-2 for encoding text into latent representations.",
    custom_args="window_size (`int`): The window size for aggregating input sequences into segments.",
)
class LanguageEncoder(LanguageEncoderBase):
    def __init__(self, config: LatentGPT2Config):
        super().__init__(config)
        self.ae_utils = LanguageEncoderUtils(
            window_size = config.window_size,
            padding_token = config.pad_token_id,
            padding_embed = None,
            wte=self.wte,
        )
        # Initialize weights and apply final processing
        self.post_init()
    def pre_process_inputs(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> PreprocessOutput:
        """
        Pre-processes and segments the inputs for the language encoder.

        Transforms input sequences by padding them to be divisible by window_size and
        reshaping them into windowed format for batch processing.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                The input token IDs to be aggregated into windows.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                The input embeddings to be aggregated into windows.

        Returns:
            `PreprocessOutput`:
                - input_ids: Aggregated input IDs with shape `(batch_size * segment_num, window_size)` or `None`.
                - inputs_embeds: Aggregated embeddings with shape `(batch_size * segment_num, window_size, hidden_size)` or `None`.
                
                Where `segment_num = ceil(seq_len / window_size)`.
        """
        if input_ids is not None:
            input_ids = self.ae_utils.agg_sequence(sequence=input_ids)
        if inputs_embeds is not None:
            inputs_embeds = self.ae_utils.agg_sequence(sequence=inputs_embeds)
        return PreprocessOutput(input_ids=input_ids, inputs_embeds=inputs_embeds)
    def post_process_outputs(
        self,
        outputs: BaseModelOutputWithPastAndCrossAttentions,
    ) -> BaseAutoencoderOutputWithPastAndCrossAttentions:
        """
        Post-processes and segment the outputs of the language encoder.

        Transforms windowed transformer outputs back to batch format and extracts
        key representations for autoencoder training.

        Args:
            outputs (`BaseModelOutputWithPastAndCrossAttentions`):
                The output from the transformer with:
                - last_hidden_state: shape `(batch_size * segment_num, window_size, hidden_size)`
                - hidden_states: tuple of `(batch_size * segment_num, window_size, hidden_size) * num_layers`

        Returns:
            `BaseAutoencoderOutputWithPastAndCrossAttentions`:
                - last_tail_hidden_state: Final position per segment, shape `(batch_size, segment_num, hidden_size)`.
                - last_window_hidden_state: Last window_size positions, shape `(batch_size, segment_num * window_size, hidden_size)`.
                - last_hidden_state: Full output reshaped to `(batch_size, segment_num * window_size, hidden_size)`.
                - hidden_states: All layer outputs reshaped to batch format with shape `(batch_size, segment_num * window_size, hidden_size) * num_hidden_layers_encoder` (if provided).
                - past_key_values, attentions, cross_attentions: Passed through unchanged.
        """
        if outputs is None:
            return outputs
        return BaseAutoencoderOutputWithPastAndCrossAttentions(
            # Only keep the last hidden_state of hidden_state of the last layer
            last_tail_hidden_state=self.ae_utils.split_sequence(sequence=outputs.last_hidden_state[:, -1:, ...]) if outputs.last_hidden_state is not None else None,
            # Only keep the last self.config.window_size hidden_state of hidden_state of the last layer
            last_window_hidden_state=self.ae_utils.split_sequence(sequence=outputs.last_hidden_state[:, -self.config.window_size:, ...]) if outputs.last_hidden_state is not None else None,
            last_hidden_state=self.ae_utils.split_sequence(sequence=outputs.last_hidden_state) if outputs.last_hidden_state is not None else None,
            past_key_values=outputs.past_key_values,
            hidden_states=tuple(self.ae_utils.split_sequence(sequence=hidden_state) for hidden_state in outputs.hidden_states) if outputs.hidden_states is not None else None,
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
        
        # Re-index blocks to match the new layer positions (0 to num_hidden_layers_decoder-1)
        for new_idx, block in enumerate(self.h):
            block.attn.layer_idx = new_idx
            if hasattr(block, 'crossattention'):
                block.crossattention.layer_idx = new_idx
        return self

    @auto_docstring(
        custom_intro="Forward pass for the language encoder. Processes input sequences through the encoder transformer, segmenting them into fixed-size windows and returning compressed representations.",
        custom_args=r"""
            input_ids (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Input token IDs. Sequence will be padded to be divisible by `window_size` if needed. After padding, `segment_num = ceil(seq_len / window_size)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                Pre-computed embeddings instead of `input_ids`.
        """
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
        **kwargs,
    ) -> Union[tuple, BaseAutoencoderOutputWithPastAndCrossAttentions]:
        r"""
         Forward pass for the language encoder.

        Processes input sequences through the encoder transformer, segmenting them into fixed-size
        windows and returning compressed representations.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                    Input token IDs. Sequence will be padded to be divisible by `window_size` if needed. After padding, `segment_num = ceil(seq_len / window_size)`.
                inputs_embeds (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                    Pre-computed embeddings instead of `input_ids`.
        Returns:
            [`BaseAutoencoderOutputWithPastAndCrossAttentions`] or `tuple`:
                - `last_tail_hidden_state`: Final position hidden state per segment, 
                  with shape `(batch_size, segment_num, hidden_size)`.
                - `last_window_hidden_state`: Hidden states for window positions, 
                  with shape `(batch_size, segment_num * window_size, hidden_size)`.
                - `last_hidden_state`: Full hidden states from final layer, 
                  with shape `(batch_size, segment_num * window_size, hidden_size)`.
                - `past_key_values`: Cached key-value states.
                - `hidden_states`: Hidden states from all layers (if requested), 
                  tuple of `(batch_size, segment_num * window_size, hidden_size) * num_hidden_layers_encoder`.
                - `attentions`: Attention weights (if requested).
                - `cross_attentions`: Cross-attention weights (if applicable).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pre_process_res: PreprocessOutput = self.pre_process_inputs(input_ids=input_ids, inputs_embeds=inputs_embeds)
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
        return self.post_process_outputs(outputs=output)
        
@auto_docstring(
    custom_intro="Language encoder model with latent projection head that combines a LanguageEncoder transformer with projection heads for both latent representations and vocabulary logits. This model processes input sequences by: 1) Segmenting sequences into fixed-size windows, 2) Processing through encoder transformer layers, 3) Projecting final hidden states to latent space via latent_head, 4) Optionally producing vocabulary logits via lm_head. Key features include window-based sequence processing for compression, dual output capabilities, and support for both autoencoder and language modeling objectives.",
)
class LanguageEncoderLatentHead(GPT2PreTrainedModel, GenerationMixin):
    # No tied weights since latent_head projects to latent_dim, not vocab_size

    def __init__(self, config: LatentGPT2Config):
        super().__init__(config)
        self.transformer = LanguageEncoder(config=config)
        # Project from hidden_state to latent_dim
        self.latent_head = nn.Linear(config.n_embd, config.latent_dim, bias=False)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Initialize latent_head as identity matrix if dimensions match
        if config.n_embd == config.latent_dim:
            with torch.no_grad():
                self.latent_head.weight.copy_(torch.eye(config.n_embd))
        
    def init_weight_from_pretrained(self, pretrained_model: GPT2LMHeadModel):
        """
        Initializes the language encoder from a pre-trained model.

        Args:
            pretrained_model: The pre-trained model to use.

        Returns:
            A new instance of the class.
        """
        self.transformer.init_weight_from_pretrained(pretrained_model=pretrained_model.transformer)
        # latent_head is not copied from pretrained model since it projects to latent_dim, not vocab_size
        # It will be initialized with random weights
        self.lm_head = copy.deepcopy(pretrained_model.lm_head)
        return self

    @auto_docstring(
        custom_intro="Forward pass for the language encoder with latent head. Processes input sequences through the encoder and produces both latent representations and vocabulary logits for language modeling.",
        custom_args=r"""
            input_ids (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Input token IDs. Sequence will be padded to be divisible by `window_size` if needed. After padding, `segment_num = ceil(seq_len / window_size)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                Pre-computed embeddings instead of `input_ids`.
            labels (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Labels for language modeling. Note that the labels **are shifted** inside the model.
            logits_to_keep (`int` or `torch.Tensor`, *optional*, defaults to 0):
                Number of logits to keep from the end of the sequence for memory efficiency.
        """
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
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[tuple, CausalLMAutoencoderOutputWithCrossAttentions]:
        r"""
        Forward pass for the language encoder with latent head.

        Processes input sequences through the encoder and produces both latent representations
        and vocabulary logits for language modeling.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Input token IDs. Sequence will be padded to be divisible by `window_size` if needed.
                After padding, `segment_num = ceil(seq_len / window_size)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                Pre-computed embeddings instead of `input_ids`.
            labels (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Labels for language modeling. Note that the labels **are shifted** inside the model.

        Returns:
            [`CausalLMAutoencoderOutputWithCrossAttentions`] or `tuple`:
                - `logits`: Latent projection logits with shape `(batch_size, logits_to_keep, latent_dim)` 
                  if `logits_to_keep > 0`, otherwise shape `(batch_size, segment_num, latent_dim)` (Note: these are latent projections, not vocab logits)
                - `last_tail_hidden_state`: Final position hidden state per segment, 
                  with shape `(batch_size, segment_num, hidden_size)` (if `output_hidden_states=True`).
                - `last_window_hidden_state`: Hidden states for window positions, 
                  with shape `(batch_size, segment_num * window_size, hidden_size)` (if `output_hidden_states=True`).
                - `last_hidden_state`: Full hidden states from final layer, 
                  with shape `(batch_size, segment_num * window_size, hidden_size)` (if `output_hidden_states=True`).
                - `hidden_states`: Hidden states from all layers (if requested), 
                  tuple of `(batch_size, segment_num * window_size, hidden_size) * num_hidden_layers_encoder`.
                - `attentions`: Attention weights (if requested).
                - `cross_attentions`: Cross-attention weights (if applicable).
                - `latent_embeds`: Latent embeddings with shape `(batch_size, segment_num, hidden_size)`.
                - `latents`: Latent representations with shape `(batch_size, segment_num, latent_dim)`.
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
            output_hidden_states=True,
            return_dict=True,
        )
        # Only use the last hidden_state of the last layer as latent
        last_tail_hidden_state = transformer_outputs.last_tail_hidden_state
        
        # Project the latents to logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.latent_head(last_tail_hidden_state[:, slice_indices, :])
        latents = self.latent_head(last_tail_hidden_state)

        loss = None

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMAutoencoderOutputWithCrossAttentions(
            last_tail_hidden_state=transformer_outputs.last_tail_hidden_state if output_hidden_states else None,
            last_window_hidden_state=transformer_outputs.last_window_hidden_state if output_hidden_states else None,
            last_hidden_state=transformer_outputs.last_hidden_state if output_hidden_states else None,
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states if output_hidden_states else None,
            attentions=transformer_outputs.attentions if output_attentions else None,
            cross_attentions=transformer_outputs.cross_attentions if output_attentions else None,
            latent_embeds=transformer_outputs.last_tail_hidden_state,
            latents=latents,
        )
        
@auto_docstring(
    custom_intro="Language decoder model based on GPT-2 for decoding latent representations back to text.",
    custom_args="window_size (`int`): The window size for disaggregating latent representations back to sequences.",
)
class LanguageDecoder(LanguageDecoderBase):
    def __init__(self, config: LatentGPT2Config):
        super().__init__(config)
        self.ae_utils = LanguageDecoderUtils(
            window_size=config.window_size,
            padding_token=config.pad_token_id,
            masking_token=getattr(config, "mask_token_id", config.pad_token_id),
            wte=self.wte,
        )

        # Initialize weights and apply final processing
        self.post_init()
        
    def pre_process_inputs(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_latents: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> PreprocessOutput:
        """
        Pre-processes and segments the inputs for the language decoder.

        Transforms input sequences by aggregating them into single-element segments for
        autoregressive processing. Converts latent vectors to embeddings when provided.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                The input token IDs to be aggregated (typically not used in decoder).
            inputs_latents (`torch.FloatTensor` of shape `(batch_size, segment_num, latent_dim)`, *optional*):
                Latent representations to be converted to embeddings and aggregated.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                Pre-computed embeddings to be aggregated.

        Returns:
            `PreprocessOutput`:
                - input_ids: Aggregated input IDs with shape `(batch_size * seq_len, 1)` or `None`.
                - inputs_latents: Aggregated latents with shape `(batch_size * segment_num, 1, latent_dim)` or `None`.
                - inputs_embeds: Aggregated embeddings with shape `(batch_size * seq_len, 1, hidden_size)` or converted latent embeddings.
        """
        if inputs_embeds is not None:
            if input_ids is not None:
                raise ValueError("input_ids and inputs_embeds are provided at the same time, only one of them is accepted.")
            if inputs_latents is not None:
                raise ValueError("inputs_latents and inputs_embeds are provided at the same time, only one of them is accepted.")
            inputs_embeds = self.ae_utils.agg_sequence(sequence=inputs_embeds)
            return PreprocessOutput(input_ids=input_ids, inputs_latents=inputs_latents, inputs_embeds=inputs_embeds)
        else:
            if input_ids is not None:
                input_ids = self.ae_utils.agg_sequence(sequence=input_ids)
            if inputs_latents is not None:
                inputs_latents = self.ae_utils.agg_sequence(sequence=inputs_latents)
                inputs_embeds = self.wte_latent(inputs_latents)
        return PreprocessOutput(input_ids=input_ids, inputs_latents=inputs_latents, inputs_embeds=inputs_embeds)
    
    def post_process_outputs(
        self,
        outputs: CausalLMAutoencoderOutputWithCrossAttentions,
    ) -> BaseAutoencoderOutputWithPastAndCrossAttentions:
        """
        Post-processes and segment the outputs of the language decoder.

        Transforms single-element segmented transformer outputs back to batch format
        and extracts key representations for autoencoder training.

        Args:
            outputs (`CausalLMAutoencoderOutputWithCrossAttentions`):
                The output from the decoder transformer with:
                - last_hidden_state: shape `(batch_size * segment_num, 1, hidden_size)`
                - hidden_states: tuple of `(batch_size * segment_num, 1, hidden_size) * num_layers`

        Returns:
            `BaseAutoencoderOutputWithPastAndCrossAttentions`:
                - last_tail_hidden_state: Final position per segment, shape `(batch_size, segment_num, hidden_size)`.
                - last_window_hidden_state: Last window_size positions, shape `(batch_size, segment_num, hidden_size)`.
                - last_hidden_state: Full output reshaped to `(batch_size, segment_num, hidden_size)`.
                - hidden_states: All layer outputs reshaped to batch format with shape `(batch_size, segment_num, hidden_size) * num_hidden_layers_decoder` (if provided).
                - past_key_values, attentions, cross_attentions: Passed through unchanged.
        """
        if outputs is None:
            return outputs
        # The input last_hidden_state have already the last hidden_states of the last layer, no need to slice
        return BaseAutoencoderOutputWithPastAndCrossAttentions(
            # Only keep the last hidden_state of hidden_state of the last layer
            last_tail_hidden_state=self.ae_utils.split_sequence(sequence=outputs.last_hidden_state[:, -1:, ...]) if outputs.last_hidden_state is not None else None,
            # Only keep the last self.config.window_size hidden_state of hidden_state of the last layer
            last_window_hidden_state=self.ae_utils.split_sequence(sequence=outputs.last_hidden_state[:, -self.config.window_size:, ...]) if outputs.last_hidden_state is not None else None,
            last_hidden_state=self.ae_utils.split_sequence(sequence=outputs.last_hidden_state) if outputs.last_hidden_state is not None else None,
            past_key_values=outputs.past_key_values,
            hidden_states=tuple(self.ae_utils.split_sequence(sequence=hidden_state) for hidden_state in outputs.hidden_states) if outputs.hidden_states is not None else None,
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
        # wte is not copied since it's now a Linear(latent_dim -> hidden_size), not Embedding
        # It will be initialized with random weights
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

    @auto_docstring(
        custom_intro="Forward pass for the language decoder. Processes latent representations through the decoder transformer to generate hidden states.",
        custom_args=r"""
            input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
                Input token IDs. Note: LanguageDecoder typically processes `inputs_latents` instead of `input_ids`. If provided, a warning will be issued.
            inputs_latents (`torch.FloatTensor` of shape `(batch_size, segment_num, latent_dim)`, *optional*):
                Latent representations to be decoded. These are the primary input for the decoder, typically obtained from the encoder component.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                Pre-computed embeddings to use instead of `inputs_latents`. Cannot be used together with `inputs_latents`.
        """
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_latents: Optional[torch.FloatTensor] = None,
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
    ) -> Union[tuple, CausalLMAutoencoderOutputWithCrossAttentions]:
        r"""
        
        
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
                Input token IDs. Note: LanguageDecoder typically processes `inputs_latents` instead of `input_ids`. If provided, a warning will be issued.
            inputs_latents (`torch.FloatTensor` of shape `(batch_size, segment_num, latent_dim)`, *optional*):
                Latent representations to be decoded. These are the primary input for the decoder, typically obtained from the encoder component.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, segment_num, hidden_size)`, *optional*):
                Pre-computed embeddings to use instead of `inputs_latents`. Cannot be used together with `inputs_latents`.

        Returns:
            [`BaseAutoencoderOutputWithPastAndCrossAttentions`] or `tuple`:
                - last_tail_hidden_state: Final position per segment, shape `(batch_size, segment_num, hidden_size)`.
                - last_window_hidden_state: Last window_size positions, shape `(batch_size, segment_num, hidden_size)`.
                - last_hidden_state: Full output reshaped to `(batch_size, segment_num, hidden_size)`.
                - hidden_states: All layer outputs reshaped to batch format with shape `(batch_size, segment_num, hidden_size) * num_hidden_layers_decoder` (if provided).
                - past_key_values, attentions, cross_attentions: Passed through unchanged.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            warnings.warn("Decoder only processes inputs_latents and inputs_embeds but not input_ids")
        pre_process_res: PreprocessOutput = self.pre_process_inputs(input_ids=input_ids, inputs_latents=inputs_latents, inputs_embeds=inputs_embeds)
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
        return self.post_process_outputs(outputs=output)

@auto_docstring(
    custom_intro="Language decoder model with multi-head language modeling projection that generates vocabulary logits from latent representations. This model processes latent inputs by: 1) Converting latent vectors to embeddings via wte_latent, 2) Processing through decoder transformer layers, 3) Generating logits using window_size separate LM heads, 4) Flattening multi-head outputs into sequential predictions. Key features include multi-head architecture for parallel token prediction, latent-to-sequence decoding capabilities, autoregressive language generation support, and window-based processing for efficiency.",
)
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
        return self.transformer.ae_utils.flatten_multi_heads_logits(logits=multi_head_logits)

    @auto_docstring(
        custom_intro="Forward pass for the language decoder with LM head. Processes latent representations through the decoder transformer and generates vocabulary logits for language modeling using multi-head projection.",
        custom_args=r"""
            input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
                LanguageDecoder and LanguageDecoderLMHead don't process input_ids, please use inputs_latents instead.
            inputs_latents (`torch.FloatTensor` of shape `(batch_size, segment_num, latent_dim)`, *optional*):
                Latent representations to be decoded. These are typically output from the encoder component and represent compressed semantic information to be expanded back into token sequences.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                Pre-computed embeddings instead of `inputs_latents`.
            labels (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Labels for language modeling. Note that the labels **are shifted** inside the model. Indices are selected in `[-100, 0, ..., config.vocab_size]`. All labels set to `-100` are ignored (masked).
            logits_to_keep (`int` or `torch.Tensor`, *optional*, defaults to 0):
                Number of logits to keep from the end of the sequence for memory efficiency.
        """
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_latents: Optional[torch.FloatTensor] = None,
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
        Forward pass for the language decoder with LM head.

        Processes latent representations through the decoder transformer and generates vocabulary
        logits for language modeling using multi-head projection.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
                LanguageDecoder and LanguageDecoderLMHead don't process input_ids, please use inputs_latents instead.
            inputs_latents (`torch.FloatTensor` of shape `(batch_size, segment_num, latent_dim)`, *optional*):
                Latent representations to be decoded. These are typically output from the encoder component of the
                autoencoder and represent compressed semantic information to be expanded back into token sequences.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, segment_num, hidden_size)`, *optional*):
                Pre-computed embeddings instead of `inputs_latents`.
            labels (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
                `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
                are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
            logits_to_keep (`int` or `torch.Tensor`, *optional*, defaults to 0):
                Number of logits to keep from the end of the sequence for memory efficiency.

        Returns:
            [`CausalLMAutoencoderOutputWithCrossAttentions`] or `tuple`:
                - `logits`: Token prediction logits with shape `(batch_size, segment_num * window_size, vocab_size)`.
                - `last_tail_hidden_state`: Final position hidden state per segment, 
                  with shape `(batch_size, segment_num, hidden_size)` (if `output_hidden_states=True`).
                - `last_window_hidden_state`: Hidden states for window positions, 
                  with shape `(batch_size, segment_num, hidden_size)` (if `output_hidden_states=True`).
                - `last_hidden_state`: Full hidden states from final layer, 
                  with shape `(batch_size, segment_num, hidden_size)` (if `output_hidden_states=True`).
                - `hidden_states`: Hidden states from all layers (if requested), 
                  tuple of `(batch_size, segment_num, hidden_size) * num_hidden_layers_decoder`.
                - `attentions`: Attention weights (if requested).
                - `cross_attentions`: Cross-attention weights (if applicable).
                - `latent_embeds`: Latent embeddings with shape `(batch_size, segment_num, hidden_size)`.
                - `latents`: Latent representations with shape `(batch_size, segment_num, latent_dim)`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if inputs_latents is not None and inputs_embeds is None:
            inferred_inputs_embeds = self.transformer.wte_latent(inputs_latents)
        else:
            inferred_inputs_embeds = inputs_embeds

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            inputs_latents=inputs_latents,
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
            output_hidden_states=True,
            return_dict=True,
        )
        latents = transformer_outputs.last_tail_hidden_state

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # logits = self.lm_head(latents[:, slice_indices, :])
        logits = self._project_with_multi_heads(latents[:, slice_indices, :])

        loss = None

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMAutoencoderOutputWithCrossAttentions(
            last_tail_hidden_state=transformer_outputs.last_tail_hidden_state if output_hidden_states else None,
            last_window_hidden_state=transformer_outputs.last_window_hidden_state if output_hidden_states else None,
            last_hidden_state=transformer_outputs.last_hidden_state if output_hidden_states else None,
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states if output_hidden_states else None,
            attentions=transformer_outputs.attentions if output_attentions else None,
            cross_attentions=transformer_outputs.cross_attentions if output_attentions else None,
            latent_embeds=inferred_inputs_embeds,
            latents=inputs_latents,
        )

@auto_docstring(
    custom_intro="Complete language autoencoder model combining encoder and decoder components for sequence-to-sequence autoencoding with configurable compression. This model implements a full autoencoder architecture that processes sequences through a compress-decompress cycle: 1) Input sequences are segmented into fixed-size windows, 2) Each window is encoded into a single latent vector, 3) Latent vectors are decoded back to token predictions, 4) Multi-head decoding generates multiple tokens per latent. Architecture includes LanguageEncoderLatentHead for encoding and LanguageDecoderLMHead for decoding with tied weights between components. Key features include end-to-end sequence autoencoding, configurable compression ratio via window_size, support for reconstruction and generation, efficient training with label smoothing loss, and weight sharing between encoder and decoder.",
)
class LanguageAutoencoder(GPT2PreTrainedModel, GenerationMixin):
    # Tieing weights doesn't work
    # _tied_weights_keys = {"encoder.latent_head.weight": "decoder.transformer.wte_latent.weight"}
    def __init__(self, config: LatentGPT2Config):
        super().__init__(config)
        # self.window_size: int = window_size
        self.encoder: LanguageEncoderLatentHead = LanguageEncoderLatentHead(config=config)
        self.decoder: LanguageDecoderLMHead = LanguageDecoderLMHead(config=config)
        
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
        self.encoder.init_weight_from_pretrained(pretrained_model=pretrained_model)
        self.decoder.init_weight_from_pretrained(pretrained_model=pretrained_model)
        return self
    
    def encode(
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
        """
        Encode input sequences into latent representations.

        This method processes input sequences through the encoder transformer, segmenting them into
        fixed-size windows and producing compressed latent representations.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Input token IDs to encode. Sequence length should be divisible by
                `window_size` for optimal processing (padding is applied if not).
                After padding, `segment_num = ceil(seq_len / window_size)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                Pre-computed embeddings instead of `input_ids`.
            labels (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Not used in encoding, included for API consistency.

            logits_to_keep (`int` or `torch.Tensor`, *optional*, defaults to 0):
                Number of logits to keep from the end of the sequence for memory efficiency.

        Returns:
            [`CausalLMAutoencoderOutputWithCrossAttentions`] or `tuple`:
                - `logits`: Latent projection logits with shape `(batch_size, logits_to_keep, latent_dim)` 
                  if `logits_to_keep > 0`, otherwise shape `(batch_size, segment_num, latent_dim)` (Note: these are latent projections, not vocab logits)
                - `last_tail_hidden_state`: Final position hidden state per segment, 
                  with shape `(batch_size, segment_num, hidden_size)` (if `output_hidden_states=True`).
                - `last_window_hidden_state`: Hidden states for window positions, 
                  with shape `(batch_size, segment_num * window_size, hidden_size)` (if `output_hidden_states=True`).
                - `last_hidden_state`: Full hidden states from final layer, 
                  with shape `(batch_size, segment_num * window_size, hidden_size)` (if `output_hidden_states=True`).
                - `hidden_states`: Hidden states from all layers (if requested), 
                  tuple of `(batch_size, segment_num * window_size, hidden_size) * num_hidden_layers_encoder`.
                - `attentions`: Attention weights (if requested).
                - `cross_attentions`: Cross-attention weights (if applicable).
                - `latent_embeds`: Latent embeddings with shape `(batch_size, segment_num, hidden_size)`.
                - `latents`: Latent representations with shape `(batch_size, segment_num, latent_dim)`.

        Example:
            ```python
            # Encode text to latents
            encoder_output = model.encode(input_ids=input_ids)
            latents = encoder_output.latents  # (batch_size, segment_num, latent_dim)
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_output = self.encoder(
            input_ids=input_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        return encoder_output
    
    def decode(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_latents: Optional[torch.FloatTensor] = None,
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
        """
        Decode latent representations into token sequences.

        This method processes latent vectors through the decoder transformer to generate
        token logits for language modeling. Each latent is expanded to `window_size` token predictions.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
                Not typically used in decoding, included for API consistency.
            inputs_latents (`torch.FloatTensor` of shape `(batch_size, segment_num, latent_dim)`, *optional*):
                Latent representations to decode. Each latent is expanded to `window_size` token predictions.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, segment_num, hidden_size)`, *optional*):
                Pre-computed embeddings to use instead of latents.
            labels (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Not used in decoding, included for API consistency.
            logits_to_keep (`int` or `torch.Tensor`, *optional*, defaults to 0):
                Number of logits to keep from the end of the sequence for memory efficiency.

        Returns:
            [`CausalLMAutoencoderOutputWithCrossAttentions`] or `tuple`:
                - `logits`: Token prediction logits with shape `(batch_size, segment_num * window_size, vocab_size)`.
                - `last_tail_hidden_state`: Final position hidden state per segment, 
                  with shape `(batch_size, segment_num, hidden_size)` (if `output_hidden_states=True`).
                - `last_window_hidden_state`: Hidden states for window positions, 
                  with shape `(batch_size, segment_num, hidden_size)` (if `output_hidden_states=True`).
                - `last_hidden_state`: Full hidden states from final layer, 
                  with shape `(batch_size, segment_num, hidden_size)` (if `output_hidden_states=True`).
                - `hidden_states`: Hidden states from all layers (if requested), 
                  tuple of `(batch_size, segment_num, hidden_size) * num_hidden_layers_decoder`.
                - `attentions`: Attention weights (if requested).
                - `cross_attentions`: Cross-attention weights (if applicable).
                - `latent_embeds`: Latent embeddings with shape `(batch_size, segment_num, hidden_size)`.
                - `latents`: Latent representations with shape `(batch_size, segment_num, latent_dim)`.

        Example:
            ```python
            # First encode, then decode
            encoder_output = model.encode(input_ids=input_ids)
            decoder_output = model.decode(inputs_latents=encoder_output.latents)
            logits = decoder_output.logits  # (batch_size, segment_num * window_size, vocab_size)
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        decoder_output = self.decoder(
            input_ids=input_ids,
            inputs_latents=inputs_latents,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            cache_position=cache_position,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        return decoder_output

    def _format_dict_output(
        self,
        dict_output,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> CausalLMAutoencoderOutputWithCrossAttentions:
        return CausalLMAutoencoderOutputWithCrossAttentions(
            last_tail_hidden_state=dict_output.last_tail_hidden_state if output_hidden_states else None,
            last_window_hidden_state=dict_output.last_window_hidden_state if output_hidden_states else None,
            last_hidden_state=dict_output.last_hidden_state if output_hidden_states else None,
            loss=dict_output.loss,
            logits=dict_output.logits,
            past_key_values=dict_output.past_key_values,
            hidden_states=dict_output.hidden_states if output_hidden_states else None,
            attentions=dict_output.attentions if output_attentions else None,
            cross_attentions=dict_output.cross_attentions if output_attentions else None,
            latent_embeds=dict_output.latent_embeds,
            latents=dict_output.latents,
        )

    @auto_docstring(
        custom_intro="Forward pass for the complete language autoencoder. Processes input sequences through encode-decode cycle for reconstruction or generation tasks.",
        custom_args=r"""
            input_ids (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Input token IDs to be encoded and decoded. Sequence will be padded to be divisible by `window_size` if needed.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                Pre-computed embeddings instead of `input_ids`.
            labels (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Labels for computing reconstruction loss. Note that labels **are shifted** inside the model.
            logits_to_keep (`int` or `torch.Tensor`, *optional*, defaults to 0):
                Number of logits to keep from the end of the sequence for memory efficiency.
            return_encoder_decoder_res (`bool`, *optional*, defaults to `False`):
                Whether to return separate encoder and decoder results in addition to combined autoencoder output.
        """
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
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        return_encoder_decoder_res: bool = False,
        **kwargs,
    ) -> Union[tuple, CausalLMAutoencoderOutputWithCrossAttentions]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Input token IDs to be encoded and decoded. Sequence will be padded to be divisible by `window_size` if needed.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                Pre-computed embeddings instead of `input_ids`.
            labels (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Labels for computing reconstruction loss. Note that labels **are shifted** inside the model.
            logits_to_keep (`int` or `torch.Tensor`, *optional*, defaults to 0):
                Number of logits to keep from the end of the sequence for memory efficiency.
            return_encoder_decoder_res (`bool`, *optional*, defaults to `False`):
                Whether to return separate encoder and decoder results in addition to combined autoencoder output.

        Returns:
            [`CausalLMAutoencoderOutputWithCrossAttentions`] or `tuple`:
                - `logits`: Token prediction logits from decoder with shape `(batch_size, segment_num * window_size, vocab_size)`.
                - `last_tail_hidden_state`: Final position hidden state per segment from decoder, with shape `(batch_size, segment_num, hidden_size)`.
                - `last_window_hidden_state`: Hidden states for window positions from decoder, with shape `(batch_size, segment_num, hidden_size)`.
                - `last_hidden_state`: Full hidden states from decoder final layer, with shape `(batch_size, segment_num, hidden_size)`.
                - `hidden_states`: Hidden states from all layers (if requested), 
                  tuple of `(batch_size, segment_num, hidden_size) * num_hidden_layers_decoder`.
                - `attentions`: Attention weights (if requested).
                - `cross_attentions`: Cross-attention weights (if applicable).
                - `latent_embeds`: Latent embeddings with shape `(batch_size, segment_num, hidden_size)`.
                - `latents`: Encoded latent representations with shape `(batch_size, segment_num, latent_dim)`.
                - `loss`: Reconstruction loss (if labels provided).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_output = self.encode(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            attention_mask=None,
            cache_position=None,
            token_type_ids=None,
            position_ids=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            logits_to_keep=0,
        )
        # hidden_states = encoder_output[0]
        decoder_output = self.decode(
            input_ids=None,
            inputs_latents=encoder_output.latents,
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
            logits_to_keep=0,
        )

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = decoder_output.logits[:, slice_indices, :]
        # print(f"decoder_output.logits: {decoder_output.logits.shape}")
        # print(f"logits: {logits.shape}")

        loss = None
        if labels is not None:
            # Pad labels to make it can be divided by self.config.window_size
            padded_labels: torch.LongTensor = self.decoder.transformer.ae_utils.pad(sequence=labels)
            # Flatten the tokens
            loss = self.loss_function(
                logits=logits,
                labels=padded_labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + decoder_output[1:]
            return ((loss,) + output) if loss is not None else output

        ae_output = CausalLMAutoencoderOutputWithCrossAttentions(
            last_tail_hidden_state=decoder_output.last_tail_hidden_state if output_hidden_states else None,
            last_window_hidden_state=decoder_output.last_window_hidden_state if output_hidden_states else None,
            last_hidden_state=decoder_output.last_hidden_state if output_hidden_states else None,
            loss=loss,
            logits=logits,
            past_key_values=decoder_output.past_key_values,
            hidden_states=decoder_output.hidden_states if output_hidden_states else None,
            attentions=decoder_output.attentions if output_attentions else None,
            cross_attentions=decoder_output.cross_attentions if output_attentions else None,
            latent_embeds=encoder_output.latent_embeds,
            latents=encoder_output.latents,
        )
        # Handle encoder and decoder dict output format
        encoder_output = self._format_dict_output(dict_output=encoder_output, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        decoder_output = self._format_dict_output(dict_output=decoder_output, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        
        # When return_encoder_decoder_res=True, return tuple of (ar_outputs, ltar_output, encoder_output)
        if return_encoder_decoder_res:
            # ltar_output is the latent AR prediction (placeholder: using encoder_output for now)
            # encoder_output contains the actual encoder latents
            return (ae_output, encoder_output, decoder_output)
        return ae_output
    
    def loss_function(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        vocab_size: int,
        shift_labels: int = 0,
        epsilon: float = 0.1,
        ignore_index: int = -100,
        **kwargs,
    ) -> torch.FloatTensor:
        # Copy from LTLMTrainer.compute_smoothed_loss
        # Either do not shift, for reconstruction, or shift by the window_size
        if shift_labels:
            logits = logits[..., :-shift_labels, :].contiguous()
            labels = labels[..., shift_labels:].contiguous()

        # Claude Code update, double check, Ensure logits and labels have the same sequence length
        # I've added a padding function for labels, making logits and labels have the same length. Labels sometimes cannot be devided by the self.config.window_size
        # logits_seq_len = logits.size(-2)
        # labels_seq_len = labels.size(-1)
        # if logits_seq_len != labels_seq_len:
        #     min_len = min(logits_seq_len, labels_seq_len)
        #     logits = logits[..., :min_len, :].contiguous()
        #     labels = labels[..., :min_len].contiguous()
        # Claude Code update ends

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        # print(f"log_probs: {log_probs.shape}, labels: {labels.shape}")
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - epsilon) * nll_loss + epsilon * smoothed_loss

__all__ = [
    "LanguageAutoencoder",
    "LanguageDecoderLMHead",
    "LanguageEncoderLatentHead",
]
