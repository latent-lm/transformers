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

"""
Language Autoencoder based on GPT-2.

This module implements a language autoencoder that compresses text into latent representations
using a windowed encoding scheme and reconstructs text using multi-head decoding.

Architecture Overview:
    - Encoder: Processes input tokens in windows of `window_size` and outputs one latent per window
    - Decoder: Takes one latent per window and outputs `window_size` tokens using multi-head projection

Tensor Shape Flow (where num_segments = ceil(seq_len / window_size)):
    Encoding:
        input_ids: (batch_size, seq_len)
        -> pad to multiple of window_size: (batch_size, padded_seq_len)
        -> reshape for windowed processing: (batch_size * num_segments, window_size)
        -> transformer: (batch_size * num_segments, window_size, hidden_size)
        -> take last position: (batch_size * num_segments, 1, hidden_size)
        -> reshape: (batch_size, num_segments, hidden_size)
        -> project to latent: (batch_size, num_segments, latent_dim)

    Decoding:
        latents: (batch_size, num_segments, latent_dim)
        -> project to hidden: (batch_size, num_segments, hidden_size)
        -> reshape for transformer: (batch_size * num_segments, 1, hidden_size)
        -> transformer: (batch_size * num_segments, 1, hidden_size)
        -> reshape back: (batch_size, num_segments, hidden_size)
        -> multi-head projection: window_size heads, each (batch_size, num_segments, vocab_size)
        -> interleave and reshape: (batch_size, num_segments * window_size, vocab_size)
"""

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
from ...utils import logging
from .configuration_latent_gpt2 import LatentGPT2Config
from .modeling_gpt2 import GPT2ModelBase, GPT2PreTrainedModel, GPT2LMHeadModel, GPT2Block


logger = logging.get_logger(__name__)


# =============================================================================
# Base Classes for Encoder and Decoder
# =============================================================================


class LanguageEncoderBase(GPT2ModelBase):
    """
    Base encoder model with configurable number of transformer layers.

    Inherits from GPT2ModelBase and uses the first `num_hidden_layers_encoder` layers.
    """

    def __init__(self, config: LatentGPT2Config):
        super().__init__(config)
        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([
            GPT2Block(config, layer_idx=i)
            for i in range(config.num_hidden_layers_encoder)
        ])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation
        self.post_init()


class LanguageDecoderBase(GPT2ModelBase):
    """
    Base decoder model with configurable number of transformer layers.

    Uses `wte_latent` to project from latent space to hidden dimension,
    plus standard `wte` for token embeddings (used in diffusion decoding).
    """

    def __init__(self, config: LatentGPT2Config):
        super().__init__(config)
        self.embed_dim = config.hidden_size

        # Latent-to-hidden projection (replaces token embedding for latent inputs)
        self.wte_latent = nn.Linear(config.latent_dim, self.embed_dim, bias=False)
        # Standard token embedding (used for diffusion/mask decoding)
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([
            GPT2Block(config, layer_idx=i)
            for i in range(config.num_hidden_layers_decoder)
        ])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation
        self.post_init()

        # Initialize wte_latent as identity if dimensions match
        if config.latent_dim == self.embed_dim:
            with torch.no_grad():
                self.wte_latent.weight.copy_(torch.eye(self.embed_dim))


# =============================================================================
# Sequence Window Utilities
# =============================================================================


class SequenceWindowUtilsBase:
    """
    Base utility class for windowed sequence operations.

    Provides methods to:
    - Pad sequences to be divisible by window_size
    - Split/aggregate sequences into windows
    - Handle padding and masking tokens

    Attributes:
        TOKEN_TYPE_PADDING: Use padding token for filling
        TOKEN_TYPE_MASKING: Use mask token for filling (diffusion models)
    """

    TOKEN_TYPE_PADDING: str = "padding"
    TOKEN_TYPE_MASKING: str = "masking"
    SUPPORTED_TOKEN_TYPES: List[str] = [TOKEN_TYPE_PADDING, TOKEN_TYPE_MASKING]

    def __init__(
        self,
        window_size: int,
        padding_token: Optional[int] = None,
        padding_embed: Optional[torch.FloatTensor] = None,
        masking_token: Optional[int] = None,
        masking_embed: Optional[torch.FloatTensor] = None,
        wte: Optional[nn.Embedding] = None,
    ):
        """
        Args:
            window_size: Number of tokens per window/segment
            padding_token: Token ID used for padding
            padding_embed: Pre-computed embedding for padding (optional)
            masking_token: Token ID used for masking (diffusion)
            masking_embed: Pre-computed embedding for masking (optional)
            wte: Token embedding layer (used to compute embeddings if not provided)
        """
        self._batch_size: Optional[int] = None
        self._seq_len: Optional[int] = None
        self._segment_num: Optional[int] = None
        self._window_size: int = window_size
        self._padding_token = padding_token
        self._padding_embed = padding_embed
        self._masking_token = masking_token
        self._masking_embed = masking_embed
        self._wte: Optional[nn.Embedding] = wte

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    @property
    def seq_len(self) -> Optional[int]:
        return self._seq_len

    @property
    def segment_num(self) -> Optional[int]:
        return self._segment_num

    @property
    def window_size(self) -> int:
        return self._window_size

    def _get_token(self, token_type: str) -> int:
        """Get token ID for the specified type."""
        if token_type == self.TOKEN_TYPE_PADDING:
            if self._padding_token is None:
                raise ValueError("padding_token is not set")
            return self._padding_token
        elif token_type == self.TOKEN_TYPE_MASKING:
            if self._masking_token is None:
                raise ValueError("masking_token is not set")
            return self._masking_token
        raise ValueError(f"Unsupported token_type: {token_type}. Supported: {self.SUPPORTED_TOKEN_TYPES}")

    def _get_embed(self, token_type: str) -> torch.FloatTensor:
        """Get embedding for the specified token type (computes if not cached)."""
        if token_type == self.TOKEN_TYPE_PADDING:
            if self._padding_embed is None:
                if self._wte is None:
                    raise ValueError("Provide padding_embed or wte to compute embedding")
                self._padding_embed = self._wte(torch.tensor(self._padding_token))
            return self._padding_embed
        elif token_type == self.TOKEN_TYPE_MASKING:
            if self._masking_embed is None:
                if self._wte is None:
                    raise ValueError("Provide masking_embed or wte to compute embedding")
                self._masking_embed = self._wte(torch.tensor(self._masking_token))
            return self._masking_embed
        raise ValueError(f"Unsupported token_type: {token_type}. Supported: {self.SUPPORTED_TOKEN_TYPES}")

    def pad(
        self,
        sequence: Optional[torch.Tensor],
        token_type: str = TOKEN_TYPE_PADDING
    ) -> Optional[torch.Tensor]:
        """
        Pad sequence length to be divisible by window_size.

        Args:
            sequence: Input tensor
                - 2D (batch_size, seq_len): pads with token IDs
                - 3D (batch_size, seq_len, embed_dim): pads with embeddings
            token_type: "padding" or "masking"

        Returns:
            Padded sequence with seq_len divisible by window_size
        """
        if sequence is None:
            return None

        batch_size, seq_len = sequence.shape[0], sequence.shape[1]
        # Calculate padding needed to make seq_len divisible by window_size
        # Example: seq_len=10, window_size=4 -> pad_len = (4 - 10%4) % 4 = (4-2)%4 = 2
        pad_len = (self._window_size - (seq_len % self._window_size)) % self._window_size

        # No padding needed if already divisible
        if pad_len == 0:
            return sequence

        if sequence.dim() == 2:
            # Token IDs: (batch_size, seq_len) -> (batch_size, padded_len)
            pad = torch.full(
                (batch_size, pad_len),
                self._get_token(token_type),
                dtype=sequence.dtype,
                device=sequence.device
            )
        elif sequence.dim() == 3:
            # Embeddings: (batch_size, seq_len, embed_dim) -> (batch_size, padded_len, embed_dim)
            pad = self._get_embed(token_type).expand(batch_size, pad_len, -1).to(
                dtype=sequence.dtype, device=sequence.device
            )
        else:
            raise NotImplementedError(f"Unsupported dimension: {sequence.dim()}, shape: {sequence.shape}")

        return torch.cat((sequence, pad), dim=1)

    def split_sequence(self, sequence: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Reshape windowed sequence back to batch dimensions.

        Args:
            sequence: Shape (batch_size * segment_num, window_len, ...)

        Returns:
            Shape (batch_size, segment_num * window_len, ...)
        """
        if sequence is None:
            return None

        # Inverse of agg_sequence: restore batch dimension from flattened segments
        # (batch * num_segments, window_len, ...) -> (batch, num_segments * window_len, ...)
        # This merges all segment outputs back into a single sequence per batch item
        return sequence.reshape(
            self._batch_size,
            self._segment_num * sequence.shape[1],
            *sequence.shape[2:]
        )

    def split_context_target(
        self,
        sequence: torch.Tensor,
        is_padding: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Split sequence into context (all but last window) and target (last window).

        Args:
            sequence: Input tensor (2D or 3D)
            is_padding: Whether to pad context to be divisible by window_size

        Returns:
            (context, target) tuple
        """
        if sequence is None:
            return None, None

        if sequence.dim() == 2:
            context = sequence[..., :-self._window_size].contiguous()
            target = sequence[..., -self._window_size:].contiguous()
        elif sequence.dim() > 2:
            context = sequence[..., :-self._window_size, :].contiguous()
            target = sequence[..., -self._window_size:, :].contiguous()
        else:
            raise NotImplementedError(f"Unsupported dimension: {sequence.dim()}")

        if is_padding:
            context = self.pad(sequence=context)
        return context, target

    def split_context_target_then_cat(
        self,
        sequence: torch.Tensor,
        is_padding: bool = True,
        return_context_target: bool = False,
    ) -> Union[Optional[torch.Tensor], Tuple[Optional[torch.Tensor], ...]]:
        """
        Split into context/target, optionally pad context, then concatenate.

        Returns:
            If return_context_target=False: concatenated sequence
            If return_context_target=True: (concatenated, context, target)
        """
        if sequence is None:
            return (None, None, None) if return_context_target else None

        context, target = self.split_context_target(sequence=sequence, is_padding=is_padding)

        if sequence.dim() == 2:
            joint_seq = torch.cat((context, target), dim=-1)
        else:
            joint_seq = torch.cat((context, target), dim=-2)

        if return_context_target:
            return joint_seq, context, target
        return joint_seq


class LanguageEncoderUtils(SequenceWindowUtilsBase):
    """
    Encoder-specific utilities for aggregating sequences into fixed-size windows.

    Encoding flow:
        (batch_size, seq_len) -> pad -> (batch_size, padded_len)
        -> reshape -> (batch_size * num_segments, window_size)
    """

    def agg_sequence(self, sequence: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Aggregate sequence into windows for encoder processing.

        Args:
            sequence: Shape (batch_size, seq_len) or (batch_size, seq_len, embed_dim)

        Returns:
            Shape (batch_size * segment_num, window_size) or
                  (batch_size * segment_num, window_size, embed_dim)
        """
        if sequence is None:
            return None

        # Store original batch dimensions for later reshaping back
        self._batch_size = sequence.shape[0]
        self._seq_len = sequence.shape[1]

        # Pad sequence to be divisible by window_size
        padded = self.pad(sequence=sequence, token_type=self.TOKEN_TYPE_PADDING)

        # Calculate number of segments (windows) after padding
        self._segment_num = padded.shape[1] // self._window_size

        # Reshape: (batch, padded_len, ...) -> (batch * num_segments, window_size, ...)
        # This allows processing each window independently through the transformer
        return padded.view(
            self._batch_size * self._segment_num,
            self._window_size,
            *padded.shape[2:]
        )


class LanguageDecoderUtils(SequenceWindowUtilsBase):
    """
    Decoder-specific utilities for processing latents and aggregating outputs.

    Standard decoding flow (one latent -> window_size tokens):
        (batch_size, num_segments, latent_dim)
        -> reshape -> (batch_size * num_segments, 1, latent_dim)

    Multi-head output flow:
        window_size heads each produce (batch_size * num_segments, 1, vocab_size)
        -> interleave -> (batch_size, num_segments * window_size, vocab_size)
    """

    def agg_sequence(self, sequence: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Reshape latents for decoder: each latent becomes a single-token sequence.

        Args:
            sequence: Shape (batch_size, num_segments, ...)

        Returns:
            Shape (batch_size * num_segments, 1, ...)
        """
        if sequence is None:
            return None

        # Store dimensions for later reshaping back to batch form
        self._batch_size = sequence.shape[0]
        self._seq_len = sequence.shape[1]
        self._segment_num = sequence.shape[1]  # Each latent is one segment

        # Reshape: (batch, num_segments, latent_dim) -> (batch * num_segments, 1, latent_dim)
        # Each latent vector becomes a single-position sequence for the transformer
        # The "1" dimension is needed because transformers expect (batch, seq_len, hidden)
        return sequence.view(self._batch_size * self._segment_num, 1, *sequence.shape[2:])

    def agg_sequence_by_window_size(self, sequence: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Aggregate sequence into window_size-length segments (used for diffusion decoding).

        Args:
            sequence: Shape (batch_size, seq_len, ...)

        Returns:
            Shape (batch_size * segment_num, window_size, ...)
        """
        if sequence is None:
            return None
        self._batch_size = sequence.shape[0]
        self._seq_len = sequence.shape[1]
        padded = self.pad(sequence=sequence, token_type=self.TOKEN_TYPE_PADDING)
        self._segment_num = padded.shape[1] // self._window_size
        return padded.view(
            self._batch_size * self._segment_num,
            self._window_size,
            *padded.shape[2:]
        )

    def agg_sequence_mask_diffusion(self, sequence: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Aggregate masked sequence for diffusion decoding (pads with mask tokens).

        Args:
            sequence: Shape (batch_size, seq_len, ...)

        Returns:
            Shape (batch_size * segment_num, window_size, ...)
        """
        if sequence is None:
            return None
        self._batch_size = sequence.shape[0]
        self._seq_len = sequence.shape[1]
        padded = self.pad(sequence=sequence, token_type=self.TOKEN_TYPE_MASKING)
        self._segment_num = padded.shape[1] // self._window_size
        return padded.view(
            self._batch_size * self._segment_num,
            self._window_size,
            *padded.shape[2:]
        )

    def flatten_multi_heads_logits(self, logits: List[torch.Tensor]) -> torch.Tensor:
        """
        Interleave multi-head logits into a single sequence.

        Each head predicts one position in the output window. This method interleaves
        the predictions so position 0 of each segment comes from head 0, position 1
        from head 1, etc.

        Args:
            logits: List of length window_size, each shape (batch_size, num_segments, vocab_size)

        Returns:
            Shape (batch_size, num_segments * window_size, vocab_size)
            Positions are interleaved: [seg0_pos0, seg0_pos1, ..., seg1_pos0, seg1_pos1, ...]
        """
        # Stack all head outputs along a new dimension (dim=2)
        # Before: list of window_size tensors, each (batch, num_segments, vocab_size)
        # After stack: (batch, num_segments, window_size, vocab_size)
        stacked = torch.stack(logits, dim=2)

        # Flatten to interleave positions from each segment
        # (batch, num_segments, window_size, vocab_size) -> (batch, num_segments * window_size, vocab_size)
        # Result ordering: [seg0_tok0, seg0_tok1, ..., seg0_tokW, seg1_tok0, seg1_tok1, ...]
        # This matches the original input token ordering after padding
        return stacked.view(
            self._batch_size,
            self._segment_num * self._window_size,
            *logits[0].shape[2:]
        )

    def pad(self, sequence: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Pad with padding tokens (convenience override)."""
        return super().pad(sequence=sequence, token_type=self.TOKEN_TYPE_PADDING)


# =============================================================================
# Encoder Models
# =============================================================================


class LanguageEncoder(LanguageEncoderBase):
    """
    Language encoder that processes text in windows and produces latent representations.

    Processing flow:
        1. Input: (batch_size, seq_len)
        2. Pad to multiple of window_size: (batch_size, padded_len)
        3. Reshape to windows: (batch_size * num_segments, window_size)
        4. Transformer: (batch_size * num_segments, window_size, hidden_size)
        5. Extract last position: (batch_size, num_segments, hidden_size)
    """

    def __init__(self, config: LatentGPT2Config):
        super().__init__(config)
        self.ae_utils = LanguageEncoderUtils(
            window_size=config.window_size,
            padding_token=config.pad_token_id,
            wte=self.wte,
        )
        self.post_init()

    def pre_process_inputs(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> PreprocessOutput:
        """
        Aggregate inputs into windows for transformer processing.

        Args:
            input_ids: Shape (batch_size, seq_len)
            inputs_embeds: Shape (batch_size, seq_len, hidden_size)

        Returns:
            PreprocessOutput with shapes (batch_size * num_segments, window_size, ...)
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
        Reshape transformer outputs back to batch dimensions.

        Args:
            outputs: Transformer outputs with last_hidden_state shape
                     (batch_size * num_segments, window_size, hidden_size)

        Returns:
            Autoencoder outputs with:
            - last_tail_hidden_state: (batch_size, num_segments, hidden_size) [last position only]
            - last_window_hidden_state: (batch_size, num_segments * window_size, hidden_size) [last window]
            - last_hidden_state: (batch_size, padded_len, hidden_size) [full sequence]
        """
        if outputs is None:
            return outputs

        # Transformer output shape: (batch * num_segments, window_size, hidden_size)
        last_hidden = outputs.last_hidden_state

        return BaseAutoencoderOutputWithPastAndCrossAttentions(
            # Extract only the last position of each window as the latent representation
            # Shape: (batch * num_segments, 1, hidden) -> (batch, num_segments, hidden)
            # This is the key compression: window_size tokens -> 1 latent vector
            last_tail_hidden_state=(
                self.ae_utils.split_sequence(last_hidden[:, -1:, ...])
                if last_hidden is not None else None
            ),
            # Extract the last window_size positions (entire last window)
            # Shape: (batch * num_segments, window_size, hidden) -> (batch, num_segments * window_size, hidden)
            last_window_hidden_state=(
                self.ae_utils.split_sequence(last_hidden[:, -self.config.window_size:, ...])
                if last_hidden is not None else None
            ),
            # Full sequence reshaped back to batch form
            # Shape: (batch * num_segments, window_size, hidden) -> (batch, padded_seq_len, hidden)
            last_hidden_state=(
                self.ae_utils.split_sequence(last_hidden)
                if last_hidden is not None else None
            ),
            past_key_values=outputs.past_key_values,
            # Reshape all intermediate layer hidden states
            hidden_states=(
                tuple(self.ae_utils.split_sequence(h) for h in outputs.hidden_states)
                if outputs.hidden_states is not None else None
            ),
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def init_weight_from_pretrained(self, pretrained_model: GPT2ModelBase) -> "LanguageEncoder":
        """
        Initialize encoder weights from a pretrained GPT-2 model.

        Uses the first `num_hidden_layers_encoder` layers from the pretrained model.
        """
        self.wte = copy.deepcopy(pretrained_model.wte)
        self.wpe = copy.deepcopy(pretrained_model.wpe)
        self.drop = copy.deepcopy(pretrained_model.drop)
        self.h = copy.deepcopy(pretrained_model.h[:self.config.num_hidden_layers_encoder])
        self.ln_f = copy.deepcopy(pretrained_model.ln_f)
        self.gradient_checkpointing = pretrained_model.gradient_checkpointing

        # Re-index layer indices for attention caching
        for new_idx, block in enumerate(self.h):
            block.attn.layer_idx = new_idx
            if hasattr(block, 'crossattention'):
                block.crossattention.layer_idx = new_idx
        return self

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
        """
        Forward pass for the language encoder.

        Args:
            input_ids: Shape (batch_size, seq_len)
            inputs_embeds: Shape (batch_size, seq_len, hidden_size)
            return_segment: Whether to return segmented outputs (always True for encoder)

        Returns:
            BaseAutoencoderOutputWithPastAndCrossAttentions with windowed hidden states
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        preprocessed = self.pre_process_inputs(input_ids=input_ids, inputs_embeds=inputs_embeds)

        output = super().forward(
            input_ids=preprocessed.input_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=preprocessed.inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        return self.post_process_outputs(outputs=output)


class LanguageEncoderLatentHead(GPT2PreTrainedModel, GenerationMixin):
    """
    Language encoder with a projection head to latent space.

    Processing flow:
        1. Encode text via LanguageEncoder
        2. Project last hidden state to latent dimension

    Output shapes:
        - latents: (batch_size, num_segments, latent_dim)
        - logits: Same as latents (the latent projection)
    """

    def __init__(self, config: LatentGPT2Config):
        super().__init__(config)
        self.transformer = LanguageEncoder(config=config)
        self.latent_head = nn.Linear(config.n_embd, config.latent_dim, bias=False)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()

        # Initialize latent_head as identity if dimensions match
        if config.n_embd == config.latent_dim:
            with torch.no_grad():
                self.latent_head.weight.copy_(torch.eye(config.n_embd))

    def init_weight_from_pretrained(self, pretrained_model: GPT2LMHeadModel) -> "LanguageEncoderLatentHead":
        """Initialize from pretrained GPT-2 LM model."""
        self.transformer.init_weight_from_pretrained(pretrained_model.transformer)
        self.lm_head = copy.deepcopy(pretrained_model.lm_head)
        # latent_head is randomly initialized (projects to latent_dim, not vocab)
        return self

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
        """
        Forward pass for encoder with latent head.

        Args:
            input_ids: Shape (batch_size, seq_len)
            logits_to_keep: Number of final positions to project (0 = all)

        Returns:
            CausalLMAutoencoderOutputWithCrossAttentions with:
            - logits: (batch_size, num_segments, latent_dim) - latent projections
            - latents: Same as logits
            - latent_embeds: (batch_size, num_segments, hidden_size) - pre-projection hidden states
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

        # Use last position of each window as latent representation
        # Shape: (batch_size, num_segments, hidden_size)
        # Each segment's last token summarizes the entire window
        latent_embeds = transformer_outputs.last_tail_hidden_state

        # Handle logits_to_keep: 0 means keep all, N means keep last N positions
        # slice(-0, None) = slice(0, None) = all elements (correct behavior for 0)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        # Project hidden states to latent space
        # Shape: (batch_size, num_segments, hidden_size) -> (batch_size, num_segments, latent_dim)
        logits = self.latent_head(latent_embeds[:, slice_indices, :])

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return output

        return CausalLMAutoencoderOutputWithCrossAttentions(
            last_tail_hidden_state=transformer_outputs.last_tail_hidden_state if output_hidden_states else None,
            last_window_hidden_state=transformer_outputs.last_window_hidden_state if output_hidden_states else None,
            last_hidden_state=transformer_outputs.last_hidden_state if output_hidden_states else None,
            loss=None,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states if output_hidden_states else None,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
            latent_embeds=latent_embeds,
            latents=logits,
        )


# =============================================================================
# Decoder Models
# =============================================================================


class LanguageDecoder(LanguageDecoderBase):
    """
    Language decoder that reconstructs text from latent representations.

    Processing flow:
        1. Input latents: (batch_size, num_segments, latent_dim)
        2. Project to hidden: (batch_size, num_segments, hidden_size)
        3. Reshape: (batch_size * num_segments, 1, hidden_size)
        4. Transformer: (batch_size * num_segments, 1, hidden_size)
        5. Reshape back: (batch_size, num_segments, hidden_size)
    """

    def __init__(self, config: LatentGPT2Config):
        super().__init__(config)
        self.ae_utils = LanguageDecoderUtils(
            window_size=config.window_size,
            padding_token=config.pad_token_id,
            masking_token=getattr(config, "mask_token_id", config.pad_token_id),
            wte=self.wte,
        )
        self.post_init()

    def pre_process_inputs(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_latents: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> PreprocessOutput:
        """
        Prepare inputs for decoder transformer.

        Args:
            input_ids: Token IDs (not used in standard decoding, emits warning)
            inputs_latents: Shape (batch_size, num_segments, latent_dim)
            inputs_embeds: Shape (batch_size, num_segments, hidden_size)

        Returns:
            PreprocessOutput with inputs_embeds reshaped to
            (batch_size * num_segments, 1, hidden_size)
        """
        # Mutually exclusive: can't provide both latents and embeds
        if inputs_latents is not None and inputs_embeds is not None:
            raise ValueError("Provide either inputs_latents or inputs_embeds, not both")

        # Reshape inputs for transformer processing
        if input_ids is not None:
            # For diffusion decoding: reshape token IDs
            input_ids = self.ae_utils.agg_sequence(sequence=input_ids)

        if inputs_latents is not None:
            # Standard decoding path: latents -> embeddings
            # (batch, num_segments, latent_dim) -> (batch * num_segments, 1, latent_dim)
            inputs_latents = self.ae_utils.agg_sequence(sequence=inputs_latents)
            # Project latents to hidden dimension for transformer
            # (batch * num_segments, 1, latent_dim) -> (batch * num_segments, 1, hidden_size)
            inputs_embeds = self.wte_latent(inputs_latents)
        elif inputs_embeds is not None:
            # Direct embedding input (skip latent projection)
            inputs_embeds = self.ae_utils.agg_sequence(sequence=inputs_embeds)

        return PreprocessOutput(
            input_ids=input_ids,
            inputs_latents=inputs_latents,
            inputs_embeds=inputs_embeds
        )

    def post_process_outputs(
        self,
        outputs: BaseModelOutputWithPastAndCrossAttentions,
    ) -> BaseAutoencoderOutputWithPastAndCrossAttentions:
        """
        Reshape transformer outputs back to batch dimensions.

        Args:
            outputs: Transformer outputs with last_hidden_state shape
                     (batch_size * num_segments, 1, hidden_size)

        Returns:
            Autoencoder outputs with:
            - last_tail_hidden_state: (batch_size, num_segments, hidden_size)
            - last_window_hidden_state: Same as last_tail_hidden_state (decoder has seq_len=1)
            - last_hidden_state: (batch_size, num_segments, hidden_size)
        """
        if outputs is None:
            return outputs

        last_hidden = outputs.last_hidden_state

        # Decoder processes each latent independently (seq_len=1), so last_tail and full sequence are the same
        # last_window_hidden_state is also the same since decoder has only 1 position per segment
        reshaped_hidden = (
            self.ae_utils.split_sequence(last_hidden)
            if last_hidden is not None else None
        )

        return BaseAutoencoderOutputWithPastAndCrossAttentions(
            last_tail_hidden_state=reshaped_hidden,
            last_window_hidden_state=reshaped_hidden,
            last_hidden_state=reshaped_hidden,
            past_key_values=outputs.past_key_values,
            hidden_states=(
                tuple(self.ae_utils.split_sequence(h) for h in outputs.hidden_states)
                if outputs.hidden_states is not None else None
            ),
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def init_weight_from_pretrained(self, pretrained_model: GPT2ModelBase) -> "LanguageDecoder":
        """
        Initialize decoder weights from a pretrained GPT-2 model.

        Uses the last `num_hidden_layers_decoder` layers from the pretrained model.
        Note: wte_latent is randomly initialized (projects from latent_dim).
        """
        self.wte = copy.deepcopy(pretrained_model.wte)
        self.wpe = copy.deepcopy(pretrained_model.wpe)
        self.drop = copy.deepcopy(pretrained_model.drop)
        self.h = copy.deepcopy(pretrained_model.h[-self.config.num_hidden_layers_decoder:])
        self.ln_f = copy.deepcopy(pretrained_model.ln_f)
        self.gradient_checkpointing = pretrained_model.gradient_checkpointing

        # Re-index layer indices for attention caching
        for new_idx, block in enumerate(self.h):
            block.attn.layer_idx = new_idx
            if hasattr(block, 'crossattention'):
                block.crossattention.layer_idx = new_idx
        return self

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
        return_segment: bool = True,
        **kwargs,
    ) -> Union[tuple, BaseAutoencoderOutputWithPastAndCrossAttentions]:
        """
        Forward pass for the language decoder.

        Args:
            inputs_latents: Shape (batch_size, num_segments, latent_dim)
            inputs_embeds: Shape (batch_size, num_segments, hidden_size)
            input_ids: Not used (warning emitted if provided)

        Returns:
            BaseAutoencoderOutputWithPastAndCrossAttentions
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            warnings.warn("LanguageDecoder processes inputs_latents/inputs_embeds, not input_ids")

        preprocessed = self.pre_process_inputs(
            input_ids=input_ids,
            inputs_latents=inputs_latents,
            inputs_embeds=inputs_embeds
        )

        output = super().forward(
            input_ids=None,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=preprocessed.inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        return self.post_process_outputs(outputs=output)


class LanguageDecoderLMHead(GPT2PreTrainedModel, GenerationMixin):
    """
    Language decoder with multi-head LM projection.

    Uses `window_size` separate linear heads to project each decoder hidden state
    to `window_size` vocabulary distributions.

    Processing flow:
        1. Decode latents: (batch_size, num_segments, hidden_size)
        2. Apply window_size LM heads: each produces (batch_size, num_segments, vocab_size)
        3. Interleave outputs: (batch_size, num_segments * window_size, vocab_size)
    """

    def __init__(self, config: LatentGPT2Config):
        super().__init__(config)
        self.transformer = LanguageDecoder(config=config)

        # Multi-head projection: each head predicts one position in the window
        self.multi_lm_head = nn.ModuleList([
            nn.Linear(config.n_embd, config.vocab_size, bias=False)
            for _ in range(config.window_size)
        ])
        self.post_init()

    def init_weight_from_pretrained(self, pretrained_model: GPT2LMHeadModel) -> "LanguageDecoderLMHead":
        """Initialize from pretrained GPT-2 LM model."""
        self.transformer.init_weight_from_pretrained(pretrained_model.transformer)
        # Deep copy the pretrained lm_head weights to each multi-head
        for i in range(len(self.multi_lm_head)):
            self.multi_lm_head[i] = copy.deepcopy(pretrained_model.lm_head)
        return self

    def _project_with_multi_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project decoder hidden states to vocabulary logits using multi-head mechanism.

        Args:
            hidden_states: Shape (batch_size, num_segments, hidden_size)

        Returns:
            logits: Shape (batch_size, num_segments * window_size, vocab_size)
        """
        # Each head predicts one token position within the window
        # Head 0 predicts position 0, Head 1 predicts position 1, etc.
        # This allows one latent to expand to window_size tokens
        multi_head_logits = [head(hidden_states) for head in self.multi_lm_head]

        # Interleave outputs: [head0_seg0, head1_seg0, ..., head0_seg1, head1_seg1, ...]
        # to match original token ordering
        return self.transformer.ae_utils.flatten_multi_heads_logits(logits=multi_head_logits)

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
        """
        Forward pass for decoder with multi-head LM projection.

        Args:
            inputs_latents: Shape (batch_size, num_segments, latent_dim)
            inputs_embeds: Shape (batch_size, num_segments, hidden_size)
            logits_to_keep: Number of final positions to keep (0 = all)

        Returns:
            CausalLMAutoencoderOutputWithCrossAttentions with:
            - logits: (batch_size, num_segments * window_size, vocab_size)
            - latents: Input latents
            - latent_embeds: Projected latent embeddings
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Compute latent embeddings if needed
        latent_embeds = None
        if inputs_latents is not None and inputs_embeds is None:
            latent_embeds = self.transformer.wte_latent(inputs_latents)
        else:
            latent_embeds = inputs_embeds

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

        hidden_states = transformer_outputs.last_tail_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self._project_with_multi_heads(hidden_states[:, slice_indices, :])

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return output

        return CausalLMAutoencoderOutputWithCrossAttentions(
            last_tail_hidden_state=transformer_outputs.last_tail_hidden_state if output_hidden_states else None,
            last_window_hidden_state=transformer_outputs.last_window_hidden_state if output_hidden_states else None,
            last_hidden_state=transformer_outputs.last_hidden_state if output_hidden_states else None,
            loss=None,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states if output_hidden_states else None,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
            latent_embeds=latent_embeds,
            latents=inputs_latents,
        )


# =============================================================================
# Full Autoencoder
# =============================================================================


class LanguageAutoencoder(GPT2PreTrainedModel, GenerationMixin):
    """
    Complete language autoencoder combining encoder and decoder.

    Architecture:
        - Encoder: LanguageEncoderLatentHead (text -> latents)
        - Decoder: LanguageDecoderLMHead (latents -> text)

    Full processing flow:
        1. Encoder: (batch_size, seq_len) -> (batch_size, num_segments, latent_dim)
        2. Decoder: (batch_size, num_segments, latent_dim) -> (batch_size, padded_seq_len, vocab_size)

    Note: padded_seq_len = ceil(seq_len / window_size) * window_size
    """

    # Tie encoder's latent projection to decoder's latent-to-hidden projection
    # This ensures the latent space is shared between encoder and decoder
    _tied_weights_keys = {"encoder.latent_head.weight": "decoder.transformer.wte_latent.weight"}

    def __init__(self, config: LatentGPT2Config):
        super().__init__(config)
        self.encoder = LanguageEncoderLatentHead(config=config)
        self.decoder = LanguageDecoderLMHead(config=config)
        self.post_init()

    def init_weight_from_pretrained(self, pretrained_model: GPT2LMHeadModel) -> "LanguageAutoencoder":
        """Initialize both encoder and decoder from pretrained GPT-2 LM model."""
        self.encoder.init_weight_from_pretrained(pretrained_model)
        self.decoder.init_weight_from_pretrained(pretrained_model)
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
        Encode input text to latent representations.

        Args:
            input_ids: Shape (batch_size, seq_len)

        Returns:
            CausalLMAutoencoderOutputWithCrossAttentions with latents shape
            (batch_size, num_segments, latent_dim)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return self.encoder(
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
        Decode latent representations to vocabulary logits.

        Args:
            inputs_latents: Shape (batch_size, num_segments, latent_dim)

        Returns:
            CausalLMAutoencoderOutputWithCrossAttentions with logits shape
            (batch_size, num_segments * window_size, vocab_size)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return self.decoder(
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

    def _format_dict_output(
        self,
        dict_output: CausalLMAutoencoderOutputWithCrossAttentions,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> CausalLMAutoencoderOutputWithCrossAttentions:
        """Format output based on requested hidden states and attentions."""
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
            latents=dict_output.last_tail_hidden_state,
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
        """
        Full forward pass: encode input text and decode back to vocabulary logits.

        Args:
            input_ids: Shape (batch_size, seq_len)
            labels: Shape (batch_size, seq_len) - for computing reconstruction loss
            return_encoder_decoder_res: If True, returns (ae_output, encoder_output, decoder_output)

        Returns:
            CausalLMAutoencoderOutputWithCrossAttentions with:
            - logits: (batch_size, padded_seq_len, vocab_size)
            - latents: (batch_size, num_segments, hidden_size)
            - loss: Reconstruction loss (if labels provided)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # =====================================================================
        # ENCODING PHASE
        # Input: (batch_size, seq_len) -> Output: (batch_size, num_segments, latent_dim)
        # Compresses window_size tokens into 1 latent vector per segment
        # =====================================================================
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
            output_hidden_states=True,  # Need hidden states for latent extraction
            return_dict=True,
            logits_to_keep=0,  # Keep all latents
        )

        # =====================================================================
        # DECODING PHASE
        # Input: (batch_size, num_segments, latent_dim) -> Output: (batch_size, padded_seq_len, vocab_size)
        # Expands 1 latent vector into window_size token predictions per segment
        # =====================================================================
        decoder_output = self.decode(
            input_ids=None,  # Decoder uses latents, not token IDs
            inputs_latents=encoder_output.latents,  # Latents from encoder
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
            logits_to_keep=0,  # Keep all positions
        )

        # Apply logits_to_keep slicing (0 means keep all)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = decoder_output.logits[:, slice_indices, :]

        # =====================================================================
        # LOSS COMPUTATION (if labels provided)
        # Labels must be padded to match logits length (multiple of window_size)
        # =====================================================================
        loss = None
        if labels is not None:
            # Pad labels to match the padded sequence length from decoder
            padded_labels = self.decoder.transformer.ae_utils.pad(sequence=labels)
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
            latents=encoder_output.last_tail_hidden_state,
        )

        if return_encoder_decoder_res:
            encoder_output = self._format_dict_output(
                encoder_output, output_attentions, output_hidden_states
            )
            decoder_output = self._format_dict_output(
                decoder_output, output_attentions, output_hidden_states
            )
            return (ae_output, encoder_output, decoder_output)

        return ae_output

    def _loss_function(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        vocab_size: int,
        shift_labels: int = 0,
        epsilon: float = 0.1,
        ignore_index: int = -100,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Compute label-smoothed cross-entropy loss for reconstruction.

        Args:
            logits: Shape (batch_size, seq_len, vocab_size)
            labels: Shape (batch_size, seq_len)
            vocab_size: Size of vocabulary
            shift_labels: Number of positions to shift labels (0 for reconstruction)
            epsilon: Label smoothing factor
            ignore_index: Index to ignore in loss computation

        Returns:
            Scalar loss tensor
        """
        # Optionally shift labels for next-token prediction (shift_labels=1)
        # For reconstruction (shift_labels=0), logits and labels align directly
        if shift_labels:
            logits = logits[..., :-shift_labels, :].contiguous()
            labels = labels[..., shift_labels:].contiguous()

        # Compute negative log probabilities for all vocab tokens
        log_probs = -nn.functional.log_softmax(logits, dim=-1)

        # Ensure labels have shape (batch, seq_len, 1) for gather operation
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        # Create mask for padding/ignored positions
        padding_mask = labels.eq(ignore_index)

        # Clamp negative indices to 0 to avoid gather errors (masked out anyway)
        labels = torch.clamp(labels, min=0)

        # NLL loss: gather the log prob of the correct token at each position
        nll_loss = log_probs.gather(dim=-1, index=labels)

        # Smoothed loss: sum of all log probs (uniform distribution component)
        # Use float32 for numerical stability in the sum
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        # Zero out loss for padded positions
        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Average over non-padded positions only
        num_active = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active
        smoothed_loss = smoothed_loss.sum() / (num_active * log_probs.shape[-1])

        # Label smoothing: interpolate between NLL and uniform distribution
        # epsilon=0.1 means 90% weight on correct token, 10% spread uniformly
        return (1 - epsilon) * nll_loss + epsilon * smoothed_loss


__all__ = [
    "LanguageAutoencoder",
    "LanguageDecoderLMHead",
    "LanguageEncoderLatentHead",
    "LanguageEncoder",
    "LanguageDecoder",
    "LanguageEncoderUtils",
    "LanguageDecoderUtils",
]
