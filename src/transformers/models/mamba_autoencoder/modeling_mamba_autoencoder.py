# coding=utf-8
# Copyright 2024 state-spaces/mamba org and HuggingFace Inc. team.
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
"""PyTorch MAMBA Autoencoder models.

This module implements language autoencoders based on the MAMBA (Selective State Space Model)
architecture. The autoencoder compresses input text into latent representations and reconstructs
the original text from these latents.

Architecture Overview:
    - **MambaEncoderBase**: Base encoder using MAMBA blocks to process input sequences.
    - **MambaDecoderBase**: Base decoder using MAMBA blocks to decode latent representations.
    - **MambaEncoder**: Encoder with windowed sequence aggregation.
    - **MambaEncoderLatentHead**: Encoder with projection to latent space.
    - **MambaDecoder**: Decoder that expands latents back to token sequences.
    - **MambaDecoderLMHead**: Decoder with multi-head language modeling heads.
    - **MambaAutoencoder**: Complete autoencoder combining encoder and decoder.

Key Concepts:
    - **Window Size**: Sequences are processed in fixed-size windows. The encoder aggregates
      `window_size` tokens into a single latent vector, and the decoder expands each latent
      back to `window_size` tokens using multiple LM heads.
    - **Latent Projection**: The encoder projects hidden states to `latent_dim`, and the
      decoder projects latents back to `hidden_size` via a learned linear transformation.
"""

import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from ...generation import GenerationMixin
from ...modeling_outputs import (
    PreprocessOutput,
    BaseAutoencoderOutputWithPastAndCrossAttentions,
    CausalLMAutoencoderOutputWithCrossAttentions,
)
from ...utils import (
    ModelOutput,
    auto_docstring,
    logging,
)
from .configuration_mamba import LatentMambaConfig
from ...models.mamba.modeling_mamba import (
    MambaPreTrainedModel,
    MambaModel,
    MambaForCausalLM,
    MambaBlock,
    MambaRMSNorm,
    MambaCache,
    MambaOutput,
)


logger = logging.get_logger(__name__)


# =============================================================================
# Sequence Window Utilities (adapted from latent_gpt2)
# =============================================================================


class SequenceWindowUtilsBase:
    """Base class for sequence windowing utilities with shared properties and padding methods."""
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
        embeddings: Optional[torch.nn.Embedding] = None,
    ):
        self._batch_size: int = None
        self._seq_len: int = None
        self._segment_num: int = None
        self._window_size: int = window_size
        self._padding_token = padding_token
        self._padding_embed = padding_embed
        self._masking_token = masking_token
        self._masking_embed = masking_embed
        self._embeddings: torch.nn.Embedding = embeddings

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
        if self._padding_embed is None:
            if self._embeddings is None:
                raise ValueError("To get padding_embed, either provide padding_embed or embeddings")
            self._padding_embed = self._embeddings(
                torch.tensor(self._padding_token, device=self._embeddings.weight.device)
            )
        return self._padding_embed

    def _get_padding_token(self) -> torch.LongTensor:
        if self._padding_token is None:
            raise ValueError("padding_token is not set")
        return self._padding_token

    def _get_masking_embed(self) -> torch.FloatTensor:
        if self._masking_embed is None:
            if self._embeddings is None:
                raise ValueError("To get masking_embed, either provide masking_embed or embeddings")
            self._masking_embed = self._embeddings(
                torch.tensor(self._masking_token, device=self._embeddings.weight.device)
            )
        return self._masking_embed

    def _get_masking_token(self) -> torch.LongTensor:
        if self._masking_token is None:
            raise ValueError("masking_token is not set")
        return self._masking_token

    def _get_used_token(self, token_type: str):
        if token_type == SequenceWindowUtilsBase.TOKEN_TYPE_PADDING:
            return self._get_padding_token()
        elif token_type == SequenceWindowUtilsBase.TOKEN_TYPE_MASKING:
            return self._get_masking_token()
        else:
            raise ValueError(f"token_type, {token_type}, isn't supported, only support {SequenceWindowUtilsBase.SUPPORTED_TOKEN_TYPES}")

    def _get_used_embed(self, token_type: str):
        if token_type == SequenceWindowUtilsBase.TOKEN_TYPE_PADDING:
            return self._get_padding_embed()
        elif token_type == SequenceWindowUtilsBase.TOKEN_TYPE_MASKING:
            return self._get_masking_embed()
        else:
            raise ValueError(f"token_type, {token_type}, isn't supported, only support {SequenceWindowUtilsBase.SUPPORTED_TOKEN_TYPES}")

    def _pad(self, sequence: Optional[torch.Tensor] = None, token_type: str = "padding") -> torch.Tensor:
        """
        Pad the given sequence to be a multiple of window_size.

        If sequence is None, return None.
        For 2D sequence (batch_size, seq_len), pad with padding_token.
        For 3D sequence (batch_size, seq_len, embedding_size), pad with padding_embed.

        Returns:
            torch.Tensor: Padded sequence with length divisible by window_size.
        """
        if sequence is None:
            return sequence
        batch_size: int = sequence.shape[0]
        seq_len: int = sequence.shape[1]
        pad_len: int = (self._window_size - (seq_len % self._window_size)) % self._window_size

        if pad_len == 0:
            return sequence

        if sequence.dim() == 2:
            pad = torch.full((batch_size, pad_len), self._get_used_token(token_type=token_type),
                           dtype=sequence.dtype, device=sequence.device)
        elif sequence.dim() == 3:
            pad = self._get_used_embed(token_type=token_type).expand(batch_size, pad_len, -1).to(
                dtype=sequence.dtype, device=sequence.device)
        else:
            raise NotImplementedError(f"Unsupported input dimension: {sequence.dim()} with shape {sequence.shape}")
        return torch.cat((sequence, pad), dim=1)

    def split_sequence(self, sequence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Split a windowed sequence back to batch dimensions."""
        if sequence is None:
            return sequence
        return sequence.reshape(self._batch_size, self._segment_num * sequence.shape[1], *sequence.shape[2:])


class MambaEncoderUtils(SequenceWindowUtilsBase):
    """Encoder utilities that aggregate sequences into fixed-size windows."""

    def agg_sequence(self, sequence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Aggregate a sequence into fixed-size windows.

        Args:
            sequence: The sequence to aggregate. Shape (batch_size, seq_len) or (batch_size, seq_len, embed_dim).

        Returns:
            Aggregated sequence with shape (batch_size * segment_num, window_size, ...).
        """
        if sequence is None:
            return sequence
        self._batch_size = sequence.shape[0]
        self._seq_len = sequence.shape[1]
        padded_sequence = self._pad(sequence=sequence, token_type=SequenceWindowUtilsBase.TOKEN_TYPE_PADDING)
        self._segment_num = padded_sequence.shape[1] // self._window_size
        return padded_sequence.view(self._batch_size * self._segment_num, self._window_size, *padded_sequence.shape[2:])


class MambaDecoderUtils(SequenceWindowUtilsBase):
    """Decoder utilities that handle single-element and window-based sequence operations."""

    def agg_sequence(self, sequence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Aggregate a sequence into single-element segments.

        Args:
            sequence: The sequence to aggregate. Shape (batch_size, seq_len, ...).

        Returns:
            Aggregated sequence with shape (batch_size * seq_len, 1, ...).
        """
        if sequence is None:
            return sequence
        self._batch_size = sequence.shape[0]
        self._seq_len = sequence.shape[1]
        self._segment_num = sequence.shape[1]
        return sequence.view(self._batch_size * self._segment_num, 1, *sequence.shape[2:])

    def _agg_sequence_by_window(
        self,
        sequence: Optional[torch.Tensor] = None,
        token_type: str = SequenceWindowUtilsBase.TOKEN_TYPE_PADDING,
    ) -> torch.Tensor:
        """
        Internal method to aggregate a sequence into window_size-length segments.

        Args:
            sequence: The sequence to aggregate. Shape (batch_size, seq_len, ...).
            token_type: Token type for padding ("padding" or "masking").

        Returns:
            Aggregated sequence with shape (batch_size * segment_num, window_size, ...).
        """
        if sequence is None:
            return sequence
        self._batch_size = sequence.shape[0]
        self._seq_len = sequence.shape[1]
        padded_sequence = self._pad(sequence=sequence, token_type=token_type)
        self._segment_num = padded_sequence.shape[1] // self._window_size
        return padded_sequence.view(self._batch_size * self._segment_num, self._window_size, *padded_sequence.shape[2:])

    def agg_sequence_by_window_size(self, sequence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Aggregate a sequence into window_size-length segments, padding if necessary."""
        return self._agg_sequence_by_window(sequence, token_type=SequenceWindowUtilsBase.TOKEN_TYPE_PADDING)

    def agg_sequence_mask_diffusion(self, sequence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Aggregate a masked sequence into window_size-length segments for diffusion decoding."""
        return self._agg_sequence_by_window(sequence, token_type=SequenceWindowUtilsBase.TOKEN_TYPE_MASKING)

    def flatten_multi_heads_logits(self, logits: List[torch.Tensor]) -> torch.Tensor:
        """
        Flatten multi-head logits into a single sequence.

        Args:
            logits: List of tensors of shape (batch_size, segment_num, vocab_size) with length window_size.

        Returns:
            Flattened logits with shape (batch_size, segment_num * window_size, vocab_size).
        """
        flatten_seq_shape = (self._batch_size, self._segment_num * self._window_size, *logits[0].shape[2:])
        return torch.stack(logits, dim=2).view(flatten_seq_shape)

    def pad(self, sequence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pad the given sequence to be a multiple of window_size."""
        return self._pad(sequence=sequence, token_type=SequenceWindowUtilsBase.TOKEN_TYPE_PADDING)


# =============================================================================
# Base Encoder and Decoder Models
# =============================================================================


@auto_docstring(
    custom_intro="Base encoder model for MAMBA-based language encoding with configurable number of layers.",
)
class MambaEncoderBase(MambaPreTrainedModel):
    """
    Base encoder using MAMBA blocks. This model is similar to MambaModel but uses only
    `num_hidden_layers_encoder` layers instead of all layers.
    """

    def __init__(self, config: LatentMambaConfig):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size

        # Token embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        # MAMBA blocks for encoder (first N layers)
        self.layers = nn.ModuleList([
            MambaBlock(config, layer_idx=idx)
            for idx in range(config.num_hidden_layers_encoder)
        ])

        # Final normalization
        self.norm_f = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[MambaCache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, MambaOutput]:
        """
        Forward pass through the encoder base model.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        # MAMBA doesn't use cache during encoding for autoencoder
        cache_params = None

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None

        for mixer_block in self.layers:
            hidden_states = mixer_block(
                hidden_states,
                cache_params=cache_params,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return MambaOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )


@auto_docstring(
    custom_intro="Base decoder model for MAMBA-based language decoding with configurable number of layers.",
)
class MambaDecoderBase(MambaPreTrainedModel):
    """
    Base decoder using MAMBA blocks. This model processes latent representations
    through MAMBA blocks to produce hidden states for token prediction.
    """

    def __init__(self, config: LatentMambaConfig):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size

        # Project from latent_dim to hidden_size (instead of token embedding)
        self.embeddings_latent = nn.Linear(config.latent_dim, config.hidden_size, bias=False)

        # Token embeddings (for diffusion decoder)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        # MAMBA blocks for decoder (last N layers)
        self.layers = nn.ModuleList([
            MambaBlock(config, layer_idx=idx)
            for idx in range(config.num_hidden_layers_decoder)
        ])

        # Final normalization
        self.norm_f = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

        # Initialize embeddings_latent as identity matrix if dimensions match
        if config.latent_dim == config.hidden_size:
            with torch.no_grad():
                self.embeddings_latent.weight.copy_(torch.eye(config.hidden_size))

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[MambaCache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, MambaOutput]:
        """
        Forward pass through the decoder base model.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            if input_ids is not None:
                inputs_embeds = self.embeddings(input_ids)
            else:
                raise ValueError("You must specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        # MAMBA doesn't use cache during decoding for autoencoder
        cache_params = None

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None

        for mixer_block in self.layers:
            hidden_states = mixer_block(
                hidden_states,
                cache_params=cache_params,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return MambaOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )


# =============================================================================
# Encoder with Window Aggregation
# =============================================================================


@auto_docstring(
    custom_intro="MAMBA encoder model with window-based sequence aggregation for latent compression.",
    custom_args="window_size (`int`): The window size for aggregating input sequences into segments.",
)
class MambaEncoder(MambaEncoderBase):
    """
    MAMBA encoder that aggregates input sequences into fixed-size windows,
    processing each window to produce latent representations.
    """

    def __init__(self, config: LatentMambaConfig):
        super().__init__(config)
        self.ae_utils = MambaEncoderUtils(
            window_size=config.window_size,
            padding_token=config.pad_token_id,
            padding_embed=None,
            embeddings=self.embeddings,
        )
        # Initialize weights and apply final processing
        self.post_init()

    def pre_process_inputs(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> PreprocessOutput:
        """
        Pre-processes and segments the inputs for the encoder.

        Args:
            input_ids: The input ids.
            inputs_embeds: The input embeddings.

        Returns:
            A PreprocessOutput containing the pre-processed input ids and embeddings.
        """
        if input_ids is not None:
            input_ids = self.ae_utils.agg_sequence(sequence=input_ids)
        if inputs_embeds is not None:
            inputs_embeds = self.ae_utils.agg_sequence(sequence=inputs_embeds)
        return PreprocessOutput(input_ids=input_ids, inputs_embeds=inputs_embeds)

    def post_process_outputs(
        self,
        outputs: MambaOutput,
    ) -> BaseAutoencoderOutputWithPastAndCrossAttentions:
        """
        Post-processes and segments the outputs of the encoder.

        Args:
            outputs: The output of the encoder.

        Returns:
            The processed output of the encoder.
        """
        if outputs is None:
            return outputs
        return BaseAutoencoderOutputWithPastAndCrossAttentions(
            # Only keep the last hidden_state of the last layer
            last_tail_hidden_state=self.ae_utils.split_sequence(
                sequence=outputs.last_hidden_state[:, -1:, ...]
            ),
            # Only keep the last window_size hidden_states of the last layer
            last_window_hidden_state=self.ae_utils.split_sequence(
                sequence=outputs.last_hidden_state[:, -self.config.window_size:, ...]
            ),
            last_hidden_state=self.ae_utils.split_sequence(sequence=outputs.last_hidden_state),
            past_key_values=None,
            hidden_states=tuple(
                self.ae_utils.split_sequence(sequence=hidden_state)
                for hidden_state in outputs.hidden_states
            ) if outputs.hidden_states is not None else None,
            attentions=None,
            cross_attentions=None,
        )

    def init_weight_from_pretrained(self, pretrained_model: MambaModel) -> "MambaEncoder":
        """
        Initialize the encoder weights from a pre-trained MAMBA model.

        Args:
            pretrained_model: The pre-trained MAMBA model to copy weights from.

        Returns:
            Self for method chaining.
        """
        self.embeddings = copy.deepcopy(pretrained_model.embeddings)
        self.layers = copy.deepcopy(
            nn.ModuleList(list(pretrained_model.layers[:self.config.num_hidden_layers_encoder]))
        )
        self.norm_f = copy.deepcopy(pretrained_model.norm_f)
        self.gradient_checkpointing = pretrained_model.gradient_checkpointing

        # Re-index blocks to match the new layer positions
        for new_idx, block in enumerate(self.layers):
            block.layer_idx = new_idx
            block.mixer.layer_idx = new_idx
        return self

    @auto_docstring(
        custom_args="return_segment (`bool`, *optional*, defaults to `True`): Whether to return segmented outputs.",
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[MambaCache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        return_segment: bool = True,
        **kwargs,
    ) -> Union[Tuple, BaseAutoencoderOutputWithPastAndCrossAttentions]:
        """Forward pass through the encoder with window aggregation."""
        pre_process_res: PreprocessOutput = self.pre_process_inputs(
            input_ids=input_ids, inputs_embeds=inputs_embeds
        )
        output = super().forward(
            input_ids=pre_process_res.input_ids,
            inputs_embeds=pre_process_res.inputs_embeds,
            cache_params=cache_params,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            attention_mask=attention_mask,
            **kwargs,
        )
        return self.post_process_outputs(outputs=output)


# =============================================================================
# Encoder with Latent Head
# =============================================================================


class MambaEncoderLatentHead(MambaPreTrainedModel, GenerationMixin):
    """
    MAMBA encoder with a latent projection head that compresses hidden states
    into latent representations.

    The encoder processes input tokens through MAMBA blocks, aggregates them
    into windows, and projects the final hidden state of each window to the
    latent space.
    """

    def __init__(self, config: LatentMambaConfig):
        super().__init__(config)
        self.transformer = MambaEncoder(config=config)
        # Project from hidden_state to latent_dim
        self.latent_head = nn.Linear(config.hidden_size, config.latent_dim, bias=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Initialize latent_head as identity matrix if dimensions match
        if config.hidden_size == config.latent_dim:
            with torch.no_grad():
                self.latent_head.weight.copy_(torch.eye(config.hidden_size))

    def init_weight_from_pretrained(self, pretrained_model: MambaForCausalLM) -> "MambaEncoderLatentHead":
        """
        Initialize the encoder from a pre-trained MAMBA model.

        Args:
            pretrained_model: The pre-trained MambaForCausalLM model.

        Returns:
            Self for method chaining.
        """
        self.transformer.init_weight_from_pretrained(pretrained_model=pretrained_model.backbone)
        self.lm_head = copy.deepcopy(pretrained_model.lm_head)
        return self

    @auto_docstring(
        custom_intro="Forward pass for the MAMBA encoder with latent head.",
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[MambaCache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMAutoencoderOutputWithCrossAttentions]:
        """
        Forward pass through the encoder to produce latent representations.

        Args:
            input_ids: Input token IDs.
            inputs_embeds: Pre-computed input embeddings.
            labels: Not used in encoding, included for API consistency.
            logits_to_keep: Number of logits to keep from the end.

        Returns:
            Encoder outputs containing latent representations.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            cache_params=cache_params,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            attention_mask=attention_mask,
            **kwargs,
        )

        # Only use the last hidden_state of the last layer as latent
        latents = transformer_outputs.last_tail_hidden_state

        # Project the latents
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.latent_head(latents[:, slice_indices, :])

        loss = None

        if not return_dict:
            output = (logits,) + (transformer_outputs.last_hidden_state,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMAutoencoderOutputWithCrossAttentions(
            last_tail_hidden_state=transformer_outputs.last_tail_hidden_state,
            last_window_hidden_state=transformer_outputs.last_window_hidden_state,
            last_hidden_state=transformer_outputs.last_hidden_state,
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=transformer_outputs.hidden_states,
            attentions=None,
            cross_attentions=None,
            latent_embeds=transformer_outputs.last_tail_hidden_state,
            latents=logits,
        )


# =============================================================================
# Decoder with Window Disaggregation
# =============================================================================


@auto_docstring(
    custom_intro="MAMBA decoder model for decoding latent representations back to text.",
    custom_args="window_size (`int`): The window size for disaggregating latent representations back to sequences.",
)
class MambaDecoder(MambaDecoderBase):
    """
    MAMBA decoder that expands latent representations back to token sequences.
    Each latent vector is expanded to `window_size` token predictions.
    """

    def __init__(self, config: LatentMambaConfig):
        super().__init__(config)
        self.ae_utils = MambaDecoderUtils(
            window_size=config.window_size,
            padding_token=config.pad_token_id,
            masking_token=getattr(config, "mask_token_id", config.pad_token_id),
            embeddings=self.embeddings,
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
        Pre-processes and segments the inputs for the decoder.

        Args:
            input_ids: The input ids (not used in standard decoder).
            inputs_latents: The latent representations to decode.
            inputs_embeds: Pre-computed input embeddings.

        Returns:
            A PreprocessOutput containing the pre-processed embeddings.
        """
        if input_ids is not None:
            input_ids = self.ae_utils.agg_sequence(sequence=input_ids)
        if inputs_latents is not None:
            inputs_latents = self.ae_utils.agg_sequence(sequence=inputs_latents)
        if inputs_embeds is not None:
            inputs_embeds = self.ae_utils.agg_sequence(sequence=inputs_embeds)
            if inputs_latents is not None:
                raise ValueError("inputs_latents and inputs_embeds are provided at the same time, only one is accepted.")

        if inputs_latents is not None:
            inputs_embeds = self.embeddings_latent(inputs_latents)

        return PreprocessOutput(input_ids=input_ids, inputs_latents=inputs_latents, inputs_embeds=inputs_embeds)

    def post_process_outputs(
        self,
        outputs: MambaOutput,
    ) -> BaseAutoencoderOutputWithPastAndCrossAttentions:
        """
        Post-processes and segments the outputs of the decoder.

        Args:
            outputs: The output of the decoder.

        Returns:
            The processed output of the decoder.
        """
        if outputs is None:
            return outputs
        return BaseAutoencoderOutputWithPastAndCrossAttentions(
            # Only keep the last hidden_state of the last layer
            last_tail_hidden_state=self.ae_utils.split_sequence(
                sequence=outputs.last_hidden_state[:, -1:, ...]
            ),
            # Only keep the last window_size hidden_states of the last layer
            last_window_hidden_state=self.ae_utils.split_sequence(
                sequence=outputs.last_hidden_state[:, -self.config.window_size:, ...]
            ),
            last_hidden_state=self.ae_utils.split_sequence(sequence=outputs.last_hidden_state),
            past_key_values=None,
            hidden_states=tuple(
                self.ae_utils.split_sequence(sequence=hidden_state)
                for hidden_state in outputs.hidden_states
            ) if outputs.hidden_states is not None else None,
            attentions=None,
            cross_attentions=None,
        )

    def init_weight_from_pretrained(self, pretrained_model: MambaModel) -> "MambaDecoder":
        """
        Initialize the decoder weights from a pre-trained MAMBA model.

        Args:
            pretrained_model: The pre-trained MAMBA model to copy weights from.

        Returns:
            Self for method chaining.
        """
        self.embeddings = copy.deepcopy(pretrained_model.embeddings)
        self.layers = copy.deepcopy(
            nn.ModuleList(list(pretrained_model.layers[-self.config.num_hidden_layers_decoder:]))
        )
        self.norm_f = copy.deepcopy(pretrained_model.norm_f)
        self.gradient_checkpointing = pretrained_model.gradient_checkpointing

        # Re-index blocks to match the new layer positions
        for new_idx, block in enumerate(self.layers):
            block.layer_idx = new_idx
            block.mixer.layer_idx = new_idx
        return self

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_latents: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[MambaCache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        return_segment: bool = True,
        **kwargs,
    ) -> Union[Tuple, BaseAutoencoderOutputWithPastAndCrossAttentions]:
        """Forward pass through the decoder."""
        if input_ids is not None:
            import warnings
            warnings.warn("Decoder only processes inputs_latents and inputs_embeds but not input_ids")

        pre_process_res: PreprocessOutput = self.pre_process_inputs(
            input_ids=input_ids, inputs_latents=inputs_latents, inputs_embeds=inputs_embeds
        )
        output = super().forward(
            input_ids=None,
            inputs_embeds=pre_process_res.inputs_embeds,
            cache_params=cache_params,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            attention_mask=attention_mask,
            **kwargs,
        )
        return self.post_process_outputs(outputs=output)


# =============================================================================
# Decoder with Multi-Head LM
# =============================================================================


class MambaDecoderLMHead(MambaPreTrainedModel, GenerationMixin):
    """
    MAMBA decoder with multiple language modeling heads.

    Each latent vector is expanded to `window_size` token predictions using
    separate LM heads, allowing the model to predict all tokens in a window
    simultaneously.
    """

    def __init__(self, config: LatentMambaConfig):
        super().__init__(config)
        self.transformer = MambaDecoder(config=config)

        # Multi-head projection mechanism using window_size heads
        self.multi_lm_head = nn.ModuleList([
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            for _ in range(config.window_size)
        ])

        # Initialize weights and apply final processing
        self.post_init()

    def init_weight_from_pretrained(self, pretrained_model: MambaForCausalLM) -> "MambaDecoderLMHead":
        """
        Initialize the decoder from a pre-trained MAMBA model.

        Args:
            pretrained_model: The pre-trained MambaForCausalLM model.

        Returns:
            Self for method chaining.
        """
        self.transformer.init_weight_from_pretrained(pretrained_model=pretrained_model.backbone)
        for i in range(len(self.multi_lm_head)):
            self.multi_lm_head[i] = copy.deepcopy(pretrained_model.lm_head)
        return self

    def _project_with_multi_heads(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Project decoder hidden states to logits using multi-head mechanism.

        Args:
            latents: Shape (batch_size, sequence_length, hidden_size)

        Returns:
            logits: Shape (batch_size, sequence_length * window_size, vocab_size)
        """
        multi_head_logits: List[torch.Tensor] = []
        for i in range(self.config.window_size):
            multi_head_logits.append(self.multi_lm_head[i](latents))
        return self.transformer.ae_utils.flatten_multi_heads_logits(logits=multi_head_logits)

    @auto_docstring(
        custom_intro="Forward pass for the MAMBA decoder with multi-head LM.",
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_latents: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[MambaCache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMAutoencoderOutputWithCrossAttentions]:
        """
        Forward pass through the decoder to produce token logits.

        Args:
            inputs_latents: Latent representations to decode.
            inputs_embeds: Pre-computed input embeddings.
            labels: Not used in decoding, included for API consistency.
            logits_to_keep: Number of logits to keep from the end.

        Returns:
            Decoder outputs containing token logits.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_latents is not None and inputs_embeds is None:
            inferred_inputs_embeds = self.transformer.embeddings_latent(inputs_latents)
        else:
            inferred_inputs_embeds = inputs_embeds

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            inputs_latents=inputs_latents,
            inputs_embeds=inputs_embeds,
            cache_params=cache_params,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            attention_mask=attention_mask,
            **kwargs,
        )
        latents = transformer_outputs.last_tail_hidden_state

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self._project_with_multi_heads(latents[:, slice_indices, :])

        loss = None

        if not return_dict:
            output = (logits,) + (transformer_outputs.last_hidden_state,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMAutoencoderOutputWithCrossAttentions(
            last_tail_hidden_state=transformer_outputs.last_tail_hidden_state,
            last_window_hidden_state=transformer_outputs.last_window_hidden_state,
            last_hidden_state=transformer_outputs.last_hidden_state,
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=transformer_outputs.hidden_states,
            attentions=None,
            cross_attentions=None,
            latent_embeds=inferred_inputs_embeds,
            latents=inputs_latents,
        )


# =============================================================================
# Complete Autoencoder
# =============================================================================


@auto_docstring(
    custom_intro="""
    Complete MAMBA-based language autoencoder combining encoder and decoder.

    This model implements a full autoencoder architecture for text compression and
    reconstruction. The encoder compresses input text into latent representations,
    and the decoder reconstructs the text using multiple LM heads.

    Architecture Overview:
        - **Encoder** (`MambaEncoderLatentHead`): Processes input tokens through
          MAMBA blocks, then projects to latent space via `latent_head`.
          Each `window_size` tokens are compressed into a single latent vector.

        - **Decoder** (`MambaDecoderLMHead`): Takes latent vectors, processes them
          through MAMBA blocks, and applies multiple LM heads to predict the original
          tokens (one head per position in the window).

    Weight Tying:
        The encoder's `latent_head` and decoder's `embeddings_latent` can share weights,
        ensuring consistent latent space representations between encoding and decoding.
    """,
)
class MambaAutoencoder(MambaPreTrainedModel, GenerationMixin):
    """
    Complete MAMBA autoencoder for text compression and reconstruction.

    Attributes:
        encoder: The encoder model that compresses text to latent representations.
        decoder: The decoder that reconstructs text from latents using multi-head LM.
    """

    _tied_weights_keys = {"encoder.latent_head.weight": "decoder.transformer.embeddings_latent.weight"}

    def __init__(self, config: LatentMambaConfig):
        super().__init__(config)
        self.encoder: MambaEncoderLatentHead = MambaEncoderLatentHead(config=config)
        self.decoder: MambaDecoderLMHead = MambaDecoderLMHead(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    def init_weight_from_pretrained(self, pretrained_model: MambaForCausalLM) -> "MambaAutoencoder":
        """
        Initialize both encoder and decoder from a pre-trained MAMBA model.

        Args:
            pretrained_model: The pre-trained MambaForCausalLM model.

        Returns:
            Self for method chaining.
        """
        self.encoder.init_weight_from_pretrained(pretrained_model=pretrained_model)
        self.decoder.init_weight_from_pretrained(pretrained_model=pretrained_model)
        return self

    def encode(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[MambaCache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMAutoencoderOutputWithCrossAttentions]:
        """
        Encode input text into latent representations.

        Args:
            input_ids: Input token IDs to encode.
            inputs_embeds: Pre-computed embeddings instead of input_ids.

        Returns:
            Encoder outputs containing latent representations.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_output = self.encoder(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            cache_params=cache_params,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            attention_mask=attention_mask,
            **kwargs,
        )
        return encoder_output

    def decode(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_latents: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[MambaCache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMAutoencoderOutputWithCrossAttentions]:
        """
        Decode latent representations back to token logits.

        Args:
            inputs_latents: Latent representations to decode.
            inputs_embeds: Pre-computed embeddings to use instead of latents.

        Returns:
            Decoder outputs containing token logits.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        decoder_output = self.decoder(
            input_ids=input_ids,
            inputs_latents=inputs_latents,
            inputs_embeds=inputs_embeds,
            cache_params=cache_params,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            attention_mask=attention_mask,
            **kwargs,
        )
        return decoder_output

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[MambaCache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        return_encoder_decoder_res: bool = False,
        **kwargs,
    ) -> Union[Tuple, CausalLMAutoencoderOutputWithCrossAttentions]:
        """
        Complete forward pass through the autoencoder (encode + decode).

        Args:
            input_ids: Input token IDs to encode.
            inputs_embeds: Pre-computed input embeddings.
            labels: Target token IDs for computing reconstruction loss.
            return_encoder_decoder_res: If True, returns tuple of (ae_output, encoder_output, decoder_output).

        Returns:
            Autoencoder outputs containing loss, logits, and latent representations.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_output = self.encoder(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            cache_params=None,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True,
            cache_position=None,
            attention_mask=None,
        )

        decoder_output = self.decoder(
            input_ids=None,
            inputs_latents=encoder_output.latents,
            inputs_embeds=None,
            cache_params=None,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=None,
            attention_mask=None,
        )

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = decoder_output.logits[:, slice_indices, :]

        loss = None
        if labels is not None:
            # Pad labels to make it divisible by window_size
            padded_labels: torch.LongTensor = self.decoder.transformer.ae_utils.pad(sequence=labels)
            # Compute loss
            loss = self.loss_function(
                logits=logits,
                labels=padded_labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + (decoder_output.last_hidden_state,)
            return ((loss,) + output) if loss is not None else output

        ae_output = CausalLMAutoencoderOutputWithCrossAttentions(
            last_tail_hidden_state=decoder_output.last_tail_hidden_state,
            last_window_hidden_state=decoder_output.last_window_hidden_state,
            last_hidden_state=decoder_output.last_hidden_state,
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=decoder_output.hidden_states,
            attentions=None,
            cross_attentions=None,
            latents=encoder_output.last_tail_hidden_state,
        )

        if return_encoder_decoder_res:
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
            logits: Predicted token logits.
            labels: Target token IDs.
            vocab_size: Size of the vocabulary.
            shift_labels: Number of positions to shift labels.
            epsilon: Label smoothing factor.
            ignore_index: Label value to ignore.

        Returns:
            Scalar loss value.
        """
        # Either do not shift (for reconstruction) or shift by the window_size
        if shift_labels:
            logits = logits[..., :-shift_labels, :].contiguous()
            labels = labels[..., shift_labels:].contiguous()

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(ignore_index)
        # Clamp labels to valid range
        labels = torch.clamp(labels, min=0, max=vocab_size - 1)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - epsilon) * nll_loss + epsilon * smoothed_loss


__all__ = [
    "MambaAutoencoder",
    "MambaDecoderLMHead",
    "MambaEncoderLatentHead",
    "MambaEncoder",
    "MambaDecoder",
    "MambaEncoderBase",
    "MambaDecoderBase",
    "LatentMambaConfig",
]
