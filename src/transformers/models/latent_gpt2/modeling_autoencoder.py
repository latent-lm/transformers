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
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from ...cache_utils import Cache, DynamicCache, EncoderDecoderCache
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
        wte: Optional[torch.nn.Embedding] = None,
    ):
        self._batch_size: int = None
        self._seq_len: int = None
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
        if self._padding_embed is None:
            if self._wte is None:
                raise ValueError("To get padding_embed, either provide padding_embed or wte")
            self._padding_embed = self._wte(self._padding_token)
        return self._padding_embed

    def _get_padding_token(self) -> torch.LongTensor:
        if self._padding_token is None:
            raise ValueError("padding_token is not set")
        return self._padding_token
    
    def _get_masking_embed(self) -> torch.FloatTensor:
        if self._masking_embed is None:
            if self._wte is None:
                raise ValueError("To get masking_embed, either provide masking_embed or wte")
            self._masking_embed = self._wte(self._masking_token)
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

    def pad(self, sequence: Optional[torch.Tensor] = None, token_type: str = "padding") -> torch.Tensor:
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
    
    def split_context_target(
        self,
        sequence: torch.Tensor,
        is_padding: bool = True,
    ) -> Tuple[torch.Tensor]:
        if sequence is None:
            return None
        if sequence.dim() == 2:
            context: torch.Tensor = sequence[..., :-self._window_size].contiguous()
            target: torch.Tensor = sequence[..., -self._window_size:].contiguous()
            if is_padding:
                context: torch.Tensor = self.pad(sequence=context)
        elif sequence.dim() > 2:
            context: torch.Tensor = sequence[..., :-self._window_size, :].contiguous()
            target: torch.Tensor = sequence[..., -self._window_size:, :].contiguous()
            if is_padding:
                context: torch.Tensor = self.pad(sequence=context)
        else:
            raise NotImplementedError(f"Unsupported input dimension: {sequence.dim()} with shape {sequence.shape}")
        return context, target
    
    def split_context_target_then_cat(
        self,
        sequence: torch.Tensor,
        is_padding: bool = True,
        return_context_target: bool = False,
    ) -> Tuple[torch.Tensor]:
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
        padded_sequence = self.pad(sequence=sequence, token_type=SequenceWindowUtilsBase.TOKEN_TYPE_PADDING)
        self._segment_num = padded_sequence.shape[1] // self._window_size
        return padded_sequence.view(self._batch_size * self._segment_num, self._window_size, *padded_sequence.shape[2:])


@auto_docstring(custom_intro="Utility class for language decoding with fixed-size window disaggregation.")
class LanguageDecoderUtils(SequenceWindowUtilsBase):
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
        padded_sequence = self.pad(sequence=sequence, token_type=token_type)
        self._segment_num = padded_sequence.shape[1] // self._window_size
        return padded_sequence.view(self._batch_size * self._segment_num, self._window_size, *padded_sequence.shape[2:])

    def agg_sequence_by_window_size(self, sequence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Aggregate a sequence into window_size-length segments, padding if necessary.

        Args:
            sequence: The sequence to aggregate. Shape (batch_size, seq_len, ...).

        Returns:
            Aggregated sequence with shape (batch_size * segment_num, window_size, ...).
        """
        return self._agg_sequence_by_window(sequence, token_type=SequenceWindowUtilsBase.TOKEN_TYPE_PADDING)

    def agg_sequence_mask_diffusion(self, sequence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Aggregate a masked sequence into window_size-length segments for diffusion decoding.

        Used in mask diffusion models where the decoder receives masked token IDs
        that need to be reshaped to match the windowed structure. Pads with mask
        tokens if sequence length is not divisible by window_size.

        Args:
            sequence: The masked token sequence. Shape (batch_size, seq_len, ...).

        Returns:
            Aggregated sequence with shape (batch_size * segment_num, window_size, ...).
        """
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
        """
        Pad the given sequence to be a multiple of window_size.

        If sequence is None, return None.
        For 2D sequence (batch_size, seq_len), pad with padding_token.
        For 3D sequence (batch_size, seq_len, embedding_size), pad with padding_embed.

        Returns:
            torch.Tensor: Padded sequence with length divisible by window_size.
        """
        return super().pad(sequence=sequence, token_type=SequenceWindowUtilsBase.TOKEN_TYPE_PADDING)

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

        Args:
            input_ids (Optional[torch.LongTensor]): The input ids.
            inputs_embeds (Optional[torch.FloatTensor]): The input embeddings.

        Returns:
            A tuple containing the pre-processed input ids and embeddings.
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

        Args:
            outputs: The output of the language encoder.

        Returns:
            The processed output of the language encoder.
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
        custom_intro="Forward pass for the language encoder with LM head.",
        checkpoint="GPT2PreTrainedModel",
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
            output_hidden_states=True,
            return_dict=True,
        )
        # Only use the last hidden_state of the last layer as latent
        latents = transformer_outputs.last_tail_hidden_state
        
        # Project the latents to logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.latent_head(latents[:, slice_indices, :])

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
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
            latent_embeds=transformer_outputs.last_tail_hidden_state,
            latents=logits,
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

        Args:
            input_ids (Optional[torch.LongTensor]): The input ids.
            inputs_embeds (Optional[torch.FloatTensor]): The input embeddings.

        Returns:
            A PreprocessOutput containing the pre-processed input ids and embeddings.
        """
        if input_ids is not None:
            input_ids = self.ae_utils.agg_sequence(sequence=input_ids)
        if inputs_latents is not None:
            inputs_latents = self.ae_utils.agg_sequence(sequence=inputs_latents)
            # if input_ids is not None:
            #     raise ValueError(f"input_ids and inputs_latents are provided at the same time, only one of them is accepted.")
        if inputs_embeds is not None:
            inputs_embeds = self.ae_utils.agg_sequence(sequence=inputs_embeds)
            # if input_ids is not None:
            #     raise ValueError(f"input_ids and inputs_embeds are provided at the same time, only one of them is accepted.")
            if inputs_latents is not None:
                raise ValueError(f"inputs_latents and inputs_embeds are provided at the same time, only one of them is accepted.")

        if inputs_latents is not None:
            inputs_embeds = self.wte_latent(inputs_latents)
        # if input_ids is not None:
        #     inputs_embeds = self.wte_latent(input_ids)
        return PreprocessOutput(input_ids=input_ids, inputs_latents=inputs_latents, inputs_embeds=inputs_embeds)
    
    def post_process_outputs(
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

    # @auto_docstring
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
    ) -> Union[tuple, CausalLMAutoencoderOutputWithCrossAttentions]:
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
        custom_intro="Forward pass for the language decoder with LM head.",
        checkpoint="GPT2PreTrainedModel",
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
        inputs_latents (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Latent representations to be decoded. These are typically output from the encoder component of the
            autoencoder and represent compressed semantic information to be expanded back into token sequences.
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            LangaugeDeocder and LanuageDecoderLMHead don't process input_ides, please use input_latents instead.
        labels (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
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
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
            latent_embeds=inferred_inputs_embeds,
            latents=inputs_latents,
        )

class LanguageAutoencoder(GPT2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"encoder.latent_head.weight": "decoder.transformer.wte_latent.weight"}
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
            latents=encoder_output.last_tail_hidden_state,
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
