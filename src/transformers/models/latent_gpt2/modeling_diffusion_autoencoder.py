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
PyTorch Diffusion Autoencoder models based on GPT-2 architecture.

This module implements a diffusion-based language autoencoder that uses masked diffusion
for decoding latent representations back to text. The architecture consists of:

1. **LanguageDiffusionDecoder**: A GPT-2 based decoder that takes latent representations
   and partially/fully masked token sequences as input, using diffusion-style denoising
   to reconstruct the original text.

2. **LanguageDiffusionDecoderLMHead**: Wraps the decoder with a language modeling head
   to produce vocabulary logits from the decoder hidden states.

3. **LanguageDiffusionAutoencoder**: The complete autoencoder combining an encoder
   (LanguageEncoderLatentHead) that compresses text into latent representations and
   a diffusion decoder that reconstructs text from these latents.

Key Concepts:
    - **Window Size**: Sequences are processed in fixed-size windows. The encoder aggregates
      `window_size` tokens into a single latent vector, and the decoder expands each latent
      back to `window_size` tokens.

    - **Masked Diffusion**: Unlike the multi-head autoencoder variant, the diffusion decoder
      receives masked input tokens and learns to predict the original tokens. During training,
      tokens are randomly masked according to a masking ratio determined by the timestep.

    - **Latent Projection**: The encoder projects hidden states to `latent_dim`, and the
      decoder projects latents back to `hidden_size` via a learned linear transformation.

Example usage:
    ```python
    from transformers import LatentGPT2Config
    from transformers.models.latent_gpt2 import LanguageDiffusionAutoencoder

    # Create configuration
    config = LatentGPT2Config(
        window_size=4,
        latent_dim=768,
        num_hidden_layers_encoder=6,
        num_hidden_layers_decoder=6,
        autoencoder_type="diffusion",
    )

    # Initialize model
    model = LanguageDiffusionAutoencoder(config)

    # Forward pass (encoding + decoding)
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    logits = outputs.logits
    latents = outputs.latents
    ```
"""

import copy
import math
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
from .modeling_autoencoder import LanguageEncoderLatentHead, LanguageDecoderBase, LanguageDecoderUtils

logger = logging.get_logger(__name__)


@auto_docstring(
    custom_intro="""
    Language diffusion decoder model based on GPT-2 for decoding latent representations back to text.

    This decoder implements a masked diffusion approach where the model receives:
    1. Latent representations (compressed semantic information from the encoder)
    2. Partially or fully masked token sequences

    The decoder learns to predict the original tokens from the masked input conditioned
    on the latent representations. This differs from the multi-head decoder which uses
    separate prediction heads for each position in the window.

    Architecture:
        - Inherits from LanguageDecoderBase which provides the core transformer layers
        - Uses `wte_latent` (Linear layer) to project latents from `latent_dim` to `hidden_size`
        - Uses `wte` (Embedding layer) to embed masked/unmasked input tokens
        - Concatenates latent embeddings with token embeddings before processing

    Input Processing:
        The `pre_process_inputs` method handles the input preparation:
        - Latent representations are projected to embeddings via `wte_latent`
        - Masked token sequences are embedded via `wte`
        - If no input_ids provided, a fully masked sequence is generated
        - Latent embeddings and token embeddings are concatenated

    Output Processing:
        The `post_process_outputs` method reshapes outputs back to batch dimensions:
        - `last_tail_hidden_state`: The final hidden state (used as the next latent)
        - `last_window_hidden_state`: Hidden states for the window_size positions
        - Handles the segment-to-batch dimension conversion
    """,
    custom_args="""
        window_size (`int`):
            The window size for disaggregating latent representations back to sequences.
            Each latent vector is expanded to `window_size` token predictions.
        mask_token_id (`int`):
            Token ID used for masking during diffusion training and inference.
    """,
)
class LanguageDiffusionDecoder(LanguageDecoderBase):
    """
    Attributes:
        ae_utils (`LanguageDecoderUtils`):
            Utility class for sequence windowing operations including padding,
            aggregation, and splitting of sequences.
        wte_latent (`nn.Linear`):
            Linear projection from latent_dim to hidden_size (inherited from base).
        wte (`nn.Embedding`):
            Token embedding layer (inherited from base).
        h (`nn.ModuleList`):
            List of GPT2Block transformer layers (inherited from base).

    Example:
        ```python
        config = LatentGPT2Config(window_size=4, latent_dim=768)
        decoder = LanguageDiffusionDecoder(config)

        # Decode from latents with masked input
        outputs = decoder(
            inputs_latents=latent_vectors,  # (batch, seq_len, latent_dim)
            input_ids=masked_tokens,         # (batch, seq_len * window_size)
        )
        hidden_states = outputs.last_window_hidden_state
        ```
    """

    def __init__(self, config: LatentGPT2Config):
        """
        Initialize the LanguageDiffusionDecoder.

        Args:
            config ([`LatentGPT2Config`]):
                Configuration object containing model hyperparameters.
        """
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
        Pre-process and segment inputs for the diffusion decoder.

        This method prepares the inputs for the transformer by:
        1. Aggregating input sequences into window-sized segments for parallel processing
        2. Converting latent vectors to embeddings via `wte_latent`
        3. Converting masked token IDs to embeddings via `wte`
        4. Concatenating latent embeddings with token embeddings

        The diffusion decoder expects both latent representations AND masked token sequences.
        The latent provides the conditioning signal, while the masked tokens are progressively
        denoised during inference.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size * segment_num, window_size) or (batch_size, seq_len)`, *optional*):
                Masked token sequence where some positions contain `mask_token_id`.
                If None and `inputs_latents` is provided, a fully masked sequence
                of length `window_size` is generated automatically.
            inputs_latents (`torch.FloatTensor` of shape `(batch_size, segment_num, latent_dim)`, *optional*):
                Latent representations from the encoder. These are projected to
                `hidden_size` via `wte_latent` before being concatenated with token embeddings.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size * segment_num, 1 + window_size, hidden_size)`, *optional*):
                Pre-computed embeddings. If provided, `inputs_latents` and `input_ids`
                should not be provided simultaneously.

        Returns:
            [`PreprocessOutput`]:
                A named tuple containing:
                - `input_ids`: The aggregated and potentially masked input IDs with shape (batch_size * segment_num, window_size)
                - `inputs_latents`: The aggregated latent representations with shape (batch_size * segment_num, 1, latent_dim)
                - `inputs_embeds`: The concatenated embeddings ready for the transformer,
                  with shape `(batch_size * segment_num, 1 + window_size, hidden_size)`

        Raises:
            ValueError:
                If both `inputs_latents` and `inputs_embeds` are provided simultaneously.
            ValueError:
                If both `input_ids` and `inputs_embeds` are provided simultaneously.

        Note:
            The output `inputs_embeds` has the latent embeddings prepended to the token
            embeddings: `[latent_embed, masked_token_embed_1, ..., masked_token_embed_n]`
        """
        # CAUTION: Complex input handling and reshaping
        if inputs_embeds is not None:
            if inputs_embeds.shape[-2] != (1 + self.config.window_size):
                raise ValueError(f"The last second dimension, inputs_embeds.shape[-2], should be {1 + self.config.window_size}, but got {inputs_embeds.shape[-2]}.")
            if inputs_latents is not None:
                raise ValueError(f"inputs_latents and inputs_embeds are provided at the same time, only one of them is accepted.")
            if input_ids is not None:
                raise ValueError(f"input_ids and inputs_embeds are provided at the same time, only one of them is accepted.")
            return PreprocessOutput(input_ids=input_ids, inputs_latents=inputs_latents, inputs_embeds=inputs_embeds)
        else:
            if inputs_latents is None:
                raise ValueError(f"inputs_latents should be provided if inputs_embeds is None.")
            # Convert latent sequence from (batch_size, segment_num, latent_dim) to (batch_size * segment_num, 1, latent_dim)
            inputs_latents = self.ae_utils.agg_sequence(sequence=inputs_latents)
            # Convert from (batch_size * segment_num, 1, latent_dim) to (batch_size * segment_num, 1, hidden_size)
            inputs_embeds = self.wte_latent(inputs_latents)

            # Required input_ids shape (batch_size * segment_num, window_size)
            input_ids_req_shape: Tuple[int, int] = inputs_latents.shape[:-2] + (self.config.window_size, )
            if input_ids is not None:
                # If there is input_ids, aggregate and convert it into embeddings
                # Only accept 2 shapes, (batch_size * segment_num, window_size) or (batch_size, seq_len)
                if tuple(input_ids.shape) != input_ids_req_shape:
                    # if input_ides is not (batch_size * segment_num, window_size), pad and aggregate and return (batch_size * segment_num, window_size). 
                    input_ids = self.ae_utils.agg_sequence_mask_diffusion(sequence=input_ids)
                    if tuple(input_ids.shape) != input_ids_req_shape:
                        raise ValueError(f"input_ids with shape {input_ids.shape} cannot be padded and reshaped into required shape, {input_ids_req_shape}.")
                # If there is input_ids, convert it into embeddings
                masked_input_embeds = self.wte(input_ids)
            else:
                # If there is no input masked sequecne, random sample a fully masked sequence with shape (batch_size * segment_num, window_size)
                # Concatenate Partially / Fully Masked target sequence
                masked_input_ids: torch.LongTensor = (torch.ones(*input_ids_req_shape) * self.config.mask_token_id).long().to(inputs_embeds.device)
                masked_input_embeds: torch.FloatTensor = self.wte(masked_input_ids)

            # full_inputs_embeds shape, (batch_size * segment_num, 1 + window_size, latent_dim)
            full_inputs_embeds: torch.FloatTensor = torch.cat([inputs_embeds, masked_input_embeds], dim=-2)
        return PreprocessOutput(input_ids=input_ids, inputs_latents=inputs_latents, inputs_embeds=full_inputs_embeds)

    def post_process_outputs(
        self,
        outputs: CausalLMAutoencoderOutputWithCrossAttentions,
    ) -> BaseAutoencoderOutputWithPastAndCrossAttentions:
        """
        Post-process and reshape decoder outputs back to batch dimensions.

        This method reverses the segment-based processing done in `pre_process_inputs`,
        converting the outputs from shape `(batch_size * segment_num, 1 + window_size, hidden_size)`
        back to `(batch_size, segment_num * (1 + window_size), hidden_size)`.

        It also extracts specific hidden states needed for downstream processing:
        - `last_tail_hidden_state`: The final position's hidden state, which can serve
          as the latent representation for the next segment in autoregressive generation.
        - `last_window_hidden_state`: Hidden states for the last `window_size` positions,
          which correspond to the decoded token positions (excluding the latent position).

        Args:
            outputs ([`CausalLMAutoencoderOutputWithCrossAttentions`]):
                The raw output from the parent transformer forward pass, containing:`
                - `last_hidden_state`: Full hidden states from the final layer (if requested), with shape `(batch_size * segment_num, 1 + window_size, hidden_size)`
                - `past_key_values`: Cached key-value states for generation
                - `hidden_states`: Hidden states from all layers (if requested), with shape `(batch_size * segment_num, 1 + window_size, hidden_size) * num_hidden_layers_decoder`
                - `attentions`: Attention weights (if requested)
                - `cross_attentions`: Cross-attention weights (if applicable)

        Returns:
            [`BaseAutoencoderOutputWithPastAndCrossAttentions`]:
                Processed outputs with reshaped tensors containing:
                - `last_tail_hidden_state`:  Tail of last hidden state (if applicable) with shape `(batch_size, segment_num, hidden_size)`
                - `last_window_hidden_state`: Last window of last hidden state (if applicable), with shape `(batch_size, segment_num * window_size, hidden_size)`
                - `last_hidden_state`: Full last hidden states (if applicable), with shape `(batch_size, (1 + window_size) * segment_num, hidden_size)`
                - `past_key_values`: Unchanged cache states
                - `hidden_states`: Tuple of hidden states per layer (if applicable), with shape `(batch_size, (1 + window_size) * segment_num, hidden_size) * num_hidden_layers_decoder`
                - `attentions`: Unchanged attention weights
                - `cross_attentions`: Unchanged cross-attention weights

        Note:
            If `outputs` is None, returns None without processing.
        """
        if outputs is None:
            return outputs
        return BaseAutoencoderOutputWithPastAndCrossAttentions(
            # Only keep the last hidden_state of hidden_state of the last layer
            # from last_hidden_state, (batch_size * segment_num, 1 + window_size, hidden_size), to (batch_size * segment_num, 1, hidden_size) after slicing, to last_tail_hidden_state, (batch_size, segment_num, hidden_size) after splitting
            last_tail_hidden_state=self.ae_utils.split_sequence(sequence=outputs.last_hidden_state[:, -1:, ...]),
            # Only keep the last self.config.window_size hidden_state of hidden_state of the last layer, 
            # from last_hidden_state, (batch_size * segment_num, 1 + window_size, hidden_size), to (batch_size * segment_num, window_size, hidden_size) after slicing, to last_window_hidden_state, (batch_size, segment_num * window_size, hidden_size) after splitting
            last_window_hidden_state=self.ae_utils.split_sequence(sequence=outputs.last_hidden_state[:, -self.config.window_size:, ...]),
            # from last_hidden_state, (batch_size * segment_num, 1 + window_size, hidden_size), to last_hidden_state (batch_size, (1 + window_size) * segment_num, hidden_size) after splitting
            last_hidden_state=self.ae_utils.split_sequence(sequence=outputs.last_hidden_state),
            past_key_values=outputs.past_key_values,
            # from hidden_state, (batch_size * segment_num, 1 + window_size, hidden_size) * num_hidden_layers_decoder, to hidden_state (batch_size, (1 + window_size) * segment_num, hidden_size) * num_hidden_layers_decoder after splitting
            hidden_states=tuple(self.ae_utils.split_sequence(sequence=hidden_state) for hidden_state in outputs.hidden_states) if outputs.hidden_states is not None else None,
            # TODO: Handle attentions and cross_attentions reshaping
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def init_weight_from_pretrained(self, pretrained_model: GPT2ModelBase) -> "LanguageDiffusionDecoder":
        """
        Initialize the decoder weights from a pre-trained GPT-2 model.

        This method copies weights from a pre-trained GPT-2 model to initialize
        the decoder. It uses the last `num_hidden_layers_decoder` layers from
        the pretrained model, which typically contain higher-level representations
        suitable for generation tasks.

        The following components are copied:
        - `wte`: Token embeddings (for embedding masked input tokens)
        - `wpe`: Position embeddings
        - `drop`: Dropout layer
        - `h`: Transformer blocks (last N layers)
        - `ln_f`: Final layer normalization

        Note that `wte_latent` is NOT copied since it's a Linear projection
        from `latent_dim` to `hidden_size`, not an embedding layer. It retains
        its random initialization (or identity if `latent_dim == hidden_size`).

        Args:
            pretrained_model ([`GPT2ModelBase`]):
                The pre-trained GPT-2 model to copy weights from. Should be a
                `GPT2Model` or similar base model containing the transformer blocks.

        Returns:
            [`LanguageDiffusionDecoder`]:
                Returns `self` for method chaining.

        Example:
            ```python
            from transformers import GPT2Model

            pretrained = GPT2Model.from_pretrained("gpt2")
            decoder = LanguageDiffusionDecoder(config)
            decoder.init_weight_from_pretrained(pretrained)
            ```
        """
        # wte is copied for embedding masked input tokens
        # wte_latent is not copied since it's a Linear(latent_dim -> hidden_size)
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

    @auto_docstring
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
    ) -> Union[tuple, BaseAutoencoderOutputWithPastAndCrossAttentions]:
        r"""
        Forward pass through the diffusion decoder.

        This method processes latent representations and masked token sequences through
        the transformer to produce hidden states that can be used for token prediction.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size * segment_num, window_size) or (batch_size, seq_len)`, *optional*):
                Masked token sequence where some positions contain `mask_token_id`.
                If None and `inputs_latents` is provided, a fully masked sequence
                of length `window_size` is generated automatically.
            inputs_latents (`torch.FloatTensor` of shape `(batch_size, segment_num, latent_dim)`, *optional*):
                Latent representations from the encoder. These are projected to
                `hidden_size` via `wte_latent` before being concatenated with token embeddings.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size * segment_num, 1 + window_size, hidden_size)`, *optional*):
                Pre-computed embeddings. If provided, `inputs_latents` and `input_ids`
                should not be provided simultaneously.

        Returns:
            [`BaseAutoencoderOutputWithPastAndCrossAttentions`]:
                Processed outputs with tensors containing:
                - `last_tail_hidden_state`:  Tail of last hidden state (if applicable) with shape `(batch_size, segment_num, hidden_size)`
                - `last_window_hidden_state`: Last window of last hidden state (if applicable), with shape `(batch_size, segment_num * window_size, hidden_size)`
                - `last_hidden_state`: Full last hidden states (if applicable), with shape `(batch_size, (1 + window_size) * segment_num, hidden_size)`
                - `past_key_values`: Unchanged cache states
                - `hidden_states`: Tuple of hidden states per layer (if applicable), with shape `(batch_size, (1 + window_size) * segment_num, hidden_size) * num_hidden_layers_decoder`
                - `attentions`: Unchanged attention weights
                - `cross_attentions`: Unchanged cross-attention weights
        """
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
    custom_intro="""
    Language diffusion decoder with language modeling head for text generation.

    This model wraps the `LanguageDiffusionDecoder` with a linear projection head
    that maps hidden states to vocabulary logits. Unlike the multi-head variant
    which uses separate heads for each position in the window, this model uses
    a single shared LM head applied to all positions.

    The model processes latent representations concatenated with masked token
    embeddings, then applies the LM head to the hidden states corresponding to
    the token positions (the last `window_size` positions in each segment).

    Architecture:
        - `transformer`: LanguageDiffusionDecoder for processing inputs
        - `lm_head`: Linear projection from hidden_size to vocab_size

    Key Differences from Multi-Head Decoder:
        1. Uses a single shared LM head instead of `window_size` separate heads
        2. Expects masked input tokens (diffusion-style) rather than just latents
        3. Applies LM head to `last_window_hidden_state` (token positions only)
    """,
)
class LanguageDiffusionDecoderLMHead(GPT2PreTrainedModel, GenerationMixin):
    """
    Attributes:
        transformer ([`LanguageDiffusionDecoder`]):
            The diffusion decoder transformer model.
        lm_head (`nn.Linear`):
            Linear layer projecting hidden states to vocabulary logits.
            Shape: (hidden_size, vocab_size).

    Example:
        ```python
        config = LatentGPT2Config(window_size=4, latent_dim=768)
        model = LanguageDiffusionDecoderLMHead(config)

        # Decode latents to token logits
        outputs = model(
            inputs_latents=latent_vectors,
            input_ids=masked_tokens,
        )
        logits = outputs.logits  # (batch_size, seq_len * window_size, vocab_size)
        ```
    """

    def __init__(self, config: LatentGPT2Config):
        """
        Initialize the LanguageDiffusionDecoderLMHead.

        Args:
            config ([`LatentGPT2Config`]):
                Configuration object containing model hyperparameters.
        """
        super().__init__(config)
        self.transformer = LanguageDiffusionDecoder(config=config)

        # Single head projection mechanism (shared across all positions)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def init_weight_from_pretrained(self, pretrained_model: GPT2LMHeadModel) -> "LanguageDiffusionDecoderLMHead":
        """
        Initialize model weights from a pre-trained GPT-2 LM model.

        This method copies the transformer weights and LM head from a pre-trained
        GPT-2 model. The single LM head is copied directly, unlike the multi-head
        variant which would replicate it for each position.

        Args:
            pretrained_model ([`GPT2LMHeadModel`]):
                The pre-trained GPT-2 language model to copy weights from.

        Returns:
            [`LanguageDiffusionDecoderLMHead`]:
                Returns `self` for method chaining.

        Example:
            ```python
            from transformers import GPT2LMHeadModel

            pretrained = GPT2LMHeadModel.from_pretrained("gpt2")
            model = LanguageDiffusionDecoderLMHead(config)
            model.init_weight_from_pretrained(pretrained)
            ```
        """
        self.transformer.init_weight_from_pretrained(pretrained_model=pretrained_model.transformer)
        self.lm_head = copy.deepcopy(pretrained_model.lm_head)
        return self

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
        Forward pass for the diffusion decoder with LM head.

        This method processes latent representations through the diffusion decoder transformer,
        then applies the LM head to produce vocabulary logits.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size * segment_num, window_size) or (batch_size, seq_len)`, *optional*):
                Masked token sequence for diffusion decoding. If None, a fully masked sequence is generated.
            inputs_latents (`torch.FloatTensor` of shape `(batch_size, segment_num, latent_dim)`, *optional*):
                Latent representations to be decoded. These are typically output from the encoder component of the
                autoencoder and represent compressed semantic information to be expanded back into token sequences.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size * segment_num, 1 + window_size, hidden_size)`, *optional*):
                Pre-computed embeddings (alternative to `inputs_latents` and `input_ids`).
            labels (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
                `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
                are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`

        Returns:
            [`CausalLMAutoencoderOutputWithCrossAttentions`] or `tuple`:
                - `logits`: Token prediction logits, with shape `(batch_size, segment_num * window_size, vocab_size)`
                - `last_tail_hidden_state`: Final position hidden state per segment, with shape `(batch_size, segment_num, hidden_size)` (if `output_hidden_states=True`)
                - `last_window_hidden_state`: Hidden states for decoded token positions, with shape `(batch_size, segment_num * window_size, hidden_size)` (if `output_hidden_states=True`)
                - `last_hidden_state`: Full hidden states from final layer, with shape `(batch_size, (1 + window_size) * segment_num, hidden_size)` (if `output_hidden_states=True`)
                - `past_key_values`: Cached key-value states
                - `hidden_states`: All layer hidden states, tuple of `(batch_size, (1 + window_size) * segment_num, hidden_size) * num_hidden_layers_decoder` (if `output_hidden_states=True`)
                - `attentions`: Attention weights (if `output_attentions=True`)
                - `cross_attentions`: Cross-attention weights (if applicable)
                - `latent_embeds`: Latent embeddings, with shape `(batch_size, segment_num, hidden_size)`
                - `latents`: Input latent representations, with shape `(batch_size, segment_num, latent_dim)`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        inferred_inputs_embeds = None
        if inputs_latents is not None and inputs_embeds is None:
            # inferred_latent_embeds is the embedding of latent, with shape `(batch_size * segment_num, 1, hidden_size)`
            inferred_latent_embeds = self.transformer.wte_latent(inputs_latents)
        else:
            # inputs_embeds is (`torch.FloatTensor` of shape `(batch_size * segment_num, 1 + window_size, hidden_size)`, *optional*)
            inferred_latent_embeds = inputs_embeds[:, 1, ...]

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
        last_window_hidden_state = transformer_outputs.last_window_hidden_state

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(last_window_hidden_state[:, slice_indices, :])

        loss = None
        # No loss function provided
        # if labels is not None:
        #     # Flatten the tokens
        #     loss = self.loss_function(
        #         logits,
        #         labels,
        #         vocab_size=self.config.vocab_size,
        #         **kwargs,
        #     )

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
    custom_intro="""
    Complete diffusion-based language autoencoder combining encoder and decoder.

    This model implements a full autoencoder architecture for text compression and
    reconstruction using a diffusion-style decoder. The encoder compresses input
    text into latent representations, and the decoder reconstructs the text using
    masked diffusion.

    Architecture Overview:
        - **Encoder** (`LanguageEncoderLatentHead`): Processes input tokens through
          GPT-2 transformer layers, then projects to latent space via `latent_head`.
          Each `window_size` tokens are compressed into a single latent vector.

        - **Decoder** (`LanguageDiffusionDecoderLMHead`): Takes latent vectors and
          masked token sequences, processes them through transformer layers, and
          applies an LM head to predict the original tokens.

    Weight Tying:
        The encoder's `latent_head` and decoder's `wte_latent` share weights (tied),
        ensuring consistent latent space representations between encoding and decoding.

    Training:
        During training, the decoder receives masked versions of the target tokens
        with masking ratio determined by `dae_num_train_timesteps`. At timestep 0,
        tokens are fully masked; at the maximum timestep, no tokens are masked.

    Inference:
        For generation, start with fully masked tokens and iteratively denoise by
        sampling from the predicted distribution and reducing the masking ratio.
    """,
    custom_args="""
        return_encoder_decoder_res (`bool`, *optional*, defaults to `False`):
            If True, returns a tuple of (autoencoder_output, encoder_output, decoder_output)
            for debugging and analysis purposes.
    """,
)
class LanguageDiffusionAutoencoder(GPT2PreTrainedModel, GenerationMixin):
    """
    Attributes:
        encoder ([`LanguageEncoderLatentHead`]):
            The encoder model that compresses text to latent representations.
        decoder ([`LanguageDiffusionDecoderLMHead`]):
            The diffusion decoder that reconstructs text from latents.

    Example:
        ```python
        from transformers import LatentGPT2Config
        from transformers.models.latent_gpt2 import LanguageDiffusionAutoencoder

        # Create and initialize model
        config = LatentGPT2Config(
            window_size=4,
            latent_dim=768,
            autoencoder_type="diffusion",
        )
        model = LanguageDiffusionAutoencoder(config)

        # Training forward pass
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        # Access latent representations
        latents = outputs.latents  # (batch_size, segment_num, latent_dim)
        ```
    """
    # Tieing weights doesn't work
    # _tied_weights_keys = {"encoder.latent_head.weight": "decoder.transformer.wte_latent.weight"}

    def __init__(self, config: LatentGPT2Config):
        """
        Initialize the LanguageDiffusionAutoencoder.

        Args:
            config ([`LatentGPT2Config`]):
                Configuration object containing model hyperparameters including:
                - `window_size`: Tokens per latent vector
                - `latent_dim`: Latent space dimensionality
                - `num_hidden_layers_encoder`: Encoder transformer layers
                - `num_hidden_layers_decoder`: Decoder transformer layers
                - `dae_num_train_timesteps`: Diffusion training timesteps
        """
        super().__init__(config)
        self.encoder: LanguageEncoderLatentHead = LanguageEncoderLatentHead(config=config)
        self.decoder: LanguageDiffusionDecoderLMHead = LanguageDiffusionDecoderLMHead(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    def init_weight_from_pretrained(self, pretrained_model: GPT2LMHeadModel) -> "LanguageDiffusionAutoencoder":
        """
        Initialize both encoder and decoder from a pre-trained GPT-2 model.

        This method initializes both the encoder and decoder components from
        a single pre-trained GPT-2 model:
        - Encoder uses the first `num_hidden_layers_encoder` layers
        - Decoder uses the last `num_hidden_layers_decoder` layers

        Args:
            pretrained_model ([`GPT2LMHeadModel`]):
                The pre-trained GPT-2 language model to copy weights from.

        Returns:
            [`LanguageDiffusionAutoencoder`]:
                Returns `self` for method chaining.

        Example:
            ```python
            from transformers import GPT2LMHeadModel

            pretrained = GPT2LMHeadModel.from_pretrained("gpt2")
            model = LanguageDiffusionAutoencoder(config)
            model.init_weight_from_pretrained(pretrained)
            ```
        """
        self.encoder.init_weight_from_pretrained(pretrained_model=pretrained_model)
        self.decoder.init_weight_from_pretrained(pretrained_model=pretrained_model)
        return self

    def _get_decoder_inputs(
        self,
        labels: torch.LongTensor,
        timestep: Optional[int] = None,
    ) -> Optional[torch.LongTensor]:
        """
        Generate masked decoder inputs from labels for diffusion-style training.

        This method creates masked versions of the target sequence for training
        the diffusion decoder. The masking ratio is determined by the timestep:
        - At timestep 0: All tokens are masked (ratio = 1.0)
        - At timestep T: No tokens are masked (ratio = 0.0)
        - Intermediate timesteps: Linear interpolation of masking ratio

        The masking schedule follows discrete steps:
        `ratio = t / T` where `t` is uniformly sampled from `{0, 1, ..., T}`
        and `T = dae_num_train_timesteps`.

        Note:
            This method currently returns None (masking disabled). The commented
            code shows the full implementation for reference. When enabled, it
            would apply random per-position masking based on the sampled ratio.

        Args:
            labels (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                Target token sequence to be masked. Tokens exceeding `vocab_size`
                are clamped to valid range.
            timestep (`int`, *optional*):
                Specific timestep to use for determining masking ratio. If None,
                a random timestep is sampled uniformly from `[0, dae_num_train_timesteps]`
                for each sample in the batch.

        Returns:
            `torch.LongTensor` of shape `(batch_size, seq_len)` or `None`:
                Masked token sequence where positions are randomly replaced with
                `mask_token_id` based on the masking ratio. Currently returns None
                (masking disabled, decoder receives no input_ids).

        Algorithm (when enabled):
            1. Sample timestep `t ~ Uniform(0, T)` per batch item
            2. Compute masking ratio `r = t / T`
            3. For each position, mask if `random() < r`
            4. Replace masked positions with `mask_token_id`
        """
        # Currently returns None - masking is handled differently or disabled
        return None

        # # TODO: Clip the label to avoid the token id execeed vocab_size, but why labels has token execeed voacb_size?
        # # Clamp labels to be within the vocab size
        # # print(f"Before labels: {labels.shape}, max: {torch.max(labels)}, min: {torch.min(labels)}")
        # labels = torch.clamp(labels, min=0, max=self.config.vocab_size - 1)
        # # print(f"After labels: {labels.shape}, max: {torch.max(labels)}, min: {torch.min(labels)}")

        # batch_size, seq_len = labels.shape
        # device = labels.device
        # num_timesteps: int = self.config.dae_num_train_timesteps
        # mask_token_id: int = self.config.mask_token_id

        # masking_ratios: torch.FloatTensor = None
        # if num_timesteps is None:
        #     # If num_timesteps is None, use continuous timestep
        #     masking_ratios = torch.rand((batch_size, 1), device=device, dtype=torch.float)
        # else:
        #     # If num_timesteps is None, use discrete timestep
        #     # Sample timestep uniformly from [0, num_timesteps] for each sample in batch
        #     if timestep is not None:
        #         # Use provided timestep for all samples
        #         timesteps = torch.full((batch_size, 1), timestep, device=device, dtype=torch.long)
        #     else:
        #         # Sample random timestep for each sample: t in {0, 1, 2, ..., num_timesteps}
        #         timesteps = torch.randint(0, num_timesteps + 1, (batch_size, 1), device=device)

        #     # Compute masking ratio for each sample: ratio = t / num_timesteps
        #     # Shape: (batch_size, 1) for broadcasting
        #     if num_timesteps == 0:
        #         # If num_timesteps == 0, only trained on fully masked sequence
        #         masking_ratios = torch.ones((batch_size, 1), device=device, dtype=torch.float)
        #     else:
        #         # If num_timesteps > 0, trained on fully and paritally masked sequence
        #         masking_ratios = timesteps.float() / num_timesteps
        #     # masking_ratios = masking_ratios.unsqueeze(-1)  # (batch_size, 1)

        # # Generate random values for each position
        # random_probs = torch.rand(batch_size, seq_len, device=device)
        # # Create mask: True where we should mask (random < masking_ratio)
        # mask = random_probs < masking_ratios

        # # Apply masking: replace masked positions with mask_token_id
        # masked_labels = labels.clone()
        # masked_labels.masked_fill_(mask, mask_token_id)

        # return masked_labels

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
        Encode input text into latent representations.

        This method runs only the encoder portion of the autoencoder, compressing
        input tokens into latent vectors. Each `window_size` tokens are aggregated
        into a single latent vector of dimension `latent_dim`.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Input token IDs to encode. Sequence length should be divisible by
                `window_size` for optimal processing (padding is applied if not).
                After padding, `segment_num = ceil(seq_len / window_size)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                Pre-computed embeddings instead of `input_ids`.
            labels (`torch.LongTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                Not used in encoding, included for API consistency.
            use_cache (`bool`, *optional*):
                Whether to return cached key-value states.

        Returns:
            [`CausalLMAutoencoderOutputWithCrossAttentions`] or `tuple`:
                Encoder outputs containing:
                - `latents`: Compressed latent vectors, with shape `(batch_size, segment_num, latent_dim)`
                - `last_tail_hidden_state`: Final hidden state per segment, with shape `(batch_size, segment_num, hidden_size)`
                - `last_window_hidden_state`: Hidden states for window positions, with shape `(batch_size, segment_num * window_size, hidden_size)`
                - `last_hidden_state`: Full hidden states from final layer, with shape `(batch_size, segment_num * window_size, hidden_size)`
                - `hidden_states`: All layer outputs (if requested), tuple of `(batch_size, segment_num * window_size, hidden_size) * num_hidden_layers_encoder`
                - `attentions`: Attention weights (if requested)

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
        Decode latent representations back to token logits.

        This method runs only the decoder portion of the autoencoder, reconstructing
        token logits from latent vectors. For diffusion decoding, it also takes
        masked token sequences as conditioning input.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size * segment_num, window_size) or (batch_size, seq_len)`, *optional*):
                Masked token sequence for diffusion decoding. Positions with
                `mask_token_id` will be predicted by the model. If None, a fully masked
                sequence is generated internally.
            inputs_latents (`torch.FloatTensor` of shape `(batch_size, segment_num, latent_dim)`, *optional*):
                Latent representations to decode. Each latent is expanded to
                `window_size` token predictions.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size * segment_num, 1 + window_size, hidden_size)`, *optional*):
                Pre-computed embeddings to use instead of latents and input_ids.
            labels (`torch.LongTensor`, *optional*):
                Not used in decoding, included for API consistency.

        Returns:
            [`CausalLMAutoencoderOutputWithCrossAttentions`] or `tuple`:
                Decoder outputs containing:
                - `logits`: Token prediction logits, with shape `(batch_size, segment_num * window_size, vocab_size)`
                - `last_tail_hidden_state`: Final position hidden state per segment, with shape `(batch_size, segment_num, hidden_size)` (if `output_hidden_states=True`)
                - `last_window_hidden_state`: Hidden states for decoded positions, with shape `(batch_size, segment_num * window_size, hidden_size)` (if `output_hidden_states=True`)
                - `last_hidden_state`: Full hidden states from final layer, with shape `(batch_size, (1 + window_size) * segment_num, hidden_size)` (if `output_hidden_states=True`)
                - `hidden_states`: All layer outputs (if requested), tuple of `(batch_size, (1 + window_size) * segment_num, hidden_size) * num_hidden_layers_decoder`
                - `attentions`: Attention weights (if requested)
                - `latent_embeds`: Latent embeddings, with shape `(batch_size, segment_num, hidden_size)`
                - `latents`: Input latent representations, with shape `(batch_size, segment_num, latent_dim)`

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
        return decoder_output
    
    def _format_dict_output(
        self,
        dict_output,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> CausalLMAutoencoderOutputWithCrossAttentions:
        """
        Format encoder or decoder output into a standardized CausalLMAutoencoderOutputWithCrossAttentions.

        This helper method standardizes the output format by conditionally including
        hidden states and attention weights based on the output flags.

        Args:
            dict_output: The raw output from encoder or decoder, containing:
                - `last_tail_hidden_state`: Shape `(batch_size, segment_num, hidden_size)`
                - `last_window_hidden_state`: Shape `(batch_size, segment_num * window_size, hidden_size)`
                - `last_hidden_state`: Shape varies by encoder/decoder
                - `logits`: Shape `(batch_size, segment_num * window_size, vocab_size)` (decoder only)
                - `hidden_states`: Tuple of hidden states per layer
                - `attentions`: Attention weights (if requested)
            output_attentions (`bool`, *optional*):
                Whether to include attention weights in output.
            output_hidden_states (`bool`, *optional*):
                Whether to include hidden states in output.

        Returns:
            [`CausalLMAutoencoderOutputWithCrossAttentions`]:
                Formatted output with conditionally included fields.
        """
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
        return_encoder_decoder_res: bool = False,
        **kwargs,
    ) -> Union[tuple, CausalLMAutoencoderOutputWithCrossAttentions, Tuple]:
        r"""
        Complete forward pass through the diffusion autoencoder (encode + decode).

        This method performs the full autoencoder forward pass:
        1. Encodes input tokens into latent representations
        2. Optionally generates masked decoder inputs from labels
        3. Decodes latents back to token logits using the diffusion decoder
        4. Computes reconstruction loss if labels are provided

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Input token IDs to encode. The sequence is padded to be divisible by
                `window_size`, resulting in `segment_num = ceil(seq_len / window_size)`.
            inputs_embeds (`torch.FloatTensor`, *optional*):
                Not used in current implementation.
            labels (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Target token IDs for computing reconstruction loss. Labels are
                padded to be divisible by `window_size` before loss computation.
                Positions with value -100 are ignored in the loss.
            return_encoder_decoder_res (`bool`, *optional*, defaults to `False`):
                If True, returns a tuple of (autoencoder_output, encoder_output, decoder_output)
                for debugging and detailed analysis.

        Returns:
            [`CausalLMAutoencoderOutputWithCrossAttentions`] or `tuple`:
                If `return_encoder_decoder_res=False` (default):
                    - `loss`: Reconstruction loss scalar (if labels provided)
                    - `logits`: Token prediction logits, with shape `(batch_size, segment_num * window_size, vocab_size)`
                    - `latents`: Encoded latent vectors, with shape `(batch_size, segment_num, latent_dim)`
                    - `last_tail_hidden_state`: Final position hidden state per segment, with shape `(batch_size, segment_num, hidden_size)` (if `output_hidden_states=True`)
                    - `last_window_hidden_state`: Hidden states for decoded token positions, with shape `(batch_size, segment_num * window_size, hidden_size)` (if `output_hidden_states=True`)
                    - `last_hidden_state`: Decoder final layer hidden states, with shape `(batch_size, (1 + window_size) * segment_num, hidden_size)` (if `output_hidden_states=True`)
                    - `hidden_states`: All layer hidden states, tuple of `(batch_size, (1 + window_size) * segment_num, hidden_size) * num_hidden_layers_decoder` (if `output_hidden_states=True`)
                    - `attentions`: Attention weights (if `output_attentions=True`)

                If `return_encoder_decoder_res=True`:
                    Tuple of (autoencoder_output, encoder_output, decoder_output)

        Example:
            ```python
            # Training forward pass
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()

            # Inference (get logits and latents)
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                logits = outputs.logits  # (batch_size, segment_num * window_size, vocab_size)
                latents = outputs.latents  # (batch_size, segment_num, latent_dim)
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_output = self.encoder(
            input_ids=input_ids,
            past_key_values=None,
            cache_position=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            logits_to_keep=0,
        )

        decoder_input_ids = self._get_decoder_inputs(labels=labels)

        decoder_output = self.decoder(
            input_ids=decoder_input_ids,
            inputs_latents=encoder_output.latents,
            past_key_values=None,
            cache_position=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            logits_to_keep=0,
        )

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # logits = self.lm_head(decoder_output.hidden_states[:, slice_indices, :])
        logits = decoder_output.logits[:, slice_indices, :]

        loss = None
        if labels is not None:
            # Pad labels to make it can be divided by self.config.window_size
            padded_labels: torch.LongTensor = self.decoder.transformer.ae_utils.pad(sequence=labels)
            # Flatten the tokens
            loss = self.loss_function(
                # Skip the logit at the first dimension, which is the output corresponsing to the latent
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

        This loss function implements label smoothing, which helps prevent
        overconfident predictions and improves generalization. The smoothed
        loss is a weighted combination of:
        - Standard negative log-likelihood loss (weight: 1 - epsilon)
        - Uniform distribution penalty (weight: epsilon)

        Label Smoothing Formula:
            `loss = (1 - epsilon) * NLL + epsilon * uniform_penalty`

        Where:
        - NLL = -log(p(y_true))
        - uniform_penalty = mean(-log(p)) over all classes

        Args:
            logits (`torch.FloatTensor` of shape `(batch_size, window_size * segment_num, vocab_size)`):
                Predicted token logits from the decoder.
            labels (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                Target token IDs. Values equal to `ignore_index` are masked.
            vocab_size (`int`):
                Size of the vocabulary for clamping label indices.
            shift_labels (`int`, *optional*, defaults to 0):
                Number of positions to shift labels for autoregressive loss.
                If > 0, logits[:-shift] are compared with labels[shift:].
                For reconstruction (not prediction), use 0.
            epsilon (`float`, *optional*, defaults to 0.1):
                Label smoothing factor. 0.0 = no smoothing (standard CE),
                1.0 = uniform distribution.
            ignore_index (`int`, *optional*, defaults to -100):
                Label value to ignore in loss computation (typically padding).

        Returns:
            `torch.FloatTensor`:
                Scalar loss value averaged over non-ignored positions.

        Note:
            - Labels are clamped to [0, vocab_size - 1] to handle any out-of-range values
            - Loss is computed only on positions where label != ignore_index
            - Supports fp16 inputs via internal upcasting to fp32
        """
        # Either do not shift (for reconstruction) or shift by the window_size
        if shift_labels:
            logits = logits[..., :-shift_labels, :].contiguous()
            labels = labels[..., shift_labels:].contiguous()

        # Ensure logits and labels have the same sequence length
        # logits_seq_len = logits.size(-2)
        # labels_seq_len = labels.size(-1)
        # if logits_seq_len != labels_seq_len:
        #     min_len = min(logits_seq_len, labels_seq_len)
        #     logits = logits[..., :min_len, :].contiguous()
        #     labels = labels[..., :min_len].contiguous()

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        # TODO: Clip the label to avoid the token id execeed vocab_size, but why labels has token execeed voacb_size?
        labels = torch.clamp(labels, min=0, max=vocab_size - 1)
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
    "LanguageDiffusionDecoder",
    "LanguageDiffusionDecoderLMHead",
    "LanguageDiffusionAutoencoder",
]
