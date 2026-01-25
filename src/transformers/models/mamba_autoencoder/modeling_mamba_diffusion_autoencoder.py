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
"""
PyTorch MAMBA Diffusion Autoencoder models.

This module implements a diffusion-based language autoencoder that uses masked diffusion
for decoding latent representations back to text. The architecture consists of:

1. **MambaDiffusionDecoder**: A MAMBA-based decoder that takes latent representations
   and partially/fully masked token sequences as input, using diffusion-style denoising
   to reconstruct the original text.

2. **MambaDiffusionDecoderLMHead**: Wraps the decoder with a language modeling head
   to produce vocabulary logits from the decoder hidden states.

3. **MambaDiffusionAutoencoder**: The complete autoencoder combining an encoder
   (MambaEncoderLatentHead) that compresses text into latent representations and
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
    from transformers import LatentMambaConfig
    from transformers.models.mamba_autoencoder import MambaDiffusionAutoencoder

    # Create configuration
    config = LatentMambaConfig(
        window_size=4,
        latent_dim=768,
        num_hidden_layers_encoder=6,
        num_hidden_layers_decoder=6,
        autoencoder_type="diffusion",
    )

    # Initialize model
    model = MambaDiffusionAutoencoder(config)

    # Forward pass (encoding + decoding)
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    logits = outputs.logits
    latents = outputs.latents
    ```
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
from .modeling_mamba_autoencoder import (
    MambaEncoderLatentHead,
    MambaDecoderBase,
    MambaDecoderUtils,
)


logger = logging.get_logger(__name__)


# =============================================================================
# Diffusion Decoder
# =============================================================================


@auto_docstring(
    custom_intro="""
    MAMBA diffusion decoder model for decoding latent representations back to text.

    This decoder implements a masked diffusion approach where the model receives:
    1. Latent representations (compressed semantic information from the encoder)
    2. Partially or fully masked token sequences

    The decoder learns to predict the original tokens from the masked input conditioned
    on the latent representations. This differs from the multi-head decoder which uses
    separate prediction heads for each position in the window.

    Architecture:
        - Inherits from MambaDecoderBase which provides the core MAMBA layers
        - Uses `embeddings_latent` (Linear layer) to project latents from `latent_dim` to `hidden_size`
        - Uses `embeddings` (Embedding layer) to embed masked/unmasked input tokens
        - Concatenates latent embeddings with token embeddings before processing

    Input Processing:
        The `pre_process_inputs` method handles the input preparation:
        - Latent representations are projected to embeddings via `embeddings_latent`
        - Masked token sequences are embedded via `embeddings`
        - If no input_ids provided, a fully masked sequence is generated
        - Latent embeddings and token embeddings are concatenated
    """,
    custom_args="""
        window_size (`int`):
            The window size for disaggregating latent representations back to sequences.
            Each latent vector is expanded to `window_size` token predictions.
        mask_token_id (`int`):
            Token ID used for masking during diffusion training and inference.
    """,
)
class MambaDiffusionDecoder(MambaDecoderBase):
    """
    MAMBA diffusion decoder that expands latent representations back to token sequences
    using a masked diffusion approach.

    Attributes:
        ae_utils: Utility class for sequence windowing operations.
        embeddings_latent: Linear projection from latent_dim to hidden_size.
        embeddings: Token embedding layer.
        layers: List of MambaBlock transformer layers.
        norm_f: Final RMS normalization layer.
    """

    def __init__(self, config: LatentMambaConfig):
        """
        Initialize the MambaDiffusionDecoder.

        Args:
            config: Configuration object containing model hyperparameters.
        """
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
        Pre-process and segment inputs for the diffusion decoder.

        This method prepares the inputs for the MAMBA blocks by:
        1. Aggregating input sequences into window-sized segments for parallel processing
        2. Converting latent vectors to embeddings via `embeddings_latent`
        3. Converting masked token IDs to embeddings via `embeddings`
        4. Concatenating latent embeddings with token embeddings

        The diffusion decoder expects both latent representations AND masked token sequences.
        The latent provides the conditioning signal, while the masked tokens are progressively
        denoised during inference.

        Args:
            input_ids: Masked token sequence where some positions contain `mask_token_id`.
                If None and `inputs_latents` is provided, a fully masked sequence
                of length `window_size` is generated automatically.
            inputs_latents: Latent representations from the encoder. These are projected to
                `hidden_size` via `embeddings_latent` before being concatenated with token embeddings.
            inputs_embeds: Pre-computed embeddings. If provided, `inputs_latents` and `input_ids`
                should not be provided simultaneously.

        Returns:
            PreprocessOutput containing:
                - `input_ids`: The aggregated and potentially masked input IDs
                - `inputs_latents`: The aggregated latent representations
                - `inputs_embeds`: The concatenated embeddings ready for the MAMBA blocks,
                  with shape `(batch_size * segment_num, 1 + window_size, hidden_size)`

        Raises:
            ValueError: If both `inputs_latents` and `inputs_embeds` are provided simultaneously.
            ValueError: If both `input_ids` and `inputs_embeds` are provided simultaneously.

        Note:
            The output `inputs_embeds` has the latent embeddings prepended to the token
            embeddings: `[latent_embed, masked_token_embed_1, ..., masked_token_embed_n]`
        """
        if input_ids is not None:
            input_ids = self.ae_utils.agg_sequence_mask_diffusion(sequence=input_ids)
        if inputs_latents is not None:
            inputs_latents = self.ae_utils.agg_sequence(sequence=inputs_latents)
        if inputs_embeds is not None:
            inputs_embeds = self.ae_utils.agg_sequence(sequence=inputs_embeds)
            if inputs_latents is not None:
                raise ValueError("inputs_latents and inputs_embeds are provided at the same time, only one is accepted.")
            if input_ids is not None:
                raise ValueError("input_ids and inputs_embeds are provided at the same time, only one is accepted.")
            return PreprocessOutput(input_ids=input_ids, inputs_latents=inputs_latents, inputs_embeds=inputs_embeds)

        if inputs_latents is not None:
            inputs_embeds = self.embeddings_latent(inputs_latents)

        if input_ids is not None:
            # If there is input_ids, convert it into embeddings
            masked_input_embeds = self.embeddings(input_ids)
        else:
            # If there is no input masked sequence, generate a fully masked sequence
            masked_input_ids: torch.LongTensor = (
                torch.ones(*(inputs_embeds.shape[:-2] + (self.config.window_size,))) * self.config.mask_token_id
            ).long().to(inputs_embeds.device)
            masked_input_embeds: torch.FloatTensor = self.embeddings(masked_input_ids)

        full_inputs_embeds: torch.FloatTensor = torch.cat([inputs_embeds, masked_input_embeds], dim=-2)
        return PreprocessOutput(input_ids=input_ids, inputs_latents=inputs_latents, inputs_embeds=full_inputs_embeds)

    def post_process_outputs(
        self,
        outputs: MambaOutput,
    ) -> BaseAutoencoderOutputWithPastAndCrossAttentions:
        """
        Post-process and reshape decoder outputs back to batch dimensions.

        This method reverses the segment-based processing done in `pre_process_inputs`,
        converting the outputs from shape `(batch_size * segment_num, seq_len, hidden_size)`
        back to `(batch_size, segment_num * seq_len, hidden_size)`.

        It also extracts specific hidden states needed for downstream processing:
        - `last_tail_hidden_state`: The final position's hidden state
        - `last_window_hidden_state`: Hidden states for the last `window_size` positions

        Args:
            outputs: The raw output from the parent forward pass.

        Returns:
            BaseAutoencoderOutputWithPastAndCrossAttentions with reshaped tensors.
        """
        if outputs is None:
            return outputs
        return BaseAutoencoderOutputWithPastAndCrossAttentions(
            last_tail_hidden_state=self.ae_utils.split_sequence(
                sequence=outputs.last_hidden_state[:, -1:, ...]
            ),
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

    def init_weight_from_pretrained(self, pretrained_model: MambaModel) -> "MambaDiffusionDecoder":
        """
        Initialize the decoder weights from a pre-trained MAMBA model.

        This method copies weights from a pre-trained MAMBA model to initialize
        the decoder. It uses the last `num_hidden_layers_decoder` layers from
        the pretrained model.

        The following components are copied:
        - `embeddings`: Token embeddings (for embedding masked input tokens)
        - `layers`: MAMBA blocks (last N layers)
        - `norm_f`: Final RMS normalization

        Note that `embeddings_latent` is NOT copied since it's a Linear projection
        from `latent_dim` to `hidden_size`, not an embedding layer.

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

    @auto_docstring
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
        """
        Forward pass through the diffusion decoder.

        Args:
            input_ids: Masked token sequence. Positions containing `mask_token_id` will be
                predicted by the model. If None, a fully masked sequence is generated.
            inputs_latents: Latent representations from the encoder to condition the decoding.
            inputs_embeds: Pre-computed embeddings (alternative to `inputs_latents`).
            return_segment: Whether to reshape outputs back to batch dimensions.

        Returns:
            BaseAutoencoderOutputWithPastAndCrossAttentions containing:
                - `last_tail_hidden_state`: Final position hidden state per segment
                - `last_window_hidden_state`: Hidden states for decoded token positions
                - `last_hidden_state`: Full hidden states from final layer
                - `hidden_states`: All layer hidden states (if requested)
        """
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
# Diffusion Decoder with LM Head
# =============================================================================


@auto_docstring(
    custom_intro="""
    MAMBA diffusion decoder with language modeling head for text generation.

    This model wraps the `MambaDiffusionDecoder` with a linear projection head
    that maps hidden states to vocabulary logits. Unlike the multi-head variant
    which uses separate heads for each position in the window, this model uses
    a single shared LM head applied to all positions.

    Architecture:
        - `transformer`: MambaDiffusionDecoder for processing inputs
        - `lm_head`: Linear projection from hidden_size to vocab_size

    Key Differences from Multi-Head Decoder:
        1. Uses a single shared LM head instead of `window_size` separate heads
        2. Expects masked input tokens (diffusion-style) rather than just latents
        3. Applies LM head to `last_window_hidden_state` (token positions only)
    """,
)
class MambaDiffusionDecoderLMHead(MambaPreTrainedModel, GenerationMixin):
    """
    MAMBA diffusion decoder with a single shared language modeling head.

    Attributes:
        transformer: The diffusion decoder transformer model.
        lm_head: Linear layer projecting hidden states to vocabulary logits.
    """

    def __init__(self, config: LatentMambaConfig):
        """
        Initialize the MambaDiffusionDecoderLMHead.

        Args:
            config: Configuration object containing model hyperparameters.
        """
        super().__init__(config)
        self.transformer = MambaDiffusionDecoder(config=config)

        # Single head projection mechanism (shared across all positions)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def init_weight_from_pretrained(self, pretrained_model: MambaForCausalLM) -> "MambaDiffusionDecoderLMHead":
        """
        Initialize model weights from a pre-trained MAMBA LM model.

        Args:
            pretrained_model: The pre-trained MambaForCausalLM model.

        Returns:
            Self for method chaining.
        """
        self.transformer.init_weight_from_pretrained(pretrained_model=pretrained_model.backbone)
        self.lm_head = copy.deepcopy(pretrained_model.lm_head)
        return self

    @auto_docstring(
        custom_intro="Forward pass for the MAMBA diffusion decoder with LM head.",
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
        Forward pass to produce token logits from latent representations.

        Args:
            inputs_latents: Latent representations to be decoded.
            input_ids: Masked token sequence for conditioning.
            labels: Not used, included for API consistency.
            logits_to_keep: Number of logits to keep from the end.

        Returns:
            CausalLMAutoencoderOutputWithCrossAttentions containing logits and hidden states.
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
        last_window_hidden_state = transformer_outputs.last_window_hidden_state

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(last_window_hidden_state[:, slice_indices, :])

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
# Complete Diffusion Autoencoder
# =============================================================================


@auto_docstring(
    custom_intro="""
    Complete diffusion-based MAMBA language autoencoder combining encoder and decoder.

    This model implements a full autoencoder architecture for text compression and
    reconstruction using a diffusion-style decoder. The encoder compresses input
    text into latent representations, and the decoder reconstructs the text using
    masked diffusion.

    Architecture Overview:
        - **Encoder** (`MambaEncoderLatentHead`): Processes input tokens through
          MAMBA blocks, then projects to latent space via `latent_head`.
          Each `window_size` tokens are compressed into a single latent vector.

        - **Decoder** (`MambaDiffusionDecoderLMHead`): Takes latent vectors and
          masked token sequences, processes them through MAMBA blocks, and
          applies an LM head to predict the original tokens.

    Weight Tying:
        The encoder's `latent_head` and decoder's `embeddings_latent` share weights (tied),
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
class MambaDiffusionAutoencoder(MambaPreTrainedModel, GenerationMixin):
    """
    Complete MAMBA diffusion autoencoder for text compression and reconstruction.

    Attributes:
        encoder: The encoder model that compresses text to latent representations.
        decoder: The diffusion decoder that reconstructs text from latents.
    """

    _tied_weights_keys = {"encoder.latent_head.weight": "decoder.transformer.embeddings_latent.weight"}

    def __init__(self, config: LatentMambaConfig):
        """
        Initialize the MambaDiffusionAutoencoder.

        Args:
            config: Configuration object containing model hyperparameters including:
                - `window_size`: Tokens per latent vector
                - `latent_dim`: Latent space dimensionality
                - `num_hidden_layers_encoder`: Encoder MAMBA layers
                - `num_hidden_layers_decoder`: Decoder MAMBA layers
                - `dae_num_train_timesteps`: Diffusion training timesteps
        """
        super().__init__(config)
        self.encoder: MambaEncoderLatentHead = MambaEncoderLatentHead(config=config)
        self.decoder: MambaDiffusionDecoderLMHead = MambaDiffusionDecoderLMHead(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    def init_weight_from_pretrained(self, pretrained_model: MambaForCausalLM) -> "MambaDiffusionAutoencoder":
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

        Note:
            This method currently returns None (masking disabled). Enable the
            commented code to implement actual masking during training.

        Args:
            labels: Target token sequence to be masked.
            timestep: Specific timestep to use for determining masking ratio.

        Returns:
            Masked token sequence or None (masking disabled).
        """
        # Currently returns None - masking is handled differently or disabled
        return None

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
            input_ids: Masked token sequence for diffusion decoding.
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
    ) -> Union[Tuple, CausalLMAutoencoderOutputWithCrossAttentions, Tuple]:
        """
        Complete forward pass through the diffusion autoencoder (encode + decode).

        Args:
            input_ids: Input token IDs to encode.
            inputs_embeds: Pre-computed input embeddings.
            labels: Target token IDs for computing reconstruction loss.
            return_encoder_decoder_res: If True, returns tuple of
                (autoencoder_output, encoder_output, decoder_output).

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

        encoder_input_ids = self._get_decoder_inputs(labels=labels)

        decoder_output = self.decoder(
            input_ids=encoder_input_ids,
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

        This loss function implements label smoothing, which helps prevent
        overconfident predictions and improves generalization.

        Args:
            logits: Predicted token logits.
            labels: Target token IDs.
            vocab_size: Size of the vocabulary for clamping label indices.
            shift_labels: Number of positions to shift labels.
            epsilon: Label smoothing factor. 0.0 = no smoothing.
            ignore_index: Label value to ignore in loss computation.

        Returns:
            Scalar loss value averaged over non-ignored positions.
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
    "MambaDiffusionDecoder",
    "MambaDiffusionDecoderLMHead",
    "MambaDiffusionAutoencoder",
]
