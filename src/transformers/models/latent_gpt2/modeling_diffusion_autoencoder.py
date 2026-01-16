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
# from collections.abc import Callable
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
    # CausalLMOutputWithCrossAttentions,
    # QuestionAnsweringModelOutput,
    # SequenceClassifierOutputWithPast,
    # TokenClassifierOutput,
    CausalLMAutoencoderOutputWithCrossAttentions,
)

from ...utils import (
    # ModelOutput,
    auto_docstring,
    logging,
)
from .configuration_latent_gpt2 import LatentGPT2Config
from .modeling_gpt2 import GPT2ModelBase, GPT2PreTrainedModel, GPT2LMHeadModel, GPT2Block
from .modeling_autoencoder import LanguageEncoderLatentHead, LanguageDecoderBase, LanguageDecoderUtils

logger = logging.get_logger(__name__)
        
@auto_docstring(
    custom_intro="Language decoder model based on GPT-2 for decoding latent representations back to text.",
    custom_args="window_size (`int`): The window size for disaggregating latent representations back to sequences.",
)
class LanguageDiffusionDecoder(LanguageDecoderBase):
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
            input_ids = self.ae_utils.agg_sequence_mask_diffusion(sequence=input_ids)
        if inputs_latents is not None:
            inputs_latents = self.ae_utils.agg_sequence(sequence=inputs_latents)
        if inputs_embeds is not None:
            inputs_embeds = self.ae_utils.agg_sequence(sequence=inputs_embeds)
            if inputs_latents is not None:
                raise ValueError(f"inputs_latents and inputs_embeds are provided at the same time, only one of them is accepted.")

        if inputs_latents is not None:
            inputs_embeds = self.wte_latent(inputs_latents)
        if input_ids is not None:
            # If there is input_ids, convert it into embeddings
            masked_input_embeds = self.wte(input_ids)
        else:
            # If there is no input masked sequecne, random sample a fully masked sequence
            # Concatenate Partially / Fully Masked target sequence
            masked_input_ids: torch.LongTensor = (torch.ones(*inputs_embeds.shape[:-1]) * self.config.pad_token_id).long().to(self.device)
            masked_input_embeds: torch.FloatTensor = self.wte(masked_input_ids)

        full_inputs_embeds: torch.FloatTensor = torch.cat([inputs_embeds, masked_input_embeds], dim=-2)
        return PreprocessOutput(input_ids=input_ids, inputs_latents=inputs_latents, inputs_embeds=full_inputs_embeds)
    
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
        return BaseAutoencoderOutputWithPastAndCrossAttentions(
            # Only keep the last hidden_state of hidden_state of the last layer
            last_tail_hidden_state=self.ae_utils.split_sequence(sequence=outputs.last_hidden_state[:, -1:, ...]),
            # Only keep the last self.config.window_size hidden_state of hidden_state of the last layer
            last_window_hidden_state=self.ae_utils.split_sequence(sequence=outputs.last_hidden_state[:, -self.config.window_size:, ...]),
            last_hidden_state=self.ae_utils.split_sequence(sequence=outputs.last_hidden_state),
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

class LanguageDiffusionDecoderLMHead(GPT2PreTrainedModel, GenerationMixin):
    # _tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}

    def __init__(self, config: LatentGPT2Config):
        super().__init__(config)
        self.transformer = LanguageDiffusionDecoder(config=config)
        
        # Single head projection mechanism
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

    # def _project_with_multi_heads(self, latents: torch.Tensor) -> torch.Tensor:
    #     """
    #     Project decoder hidden states to logits using multi-head mechanism.
        
    #     Args:
    #         latents: Shape (batch_size, sequence_length, hidden_size)
        
    #     Returns:
    #         logits: Shape (batch_size, sequence_length * config.window_size, vocab_size)
    #     """
    #     multi_head_logits: List[torch.Tensor] = []
    #     for i in range(self.config.window_size):
    #         multi_head_logits.append(self.multi_lm_head[i](latents))
    #     return self.transformer.ae_utils.flatten_multi_heads_logits(logits=multi_head_logits)

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
            output_hidden_states=output_hidden_states,
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
            last_tail_hidden_state=transformer_outputs.last_tail_hidden_state,
            last_window_hidden_state=transformer_outputs.last_window_hidden_state,
            last_hidden_state=transformer_outputs.last_hidden_state,
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
            latent_embeds=inferred_inputs_embeds,
            latents=inputs_latents,
        )

class LanguageDiffusionAutoencoder(GPT2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"encoder.latent_head.weight": "decoder.transformer.wte_latent.weight"}
    def __init__(self, config: LatentGPT2Config):
        super().__init__(config)
        # self.window_size: int = window_size
        self.encoder: LanguageEncoderLatentHead = LanguageEncoderLatentHead(config=config)
        self.decoder: LanguageDiffusionDecoderLMHead = LanguageDiffusionDecoderLMHead(config=config)
        
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

    def _get_decoder_inputs(
        self,
        labels: torch.LongTensor,
        timestep: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Generate masked decoder inputs from labels using random masking for diffusion training.

        The masking ratio is drawn uniformly from discrete steps:
        0/T, 1/T, 2/T, ..., T/T where T = fm_num_train_timesteps.

        Args:
            labels: Target token sequence of shape (batch_size, seq_len).
            timestep: Optional specific timestep to use. If None, samples uniformly
                from [0, fm_num_train_timesteps].

        Returns:
            Masked token sequence of shape (batch_size, seq_len) where some tokens
            are replaced with mask_token_id based on the masking ratio.
        """
        if labels is None:
            return None

        # TODO: Clip the label to avoid the token id execeed vocab_size, but why labels has token execeed voacb_size?
        # Clamp labels to be within the vocab size
        # print(f"Before labels: {labels.shape}, max: {torch.max(labels)}, min: {torch.min(labels)}")
        labels = torch.clamp(labels, min=0, max=self.config.vocab_size - 1)
        # print(f"After labels: {labels.shape}, max: {torch.max(labels)}, min: {torch.min(labels)}")

        batch_size, seq_len = labels.shape
        device = labels.device
        num_timesteps: int = self.config.dae_num_train_timesteps
        mask_token_id: int = self.config.mask_token_id

        masking_ratios: torch.FloatTensor = None
        if num_timesteps is None:
            # If num_timesteps is None, use continuous timestep
            masking_ratios = torch.rand((batch_size, 1), device=device, dtype=torch.float)
        else:
            # If num_timesteps is None, use discrete timestep
            # Sample timestep uniformly from [0, num_timesteps] for each sample in batch
            if timestep is not None:
                # Use provided timestep for all samples
                timesteps = torch.full((batch_size, 1), timestep, device=device, dtype=torch.long)
            else:
                # Sample random timestep for each sample: t in {0, 1, 2, ..., num_timesteps}
                timesteps = torch.randint(0, num_timesteps + 1, (batch_size, 1), device=device)

            # Compute masking ratio for each sample: ratio = t / num_timesteps
            # Shape: (batch_size, 1) for broadcasting
            if num_timesteps == 0:
                # If num_timesteps == 0, only trained on fully masked sequence
                masking_ratios = torch.ones((batch_size, 1), device=device, dtype=torch.float)
            else:
                # If num_timesteps > 0, trained on fully and paritally masked sequence
                masking_ratios = timesteps.float() / num_timesteps
            # masking_ratios = masking_ratios.unsqueeze(-1)  # (batch_size, 1)

        # Generate random values for each position
        random_probs = torch.rand(batch_size, seq_len, device=device)
        # Create mask: True where we should mask (random < masking_ratio)
        mask = random_probs < masking_ratios

        # Apply masking: replace masked positions with mask_token_id
        masked_labels = labels.clone()
        masked_labels.masked_fill_(mask, mask_token_id)

        return masked_labels

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
            attention_mask=attention_mask,
            cache_position=cache_position,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            # past_key_values=None,
            # attention_mask=None,
            # cache_position=None,
            # token_type_ids=None,
            # position_ids=None,
            # inputs_embeds=None,
            # encoder_hidden_states=None,
            # encoder_attention_mask=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
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
            cache_position=cache_position,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            # past_key_values=None,
            # attention_mask=None,
            # cache_position=None,
            # token_type_ids=None,
            # position_ids=None,
            inputs_embeds=encoder_output.last_tail_hidden_state,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            # encoder_hidden_states=None,
            # encoder_attention_mask=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        return decoder_output

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
        # print(f"labels: {labels.shape}")
        encoder_input_ids = self._get_decoder_inputs(labels=labels)
        # print(f"encoder_input_ids: {encoder_input_ids.shape}")
        decoder_output = self.decoder(
            input_ids=encoder_input_ids,
            # inputs_latents=None,
            inputs_latents=encoder_output.latents,
            # past_key_values=past_key_values,
            # attention_mask=attention_mask,
            # cache_position=cache_position,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            past_key_values=None,
            attention_mask=None,
            cache_position=None,
            token_type_ids=None,
            position_ids=None,
            # inputs_embeds=encoder_output.latent_embeds,
            inputs_embeds=None,
            # encoder_hidden_states=encoder_hidden_states,
            # encoder_attention_mask=encoder_attention_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
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
                # Skip the logit at the first dimension, which is the output corresponsing to the latent
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + decoder_output[1:]
            return ((loss,) + output) if loss is not None else output

        ae_output = CausalLMAutoencoderOutputWithCrossAttentions(
            last_tail_hidden_state=decoder_output.last_tail_hidden_state,
            last_window_hidden_state=decoder_output.last_window_hidden_state,
            last_hidden_state=decoder_output.last_hidden_state,
            loss=loss,
            logits=logits,
            past_key_values=decoder_output.past_key_values,
            hidden_states=decoder_output.hidden_states,
            attentions=decoder_output.attentions,
            cross_attentions=decoder_output.cross_attentions,
            latents=encoder_output.last_tail_hidden_state,
        )
        
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

        # TODO: Claude Code update, double check, Ensure logits and labels have the same sequence length
        logits_seq_len = logits.size(-2)
        labels_seq_len = labels.size(-1)
        if logits_seq_len != labels_seq_len:
            min_len = min(logits_seq_len, labels_seq_len)
            logits = logits[..., :min_len, :].contiguous()
            labels = labels[..., :min_len].contiguous()
        # TODO: Claude Code update ends

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
