from typing import Optional, Union

import torch
from torch import nn

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import (
    CausalLMAutoencoderOutputWithCrossAttentions,
    LatentCausalLMOutputWithCrossAttentions,
)
from .configuration_latent_gpt2 import LatentGPT2Config
from .modeling_autoencoder import LanguageAutoencoder, GPT2PreTrainedModel
from .modeling_diffusion_autoencoder import LanguageDiffusionAutoencoder
from .modeling_flow_matching import LanguageFlowMatching

class LatnetAutoregressive(GPT2PreTrainedModel, GenerationMixin):
    def __init__(self, config: LatentGPT2Config):
        super().__init__(config)
        if config.autoencoder_type == LatentGPT2Config.AUTOENCODER_TYPE_DIFFUSION:
            self.autoencoder: LanguageDiffusionAutoencoder = LanguageDiffusionAutoencoder(config=config)
        elif config.autoencoder_type == LatentGPT2Config.AUTOENCODER_TYPE_MULTI_HEAD:
            self.autoencoder: LanguageAutoencoder = LanguageAutoencoder(config=config)
        else:
            raise ValueError(f"Unknown autoencoder type: {config.autoencoder_type}")
            
        self.fm: LanguageFlowMatching = LanguageFlowMatching(config=config)
        # self.fm = None

        # Initialize weights and apply final processing
        self.post_init()

    def init_weight_from_pretrained(self, pretrained_model):
        """
        Initializes the autoencoder from a pre-trained model.

        Args:
            pretrained_model: The pre-trained model to use.

        Returns:
            self
        """
        self.autoencoder.init_weight_from_pretrained(pretrained_model=pretrained_model)
        if self.fm is not None:
            self.fm.init_weight_from_pretrained(pretrained_model=pretrained_model)
        return self
    
    def _shift_logits_labels(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        shift_by: int = 0,
        cut_eq_len: bool = False,
    ):
        # Either do not shift, for reconstruction, or shift by the window_size
        if shift_by:
            logits = logits[..., :-shift_by, :].contiguous()
            labels = labels[..., shift_by:].contiguous()
        #  Cut off to get sequences with the same length
        if cut_eq_len:
            logits_seq_len = logits.size(-2)
            labels_seq_len = labels.size(-1)
            if logits_seq_len != labels_seq_len:
                min_len = min(logits_seq_len, labels_seq_len)
                logits = logits[..., :min_len, :].contiguous()
                labels = labels[..., :min_len].contiguous()
        return logits, labels
    
    def _format_autoencoder_dict_output(
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
    
    def forward_autoencoder_fm(
        self,
        inputs_prev_latent: Optional[torch.FloatTensor] = None,
        inputs_latent_timestep: Optional[torch.FloatTensor] = None,
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
        return_encoder_decoder_res: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[tuple, LatentCausalLMOutputWithCrossAttentions]:
        """
        Forward pass autoencoder and flow matching, take gradients on both, use autoencoder + flow matching + end-to-end loss
        Assume input sequence is the same as labels sequence
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Extract last self.config.window_size tokens as target, the remaining as context. Pad the context to make it divisible to self.config.window_size
        # Treat the last chunk of the input sequence as target
        input_ids_joint = self.autoencoder.encoder.transformer.ae_utils.split_context_target_then_cat(
            sequence=input_ids, is_padding=True, return_context_target=False)
        inputs_embeds_joint = self.autoencoder.encoder.transformer.ae_utils.split_context_target_then_cat(
            sequence=inputs_embeds, is_padding=True, return_context_target=False)
        labels_joint, labels_context, labels_target = self.autoencoder.encoder.transformer.ae_utils.split_context_target_then_cat(
            sequence=labels, is_padding=True, return_context_target=True)

        ae_output_combined = self.autoencoder(
            input_ids=input_ids_joint,
            past_key_values=None,
            cache_position=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=inputs_embeds_joint,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=labels_joint,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            logits_to_keep=0,
            return_encoder_decoder_res=return_encoder_decoder_res,
        )
        if return_encoder_decoder_res:
            ae_output, encoder_output, decoder_output = ae_output_combined
        else:
            ae_output = ae_output_combined
        
        fm_output = self.fm(
            inputs_prev_latent=None,
            inputs_latent_timestep=None,
            input_ids=None,
            past_key_values=None,
            cache_position=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=ae_output.latents[:, :-1, ...],
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=ae_output.latents[:, -1:, ...],
            use_cache=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            logits_to_keep=0,
        )
        
        decoder_output = self.autoencoder.decode(
            input_ids=None,
            past_key_values=None,
            cache_position=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=fm_output.estimates,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=labels_target,
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
            # Flatten the tokens
            loss = self.loss_function(
                # Skip the logit at the first dimension, which is the output corresponsing to the latent
                logits=logits,
                labels=labels_target,
                vocab_size=self.config.vocab_size,
                **kwargs,
            ) + fm_output.loss + ae_output.loss

        latent_lm_output = LatentCausalLMOutputWithCrossAttentions(
            last_tail_hidden_state=decoder_output.last_tail_hidden_state if output_hidden_states else None,
            last_hidden_state=decoder_output.last_hidden_state if output_hidden_states else None,
            loss=loss,
            logits=logits,
            past_key_values=decoder_output.past_key_values,
            hidden_states=decoder_output.hidden_states if output_hidden_states else None,
            attentions=decoder_output.attentions if output_attentions else None,
            cross_attentions=decoder_output.cross_attentions if output_attentions else None,
            latents=ae_output.latents,
        )
        
        if not return_dict:
            output = (logits,) + decoder_output[1:]
            return ((loss,) + output) if loss is not None else output

        # When return_encoder_decoder_res=True, return tuple of (ar_outputs, ltar_output, encoder_output)
        if return_encoder_decoder_res:
            # ltar_output is the latent AR prediction (placeholder: using encoder_output for now)
            # encoder_output contains the actual encoder latents
            ret_ae_output = self._format_autoencoder_dict_output(dict_output=ae_output, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
            ret_encoder_output = self._format_autoencoder_dict_output(dict_output=encoder_output, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
            ret_decoder_output = self._format_autoencoder_dict_output(dict_output=decoder_output, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
            return (latent_lm_output, ret_ae_output, ret_encoder_output, ret_decoder_output)
        return latent_lm_output
    
    def forward_fm(
        self,
        inputs_prev_latent: Optional[torch.FloatTensor] = None,
        inputs_latent_timestep: Optional[torch.FloatTensor] = None,
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
        return_encoder_decoder_res: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[tuple, LatentCausalLMOutputWithCrossAttentions]:
        """
        Forward pass autoencoder and flow matching, take gradients on flow matching, use flow matching loss
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Extract last self.config.window_size tokens as target, the remaining as context. Pad the context to make it divisible to self.config.window_size
        # Treat the last chunk of the input sequence as target
        if labels is not None:
            labels_target = labels[..., -self.config.window_size:]

        with torch.no_grad():
            encoder_output = self.autoencoder.encode(
                input_ids=input_ids,
                past_key_values=None,
                cache_position=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                logits_to_keep=0,
            )
        
        fm_output = self.fm(
            inputs_prev_latent=None,
            inputs_latent_timestep=None,
            input_ids=None,
            past_key_values=None,
            cache_position=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=encoder_output.latents[:, :-1, ...],
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=encoder_output.latents[:, -1:, ...],
            use_cache=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            logits_to_keep=0,
        )
        with torch.no_grad():
            decoder_output = self.autoencoder.decode(
                input_ids=None,
                past_key_values=None,
                cache_position=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=fm_output.estimates,
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

        loss = fm_output.loss

        latent_lm_output = LatentCausalLMOutputWithCrossAttentions(
            last_tail_hidden_state=decoder_output.last_tail_hidden_state if output_hidden_states else None,
            last_hidden_state=decoder_output.last_hidden_state if output_hidden_states else None,
            loss=loss,
            logits=logits,
            past_key_values=decoder_output.past_key_values,
            hidden_states=decoder_output.hidden_states if output_hidden_states else None,
            attentions=decoder_output.attentions if output_attentions else None,
            cross_attentions=decoder_output.cross_attentions if output_attentions else None,
            latents=encoder_output.latents,
        )
        
        if not return_dict:
            output = (logits,) + decoder_output[1:]
            return ((loss,) + output) if loss is not None else output

        # When return_encoder_decoder_res=True, return tuple of (ar_outputs, ltar_output, encoder_output)
        if return_encoder_decoder_res:
            # ltar_output is the latent AR prediction (placeholder: using encoder_output for now)
            # encoder_output contains the actual encoder latents
            ret_encoder_output = self._format_autoencoder_dict_output(dict_output=encoder_output, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
            ret_decoder_output = self._format_autoencoder_dict_output(dict_output=decoder_output, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
            return (latent_lm_output, ret_encoder_output, ret_decoder_output)
        return latent_lm_output
    
    def forward_latent_fm(
        self,
        inputs_prev_latent: Optional[torch.FloatTensor] = None,
        inputs_latent_timestep: Optional[torch.FloatTensor] = None,
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
        return_encoder_decoder_res: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[tuple, LatentCausalLMOutputWithCrossAttentions]:
        """
        Forward pass latent flow matching end-to-end, take gradients on flow matching, use end-to-end loss
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Extract last self.config.window_size tokens as target, the remaining as context. Pad the context to make it divisible to self.config.window_size
        # Treat the last chunk of the input sequence as target
        if labels is not None:
            labels_target = labels[..., -self.config.window_size:]

        with torch.no_grad():
            encoder_output = self.autoencoder.encode(
                input_ids=input_ids,
                past_key_values=None,
                cache_position=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                logits_to_keep=0,
            )

        fm_output = self.fm(
            inputs_prev_latent=None,
            inputs_latent_timestep=None,
            input_ids=None,
            past_key_values=None,
            cache_position=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=encoder_output.latents[:, :-1, ...],
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=encoder_output.latents[:, -1:, ...],
            use_cache=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            logits_to_keep=0,
        )

        with torch.no_grad():
            decoder_output = self.autoencoder.decode(
                input_ids=None,
                past_key_values=None,
                cache_position=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=fm_output.estimates,
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
            # Flatten the tokens
            loss = self.loss_function(
                # Skip the logit at the first dimension, which is the output corresponsing to the latent
                logits=logits,
                labels=labels_target,
                vocab_size=self.config.vocab_size,
                **kwargs,
            ) + fm_output.loss

        latent_lm_output = LatentCausalLMOutputWithCrossAttentions(
            last_tail_hidden_state=decoder_output.last_tail_hidden_state if output_hidden_states else None,
            last_hidden_state=decoder_output.last_hidden_state if output_hidden_states else None,
            loss=loss,
            logits=logits,
            past_key_values=decoder_output.past_key_values,
            hidden_states=decoder_output.hidden_states if output_hidden_states else None,
            attentions=decoder_output.attentions if output_attentions else None,
            cross_attentions=decoder_output.cross_attentions if output_attentions else None,
            latents=encoder_output.latents,
        )

        if not return_dict:
            output = (logits,) + decoder_output[1:]
            return ((loss,) + output) if loss is not None else output

        # When return_encoder_decoder_res=True, return tuple of (ar_outputs, ltar_output, encoder_output)
        if return_encoder_decoder_res:
            # ltar_output is the latent AR prediction (placeholder: using encoder_output for now)
            # encoder_output contains the actual encoder latents
            ret_encoder_output = self._format_autoencoder_dict_output(dict_output=encoder_output, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
            ret_decoder_output = self._format_autoencoder_dict_output(dict_output=decoder_output, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
            return (latent_lm_output, ret_encoder_output, ret_decoder_output)
        return latent_lm_output
    
    def forward_autoencoder(
        self,
        inputs_prev_latent: Optional[torch.FloatTensor] = None,
        inputs_latent_timestep: Optional[torch.FloatTensor] = None,
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
    ) -> Union[tuple, LatentCausalLMOutputWithCrossAttentions]:
        """
        Forward pass autoencoder, take gradients on autoencoder, use autoencoder loss
        Assume input sequence is the same as labels sequence
        """
        ae_output = self.autoencoder(
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
            **kwargs
        )
        return LatentCausalLMOutputWithCrossAttentions(
            last_tail_hidden_state=ae_output.last_tail_hidden_state,
            last_hidden_state=ae_output.last_hidden_state,
            loss=ae_output.loss,
            logits=ae_output.logits,
            past_key_values=ae_output.past_key_values,
            hidden_states=ae_output.hidden_states,
            attentions=ae_output.attentions,
            cross_attentions=ae_output.cross_attentions,
            latents=ae_output.latents,
        )

    def forward(
        self,
        inputs_prev_latent: Optional[torch.FloatTensor] = None,
        inputs_latent_timestep: Optional[torch.FloatTensor] = None,
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
        forward_type: str = "latent_fm",
        **kwargs,
    ) -> Union[tuple, LatentCausalLMOutputWithCrossAttentions]:
        """
        labels: The length should be inputs sequence + output segment 
        """
        if forward_type == "autoencoder_fm":
            return self.forward_autoencoder_fm(
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
                **kwargs
            )
        elif forward_type == "fm":
            return self.forward_fm(
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
                **kwargs
            )
        elif forward_type == "latent_fm":
            return self.forward_latent_fm(
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
                **kwargs
            )
        elif forward_type == "autoencoder":
            return self.forward_autoencoder(
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
                **kwargs
            )
        else:
            raise NotImplementedError(f"forward_type = {forward_type} is not supported.")

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
            logits (`torch.FloatTensor` of shape `(batch_size, seq_len, vocab_size)`):
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
        logits_seq_len = logits.size(-2)
        labels_seq_len = labels.size(-1)
        if logits_seq_len != labels_seq_len:
            min_len = min(logits_seq_len, labels_seq_len)
            logits = logits[..., :min_len, :].contiguous()
            labels = labels[..., :min_len].contiguous()

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
