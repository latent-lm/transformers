from typing import Optional, Union

import torch

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import LatentCausalLMOutputWithCrossAttentions
from .configuration_latent_gpt2 import LatentGPT2Config
from .modeling_autoencoder import LanguageAutoencoder, GPT2PreTrainedModel
from .modeling_flow_matching import LanguageFlowMatching


class LatnetAutoregressive(GPT2PreTrainedModel, GenerationMixin):
    def __init__(self, config: LatentGPT2Config):
        super().__init__(config)
        self.autoencoder: LanguageAutoencoder = LanguageAutoencoder(config=config)
        # self.fm: LanguageFlowMatching = LanguageFlowMatching(config=config)
        self.fm = None

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
    
    def _chunk_pad_cat(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
    ):
        if input_ids is not None:
            context_input_ids = input_ids[..., :-self.config.window_size].contiguous()
            target_input_ids = input_ids[..., -self.config.window_size:].contiguous()
        
        if inputs_embeds is not None:
            context_inputs_embeds = inputs_embeds[..., :-self.config.window_size, :].contiguous()
            target_inputs_embeds = inputs_embeds[..., -self.config.window_size:, :].contiguous()
        
        preporcess_output = self.autoencoder.encoder.transformer.pre_process_inputs(
            input_ids=context_input_ids, inputs_embeds=context_inputs_embeds)
        
        res_input_ids = torch.cat((preporcess_output.input_ids, target_input_ids), dim=-1)
        res_inputs_embeds = torch.cat((preporcess_output.inputs_embeds, target_inputs_embeds), dim=-2)
        
        return res_input_ids, res_inputs_embeds
    
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
        use_latent_ar: bool = False,
        **kwargs,
    ) -> Union[tuple, LatentCausalLMOutputWithCrossAttentions]:
        """
        labels: The length should be inputs sequence + output segment 
        """
        if use_latent_ar:
            input_ids, inputs_embeds = self._chunk_pad_cat(input_ids=input_ids, inputs_embeds=inputs_embeds)
            ae_output, encoder_output, decoder_output = self.autoencoder(
                input_ids=input_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                # labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                logits_to_keep=logits_to_keep,
                return_encoder_decoder_res=True,
                **kwargs
            )
            # encoder_output = self.autoencoder.encode(
            #     input_ids=input_ids,
            #     past_key_values=past_key_values,
            #     cache_position=cache_position,
            #     attention_mask=attention_mask,
            #     token_type_ids=token_type_ids,
            #     position_ids=position_ids,
            #     inputs_embeds=inputs_embeds,
            #     encoder_hidden_states=encoder_hidden_states,
            #     encoder_attention_mask=encoder_attention_mask,
            #     labels=None,
            #     use_cache=use_cache,
            #     output_attentions=output_attentions,
            #     output_hidden_states=output_hidden_states,
            #     return_dict=return_dict,
            #     logits_to_keep=logits_to_keep,
            #     **kwargs
            # )
            
            fm_output = self.fm(
                inputs_prev_latent=None,
                inputs_latent_timestep=None,
                input_ids=None,
                past_key_values=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=encoder_output.last_tail_hidden_state,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                labels=ae_output.latents,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                logits_to_keep=logits_to_keep,
            )
            
            # decoder_output = self.autoencoder.decode(
            #     input_ids=None,
            #     past_key_values=past_key_values,
            #     cache_position=cache_position,
            #     attention_mask=attention_mask,
            #     token_type_ids=token_type_ids,
            #     position_ids=position_ids,
            #     inputs_embeds=fm_output.last_latent,
            #     encoder_hidden_states=encoder_hidden_states,
            #     encoder_attention_mask=encoder_attention_mask,
            #     labels=labels,
            #     use_cache=use_cache,
            #     output_attentions=output_attentions,
            #     output_hidden_states=output_hidden_states,
            #     return_dict=return_dict,
            #     logits_to_keep=logits_to_keep,
            #     **kwargs
            # )
            return ae_output, encoder_output, decoder_output, fm_output
        else:
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
            return ae_output