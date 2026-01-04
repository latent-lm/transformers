from typing import Optional, Union

import torch

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import LatentCausalLMOutputWithCrossAttentions
from .configuration_latent_gpt2 import LatentGPT2Config
from .modeling_gpt2 import LanguageAutoencoder, GPT2PreTrainedModel
from .flow_matching import FlowMatchingModel


class LatnetAutoregressive(GPT2PreTrainedModel, GenerationMixin):
    def __init__(self, config: LatentGPT2Config):
        super().__init__(config)
        self.autoencoder: LanguageAutoencoder = LanguageAutoencoder(config=config)
        # self.fm: FlowMatchingModel = FlowMatchingModel(config=config)
        self.fm: FlowMatchingModel = None

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
        use_latent_ar: bool = False,
        **kwargs,
    ) -> Union[tuple, LatentCausalLMOutputWithCrossAttentions]:
        if use_latent_ar:
            raise NotImplementedError("Haven't incorporated flow matching model yet.")
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
                use_latent_ar=use_latent_ar,
                **kwargs
            )
            return ae_output