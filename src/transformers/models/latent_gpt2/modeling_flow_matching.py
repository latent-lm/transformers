# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math


# class FlowMatchingModel(nn.Module):
#     """
#     Flow Matching model for mapping latent hidden states to semantic space.
#     Uses conditional flow matching with optimal transport paths.
#     """
    
#     def __init__(
#         self,
#         hidden_dim: int,
#         semantic_dim: int,
#         time_embed_dim: int = 256,
#         num_layers: int = 4,
#         num_heads: int = 8,
#         dropout: float = 0.1
#     ):
#         """
#         Args:
#             hidden_dim: Dimension of input hidden states from language model
#             semantic_dim: Dimension of the target semantic space
#             time_embed_dim: Dimension for time step embeddings
#             num_layers: Number of transformer layers in the flow network
#             num_heads: Number of attention heads
#             dropout: Dropout rate
#         """
#         super().__init__()
        
#         self.hidden_dim = hidden_dim
#         self.semantic_dim = semantic_dim
#         self.time_embed_dim = time_embed_dim
        
#         # Time embedding network (sinusoidal + MLP)
#         self.time_mlp = nn.Sequential(
#             nn.Linear(time_embed_dim, time_embed_dim * 4),
#             nn.SiLU(),
#             nn.Linear(time_embed_dim * 4, time_embed_dim)
#         )
        
#         # Project hidden state to semantic dimension
#         self.hidden_proj = nn.Linear(hidden_dim, semantic_dim)
        
#         # Flow velocity network (predicts dx/dt)
#         self.velocity_net = VelocityNetwork(
#             semantic_dim=semantic_dim,
#             time_embed_dim=time_embed_dim,
#             num_layers=num_layers,
#             num_heads=num_heads,
#             dropout=dropout
#         )
        
#     def get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
#         """
#         Create sinusoidal time embeddings.
        
#         Args:
#             t: Time steps, shape (batch_size,)
            
#         Returns:
#             Time embeddings, shape (batch_size, time_embed_dim)
#         """
#         half_dim = self.time_embed_dim // 2
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
#         emb = t[:, None] * emb[None, :]
#         emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
#         return self.time_mlp(emb)
    
#     def forward(
#         self,
#         x_t: torch.Tensor,
#         t: torch.Tensor,
#         condition: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Predict velocity field v_t(x_t | condition).
        
#         Args:
#             x_t: Current point in flow, shape (batch_size, semantic_dim)
#             t: Time steps in [0, 1], shape (batch_size,)
#             condition: Conditioning hidden states, shape (batch_size, hidden_dim)
            
#         Returns:
#             Predicted velocity, shape (batch_size, semantic_dim)
#         """
#         # Get time embeddings
#         t_emb = self.get_time_embedding(t)
        
#         # Project condition to semantic space
#         cond_proj = self.hidden_proj(condition)
        
#         # Predict velocity
#         v_t = self.velocity_net(x_t, t_emb, cond_proj)
        
#         return v_t
    
#     def sample(
#         self,
#         condition: torch.Tensor,
#         num_steps: int = 100,
#         method: str = 'euler'
#     ) -> torch.Tensor:
#         """
#         Sample from the flow by integrating from t=0 to t=1.
        
#         Args:
#             condition: Conditioning hidden states, shape (batch_size, hidden_dim)
#             num_steps: Number of integration steps
#             method: Integration method ('euler' or 'midpoint')
            
#         Returns:
#             Final semantic space points, shape (batch_size, semantic_dim)
#         """
#         batch_size = condition.shape[0]
#         device = condition.device
        
#         # Start from Gaussian noise
#         x = torch.randn(batch_size, self.semantic_dim, device=device)
        
#         dt = 1.0 / num_steps
        
#         for i in range(num_steps):
#             t = torch.full((batch_size,), i * dt, device=device)
            
#             if method == 'euler':
#                 # Euler method
#                 v = self.forward(x, t, condition)
#                 x = x + v * dt
#             elif method == 'midpoint':
#                 # Midpoint method (more accurate)
#                 v1 = self.forward(x, t, condition)
#                 x_mid = x + v1 * (dt / 2)
#                 t_mid = t + dt / 2
#                 v2 = self.forward(x_mid, t_mid, condition)
#                 x = x + v2 * dt
#             else:
#                 raise ValueError(f"Unknown method: {method}")
        
#         return x
    
#     # TODO: Modify this
#     def compute_loss(
#         self,
#         condition: torch.Tensor,
#         target: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Compute flow matching loss using conditional optimal transport.
        
#         Args:
#             condition: Conditioning hidden states, shape (batch_size, hidden_dim)
#             target: Target points in semantic space, shape (batch_size, semantic_dim)
            
#         Returns:
#             Flow matching loss
#         """
#         batch_size = condition.shape[0]
#         device = condition.device
        
#         # Sample random time steps
#         t = torch.rand(batch_size, device=device)
        
#         # Sample noise
#         x_0 = torch.randn_like(target)
        
#         # Linear interpolation path (optimal transport for Gaussian)
#         x_t = (1 - t[:, None]) * x_0 + t[:, None] * target
        
#         # Target velocity (derivative of interpolation path)
#         u_t = target - x_0
        
#         # Predict velocity
#         v_t = self.forward(x_t, t, condition)
        
#         # MSE loss between predicted and target velocity
#         loss = F.mse_loss(v_t, u_t)
        
#         return loss


# class VelocityNetwork(nn.Module):
#     """
#     Neural network for predicting velocity field in flow matching.
#     Uses transformer architecture with time and condition modulation.
#     """
    
#     def __init__(
#         self,
#         semantic_dim: int,
#         time_embed_dim: int,
#         num_layers: int = 4,
#         num_heads: int = 8,
#         dropout: float = 0.1
#     ):
#         super().__init__()
        
#         self.semantic_dim = semantic_dim
        
#         # Input projection
#         self.input_proj = nn.Linear(semantic_dim, semantic_dim)
        
#         # Transformer layers with adaptive layer norm
#         self.layers = nn.ModuleList([
#             TransformerBlock(
#                 dim=semantic_dim,
#                 num_heads=num_heads,
#                 mlp_ratio=4,
#                 dropout=dropout,
#                 time_embed_dim=time_embed_dim
#             )
#             for _ in range(num_layers)
#         ])
        
#         # Output projection
#         self.output_proj = nn.Sequential(
#             nn.LayerNorm(semantic_dim),
#             nn.Linear(semantic_dim, semantic_dim)
#         )
        
#     def forward(
#         self,
#         x: torch.Tensor,
#         t_emb: torch.Tensor,
#         condition: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Args:
#             x: Current points, shape (batch_size, semantic_dim)
#             t_emb: Time embeddings, shape (batch_size, time_embed_dim)
#             condition: Projected conditions, shape (batch_size, semantic_dim)
            
#         Returns:
#             Velocity prediction, shape (batch_size, semantic_dim)
#         """
#         # Combine input with condition
#         h = self.input_proj(x) + condition
        
#         # Apply transformer layers
#         for layer in self.layers:
#             h = layer(h, t_emb)
        
#         # Project to output
#         v = self.output_proj(h)
        
#         return v


# class TransformerBlock(nn.Module):
#     """
#     Transformer block with adaptive layer normalization based on time.
#     """
    
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int,
#         mlp_ratio: int = 4,
#         dropout: float = 0.1,
#         time_embed_dim: int = 256
#     ):
#         super().__init__()
        
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(
#             dim, num_heads, dropout=dropout, batch_first=True
#         )
        
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, dim * mlp_ratio),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(dim * mlp_ratio, dim),
#             nn.Dropout(dropout)
#         )
        
#         # Adaptive layer norm parameters (scale and shift from time)
#         self.adaLN = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(time_embed_dim, dim * 4)
#         )
        
#     def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: Input, shape (batch_size, semantic_dim)
#             t_emb: Time embeddings, shape (batch_size, time_embed_dim)
            
#         Returns:
#             Output, shape (batch_size, semantic_dim)
#         """
#         # Get adaptive parameters
#         ada_params = self.adaLN(t_emb)
#         scale1, shift1, scale2, shift2 = ada_params.chunk(4, dim=-1)
        
#         # Self-attention with adaptive norm
#         x_norm = self.norm1(x) * (1 + scale1) + shift1
#         x_attn, _ = self.attn(
#             x_norm.unsqueeze(1),
#             x_norm.unsqueeze(1),
#             x_norm.unsqueeze(1)
#         )
#         x = x + x_attn.squeeze(1)
        
#         # MLP with adaptive norm
#         x_norm = self.norm2(x) * (1 + scale2) + shift2
#         x = x + self.mlp(x_norm)
        
#         return x


# # Example usage
# if __name__ == "__main__":
#     # Model parameters
#     hidden_dim = 768  # From language model
#     semantic_dim = 512  # Target semantic space
#     batch_size = 4
    
#     # Initialize model
#     flow_model = FlowMatchingModel(
#         hidden_dim=hidden_dim,
#         semantic_dim=semantic_dim,
#         time_embed_dim=256,
#         num_layers=4,
#         num_heads=8
#     )
    
#     # Example: Training step
#     condition = torch.randn(batch_size, hidden_dim)  # From language model
#     target = torch.randn(batch_size, semantic_dim)  # Target semantic embeddings
    
#     loss = flow_model.compute_loss(condition, target)
#     print(f"Training loss: {loss.item():.4f}")
    
#     # Example: Sampling
#     flow_model.eval()
#     with torch.no_grad():
#         samples = flow_model.sample(condition, num_steps=50, method='euler')
#         print(f"Sampled semantic points shape: {samples.shape}")


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
    # BaseAutoencoderOutputWithPastAndCrossAttentions,
    # CausalLMAutoencoderOutputWithCrossAttentions,
    CausalLatentLMOutputWithCrossAttentions,
    CausalLMFlowMatchingOutputWithCrossAttentions,
    CausalLMFlowMatchingSamplingOutputWithCrossAttentions,
)
from ...utils import (
    auto_docstring,
    logging,
)
from .configuration_latent_gpt2 import LatentGPT2Config
from .modeling_gpt2 import GPT2ModelBase, GPT2PreTrainedModel, GPT2LMHeadModel, GPT2Block

logger = logging.get_logger(__name__)
        
@auto_docstring(
    custom_intro="Language decoder model based on GPT-2 for decoding latent representations back to text.",
    custom_args="window_size (`int`): The window size for disaggregating latent representations back to sequences.",
)
class LanguageFlowMatchingBase(GPT2ModelBase):
    def __init__(self, config: LatentGPT2Config):
        super().__init__(config)
        
        self.embed_dim = config.hidden_size
        # Word Token Encoding
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        # Word Position Encoding
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        # FLow Matching TimeStep Encoding
        # self.tse = nn.Embedding(1, self.embed_dim)
        self.tse = nn.Linear(1, self.embed_dim, bias=False)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers_fm)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation

        # Initialize weights and apply final processing
        self.post_init()

    def init_weight_from_pretrained(self, pretrained_model: GPT2ModelBase):
        """
        Initializes the language decoder from a pre-trained model.
        
        Args:
            pretrained_model: The pre-trained model to use.
        
        Returns:
            A new instance of the class.
        """
        self.wte = copy.deepcopy(pretrained_model.wte)
        self.wpe = copy.deepcopy(pretrained_model.wpe)
        self.drop = copy.deepcopy(pretrained_model.drop)
        self.h = copy.deepcopy(pretrained_model.h[0:self.config.num_hidden_layers_fm])
        self.ln_f = copy.deepcopy(pretrained_model.ln_f)
        self.gradient_checkpointing = pretrained_model.gradient_checkpointing

        # Re-index blocks to match the new layer positions (0 to num_hidden_layers_decoder-1)
        for new_idx, block in enumerate(self.h):
            block.attn.layer_idx = new_idx
            if hasattr(block, 'crossattention'):
                block.crossattention.layer_idx = new_idx
        return self
    
    def _rand_latent(self, batch_size: int, device: Optional[torch.device] = None):
        return torch.randn(batch_size, 1, self.config.n_embd, device=device)
    
    def __encode_prev_latent_with_timestep(
        self,
        inputs_prev_latent: torch.FloatTensor,
        inputs_latent_timestep: torch.FloatTensor
    ):
        return inputs_prev_latent + self.tse(inputs_latent_timestep)
        
    def __pre_process_inputs(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.FloatTensor],
        inputs_prev_latent: Optional[torch.FloatTensor],
        inputs_latent_timestep: Optional[torch.FloatTensor],
    ) -> PreprocessOutput:
        """
        Pre-processes and segments the inputs for the language decoder.

        Args:
            input_ids (Optional[torch.LongTensor]): The input token ids.
            inputs_embeds (Optional[torch.FloatTensor]): The input embeddings.
            inputs_prev_latent (Optional[torch.FloatTensor]): The previous latent representation.
            inputs_latent_timestep (Optional[torch.FloatTensor]): The latent timestep with range [0, 1].

        Returns:
            A PreprocessOutput containing the pre-processed input ids and embeddings.
        """
        if input_ids is not None:
            raise Warning("input_ids isn't used, please use inputs_embeds")
        if inputs_embeds is None:
            raise Warning("inputs_embeds shouldn't be None")
        if inputs_prev_latent is None:
            inputs_prev_latent = self._rand_latent(batch_size=inputs_embeds.shape[0], device=inputs_embeds.device)
        
        encoded_prev_latent: torch.FloatTensor = self.__encode_prev_latent_with_timestep(
            inputs_prev_latent=inputs_prev_latent,
            inputs_latent_timestep=inputs_latent_timestep
        )
        return PreprocessOutput(
            input_ids=input_ids,
            inputs_embeds=torch.cat([inputs_embeds, encoded_prev_latent], dim=1)
        )
    
    # def __post_process_outputs(
    #     self,
    #     outputs: BaseModelOutputWithPastAndCrossAttentions,
    # ):
    #     """
    #     Post-processes and segment the outputs of the language decoder.

    #     Args:
    #         outputs: The output of the language decoder.

    #     Returns:
    #         The processed output of the language decoder.
    #     """
    #     if outputs is None:
    #         return outputs
    #     # The input last_hidden_state have already the last hidden_states of the last layer, no need to slice
    #     return BaseModelOutputWithPastAndCrossAttentions(
    #         last_hidden_state=outputs.last_hidden_state,
    #         past_key_values=outputs.past_key_values,
    #         hidden_states=outputs.hidden_states,
    #         # TODO: Handle attentions and cross_attentions reshaping
    #         attentions=outputs.attentions,
    #         cross_attentions=outputs.cross_attentions,
    #     )

    # @auto_docstring
    def forward(
        self,
        inputs_prev_latent: Optional[torch.FloatTensor],
        inputs_latent_timestep: Optional[torch.FloatTensor],
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
    ) -> Union[tuple, BaseModelOutputWithPastAndCrossAttentions]:
        if input_ids is not None:
            raise Warning("input_ids isn't used, please use inputs_embeds")
        pre_process_res: PreprocessOutput = self.__pre_process_inputs(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            inputs_prev_latent=inputs_prev_latent,
            inputs_latent_timestep=inputs_latent_timestep
        )
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
            return_dict=return_dict,
            **kwargs,
        )
        # return self.__post_process_outputs(outputs=output)
        return output
    
class LanguageFlowMatching(GPT2PreTrainedModel, GenerationMixin):
    # _tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}

    def __init__(self, config: LatentGPT2Config):
        super().__init__(config)
        self.transformer = LanguageFlowMatchingBase(config=config)
        
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

    @auto_docstring(
        custom_intro="Forward pass for the language decoder with LM head.",
        checkpoint="GPT2PreTrainedModel",
    )
    def forward(
        self,
        inputs_prev_latent: Optional[torch.FloatTensor],
        inputs_latent_timestep: Optional[torch.FloatTensor],
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_trajectories: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[tuple, CausalLMFlowMatchingOutputWithCrossAttentions]:
        r"""
        inputs_prev_latent (`torch.FloatTensor` of shape `(batch_size, 1, hidden_dim)`, *optional*):
            The previous latent representation. If not provided during training (when labels are given),
            it will be sampled randomly based on the timestep and labels.
        inputs_latent_timestep (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
            The latent timestep with range [0, 1]. If not provided during training (when labels are given),
            it will be randomly sampled.
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        labels (`torch.FloatTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        inputs_prev_latent, inputs_latent_timestep, velocity = self._get_inputs_outputs(
            inputs_prev_latent=inputs_prev_latent,
            inputs_latent_timestep=inputs_latent_timestep,
            inputs_embeds=inputs_embeds,
            labels=labels,
        )

        transformer_outputs = self.transformer(
            inputs_prev_latent=inputs_prev_latent,
            inputs_latent_timestep=inputs_latent_timestep,
            input_ids=input_ids,
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
            output_hidden_states=True,  # Required for accessing hidden_states
            return_dict=True,
        )

        logits = transformer_outputs.last_hidden_state[:, -1:, ...]
        if return_trajectories:
            trajectories = logits.unsqueeze(1)
        # print(f"LanguageFlowMatching - inputs_prev_latent: {inputs_prev_latent.shape}")
        # print(f"LanguageFlowMatching - inputs_latent_timestep: {inputs_latent_timestep.shape}")
        # print(f"LanguageFlowMatching - logits: {logits.shape}")
        # Since x_t := t x + (1 - (1 - \simga_{min}) t) z
        # Since v := x - (1 - \sigma_{min}) z
        # Therefore, x = v + (1 - \sigma_{min}) z = v + (1 - \sigma_{min}) (\frac{x_{t} - t x}{(1 - (1 - \simga_{min}) t)})
        # CAUTION: Be carefull to the derivation of the estimates
        # estimates = inputs_prev_latent - (inputs_latent_timestep - 1.0) * logits
        estimates = inputs_prev_latent +  (1.0 - self.config.fm_min_sigma) * logits / (1.0 - (1.0 - self.config.fm_min_sigma) * inputs_latent_timestep)
        # last_context = transformer_outputs.last_hidden_state[:, :-1, ...]
        # contexts = (s[:, :-1, ...] for s in transformer_outputs.hidden_states)

        # slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # logits = self.lm_head(latents[:, slice_indices, :])
        # logits = self._project_with_multi_heads(latents[:, slice_indices, :])

        loss = None
        if velocity is not None:
            # Flatten the tokens
            loss = self.loss_function(
                logits=logits,
                labels=velocity,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMFlowMatchingOutputWithCrossAttentions(
            last_hidden_state=transformer_outputs.last_hidden_state if output_hidden_states else None,
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states if output_hidden_states else None,
            attentions=transformer_outputs.attentions if output_attentions else None,
            cross_attentions=transformer_outputs.cross_attentions if output_attentions else None,
            estimates=estimates,
            trajectories=trajectories if return_trajectories else None,
        )
        
    def _loss_function(
        self,
        logits: torch.Tensor,
        labels: torch.FloatTensor,
        vocab_size: int,
        **kwargs,
    ) -> torch.Tensor:
        dim = list(range(1, len(logits.shape)))
        return torch.norm(logits - labels, dim=dim, p=2).mean()
    
    def _get_inputs_outputs(
        self,
        inputs_prev_latent: Optional[torch.FloatTensor],
        inputs_latent_timestep: Optional[torch.FloatTensor],
        inputs_embeds: Optional[torch.FloatTensor],
        labels: Optional[torch.FloatTensor],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        
        """
        if labels is not None:
            # Training mode
            if inputs_latent_timestep is None and inputs_prev_latent is None:
                # If both inputs_latent_timestep and inputs_prev_latent are not provided, random sample
                # CAUTION: Expand timestep for broadcasting: (batch_size,) -> (batch_size, 1, 1) for (batch_size, seq_len, hidden_dim), if not, the dimension of inputs_prev_latent will be wrong
                inputs_latent_timestep = torch.rand(labels.shape[0], 1, 1, device=labels.device)
                noise = self.transformer._rand_latent(batch_size=labels.shape[0], device=labels.device)
                inputs_prev_latent = inputs_latent_timestep * labels + (1.0 - (1.0 - self.config.fm_min_sigma) * inputs_latent_timestep) * noise
            elif inputs_latent_timestep is not None and inputs_prev_latent is not None:
                # If both inputs_latent_timestep and inputs_prev_latent are provided, use it for training
                # CAUTION: Expand timestep for broadcasting: (batch_size,) -> (batch_size, 1, 1) for (batch_size, seq_len, hidden_dim), if not, the dimension of inputs_prev_latent will be wrong
                inputs_latent_timestep = inputs_latent_timestep.view(-1, 1, 1)
                noise = (inputs_prev_latent - inputs_latent_timestep * labels) /  ((1.0 - (1.0 - self.config.fm_min_sigma) * inputs_latent_timestep))
            else:
                raise ValueError("Arguements inputs_prev_latent and inputs_latent_timestep should be both provided or not provided.")
            velocity = labels - (1.0 - self.config.fm_min_sigma) * noise
        else:
            # Inference mode
            velocity = None
            if inputs_embeds is None:
                raise ValueError("If labels is missing, inputs_embeds is required, but inputs_embeds is missing.")

            if inputs_prev_latent is None and inputs_latent_timestep is None:
                # CAUTION: Expand timestep for broadcasting: (batch_size,) -> (batch_size, 1, 1) for (batch_size, seq_len, hidden_dim), if not, the dimension of inputs_prev_latent will be wrong
                # inputs_latent_timestep = torch.rand(inputs_embeds.shape[0], 1, 1, device=inputs_embeds.device)
                # CAUTION: If there is no labels and previous timestep latent, the timestep should use 0
                inputs_latent_timestep = torch.zeros(inputs_embeds.shape[0], 1, 1, dtype=torch.float32, device=inputs_embeds.device)
                inputs_prev_latent = self.transformer._rand_latent(batch_size=inputs_embeds.shape[0], device=inputs_embeds.device)
            elif inputs_prev_latent is not None and inputs_latent_timestep is not None:
                pass
            else:
                raise ValueError("If labels is missing, inputs_prev_latent and inputs_latent_timestep shoule be provided simultaneously or neither, but only one of them is provided.")
        return inputs_prev_latent, inputs_latent_timestep, velocity
    
    def _simple_model_call(
        self,
        inputs_prev_latent: torch.FloatTensor,
        inputs_latent_timestep: torch.FloatTensor,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> CausalLMFlowMatchingOutputWithCrossAttentions:
        return self(
            inputs_prev_latent=inputs_prev_latent,
            inputs_latent_timestep=inputs_latent_timestep,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            cache_position=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
            logits_to_keep=0,            
        )
    
    def _step(
        self,
        method: str,
        dt: float,
        inputs_prev_latent: torch.FloatTensor,
        inputs_latent_timestep: torch.FloatTensor,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> CausalLMFlowMatchingSamplingOutputWithCrossAttentions:
        if method == 'euler':
            # Euler method
            output: CausalLMFlowMatchingOutputWithCrossAttentions = self._simple_model_call(
                inputs_prev_latent=inputs_prev_latent,
                inputs_latent_timestep=inputs_latent_timestep,
                inputs_embeds=inputs_embeds,
            )
            v = output.last_latent
            x = inputs_prev_latent + v * dt
        elif method == 'midpoint':
            # Mid point method (more accurate)
            output1: CausalLMFlowMatchingOutputWithCrossAttentions = self._simple_model_call(
                inputs_prev_latent=inputs_prev_latent,
                inputs_latent_timestep=inputs_latent_timestep,
                inputs_embeds=inputs_embeds,
            )
            v1 = output1.last_latent
            x_mid = inputs_prev_latent + v1 * (dt / 2)
            t_mid = inputs_latent_timestep + dt / 2
            output2: CausalLMFlowMatchingOutputWithCrossAttentions = self._simple_model_call(
                inputs_prev_latent=x_mid,
                inputs_latent_timestep=t_mid,
                inputs_embeds=inputs_embeds,
            )
            v2 = output2.last_latent
            x = inputs_prev_latent + v2 * dt
        else:
            raise ValueError(f"Unknown method: {method}")
        return CausalLMFlowMatchingSamplingOutputWithCrossAttentions(
            last_hidden_state=None,
            last_latent=x,
        )
    
    def sample(
        self,
        inputs_embeds: Optional[torch.FloatTensor],
        inputs_prev_latent: Optional[torch.FloatTensor] = None,
        inputs_latent_timestep: Optional[Union[torch.Tensor, int, float]] = None,
        num_inference_steps: int = 1000,
        method: str = "euler",
        device: Union[str, torch.device] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, CausalLMFlowMatchingSamplingOutputWithCrossAttentions]:
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
        labels (`torch.FloatTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        if inputs_embeds is None:
            raise ValueError(f"inputs_embeds should be provided.")
        batch_size: int = inputs_embeds.shape[0]
        if device is None:
            device = inputs_embeds.device
        else:
            inputs_embeds = inputs_embeds.to(device)
        
        # Start from Gaussian noise
        if inputs_prev_latent is None:
            inputs_prev_latent = self.transformer._rand_latent(batch_size=batch_size, device=device)
        # Set up inference timestep
        timestep_base: torch.FloatTensor = torch.ones(batch_size).float().to(device)
        if inputs_latent_timestep is None:
            inputs_latent_timestep: torch.FloatTensor = timestep_base * 0.0
        elif isinstance(inputs_latent_timestep, int):
            inputs_latent_timestep: torch.FloatTensor = timestep_base * (inputs_latent_timestep / self.config.fm_num_train_timesteps)
        elif isinstance(inputs_latent_timestep, float):
            inputs_latent_timestep: torch.FloatTensor = timestep_base * inputs_latent_timestep
        else:
            assert len(inputs_latent_timestep) == batch_size, f"The length of inputs_latent_timestep ({len(inputs_latent_timestep)}) should be equal to inputs_embeds ({batch_size})"
        
        dt = 1.0 / num_inference_steps
        
        for i in range(num_inference_steps):
            output: CausalLMFlowMatchingSamplingOutputWithCrossAttentions = self._step(
                method=method,
                dt=dt,
                inputs_prev_latent=inputs_prev_latent,
                inputs_latent_timestep=inputs_latent_timestep,
                inputs_embeds=inputs_embeds,
            )
            inputs_prev_latent = output.last_latent
            inputs_latent_timestep = inputs_latent_timestep + dt
        
        return CausalLMFlowMatchingSamplingOutputWithCrossAttentions(
            last_hidden_state=None,
            last_latent=inputs_prev_latent,
        )