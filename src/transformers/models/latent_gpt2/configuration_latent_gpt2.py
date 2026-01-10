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
"""Latent GPT-2 configuration"""

from typing import Optional

from ...models.gpt2.configuration_gpt2 import GPT2Config
from ...utils import logging


logger = logging.get_logger(__name__)


class LatentGPT2Config(GPT2Config):
    """
    This is the configuration class to store the configuration of a [`LatnetAutoregressive`]. It is used to
    instantiate a Latent GPT-2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the GPT-2
    [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GPT2Model`].
        n_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
        activation_function (`str`, *optional*, defaults to `"gelu_new"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        summary_type (`string`, *optional*, defaults to `"cls_index"`):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`].

            Has to be one of the following options:

                - `"last"`: Take the last token hidden state (like XLNet).
                - `"first"`: Take the first token hidden state (like BERT).
                - `"mean"`: Take the mean of all tokens hidden states.
                - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
                - `"attn"`: Not implemented now, use multi-head attention.
        summary_use_proj (`bool`, *optional*, defaults to `True`):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`].

            Whether or not to add a projection after the vector extraction.
        summary_activation (`str`, *optional*):
            Argument used when doing sequence summary. Used in for the multiple choice head in
            [`GPT2DoubleHeadsModel`].

            Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
        summary_proj_to_labels (`bool`, *optional*, defaults to `True`):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`].

            Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
        summary_first_dropout (`float`, *optional*, defaults to 0.1):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`].

            The dropout ratio to be used after the projection and activation.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by dividing by sqrt(hidden_size)..
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        bos_token_id (`int`, *optional*, defaults to 50256):
            Id of the beginning of sentence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 50256):
            Id of the end of sentence token in the vocabulary.
        scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
            Whether to additionally scale attention weights by `1 / layer_idx + 1`.
        reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
            Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
            dot-product/softmax to float() when training with mixed precision.
        window_size (`int`, *optional*, defaults to 4):
            Window size used by the latent decoder.
        num_hidden_layers_encoder (`int`, *optional*, defaults to 6):
            Number of hidden layers in the latent encoder stack.
        num_hidden_layers_decoder (`int`, *optional*, defaults to 6):
            Number of hidden layers in the latent decoder stack.
        num_hidden_layers_fm (`int`, *optional*, defaults to 10):
            Number of hidden layers in the flow matching stack.
        pad_token_id (`int`, *optional*, defaults to 50256):
            Id of the padding token in the vocabulary.
        fm_min_sigma (`float`, *optional*, defaults to 0.01):
            The minimum sigma value for the flow matching noise schedule.
        fm_num_train_timesteps (`int`, *optional*, defaults to 1000):
            The number of training timesteps for the flow matching process.

    Example:

    ```python
    >>> from transformers import GPT2Config, GPT2Model

    >>> # Initializing a GPT2 configuration
    >>> configuration = GPT2Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = GPT2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_inner: Optional[int] = None,
        activation_function: str = "gelu_new",
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        summary_type: str = "cls_index",
        summary_use_proj: bool = True,
        summary_activation: Optional[str] = None,
        summary_proj_to_labels: bool = True,
        summary_first_dropout: float = 0.1,
        scale_attn_weights: bool = True,
        use_cache: bool = True,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        scale_attn_by_inverse_layer_idx: bool = False,
        reorder_and_upcast_attn: bool = False,
        # Additional parameters for LatentGPT2
        window_size: int = 4,
        num_hidden_layers_encoder: int = 6,
        num_hidden_layers_decoder: int = 6,
        num_hidden_layers_fm: int = 10,
        pad_token_id: int = 50256,
        fm_min_sigma: float = 0.01,
        fm_num_train_timesteps: int = 1000,
        **kwargs,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            activation_function=activation_function,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range,
            summary_type=summary_type,
            summary_use_proj=summary_use_proj,
            summary_activation=summary_activation,
            summary_proj_to_labels=summary_proj_to_labels,
            summary_first_dropout=summary_first_dropout,
            scale_attn_weights=scale_attn_weights,
            use_cache=use_cache,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx,
            reorder_and_upcast_attn=reorder_and_upcast_attn,
            **kwargs,
        )
        self.window_size = window_size
        self.num_hidden_layers_encoder = num_hidden_layers_encoder
        self.num_hidden_layers_decoder = num_hidden_layers_decoder
        self.num_hidden_layers_fm = num_hidden_layers_fm
        self.pad_token_id = pad_token_id
        self.fm_min_sigma = fm_min_sigma
        self.fm_num_train_timesteps = fm_num_train_timesteps
        
    @classmethod
    def build_from_pretrained(
        cls,
        config: GPT2Config,
        window_size: int = 4,
        num_hidden_layers_encoder: int = 6,
        num_hidden_layers_decoder: int = 6,
        num_hidden_layers_fm: int = 10,
        pad_token_id: int = 50256,
        fm_min_sigma: float = 0.01,
        fm_num_train_timesteps: int = 1000,
    ) -> "LatentGPT2Config":
        """
        Builds a new instance of the LatentGPT2Config from a pre-trained model.

        Args:
            cls: The class to instantiate.
            config: The configuration of the pre-trained model to use.
            window_size: The window size for the language decoder.
            num_hidden_layers_encoder: The number of hidden layers in the encoder.
            num_hidden_layers_decoder: The number of hidden layers in the decoder.
            num_hidden_layers_fm: The number of hidden layers in the flow matching.
            pad_token_id: The id of the padding token.
            fm_min_sigma: The minimum sigma value for the flow matching.
            fm_num_train_timesteps: The training timestep for the flow matching.

        Returns:
            A new instance of the class.
        """
        base_kwargs = config.to_dict()
        base_kwargs.update(
            {
                "window_size": window_size,
                "num_hidden_layers_encoder": num_hidden_layers_encoder,
                "num_hidden_layers_decoder": num_hidden_layers_decoder,
                "num_hidden_layers_fm": num_hidden_layers_fm,
                "pad_token_id": pad_token_id,
                "fm_min_sigma": fm_min_sigma,
                "fm_num_train_timesteps": fm_num_train_timesteps,
            }
        )
        return cls(**base_kwargs)

__all__ = ["LatentGPT2Config"]
