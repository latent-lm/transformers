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
    This is the configuration class to store the configuration of a [`LatentAutoregressive`]. It is used to
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
            Window size used by the latent encoder and decoder for aggregating/disaggregating representations.
        latent_dim (`int`, *optional*, defaults to 768):
            The latent dimension of the autoencoder.
        normalized_latent_sigmoid (`bool`, *optional*, defaults to `False`):
            Whether to normalize the latent using sigmoid (maps to [-1, 1] via sigmoid(x)*2-1).
            Mutually exclusive with `normalized_latent_tanh`.
        normalized_latent_tanh (`bool`, *optional*, defaults to `True`):
            Whether to normalize the latent using tanh.
            Mutually exclusive with `normalized_latent_sigmoid`.
        num_hidden_layers_encoder (`int`, *optional*, defaults to 6):
            Number of hidden layers in the latent encoder stack.
        num_hidden_layers_decoder (`int`, *optional*, defaults to 6):
            Number of hidden layers in the latent decoder stack.
        num_hidden_layers_fm (`int`, *optional*, defaults to 10):
            Number of hidden layers in the flow matching stack.
        pad_token_id (`int`, *optional*, defaults to 50256):
            Id of the padding token in the vocabulary.
        fm_num_train_timesteps (`int`, *optional*, defaults to 1000):
            The training timesteps for the flow matching.
        fm_min_sigma (`float`, *optional*, defaults to 0.01):
            The minimum sigma value for the flow matching noise schedule.
        fm_num_train_timesteps (`int`, *optional*, defaults to 1000):
            The number of training timesteps for the flow matching process.

    Example:

    ```python
    >>> from transformers import LatentGPT2Config, LatentAutoregressive

    >>> # Initializing a LatentGPT2 configuration
    >>> configuration = LatentGPT2Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = LatentAutoregressive(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "latent_gpt2"
    AUTOENCODER_TYPE_MULTI_HEAD: str = "multi_head"
    AUTOENCODER_TYPE_DIFFUSION: str = "diffusion"

    def __init__(
        self,
        # Latent-specific parameters
        window_size: int = 4,
        latent_dim: int = 768,
        normalized_latent_sigmoid: bool = False,
        normalized_latent_tanh: bool = True,
        autoencoder_type: str = "multi_head",
        num_hidden_layers_encoder: int = 6,
        num_hidden_layers_decoder: int = 6,
        num_hidden_layers_fm: int = 12,
        pad_token_id: int = 50256,
        mask_token_id: int = 50256,
        # Diffusion Autoencoder paramters
        dae_num_train_timesteps: int = 0,
        # Flow matching parameters
        fm_min_sigma: float = 0.01,
        fm_num_train_timesteps: int = 1000,
        # All GPT2Config parameters passed via kwargs
        **kwargs,
    ) -> None:
        # Set pad_token_id in kwargs for parent class
        kwargs.setdefault("pad_token_id", pad_token_id)
        kwargs.setdefault("mask_token_id", mask_token_id)

        super().__init__(**kwargs)

        # Latent architecture parameters
        self.window_size: int = window_size
        self.latent_dim: int = latent_dim
        self.normalized_latent_sigmoid: bool = normalized_latent_sigmoid
        self.normalized_latent_tanh: bool = normalized_latent_tanh

        # Autoencoder parameters
        self.autoencoder_type: str = autoencoder_type
        self.num_hidden_layers_encoder: int = num_hidden_layers_encoder
        self.num_hidden_layers_decoder: int = num_hidden_layers_decoder
        # Diffusion Autoencoder paramters
        self.dae_num_train_timesteps: int = dae_num_train_timesteps
        # Flow matching parameters
        self.num_hidden_layers_fm: int = num_hidden_layers_fm
        self.fm_min_sigma: float = fm_min_sigma
        self.fm_num_train_timesteps: int = fm_num_train_timesteps

    @classmethod
    def build_from_pretrained(
        cls,
        config: GPT2Config,
        window_size: int = 4,
        latent_dim: int = 768,
        normalized_latent_sigmoid: bool = False,
        normalized_latent_tanh: bool = True,
        autoencoder_type: str = "multi_head",
        num_hidden_layers_encoder: int = 6,
        num_hidden_layers_decoder: int = 6,
        num_hidden_layers_fm: int = 12,
        pad_token_id: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        dae_num_train_timesteps: int = 0,
        fm_min_sigma: float = 0.01,
        fm_num_train_timesteps: int = 1000,
    ) -> "LatentGPT2Config":
        """
        Builds a new instance of the LatentGPT2Config from a pre-trained GPT2Config.

        Args:
            config (`GPT2Config`):
                The configuration of the pre-trained model to use.
            window_size (`int`, *optional*, defaults to 4):
                The window size for the latent encoder/decoder.
            latent_dim (`int`, *optional*, defaults to 768):
                The latent dimension of the autoencoder.
            normalized_latent_sigmoid (`bool`, *optional*, defaults to `False`):
                Whether to normalize the latent using sigmoid (maps to [-1, 1] via sigmoid(x)*2-1).
                Mutually exclusive with `normalized_latent_tanh`.
            normalized_latent_tanh (`bool`, *optional*, defaults to `True`):
                Whether to normalize the latent using tanh.
                Mutually exclusive with `normalized_latent_sigmoid`.
            autoencoder_type (`str`, *optional*, defaults to "multi_head"):
                The type of autoencoder to use. Can be either "multi_head" or "diffusion")
            num_hidden_layers_encoder (`int`, *optional*, defaults to 6):
                The number of hidden layers in the encoder.
            num_hidden_layers_decoder (`int`, *optional*, defaults to 6):
                The number of hidden layers in the decoder.
            num_hidden_layers_fm (`int`, *optional*, defaults to 10):
                The number of hidden layers in the flow matching.
            pad_token_id (`int`, *optional*):
                The id of the padding token. If None, uses config.pad_token_id or config.eos_token_id.
            mask_token_id (`int`, *optional*):
                The id of the masking token. If None, uses config.mask_token_id or config.eos_token_id.
            dae_num_train_timesteps (`int`, *optional*, defaults to 0):
                The training timesteps for the diffusion autoencoder.
            fm_min_sigma (`float`, *optional*, defaults to 0.01):
                The minimum sigma value for the flow matching.
            fm_num_train_timesteps (`int`, *optional*, defaults to 1000):
                The training timesteps for the flow matching.

        Returns:
            `LatentGPT2Config`: A new instance of the class.
        """
        # Get base config as dict
        base_kwargs = config.to_dict()

        # Determine pad_token_id
        if pad_token_id is None:
            pad_token_id = getattr(config, "pad_token_id", None) or getattr(config, "eos_token_id", 50256)
        # Determine mask_token_id
        if mask_token_id is None:
            mask_token_id = getattr(config, "mask_token_id", None) or getattr(config, "eos_token_id", 50256)

        # Add latent-specific parameters
        base_kwargs.update({
            "window_size": window_size,
            "latent_dim": latent_dim,
            "normalized_latent_sigmoid": normalized_latent_sigmoid,
            "normalized_latent_tanh": normalized_latent_tanh,
            "autoencoder_type": autoencoder_type,
            "num_hidden_layers_encoder": num_hidden_layers_encoder,
            "num_hidden_layers_decoder": num_hidden_layers_decoder,
            "num_hidden_layers_fm": num_hidden_layers_fm,
            "pad_token_id": pad_token_id,
            "mask_token_id": mask_token_id,
            "dae_num_train_timesteps": dae_num_train_timesteps,
            "fm_min_sigma": fm_min_sigma,
            "fm_num_train_timesteps": fm_num_train_timesteps,
        })

        return cls(**base_kwargs)


__all__ = ["LatentGPT2Config"]
