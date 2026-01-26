# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""MAMBA Autoencoder configuration"""

from typing import Optional

from ...models.mamba.configuration_mamba import MambaConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class LatentMambaConfig(MambaConfig):
    """
    This is the configuration class to store the configuration of a [`MambaAutoencoder`] or
    [`MambaDiffusionAutoencoder`]. It is used to instantiate a MAMBA-based autoencoder model
    according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MAMBA
    [state-spaces/mamba-2.8b](https://huggingface.co/state-spaces/mamba-2.8b) architecture.

    Configuration objects inherit from [`MambaConfig`] and can be used to control the model outputs.
    Read the documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 50280):
            Vocabulary size of the MAMBA model.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        state_size (`int`, *optional*, defaults to 16):
            Shape of the state space latents.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the base model.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0):
            The id of the beginning of sentence token.
        eos_token_id (`int`, *optional*, defaults to 0):
            The id of the end of sentence token.
        expand (`int`, *optional*, defaults to 2):
            Expanding factor used to determine the intermediate size.
        conv_kernel (`int`, *optional*, defaults to 4):
            Size of the convolution kernel.
        use_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in ["in_proj", "out_proj"] of the mixer block.
        use_conv_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the convolution layer of the mixer block.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the decoder.
        initializer_range (`float`, *optional*, defaults to 0.1):
            The standard deviation of the initializer for weight matrices.
        residual_in_fp32 (`bool`, *optional*, defaults to `True`):
            Whether residuals should be in float32.
        time_step_rank (`Union[int,str]`, *optional*, defaults to `"auto"`):
            Rank of the discretization projection matrix.
        time_step_scale (`float`, *optional*, defaults to 1.0):
            Scale used to scale dt_proj.bias.
        time_step_min (`float`, *optional*, defaults to 0.001):
            Minimum time_step used to bound dt_proj.bias.
        time_step_max (`float`, *optional*, defaults to 0.1):
            Maximum time_step used to bound dt_proj.bias.
        time_step_init_scheme (`str`, *optional*, defaults to `"random"`):
            Init scheme for dt_proj.weight. One of ["random", "uniform"].
        time_step_floor (`float`, *optional*, defaults to 0.0001):
            Minimum clamping value of dt_proj.bias initialization.
        rescale_prenorm_residual (`bool`, *optional*, defaults to `False`):
            Whether to rescale out_proj weights when initializing.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the cache should be used.
        use_mambapy (`bool`, *optional*, defaults to `False`):
            Whether to use mamba.py fallback if CUDA unavailable.
        window_size (`int`, *optional*, defaults to 4):
            Window size used by the latent encoder and decoder for aggregating/disaggregating representations.
            Each `window_size` tokens are compressed into a single latent vector.
        latent_dim (`int`, *optional*, defaults to 768):
            The latent dimension of the autoencoder.
        num_hidden_layers_encoder (`int`, *optional*, defaults to 6):
            Number of hidden layers in the latent encoder stack.
        num_hidden_layers_decoder (`int`, *optional*, defaults to 6):
            Number of hidden layers in the latent decoder stack.
        mask_token_id (`int`, *optional*, defaults to 0):
            Id of the masking token in the vocabulary.
        autoencoder_type (`str`, *optional*, defaults to "multi_head"):
            The type of autoencoder. Can be either "multi_head" or "diffusion".
        dae_num_train_timesteps (`int`, *optional*, defaults to 0):
            The training timesteps for the diffusion autoencoder.
            0 means fully masked during training.

    Example:

    ```python
    >>> from transformers import LatentMambaConfig, MambaAutoencoder

    >>> # Initializing a Mamba autoencoder configuration
    >>> configuration = LatentMambaConfig(
    ...     window_size=4,
    ...     latent_dim=768,
    ...     num_hidden_layers_encoder=6,
    ...     num_hidden_layers_decoder=6,
    ... )

    >>> # Initializing a model from the configuration
    >>> model = MambaAutoencoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "latent_mamba"
    AUTOENCODER_TYPE_MULTI_HEAD: str = "multi_head"
    AUTOENCODER_TYPE_DIFFUSION: str = "diffusion"

    def __init__(
        self,
        # Latent-specific parameters
        window_size: int = 4,
        latent_dim: int = 768,
        autoencoder_type: str = "multi_head",
        num_hidden_layers_encoder: int = 6,
        num_hidden_layers_decoder: int = 6,
        mask_token_id: int = 0,
        # Diffusion Autoencoder parameters
        dae_num_train_timesteps: int = 0,
        # All MambaConfig parameters passed via kwargs
        **kwargs,
    ) -> None:
        # Set pad_token_id default if not provided
        kwargs.setdefault("pad_token_id", 0)

        super().__init__(**kwargs)

        # Latent architecture parameters
        self.window_size: int = window_size
        self.latent_dim: int = latent_dim

        # Autoencoder parameters
        self.autoencoder_type: str = autoencoder_type
        self.num_hidden_layers_encoder: int = num_hidden_layers_encoder
        self.num_hidden_layers_decoder: int = num_hidden_layers_decoder
        self.mask_token_id: int = mask_token_id

        # Diffusion Autoencoder parameters
        self.dae_num_train_timesteps: int = dae_num_train_timesteps

        # For compatibility with some utilities (e.g., GPT-2 style APIs)
        self.n_embd = self.hidden_size

    @classmethod
    def build_from_pretrained(
        cls,
        config: MambaConfig,
        window_size: int = 4,
        latent_dim: int = 768,
        autoencoder_type: str = "multi_head",
        num_hidden_layers_encoder: int = 6,
        num_hidden_layers_decoder: int = 6,
        pad_token_id: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        dae_num_train_timesteps: int = 0,
    ) -> "LatentMambaConfig":
        """
        Builds a new instance of the LatentMambaConfig from a pre-trained MambaConfig.

        Args:
            config (`MambaConfig`):
                The configuration of the pre-trained model to use.
            window_size (`int`, *optional*, defaults to 4):
                The window size for the latent encoder/decoder.
            latent_dim (`int`, *optional*, defaults to 768):
                The latent dimension of the autoencoder.
            autoencoder_type (`str`, *optional*, defaults to "multi_head"):
                The type of autoencoder to use. Can be either "multi_head" or "diffusion".
            num_hidden_layers_encoder (`int`, *optional*, defaults to 6):
                The number of hidden layers in the encoder.
            num_hidden_layers_decoder (`int`, *optional*, defaults to 6):
                The number of hidden layers in the decoder.
            pad_token_id (`int`, *optional*):
                The id of the padding token. If None, uses config.pad_token_id or 0.
            mask_token_id (`int`, *optional*):
                The id of the masking token. If None, uses config.pad_token_id or 0.
            dae_num_train_timesteps (`int`, *optional*, defaults to 0):
                The training timesteps for the diffusion autoencoder.

        Returns:
            `LatentMambaConfig`: A new instance of the class.
        """
        # Get base config as dict
        base_kwargs = config.to_dict()

        # Determine pad_token_id
        if pad_token_id is None:
            pad_token_id = getattr(config, "pad_token_id", None) or 0

        # Determine mask_token_id
        if mask_token_id is None:
            mask_token_id = getattr(config, "pad_token_id", None) or 0

        # Add latent-specific parameters
        base_kwargs.update({
            "window_size": window_size,
            "latent_dim": latent_dim,
            "autoencoder_type": autoencoder_type,
            "num_hidden_layers_encoder": num_hidden_layers_encoder,
            "num_hidden_layers_decoder": num_hidden_layers_decoder,
            "pad_token_id": pad_token_id,
            "mask_token_id": mask_token_id,
            "dae_num_train_timesteps": dae_num_train_timesteps,
        })

        return cls(**base_kwargs)


__all__ = ["LatentMambaConfig"]
