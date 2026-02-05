import torch

class AutoencoderUtils:
    
    def beta_kl(
        self,
        logits: torch.FloatTensor,
        latents_mean: torch.FloatTensor,
        latents_logvar: torch.FloatTensor,
        labels: torch.LongTensor,
        vocab_size: int,
        variational_beta: float = 0.0,
        gaussian_prior_std: float = 0.5,
        ignore_index: int = -100,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Args:
            logits (`torch.FloatTensor` of shape `(batch_size, window_size * segment_num, vocab_size)`):
                Predicted token logits from the decoder.
            latents_mean (`torch.FloatTensor` of shape `(batch_size, segment_num, latent_dim)`):
                The predictive mean of the latents of each token block
            latents_logvar (`torch.FloatTensor` of shape `(batch_size, segment_num, latent_dim)`):
                The predictive log variance of the latents of each token block
            labels (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                Target token IDs. Values equal to `ignore_index` are masked.
            vocab_size (`int`):
                Size of the vocabulary for clamping label indices.
            shift_labels (`int`, *optional*, defaults to 0):
                Number of positions to shift labels for autoregressive loss.
                If > 0, logits[:-shift] are compared with labels[shift:].
                For reconstruction (not prediction), use 0.
            label_smooth_epsilon (`float`, *optional*, defaults to 0.1):
                Label smoothing factor. 0.0 = no smoothing (standard CE),
                1.0 = uniform distribution.
            ignore_index (`int`, *optional*, defaults to -100):
                Label value to ignore in loss computation (typically padding).
        """
        return 0.0
    
    def recon_loss(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        vocab_size: int,
        label_smooth_epsilon: float = 0.1,
        ignore_index: int = -100,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Compute label-smoothed cross-entropy loss for reconstruction.

        This loss function implements label smoothing, which helps prevent
        overconfident predictions and improves generalization. The smoothed
        loss is a weighted combination of:
        - Standard negative log-likelihood loss (weight: 1 - label_smooth_epsilon)
        - Uniform distribution penalty (weight: label_smooth_epsilon)

        Label Smoothing Formula:
            `loss = (1 - label_smooth_epsilon) * NLL + label_smooth_epsilon * uniform_penalty`

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
            label_smooth_epsilon (`float`, *optional*, defaults to 0.1):
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
        return (1 - label_smooth_epsilon) * nll_loss + label_smooth_epsilon * smoothed_loss
    
    def loss_function(
        self,
        logits: torch.FloatTensor,
        latents_mean: torch.FloatTensor,
        latents_logvar: torch.FloatTensor,
        labels: torch.LongTensor,
        vocab_size: int,
        shift_labels: int = 0,
        label_smooth_epsilon: float = 0.1,
        variational_beta: float = 0.0,
        gaussian_prior_std: float = 0.5,
        ignore_index: int = -100,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Compute label-smoothed cross-entropy loss for reconstruction.

        This loss function implements label smoothing, which helps prevent
        overconfident predictions and improves generalization. The smoothed
        loss is a weighted combination of:
        - Standard negative log-likelihood loss (weight: 1 - label_smooth_epsilon)
        - Uniform distribution penalty (weight: label_smooth_epsilon)

        Label Smoothing Formula:
            `loss = (1 - label_smooth_epsilon) * NLL + label_smooth_epsilon * uniform_penalty`

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
            label_smooth_epsilon (`float`, *optional*, defaults to 0.1):
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
        return self.recon_loss(
            logits=logits,
            labels=labels,
            vocab_size=vocab_size,
            label_smooth_epsilon=label_smooth_epsilon,
            ignore_index=ignore_index,
            **kwargs
        ) + self.beta_kl(
            logits=logits,
            latents_mean=latents_mean,
            latents_logvar=latents_logvar,
            labels=labels,
            vocab_size=vocab_size,
            label_smooth_epsilon=label_smooth_epsilon,
            variational_beta=variational_beta,
            gaussian_prior_std=gaussian_prior_std,
            ignore_index=ignore_index,
            **kwargs
        )