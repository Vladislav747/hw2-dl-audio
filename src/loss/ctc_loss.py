import torch
from torch import Tensor
from torch.nn import CTCLoss


class CTCLossWrapper(CTCLoss):
    def __init__(self, blank=0, reduction='mean', zero_infinity=True):
        super().__init__(blank=blank, reduction=reduction, zero_infinity=zero_infinity)

    def forward(
        self, log_probs, log_probs_length, text_encoded, text_encoded_length, **batch
    ) -> Tensor:
        log_probs_t = torch.transpose(log_probs, 0, 1)

        loss = super().forward(
            log_probs=log_probs_t,
            targets=text_encoded,
            input_lengths=log_probs_length,
            target_lengths=text_encoded_length,
        )

        if torch.isnan(loss):
            print(f"NaN loss detected! log_probs shape: {log_probs.shape}")
            print(f"[CTC Loss Error] NaN/Inf loss detected!")
            print(f"  log_probs stats: min={log_probs.min():.4f}, max={log_probs.max():.4f}, mean={log_probs.mean():.4f}")
            print(f"  target_lengths: {text_encoded_length}")
            print(f"  text_encoded: {text_encoded}")
            return {"loss": loss}

        return {"loss": loss}
