from typing import List

import torch
from torch import Tensor
import numpy as np

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer
from src.text_encoder.ctc_text_encoder import CTCTextEncoder

# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


class BeamCERMetric(BaseMetric):
    def __init__(self, text_encoder: CTCTextEncoder, beam_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ) -> float:
        lengths = log_probs_length.detach().cpu().numpy().flatten()
        log_probs = log_probs.detach().cpu()

        cers = []
        for idx, (seq_len, target) in enumerate(zip(lengths, text)):
            seq_len = int(seq_len)
            if seq_len <= 0:
                cers.append(1.0)
                continue

            seq_log_probs = log_probs[idx, :seq_len]
            
            if hasattr(self.text_encoder, "ctc_beam_search"):
                pred = self.text_encoder.ctc_beam_search(seq_log_probs, self.beam_size, length=seq_len)
            else:
                argmax_pred = torch.argmax(seq_log_probs, dim=-1).numpy()
                pred = self.text_encoder.ctc_decode(argmax_pred)
            
            target_norm = self.text_encoder.normalize_text(target)
            cers.append(calc_cer(target_norm, pred))

        return np.mean(cers)


class LMBeamCERMetric(BaseMetric):
    def __init__(self, text_encoder: CTCTextEncoder, beam_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size

    def __call__(
        self, logits: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ) -> float:
        lengths = log_probs_length.detach().cpu().numpy().flatten()
        logits = logits.detach().cpu()

        cers = []
        for idx, (seq_len, target) in enumerate(zip(lengths, text)):
            seq_len = int(seq_len)
            if seq_len <= 0:
                cers.append(1.0)
                continue

            seq_logits = logits[idx, :seq_len]
            
            if hasattr(self.text_encoder, 'lm_model') and self.text_encoder.lm_model is not None:
                pred = self.text_encoder.lm_ctc_beam_search(seq_logits, self.beam_size)
            else:
                argmax_pred = torch.argmax(seq_logits, dim=-1).numpy()
                pred = self.text_encoder.ctc_decode(argmax_pred)
            
            target_norm = self.text_encoder.normalize_text(target)
            cers.append(calc_cer(target_norm, pred))

        return np.mean(cers)