from typing import List
import numpy as np
import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer
from src.text_encoder import CTCTextEncoder



# TODO beam search / LM versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class BeamWERMetric(BaseMetric):
    def __init__(self, text_encoder: CTCTextEncoder, beam_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ) -> float:
        lengths = log_probs_length.detach().cpu().numpy().flatten()
        log_probs = log_probs.detach().cpu()

        wers = []
        for idx, (seq_len, target) in enumerate(zip(lengths, text)):
            seq_len = int(seq_len)
            if seq_len <= 0:
                wers.append(1.0)
                continue

            seq_log_probs = log_probs[idx, :seq_len]
            
            if hasattr(self.text_encoder, "ctc_beam_search"):
                pred = self.text_encoder.ctc_beam_search(seq_log_probs, self.beam_size, length=seq_len)
            else:
                argmax_pred = torch.argmax(seq_log_probs, dim=-1).numpy()
                pred = self.text_encoder.ctc_decode(argmax_pred)
            
            target_norm = self.text_encoder.normalize_text(target)
            wers.append(calc_wer(target_norm, pred))

        return np.mean(wers)