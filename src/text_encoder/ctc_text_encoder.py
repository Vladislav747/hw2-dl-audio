import re
from string import ascii_lowercase
from collections import defaultdict
import numpy as np
from pyctcdecode import build_ctcdecoder

import torch


# TODO add BPE
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(
            self,
            alphabet=None,
            lm_path: str = None,
            unigrams_path: str = None,
            *args,
            **kwargs,
    ):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}


        if lm_path is not None:
            print(f"""\n LM path: {lm_path}""")

            assert unigrams_path is not None, "LM and unigrams should be provided"

            print(f"""\nUnigrams path: {unigrams_path}""")

            unigrams = []
            with open(unigrams_path, "r") as file:
                for line in file:
                    word = line.split()[0]
                    unigrams.append(word)

            self.lm_model = build_ctcdecoder(
                labels=[self.EMPTY_TOK] + list(self.alphabet),
                kenlm_model_path=lm_path,
                unigrams=unigrams,
                alpha=0.6,
                beta=0.2,
            )

        else:
            print("LM path is not provided")

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.tensor([self.char2ind[char] for char in text], dtype=torch.long).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        """
        Decode CTC:
        
        Args:
            inds: последовательность индексов токенов из словаря 
        """
        if inds is None:
            return ""
        if len(inds) == 0:
            return ""

        result = []
        prev = None
        for i in range(len(inds)):
            idx = int(inds[i])
            # если индекс равен 0, то это пустой токен, пропускаем
            if idx == 0:
                prev = None
                continue
            # если индекс не равен предыдущему, то добавляем токен в результат прогоняя его через словарь(доставая оттуда значение)
            if idx != prev:
                result.append(self.ind2char[idx])
            prev = idx

        return "".join(result).strip()

    def ctc_beam_search(self, log_probs, beam_size: int = 10, length=None):
        """
        CTC beam search decoding.
        """

        if length == 0:
            return ""

        EMPTY_TOK = '^'

        probs = log_probs.exp()

        beam = defaultdict(float)
        beam[("", EMPTY_TOK)] = 1.0

        for t in range(length):
            proba = probs[t].cpu().numpy()
            beam = self._extend_and_merge_beam(beam, proba, EMPTY_TOK)
            beam = self._truncate(beam, beam_size)

        best_key = max(beam, key=lambda x: beam[x])
        prefix, last_char = best_key
        if last_char != EMPTY_TOK:
            prefix = prefix + last_char
        return prefix.strip().replace(EMPTY_TOK, "")

    def _truncate(self, beam, beam_size):
        sorted_items = sorted(beam.items(), key=lambda x: -x[1])[:beam_size]
        return dict(sorted_items)

    def _extend_and_merge_beam(self, beam, proba, empty_tok):
        new_beam = defaultdict(float)
        for (prefix, last_char), beam_prob in beam.items():
            for ind, pr in enumerate(proba):
                new_char = self.ind2char[ind]

                if ind == 0:
                    # Blank token - don't add to prefix, reset last_char
                    new_beam[(prefix, empty_tok)] += beam_prob * pr
                elif last_char == new_char:
                    # Same character - extend without adding
                    new_beam[(prefix, new_char)] += beam_prob * pr
                else:
                    # Different character - add last_char to prefix
                    if last_char == empty_tok:
                        new_prefix = prefix
                    else:
                        new_prefix = prefix + last_char
                    new_beam[(new_prefix, new_char)] += beam_prob * pr
        return new_beam

    def lm_ctc_beam_search(self, logits: np.ndarray, beam_size: int = 20):
        return self.lm_model.decode(logits=logits, beam_width=beam_size)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
