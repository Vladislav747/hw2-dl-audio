import re
from string import ascii_lowercase

import torch

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
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

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
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
        if inds is None:
            return ""
        if len(inds) == 0:
            return ""
        
        result_list = []
        if isinstance(inds, torch.Tensor):
            temp_list = []
            for i in range(inds.shape[0]):
                temp_list.append(int(inds[i].item()))
            inds = temp_list
        else:
            new_list = []
            for item in inds:
                if hasattr(item, 'item'):
                    new_list.append(int(item.item()))
                else:
                    new_list.append(int(item))
            inds = new_list
        
        last_char_idx = -1
        for idx in range(len(inds)):
            current_idx = int(inds[idx])
            current_char = self.ind2char[current_idx]
            if current_idx == 0:
                last_char_idx = -1
            else:
                if last_char_idx != current_idx:
                    result_list.append(current_char)
                    last_char_idx = current_idx
        
        final_string = ""
        for ch in result_list:
            final_string = final_string + ch
        
        return final_string.strip()

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
