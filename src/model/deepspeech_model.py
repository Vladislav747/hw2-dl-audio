import torch.nn as nn
from torch.nn import Sequential

from torch import Tensor

from src.model.deepspeech_module.modules import NormGRU

class DeepSpeech2(nn.Module):
    def __init__(self, n_feats, n_tokens, num_rnn_layers, hidden_size, rnn_dropout, input_dim=32):
        """
        Args:
            n_feats (int)
            n_tokens (int)
            num_rnn_layers (int)
            hidden_size (int)
            rnn_dropout (float)
            input_dim (int)
        """
        super().__init__()
        
        self.input_dim = input_dim

        self.conv_params = {
            "conv1": {"padding": (20, 5), "kernel_size": (41, 11), "stride": (2, 2)},
            "conv2": {"padding": (10, 5), "kernel_size": (21, 11), "stride": (2, 2)},
            "conv3": {"padding": (10, 5), "kernel_size": (21, 11), "stride": (2, 1)},
        }

        self.conv = Sequential(
            nn.Conv2d(1, input_dim, **self.conv_params["conv1"]),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(input_dim, input_dim, **self.conv_params["conv2"]),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(input_dim, input_dim*3, **self.conv_params["conv3"]),
            nn.BatchNorm2d(input_dim*3),
            nn.ReLU(),
        )

        rnn_input_dim = self.calc_rnn_input_size(n_feats)
        rnn_input_dim *= input_dim*3

        self.rnns = Sequential(
            *[
                NormGRU(
                    (hidden_size if i > 0 else rnn_input_dim), hidden_size, rnn_dropout
                )
                for i in range(num_rnn_layers)
            ]
        )

        self.fc = nn.Linear(hidden_size, n_tokens)

    def calc_rnn_input_size(self, n_feats):
        """
        Calculates the size of the RNN input after convolutions for NormGRU
        Args:
            n_feats (int): Number of input features.
        Returns:
            int: Size of RNN input.
        """
        size = n_feats
        for conv_param in self.conv_params.values():
            size = (
                           size + 2 * conv_param["padding"][0] - conv_param["kernel_size"][0]
                   ) // conv_param["stride"][0] + 1
        return size

    def forward(self, spectrogram, spectrogram_length, **batch):
        """

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        x = self.conv(spectrogram.unsqueeze(1))
        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        x = (
            x.transpose(1, 2).transpose(0, 1).contiguous()
        )

        h = None
        for rnn in self.rnns:
            x, h = rnn(x, h)

        time_steps, batch_size = x.shape[0], x.shape[1]
        x = x.view(time_steps * batch_size, -1)
        logits = self.fc(x)
        logits = logits.view(time_steps, batch_size, -1).transpose(0, 1)

        log_probs = nn.functional.log_softmax(logits, dim=-1)
        return {
            "logits": logits,
            "log_probs": log_probs,
            "log_probs_length": self.transform_input_lengths(spectrogram_length),
        }

    def transform_input_lengths(self, input_lengths):
        """

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            transformed input_lengths (Tensor): new temporal lengths
        """
        for conv_param in self.conv_params.values():
            input_lengths = (
                                    input_lengths
                                    + 2 * conv_param["padding"][1]
                                    - conv_param["kernel_size"][1]
                            ) // conv_param["stride"][1] + 1
        return input_lengths

    def __str__(self):
        """
        Return model details including parameter counts.
        """
        all_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        result_str = super().__str__()
        result_str += f"\nAll parameters in MODEL: {all_params}"
        result_str += f"\nTrainable parameters in MODEl: {trainable_params}"
        return result_str
