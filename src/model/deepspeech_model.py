import torch
import torch.nn as nn

from torch import Tensor

from src.model.deepspeech_module.convolution import DeepSpeech2Extractor
from src.model.deepspeech_module.modules import Linear, LayerNorm
from src.model.deepspeech_module.rnn import BNReluRNN


class DeepSpeech2(nn.Module):
    """
    Args:
        input_dim (int): dimension of input vector
        n_tokens (int): number of tokens in vocabulary
        rnn_type (str, optional): type of RNN cell (default: gru)
        num_rnn_layers (int, optional): number of recurrent layers (default: 5)
        rnn_hidden_dim (int): the number of features in the hidden state `h`
        dropout_p (float, optional): dropout probability (default: 0.1)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (defulat: True)
        activation (str): type of activation function (default: hardtanh)
        device (torch.device): device - 'cuda' or 'cpu'

    Inputs: spectrogram, spectrogram_length
        - **spectrogram**: tensor of shape [batch, features, time] or [batch, time, features]
        - **spectrogram_length**: tensor of sequence lengths

    Returns: output
        - **output**: dict containing log_probs and log_probs_length
    """

    def __init__(
            self,
            input_dim: int,
            n_tokens: int,
            rnn_type: str = "gru",
            num_rnn_layers: int = 5,
            rnn_hidden_dim: int = 512,
            dropout_p: float = 0.1,
            bidirectional: bool = True,
            activation: str = "hardtanh",
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        super().__init__()

        self.device = device
        self.conv = DeepSpeech2Extractor(input_dim, activation=activation)
        self.rnn_layers = nn.ModuleList()
        # Для двунаправленного RNN размер выхода удваивается (<< 1 = * 2)
        rnn_output_size = rnn_hidden_dim << 1 if bidirectional else rnn_hidden_dim

        # Вычисляем размер выхода conv слоя: out_channels * input_dim
        # DeepSpeech2Extractor использует out_channels=32 по умолчанию
        conv_output_dim = 32 * input_dim

        # Создаем стек RNN слоев
        for idx in range(num_rnn_layers):
            self.rnn_layers.append(
                BNReluRNN(
                    input_size=conv_output_dim if idx == 0 else rnn_output_size,
                    hidden_state_dim=rnn_hidden_dim,
                    rnn_type=rnn_type,
                    bidirectional=bidirectional,
                    dropout_p=dropout_p,
                )
            )

        self.fc = nn.Sequential(
            LayerNorm(rnn_output_size),
            Linear(rnn_output_size, n_tokens, bias=False),
        )

    def forward(self, spectrogram: Tensor, spectrogram_length: Tensor, **batch) -> dict:
        outputs, output_lengths = self.conv(spectrogram, spectrogram_length)
        outputs = outputs.permute(1, 0, 2).contiguous()

        for rnn_layer in self.rnn_layers:
            outputs = rnn_layer(outputs, output_lengths)

        outputs = self.fc(outputs.transpose(0, 1)).log_softmax(dim=-1)

        log_probs_length = self.transform_input_lengths(output_lengths)

        return {"log_probs": outputs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        return input_lengths
