from torch.nn import Module, LSTM, Embedding, Linear, Sigmoid, ReLU

from torch.nn.init import kaiming_normal
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F
import torch
from torch.nn.init import xavier_normal_

NEG_INF = -10000
TINY_FLOAT = 1e-6

def mask_softmax(matrix, mask=None):

    if mask is None:
        result = F.softmax(matrix, dim=-1)
    else:
        mask_norm = ((1 - mask) * NEG_INF).to(matrix)
        for i in range(matrix.dim() - mask_norm.dim()):
            mask_norm = mask_norm.unsqueeze(1)
        result = F.softmax(matrix + mask_norm, dim=-1)

    return result

def make_sequence_mask(sequence_lengths):

    maximum_length = torch.max(sequence_lengths)

    idx = torch.arange(maximum_length).to(sequence_lengths).repeat(sequence_lengths.size(0), 1)
    mask = torch.gt(sequence_lengths.unsqueeze(-1), idx).to(sequence_lengths)

    return mask


class LSTMWithAttentionNoClassification(Module):

    def __init__(self, hidden_state_size, input_size, nb_lstm_layer):
        super().__init__()
        self.hidden_state_size = hidden_state_size
        self.input_size = input_size
        self.nb_lstm_layer = nb_lstm_layer
        self._init_layers()

    def _init_layers(self):
        self.lstm = LSTM(self.input_size, self.hidden_state_size, self.nb_lstm_layer, bidirectional=True)

        self.attention_layer = Linear(self.hidden_state_size * 2 , 1)
        self.combinaison_layer = Linear(self.hidden_state_size * 2, self.hidden_state_size)

        xavier_normal_(self.attention_layer.weight.data)
        xavier_normal_(self.combinaison_layer.weight.data)


        self.activation = ReLU()

    def forward(self, input_values,sequence_lengths ):
        lstm, attention = self._calculate_attention(input_values, sequence_lengths)
        r_attention = torch.sum(attention.unsqueeze(-1) * lstm, dim=1)
        combined_lstm = self.activation(self.combinaison_layer(r_attention))

        return combined_lstm

    def _calculate_attention(self, input_values, sequence_lengths):
        packed_batch = pack_padded_sequence(input_values, sequence_lengths, batch_first=True, enforce_sorted=False)
        lstm_output, _ = self.lstm(packed_batch)
        unpacked_lstm_output, _ = pad_packed_sequence(lstm_output, batch_first=True)

        sequence_mask = make_sequence_mask(sequence_lengths)
        attention = self.attention_layer(unpacked_lstm_output)
        soft_maxed_attention = mask_softmax(attention.squeeze(-1), sequence_mask)

        return unpacked_lstm_output, soft_maxed_attention



