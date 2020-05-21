import torch
from torch.nn import Module, Linear, Sigmoid
from torch.nn.utils.rnn import pad_sequence
from model.attention_lstm_no_classification import LSTMWithAttentionNoClassification
from torch.nn.init import xavier_normal_
from torch.nn import Dropout

class MLHAN_MultiLabel(Module):

    def __init__(self, sentence_hidden_state_size, paragraph_hidden_state_size, input_size, nb_classes, nb_lstm_layer, dropout_percentage):
        super().__init__()
        self.loss = "learned"
        # self.loss_function = upper_level_cross_entropy
        self.sentence_hidden_state_size = sentence_hidden_state_size
        self.paragraph_hidden_state_size = paragraph_hidden_state_size
        self.input_size = input_size
        self.nb_classes_multiclass = nb_classes[0]
        self.nb_classes_bin = nb_classes[1]
        self.nb_lstm_layer = nb_lstm_layer
        self.dropout_percentage = dropout_percentage
        self._init_layers()

    def _init_layers(self):
        self.lstm_sentence = LSTMWithAttentionNoClassification(self.sentence_hidden_state_size, self.input_size,1)
        self.lstm_paragraph = LSTMWithAttentionNoClassification(self.paragraph_hidden_state_size, self.sentence_hidden_state_size,1)
        self.dropout = Dropout(self.dropout_percentage)

        self.paragraph_classification_mc_layer = Linear(self.paragraph_hidden_state_size, self.nb_classes_multiclass)
        self.paragraph_classification_bin_layer = Linear(self.paragraph_hidden_state_size, self.nb_classes_bin)

        self.sentence_classification_mc_layer = Linear(self.sentence_hidden_state_size, self.nb_classes_multiclass)
        self.sentence_classification_bin_layer = Linear(self.sentence_hidden_state_size, self.nb_classes_bin)

        xavier_normal_(self.paragraph_classification_mc_layer.weight.data)
        xavier_normal_(self.paragraph_classification_bin_layer.weight.data)
        xavier_normal_(self.sentence_classification_mc_layer.weight.data)
        xavier_normal_(self.sentence_classification_bin_layer.weight.data)


        self.sentence_normalizer = Sigmoid()
        self.paragraph_normalizer = Sigmoid()

    def forward(self, input_values, level_1_lengths, level_2_lengths):
        sentence_length, paragraph_length = level_1_lengths, level_2_lengths
        input_values = self.dropout(input_values)
        sentence_representation = self.lstm_sentence(input_values, sentence_length)
        reshaped_batch = self.batch_reshape(sentence_representation, paragraph_length)

        paragraph_reprensentation = self.lstm_paragraph(reshaped_batch, paragraph_length)

        sentence_classification_mc = self.sentence_classification_mc_layer(sentence_representation)
        sentence_classification_bin = self.sentence_classification_bin_layer(sentence_representation)

        paragraph_classification_mc = self.paragraph_classification_mc_layer(paragraph_reprensentation)
        paragraph_classification_bin = self.paragraph_classification_bin_layer(paragraph_reprensentation)

        return (self.sentence_normalizer(sentence_classification_mc), self.sentence_normalizer(sentence_classification_bin)), (self.paragraph_normalizer(paragraph_classification_mc), self.paragraph_normalizer(paragraph_classification_bin))

    def batch_reshape(self, reprensentation, new_dimentions):
        if new_dimentions.device.type.startswith('cuda'):
            zero_tensor = torch.IntTensor([0]).cuda()
        else:
            zero_tensor = torch.IntTensor([0])
        paragraph_indices = torch.cumsum(torch.cat((zero_tensor, new_dimentions),0),0)
        x = []
        for start, end  in zip(paragraph_indices[:-1], paragraph_indices[1:]):
            x.append(reprensentation[start:end,:])

        return pad_sequence(x, True)

    def _calculate_hierarchical_attention_for_higher_level(self, exemple_by_lower_levels, sentence_lengths):
        _, word_attention = self.lstm_sentence._calculate_attention(exemple_by_lower_levels, sentence_lengths)
        sentence_representation = self.lstm_sentence(exemple_by_lower_levels, sentence_lengths)
        reshaped_batch = self.batch_reshape(sentence_representation, torch.IntTensor([exemple_by_lower_levels.size()[0]]).cuda())
        _, sentence_attention = self.lstm_paragraph._calculate_attention(reshaped_batch, torch.IntTensor([exemple_by_lower_levels.size()[0]]).cuda())

        return word_attention, sentence_attention
