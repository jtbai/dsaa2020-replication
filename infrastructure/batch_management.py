from torch import from_numpy, LongTensor
import numpy as np
from torch.nn.utils.rnn import pad_sequence

def reformat_and_pad_batch_1lvl(batch):
    x = []
    y_paragraph = []
    real_batch_size = 0
    sentence_true_length = []
    paragraph_true_length = []

    for paragraph, target_for_paragraph in batch:
        real_batch_size += len(paragraph)
        paragraph_true_length.append(len(paragraph))
        y_paragraph.append(target_for_paragraph)
        for index, sentence in enumerate(paragraph):
            if sentence[0]:
                x.append(sentence[0][4][0])
                sentence_true_length.append(sentence[0][4][0].size()[0])


    y_paragraph = np.array(y_paragraph, dtype=np.int64)
    sentence_true_length = np.array(sentence_true_length, dtype=np.int32)
    paragraph_true_length = np.array(paragraph_true_length, dtype=np.int32)

    sentence_true_length_tensor = from_numpy(sentence_true_length)
    paragraph_true_length_tensor = from_numpy(paragraph_true_length)

    y_paragraph_tensor = from_numpy(y_paragraph)

    return (pad_sequence(x, batch_first=True), sentence_true_length_tensor, paragraph_true_length_tensor), y_paragraph_tensor

def reformat_and_pad_batch_2lvl(batch):
    x = []
    y_sentence = []
    y_paragraph = []
    real_batch_size = 0
    sentence_true_length = []
    paragraph_true_length = []


    for paragraph, target_for_paragraph in batch:
        real_batch_size += len(paragraph)
        paragraph_true_length.append(len(paragraph))
        y_paragraph.append(target_for_paragraph)
        for index, sentence in enumerate(paragraph):
            if sentence[0]:
                x.append(sentence[0][4][0])
                y_sentence.append(sentence[1])
                sentence_true_length.append(sentence[0][4][0].size()[0])

    y_sentence = np.array(y_sentence, dtype=np.int64)
    y_paragraph = np.array(y_paragraph, dtype=np.int64)
    sentence_true_length = np.array(sentence_true_length, dtype=np.int32)
    paragraph_true_length = np.array(paragraph_true_length, dtype=np.int32)

    sentence_true_length_tensor = from_numpy(sentence_true_length)
    paragraph_true_length_tensor = from_numpy(paragraph_true_length)

    y_sentence_tensor = from_numpy(y_sentence)
    y_paragraph_tensor = from_numpy(y_paragraph)

    return (pad_sequence(x, batch_first=True), sentence_true_length_tensor, paragraph_true_length_tensor), (y_sentence_tensor, y_paragraph_tensor)

nb_multiclass=3

def reformat_and_pad_batch_for_mc_and_bin_2lvl(batch):
    x = []
    y_sentence = []
    y_paragraph_1 = np.zeros(len(batch) * nb_multiclass, dtype=np.float32).reshape(len(batch), nb_multiclass)
    y_paragraph_2 = np.zeros(len(batch), dtype=np.int64)
    real_batch_size = 0
    sentence_true_length = []
    paragraph_true_length = []

    y_sentence_1 = []
    y_sentence_2 = []

    for paragraph_index, (paragraph, target_for_paragraph) in enumerate(batch):
        real_batch_size += len(paragraph)
        paragraph_true_length.append(len(paragraph))
        y_paragraph_1[paragraph_index, :] = target_for_paragraph[0]
        y_paragraph_2[paragraph_index] = target_for_paragraph[1]

        y_sentence_1_current_paragraph = np.zeros(len(paragraph) * nb_multiclass, dtype=np.float32).reshape(len(paragraph), nb_multiclass)
        y_sentence_2_current_paragraph = np.zeros(len(paragraph) , dtype=np.int64)

        for index, sentence in enumerate(paragraph):
            x.append(sentence[0][4][0])
            y_sentence_1_current_paragraph[index, :] = sentence[1][0]
            y_sentence_2_current_paragraph[index] = sentence[1][1]

            sentence_true_length.append(sentence[0][4][0].size()[0])

        y_sentence_1.append(y_sentence_1_current_paragraph)
        y_sentence_2.append(y_sentence_2_current_paragraph)

    y_sentence_1 = np.vstack(y_sentence_1)
    y_sentence_2 = np.hstack(y_sentence_2)

    sentence_true_length = np.array(sentence_true_length, dtype=np.int32)
    paragraph_true_length = np.array(paragraph_true_length, dtype=np.int32)

    sentence_true_length_tensor = from_numpy(sentence_true_length)
    paragraph_true_length_tensor = from_numpy(paragraph_true_length)

    y_sentence_1_tensor = from_numpy(y_sentence_1)
    y_sentence_2_tensor = from_numpy(y_sentence_2)
    y_paragraph_1_tensor = from_numpy(y_paragraph_1)
    y_paragraph_2_tensor = from_numpy(y_paragraph_2)

    return (pad_sequence(x, batch_first=True), sentence_true_length_tensor, paragraph_true_length_tensor), ((y_sentence_1_tensor, y_sentence_2_tensor), (y_paragraph_1_tensor,y_paragraph_2_tensor))
