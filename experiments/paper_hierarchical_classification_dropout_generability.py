from torch import FloatTensor, from_numpy, Tensor
from torch.nn.utils.rnn import pad_sequence
from poutyne.framework import Experiment, ReduceLROnPlateau
from poutyne.framework.metrics.epoch_metrics import FBeta
import torch
from typing import Tuple, List
import pickle
import numpy as np
from infrastructure.dataset.per_class_dataset import PerClassDataset, PerClassLoader
from model.hierarchical_lstm_dropout import HierarchicalLSTMwithDropout
from poutyne import set_seeds
aggregation = "by_paragraph"
dataset = "2tfpositive_30pct"
sentence_hidden_state_size = 20
paragraph_hidden_state_size = 10
nb_layers = 1
patience = 20
cooldown = 5
batch_size=4
epoch = 75
seed = 1337
dropout = 0.0
optimizer = "adam"

test_name = "paper_baseline__phs{}_shs{}_patience{}_cd{}_{}l_bs{}_op{}_noupscale_do{}_seed{}".format(
    paragraph_hidden_state_size, sentence_hidden_state_size, patience, cooldown, nb_layers, batch_size, optimizer, dropout, seed
)

set_seeds(seed)

train_pickled = pickle.load(open("./data/{}/{}/preprocessed_train.pkl".format(aggregation, dataset),'rb'))
valid_pickled = pickle.load(open("./data/{}/{}/preprocessed_valid.pkl".format(aggregation, dataset), 'rb'))
test_pickled = pickle.load(open("./data/{}/{}/preprocessed_test.pkl".format(aggregation, dataset), 'rb'))

train = PerClassDataset(train_pickled)
valid = PerClassDataset(valid_pickled)

def reformat_and_pad_batch(batch : List[Tuple[FloatTensor, int]]) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
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

    return (pad_sequence(x, batch_first=True), sentence_true_length_tensor, y_sentence_tensor, paragraph_true_length_tensor), y_paragraph_tensor

train_loader = PerClassLoader(train, batch_size=4, k=-1, collate_fn=reformat_and_pad_batch)
valid_loader = PerClassLoader(valid, batch_size=4, k=-1, shuffle=False, collate_fn=reformat_and_pad_batch)

lstm = HierarchicalLSTMwithDropout(sentence_hidden_state_size,paragraph_hidden_state_size,300,2,nb_layers,dropout)

experiment = Experiment("model_weights/final_{}_{}_{}".format(aggregation, dataset, test_name), lstm, optimizer="adam", loss_function="cross_entropy", device=0, task="classification", monitor_metric="val_fscore_macro", monitor_mode="max", epoch_metrics=[FBeta(average='macro')])
experiment.train(train_loader, valid_loader, lr_schedulers=[ReduceLROnPlateau(patience=3)],epochs=epoch)



import tqdm
import json
import spacy

nlp = spacy.load('en_core_web_sm')

json_docs =[]
with open('model_weights/attention_{}_{}_{}.json'.format(aggregation, dataset, test_name),'w') as output_file:

    for paragraph in tqdm.tqdm(test_pickled, total=len(test_pickled)):
        paragraph_id = paragraph[0][0]

        batch = reformat_and_pad_batch([paragraph])

        (padded_batch, sentence_true_length_tensor, _, _), y_paragraph_tensor = reformat_and_pad_batch([paragraph])

        word_attention_tensor, sentence_attention_tensor = experiment.model.model._calculate_hierarchical_attention_for_higher_level(padded_batch.cuda(), sentence_true_length_tensor.cuda())

        prediction = experiment.model.model(padded_batch.cuda(), sentence_true_length_tensor.cuda(), torch.IntTensor([0]).unsqueeze(0).cuda(),torch.IntTensor([padded_batch.size()[0]]).cuda()).squeeze(0).cpu().detach().numpy()
        word_attention = word_attention_tensor.cpu().detach().numpy()
        sentence_attention = sentence_attention_tensor.squeeze(0).cpu().detach().numpy()

        for sentence_index, ((sentence_data, sentence_y), attention_vector) in enumerate(zip(paragraph[0],sentence_attention)):
            doc = nlp(sentence_data[3])
            paragraph_id = "{}-{}".format(sentence_data[0],sentence_data[1])
            sentence_id = "{}-{}-{}".format(sentence_data[0], sentence_data[1],sentence_data[2])
            tokenized_text = [token.text for token in doc]
            tokenized_text.insert(0, "{}".format(sentence_y))
            tokenized_text.insert(0, "{:.2f}".format(attention_vector))
            tokenized_text.insert(0, sentence_id)
            attention = np.ravel(word_attention[sentence_index][:len(tokenized_text)]).tolist()
            attention.insert(0, float(0))
            attention.insert(0, float(0))
            attention.insert(0, float(0))

            json_doc = {
                # 'paragraph_id': paragraph_id,
                'id':sentence_id,
                'text':tokenized_text,
                # 'sentence_attention':float(attention_vector),
                'label':int(y_paragraph_tensor.numpy()),
                'prediction':int(np.argmax(prediction)),
                'posterior':prediction.tolist(),
                'attention':attention

            }
            json_docs.append(json_doc)
    json.dump(json_docs, output_file,indent=4)


