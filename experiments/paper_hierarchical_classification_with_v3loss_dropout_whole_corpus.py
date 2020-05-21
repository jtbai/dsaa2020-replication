from torch import FloatTensor, from_numpy, Tensor
from torch.nn.utils.rnn import pad_sequence
from poutyne.framework import Experiment, ReduceLROnPlateau
from poutyne.framework import Callback
import torch
from typing import Tuple, List
import pickle
import numpy as np
from infrastructure.dataset.per_class_dataset import PerClassDataset, PerClassLoader
from model.hierarchical_lstm_with_learnable_loss_and_dropout import  HierarchicalLSTMWithLearnableLossWithDropout
from model.loss.two_level_loss import TopLevelAccuracy, FBetaTopLevel, LearnableLossWeightNoSum#, LearnableLossWeightXavierNormal
from poutyne import set_seeds



aggregation = "by_paragraph"
dataset = "2straight"
# test_name = "newlossv0_phs30_shs10_patience10_cd5_1l_bs32_k150" # 0.89 epoch 13 (tracking loss)


#V0 is two level loss straight, no balance
#V1 is the switching loss on 2 loss (top and both level)
#V1.5 is a training on the second level, then finetune with information on first level. Model used will need to be hand picked
#V2 is the iterating version of 3 losses (top low both levels)
#V3 is module learning it's loss weights

sentence_hidden_state_size = 20
paragraph_hidden_state_size = 10
nb_layers = 1
patience = 20
cooldown = 5
batch_size=4
epoch = 70
seed = 420
drop_out = 0.5
set_seeds(seed)
optimizer = "adam"
test_name = "paper_loss_phs{}_shs{}_patience{}_cd{}_{}l_bs{}_op{}_do{}_noupscale_seed{}_test".format(
    paragraph_hidden_state_size, sentence_hidden_state_size, patience, cooldown, nb_layers, batch_size, optimizer,drop_out, seed
)




train_pickled = pickle.load(open("./data/{}/{}/preprocessed_train.pkl".format(aggregation, dataset),'rb'))
valid_pickled = pickle.load(open("./data/{}/{}/preprocessed_valid.pkl".format(aggregation, dataset), 'rb'))
test_pickled = pickle.load(open("./data/{}/{}/preprocessed_test.pkl".format(aggregation, dataset), 'rb'))

# preprocessed_train = [datapoint for _, _, _, _,datapoint in train_pickled]
# preprocessed_valid = [datapoint for _, _, _, _, datapoint in valid_pickled]


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

    return (pad_sequence(x, batch_first=True), sentence_true_length_tensor, paragraph_true_length_tensor), (y_sentence_tensor, y_paragraph_tensor)

train_loader = PerClassLoader(train, batch_size=batch_size, k=-1, collate_fn=reformat_and_pad_batch)
valid_loader = PerClassLoader(valid, batch_size=4, k=-1, shuffle=False, collate_fn=reformat_and_pad_batch)

loss_functon = LearnableLossWeightNoSum()
lstm = HierarchicalLSTMWithLearnableLossWithDropout(sentence_hidden_state_size,paragraph_hidden_state_size,300,2,nb_layers,drop_out)


class IterateLossFunctionOnEpoch(Callback):
    def __init__(self, epoch_number):
        self.epoch_number = epoch_number
        super().__init__()

    def on_epoch_end(self, epoch_number, logs):
        if epoch_number % self.epoch_number == 0:
            logs["loss"] = self.model.network.loss
            self.model.network.switch_loss_type()
            self.model.loss_function = self.model.network.loss_function


experiment = Experiment("model_weights/hc_{}_{}_{}".format(aggregation, dataset, test_name), lstm, optimizer=optimizer, device=0,loss_function=loss_functon, monitor_metric="val_fscore_macro", monitor_mode="max", epoch_metrics=[TopLevelAccuracy(), FBetaTopLevel(average='macro')])
# monitor_metric="top_level_accuracy", monitor_mode="max"
experiment.train(train_loader, valid_loader, lr_schedulers=[ReduceLROnPlateau(patience=patience, cooldown=cooldown)],epochs=epoch)
# BestModelRestoreOnEpoch(epoch=50,monitor="val_fscore_macro", mode="max" )


import tqdm
import json
import spacy

nlp = spacy.load('en_core_web_sm')

# for index, paragraph in enumerate(test_pickled):
#     current_test_ids = ["{}-{}-{}-{}".format(doc_id, par_id, sentence_id) for doc_id, par_id, sentence_id, _, _ in paragraph]
#     current_test_context = [text for _, _, _, text, _ in test_pickled]
#     current_preprocessed_test = [datapoint for _, _, _, _, datapoint in test_pickled]


experiment.model.network.eval()
json_docs =[]
with open('model_weights/attention_{}_{}_{}.json'.format(aggregation, dataset, test_name),'w') as output_file:

    for paragraph in tqdm.tqdm(test_pickled, total=len(test_pickled)):
        paragraph_id = paragraph[0][0]

        (padded_batch, sentence_true_length_tensor,_), (_,y_paragraph_tensor) = reformat_and_pad_batch([paragraph])

        word_attention_tensor, sentence_attention_tensor = experiment.model.model._calculate_hierarchical_attention_for_higher_level(padded_batch.cuda(), sentence_true_length_tensor.cuda())

        predictions = experiment.model.model(padded_batch.cuda(), sentence_true_length_tensor.cuda(),
                                             torch.IntTensor([padded_batch.size()[0]]).cuda())

        paragraph_prediction = predictions[-1].squeeze(0).cpu().detach().numpy()
        sentence_predictions = predictions[0].cpu().detach().numpy()

        word_attention = word_attention_tensor.cpu().detach().numpy()
        sentence_attention = sentence_attention_tensor.squeeze(0).cpu().detach().numpy()

        for sentence_index, ((sentence_data, sentence_y), attention_vector, sentence_prediction)  in enumerate(zip(paragraph[0],sentence_attention, sentence_predictions)):
            doc = nlp(sentence_data[3])
            prediction = np.argmax(sentence_prediction)
            prediction_probability = sentence_prediction[prediction]
            paragraph_id = "{}-{}".format(sentence_data[0],sentence_data[1])
            sentence_id = "{}-{}-{}".format(sentence_data[0], sentence_data[1],sentence_data[2])
            tokenized_text = [token.text for token in doc]
            tokenized_text.insert(0, "{}/{} ({:.2f}) ".format(sentence_y, prediction, prediction_probability))
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
                'prediction':int(np.argmax(paragraph_prediction)),
                'posterior':prediction.tolist(),
                'attention':attention

            }
            json_docs.append(json_doc)
    json.dump(json_docs, output_file,indent=4)


