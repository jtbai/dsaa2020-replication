from torch import FloatTensor, from_numpy, Tensor
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from poutyne.framework import Experiment, ReduceLROnPlateau
import torch
from typing import Tuple, List
import pickle
import numpy as np
from model.hierarchical_multilabel_lstm_with_learnable_loss_and_dropout import HierarchicalLSTMWithLearnableLossWithDropoutMultiLabel
from model.loss.two_level_loss import LearnableLossMultiClassNoSum, TopLevelMultiLabelFBeta  # , LearnableLossWeightXavierNormal
from poutyne import set_seeds

aggregation = "by_paragraph"
dataset = "3mc2bin_straight"
# test_name = "newlossv0_phs30_shs10_patience10_cd5_1l_bs32_k150" # 0.89 epoch 13 (tracking loss)


#V0 is two level loss straight, no balance
#V1 is the switching loss on 2 loss (top and both level)
#V1.5 is a training on the second level, then finetune with information on first level. Model used will need to be hand picked
#V2 is the iterating version of 3 losses (top low both levels)
#V3 is module learning it's loss weights

nb_multiclass = 3
sentence_hidden_state_size = 20
paragraph_hidden_state_size = 10
nb_class = 2
nb_layers = 1
patience = 20
cooldown = 5
batch_size = 4
epoch = 75
seed = 26
drop_out = 0.50
set_seeds(seed)
optimizer = "adam"
test_name = "hmc_loss((pb+pm)(sm+sb))_phs{}_shs{}_patience{}_cd{}_{}l_bs{}_op{}_do{}_seed{}_wd0".format(
    paragraph_hidden_state_size, sentence_hidden_state_size, patience, cooldown, nb_layers, batch_size, optimizer,int(100*drop_out), seed
)


train_pickled = pickle.load(open("./data/{}/{}/preprocessed_train.pkl".format(aggregation, dataset),'rb'))
valid_pickled = pickle.load(open("./data/{}/{}/preprocessed_valid.pkl".format(aggregation, dataset), 'rb'))
test_pickled = pickle.load(open("./data/{}/{}/preprocessed_test.pkl".format(aggregation, dataset), 'rb'))

# preprocessed_train = [datapoint for _, _, _, _,datapoint in train_pickled]
# preprocessed_valid = [datapoint for _, _, _, _, datapoint in valid_pickled]



class MultiLabelDataset(Dataset):

    def __init__(self, base_pickle):
        self.dataset = base_pickle

    def __getitem__(self, item):
        current_item = self.dataset[item]

        return current_item[0], current_item[1]

    def __len__(self):
        return len(self.dataset)


train = MultiLabelDataset(train_pickled)
valid = MultiLabelDataset(valid_pickled)

def reformat_and_pad_batch_for_multiclass_and_binary_classification(batch : List[Tuple[FloatTensor, int]]) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
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

train_loader = DataLoader(train, batch_size=batch_size, collate_fn=reformat_and_pad_batch_for_multiclass_and_binary_classification)
valid_loader = DataLoader(valid, batch_size=4, shuffle=False, collate_fn=reformat_and_pad_batch_for_multiclass_and_binary_classification)

loss_functon = LearnableLossMultiClassNoSum()
lstm = HierarchicalLSTMWithLearnableLossWithDropoutMultiLabel(sentence_hidden_state_size,paragraph_hidden_state_size,300,[nb_multiclass, nb_class] ,nb_layers,drop_out)

# optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001, weight_decay=0.01)

experiment = Experiment("model_weights/test_{}_{}_{}".format(aggregation, dataset, test_name), lstm, optimizer=optimizer, device=0,loss_function=loss_functon, monitor_metric="val_fscore_macro", monitor_mode="max", epoch_metrics=[TopLevelMultiLabelFBeta(average='macro')])
experiment.train(train_loader, valid_loader, lr_schedulers=[ReduceLROnPlateau(patience=patience, cooldown=cooldown)],epochs=epoch)


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

        (padded_batch, sentence_true_length_tensor,_), (_,y_paragraph_tensor) = reformat_and_pad_batch_for_multiclass_and_binary_classification([paragraph])

        word_attention_tensor, sentence_attention_tensor = experiment.model.model._calculate_hierarchical_attention_for_higher_level(padded_batch.cuda(), sentence_true_length_tensor.cuda())

        predictions = experiment.model.model(padded_batch.cuda(), sentence_true_length_tensor.cuda(),
                                             torch.IntTensor([padded_batch.size()[0]]).cuda())

        paragraph_prediction = predictions[-1][-1].squeeze(0).cpu().detach().numpy()
        sentence_predictions_1 = predictions[0][0].cpu().detach().numpy()
        sentence_predictions_2 = predictions[0][1].cpu().detach().numpy()

        word_attention = word_attention_tensor.cpu().detach().numpy()
        sentence_attention = sentence_attention_tensor.squeeze(0).cpu().detach().numpy()

        for sentence_index, ((sentence_data, sentence_y), attention_vector, sentence_prediction_1,  sentence_prediction_2)  in enumerate(zip(paragraph[0],sentence_attention, sentence_predictions_1, sentence_predictions_2)):
            doc = nlp(sentence_data[3])
            prediction = np.argmax(sentence_prediction_2)
            prediction_probability = sentence_prediction_2[prediction]
            paragraph_id = "{}-{}".format(sentence_data[0],sentence_data[1])
            sentence_id = "{}-{}-{}".format(sentence_data[0], sentence_data[1],sentence_data[2])
            tokenized_text = [token.text for token in doc]
            joined_sentence_mc_prediction = "-".join((sentence_prediction_1>.5).astype(int).astype(str))
            joined_sentence_mc_truth = "-".join(sentence_y[0].astype(str))
            tokenized_text.insert(0, "| {}/{} {}/{} | ({:.2f})".format(joined_sentence_mc_truth, joined_sentence_mc_prediction, sentence_y[1], prediction, prediction_probability))
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
                'label':int(y_paragraph_tensor[-1].numpy()),
                'prediction':int(np.argmax(paragraph_prediction)),
                'posterior':prediction.tolist(),
                'attention':attention

            }
            json_docs.append(json_doc)
    json.dump(json_docs, output_file,indent=4)


