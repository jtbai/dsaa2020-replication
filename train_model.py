import pickle
import argparse
import torch
from torch.utils.data import DataLoader
from poutyne.framework import Experiment, ReduceLROnPlateau
from poutyne.framework.metrics.epoch_metrics import FBeta
from poutyne import set_seeds

from infrastructure.dataset import TupleInListDataset
from infrastructure.batch_management import *
from model import HAN, MLHAN_Binary, MLHAN_MultiLabel
from model.loss import MultiLevelLoss, LevelSwitchingLoss
from model.metrics import *

parser = argparse.ArgumentParser()
parser.add_argument("-m",dest="model_name", type=str)
parser.add_argument("-r",dest="seed", default=41, type=int)
parser.add_argument("-e",dest="experiment_name", default=41, type=str)

args = parser.parse_args()
seed = args.seed
set_seeds(seed)
model_name = args.model_name
experiment_name = args.experiment_name

# Dataset management

if experiment_name == "fullcorpus":
    data_path = "2straight"
elif experiment_name == "10pct":
    data_path = "2tfpositive_10pct"
elif experiment_name == "15pct":
    data_path = "2tfpositive_15pct"
elif experiment_name == "30pct":
    data_path = "2tfpositive_30pct"
elif experiment_name == "xlearn":
    data_path = "3mc2bin_straight"
else:
    raise ValueError("Experiement {} unknown. Please use full-corpus, 10pct, 15pct, 30pct or xlearn")

train_pickled = pickle.load(open("./datasets/{}/preprocessed_train.pkl".format(data_path),'rb'))
valid_pickled = pickle.load(open("./datasets/{}/preprocessed_valid.pkl".format(data_path), 'rb'))
test_pickled = pickle.load(open("./datasets/{}/preprocessed_test.pkl".format(data_path), 'rb'))
train = TupleInListDataset(train_pickled)
valid = TupleInListDataset(valid_pickled)

# Model Hyperparameters

sentence_hidden_state_size = 20
paragraph_hidden_state_size = 10
nb_layers = 1
patience = 20
cooldown = 5
batch_size = 4
dropout = .50
epoch = 125
optimizer = "adam"
test_name = "replicability_model-{}_experiement-{}_seed-{}".format(model_name, experiment_name, seed)
print(test_name)

# Model & Loss Management
final_model_experiment_folder_template = "model_output/{}"
model_experiment_folder_template = "model_output/{}"

if model_name == "han":
    dropout = .25
    collate_function =reformat_and_pad_batch_1lvl
    loss_functon = torch.nn.functional.cross_entropy
    model = HAN(sentence_hidden_state_size,paragraph_hidden_state_size,300,2,nb_layers, dropout_percentage=dropout)
    epoch_metrics = [FBeta(average='macro')]
elif model_name == "mlhan":
    dropout = .50
    collate_function = reformat_and_pad_batch_2lvl
    loss_functon = MultiLevelLoss()
    model = MLHAN_Binary(sentence_hidden_state_size, paragraph_hidden_state_size, 300, 2, nb_layers, dropout_percentage=dropout)
    epoch_metrics = [UpperLevelAccuracy(), FBetaUpperLevel(average='macro')]

    # xlearn can only be done with mlhan
    if experiment_name == "xlearn":
        collate_function = reformat_and_pad_batch_for_mc_and_bin_2lvl
        loss_functon = LevelSwitchingLoss([0,0], 0)
        # dropout = .5
        model = MLHAN_MultiLabel(sentence_hidden_state_size, paragraph_hidden_state_size, 300, [3,2], nb_layers,dropout_percentage=dropout)
        model_experiment_folder_template = "model_output/{}/bottom_level"
        epoch_metrics = [FBetaLowerLevelMultiLabel(average='macro')]
        epoch_metrics_post_transfer= [FBetaUpperLevelMultiLabel(average='macro')]


else:
    raise ValueError("Model name {} unknown. Please use han or mlhan".format(model_name))



train_loader = DataLoader(train, batch_size=4, collate_fn=collate_function)
valid_loader = DataLoader(valid, batch_size=4,  shuffle=False, collate_fn=collate_function)

experiment = Experiment(model_experiment_folder_template.format(test_name), model, optimizer=optimizer, device=0, loss_function=loss_functon, monitor_metric="val_fscore_macro", monitor_mode="max", epoch_metrics=epoch_metrics)
experiment.train(train_loader, valid_loader, lr_schedulers=[ReduceLROnPlateau(patience=patience, cooldown=cooldown)], epochs=epoch)



if experiment_name == "xlearn":
    print("\n Now Training upper level \n ")

    loss_functon.switch_state()
    best_lower_model = experiment.model.network
    experiment = Experiment(final_model_experiment_folder_template.format(test_name),best_lower_model, optimizer=optimizer, device=0, loss_function=loss_functon,monitor_metric="val_fscore_macro", monitor_mode="max",epoch_metrics=epoch_metrics_post_transfer)
    for parameter in experiment.model.network.lstm_sentence.parameters():
        parameter.requires_grad = False
    experiment.train(train_loader, valid_loader, lr_schedulers=[ReduceLROnPlateau(patience=patience, cooldown=cooldown)],epochs=epoch)

# import tqdm
# import json
# import spacy
#
# nlp = spacy.load('en_core_web_sm')
#
# # for index, paragraph in enumerate(test_pickled):
# #     current_test_ids = ["{}-{}-{}-{}".format(doc_id, par_id, sentence_id) for doc_id, par_id, sentence_id, _, _ in paragraph]
# #     current_test_context = [text for _, _, _, text, _ in test_pickled]
# #     current_preprocessed_test = [datapoint for _, _, _, _, datapoint in test_pickled]
#
#
#
# json_docs =[]
# with open('model_weights/attention_{}_{}_{}.json'.format(aggregation, dataset, test_name),'w') as output_file:
#
#     for paragraph in tqdm.tqdm(test_pickled, total=len(test_pickled)):
#         paragraph_id = paragraph[0][0]
#
#         (padded_batch, sentence_true_length_tensor,_), (_,y_paragraph_tensor) = reformat_and_pad_batch([paragraph])
#
#         word_attention_tensor, sentence_attention_tensor = experiment.model.model._calculate_hierarchical_attention_for_higher_level(padded_batch.cuda(), sentence_true_length_tensor.cuda())
#
#         prediction = experiment.model.model(padded_batch.cuda(), sentence_true_length_tensor.cuda(), torch.IntTensor([padded_batch.size()[0]]).cuda())[-1].squeeze(0).cpu().detach().numpy()
#         word_attention = word_attention_tensor.cpu().detach().numpy()
#         sentence_attention = sentence_attention_tensor.squeeze(0).cpu().detach().numpy()
#
#         for sentence_index, ((sentence_data, sentence_y), attention_vector) in enumerate(zip(paragraph[0],sentence_attention)):
#             doc = nlp(sentence_data[3])
#             paragraph_id = "{}-{}".format(sentence_data[0],sentence_data[1])
#             sentence_id = "{}-{}-{}".format(sentence_data[0], sentence_data[1],sentence_data[2])
#             tokenized_text = [token.text for token in doc]
#             tokenized_text.insert(0, "{}".format(sentence_y))
#             tokenized_text.insert(0, "{:.2f}".format(attention_vector))
#             tokenized_text.insert(0, sentence_id)
#             attention = np.ravel(word_attention[sentence_index][:len(tokenized_text)]).tolist()
#             attention.insert(0, float(0))
#             attention.insert(0, float(0))
#             attention.insert(0, float(0))
#
#             json_doc = {
#                 # 'paragraph_id': paragraph_id,
#                 'id':sentence_id,
#                 'text':tokenized_text,
#                 # 'sentence_attention':float(attention_vector),
#                 'label':int(y_paragraph_tensor.numpy()),
#                 'prediction':int(np.argmax(prediction)),
#                 'posterior':prediction.tolist(),
#                 'attention':attention
#
#             }
#             json_docs.append(json_doc)
#     json.dump(json_docs, output_file,indent=4)
#
#
