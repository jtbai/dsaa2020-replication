#This scipt compares prediction on a "full dataset" on each model saved in model_output
from torch.utils.data import DataLoader
import pickle

from infrastructure.batch_management import *
from infrastructure.dataset import TupleInListDataset
from infrastructure.get_experiments import *
from infrastructure.experiments_scores import *

test_binary_pickled = pickle.load(open("./datasets/2straight/preprocessed_test.pkl", 'rb'))
test_3mc2bin_pickled = pickle.load(open("./datasets/3mc2bin_straight/preprocessed_test.pkl", 'rb'))

test_binary_dataset = TupleInListDataset(test_binary_pickled)
test_mcbin_dataset = TupleInListDataset(test_3mc2bin_pickled)

test_loader_1lvl = DataLoader(test_binary_dataset,batch_size=32, shuffle=False, collate_fn=reformat_and_pad_batch_1lvl)
test_loader_2lvl = DataLoader(test_binary_dataset, batch_size=32, shuffle=False, collate_fn=reformat_and_pad_batch_2lvl)
test_loader_mcbin_2lvl = DataLoader(test_mcbin_dataset,batch_size=32,shuffle=False, collate_fn=reformat_and_pad_batch_for_mc_and_bin_2lvl)

import os

output_path = 'model_output'
model_output_folders = next(os.walk(output_path))[1]

def analyse_folder_name(folder_name):
    split_name = folder_name.split("_")
    model_name = split_name[1].split("-")[1]
    experiment_name = split_name[2].split("-")[1]
    seed = split_name[3].split("-")[1]

    return model_name, experiment_name, seed

def get_experiment(model_name, experiment_name, folder):
    if model_name == "han":
        experiment = get_baseline_experiment(folder)
        dataloader =test_loader_1lvl
    elif model_name == "mlhan":
        # xlearn can only be done with mlhan
        if experiment_name == "xlearn":
            experiment = get_proposed_hmc_experiment(folder)
            dataloader = test_loader_mcbin_2lvl
        else:
            experiment = get_proposed_experiment(folder)
            dataloader =test_loader_2lvl

    return experiment, dataloader

for model_folder in model_output_folders:

    model_name, experiment_name,seed  = analyse_folder_name(model_folder)
    print("\n ~~ model: {} - experiment: {} - seed: {}".format(model_name, experiment_name,seed))
    try:
        experiment_path = os.path.join(output_path, model_folder)
        experiment, dataloader = get_experiment(model_name, experiment_name, experiment_path)
        print_results_by_experiment([experiment.model.evaluate_generator(dataloader)])
    except Exception as e:
        print("Error with this run, skipping.", e.args)

