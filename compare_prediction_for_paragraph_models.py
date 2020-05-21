#This scipt compares prediction on a "full dataset" without any "class grooming". It includes a lot of useless sentenses
from torch.utils.data import DataLoader
from scipy.stats import ttest_ind as ttest
import pickle
from poutyne.framework import Experiment
from poutyne.framework.metrics import FBeta
from infrastructure.batch_management import *
from infrastructure.dataset import TupleInListDataset
from model import HAN, MLHAN_Binary, MLHAN_MultiLabel
from model.loss import MultiLevelLoss, MultiLevelMultiLabelLoss
from model.metrics import FBetaUpperLevel, FBetaLowerLevel
from model.metrics import FBetaUpperLevelMultiLabel, FBetaLowerLevelMultiLabel

test_binary_pickled = pickle.load(open("./datasets/2straight/preprocessed_test.pkl", 'rb'))
test_3mc2bin_pickled = pickle.load(open("./datasets/3mc2bin_straight/preprocessed_test.pkl", 'rb'))

test_binary_dataset = TupleInListDataset(test_binary_pickled)
test_mcbin_dataset = TupleInListDataset(test_3mc2bin_pickled)

test_loader_1lvl = DataLoader(test_binary_dataset,batch_size=32, shuffle=False, collate_fn=reformat_and_pad_batch_1lvl)
test_loader_2lvl = DataLoader(test_binary_dataset, batch_size=32, shuffle=False, collate_fn=reformat_and_pad_batch_2lvl)
test_loader_mcbin_2lvl = DataLoader(test_mcbin_dataset,batch_size=32,shuffle=False, collate_fn=reformat_and_pad_batch_for_mc_and_bin_2lvl)

paragraph_hidden_state_size = 20
sentence_hidden_state_size = 10
nb_layers = 1


def get_baseline_experiment(experiment_name):
    generability_baseline_model = HAN(20, 10, 300, 2, nb_layers,.25).eval()
    generability_baseline_experiment = Experiment(experiment_name, generability_baseline_model,monitor_metric="val_fscore_macro", monitor_mode="max",loss_function='cross_entropy', task="classification", epoch_metrics=[FBeta(average='macro')], device=0)
    generability_baseline_experiment.load_checkpoint('best')

    return generability_baseline_experiment


def get_proposed_experiment(experiment_name):
    generability_proposed_model = MLHAN_Binary(20, 10, 300, 2, nb_layers,.50).eval()
    generability_proposed_experiment = Experiment(experiment_name, generability_proposed_model,monitor_metric="val_fscore_macro", monitor_mode="max", loss_function=MultiLevelLoss(),epoch_metrics=[FBetaLowerLevel(average='macro'), FBetaUpperLevel(average='macro')], device=0)
    generability_proposed_experiment.load_checkpoint('best')

    return generability_proposed_experiment


def get_proposed_hmc_experiment(experiment_name):
    generability_proposed_model = MLHAN_MultiLabel(20, 10, 300, [3,2], 2,.50).eval()
    generability_proposed_experiment = Experiment(experiment_name, generability_proposed_model,monitor_metric="val_fscore_macro", monitor_mode="max", loss_function=MultiLevelMultiLabelLoss(),epoch_metrics=[FBetaLowerLevelMultiLabel(average='macro'), FBetaUpperLevelMultiLabel(average='macro')], device=0)
    generability_proposed_experiment.load_checkpoint('best')

    return generability_proposed_experiment

baseline_experiment_run1 = get_baseline_experiment("model_weights/baseline_sametask/baseline_hc_run_420")
baseline_experiment_run2 = get_baseline_experiment("model_weights/baseline_sametask/baseline_hc_run_1337")
baseline_experiment_run3 = get_baseline_experiment("model_weights/baseline_sametask/baseline_hc_run_101")
proposed_experiment_run1 = get_proposed_experiment("model_weights/proposed_sametask/proposed_hc_run_420")
proposed_experiment_run2 = get_proposed_experiment("model_weights/proposed_sametask/proposed_hc_run_1337")
proposed_experiment_run3 = get_proposed_experiment("model_weights/proposed_sametask/proposed_hc_run_26")

baseline_generability_10pct_run1 = get_baseline_experiment("model_weights/generability_10pct/baseline_by_paragraph_2tfpositive_10pct_seed26")
baseline_generability_10pct_run2 = get_baseline_experiment("model_weights/generability_10pct/baseline_by_paragraph_2tfpositive_10pct_seed420")
baseline_generability_10pct_run3 = get_baseline_experiment("model_weights/generability_10pct/baseline_by_paragraph_2tfpositive_10pct_seed1337")
proposed_generability_10pct_run1 = get_proposed_experiment("model_weights/generability_10pct/proposed_by_paragraph_2tfpositive_10pct_seed26")
proposed_generability_10pct_run2 = get_proposed_experiment("model_weights/generability_10pct/proposed_by_paragraph_2tfpositive_10pct_seed420")
proposed_generability_10pct_run3 = get_proposed_experiment("model_weights/generability_10pct/proposed_by_paragraph_2tfpositive_10pct_seed1337")
proposed_generability_10pct_run4 = get_proposed_experiment("model_weights/generability_10pct/proposed_by_paragraph_2tfpositive_10pct_seed1234")
proposed_generability_10pct_run5 = get_proposed_experiment("model_weights/generability_10pct/proposed_by_paragraph_2tfpositive_10pct_seed7777")
proposed_generability_10pct_run6 = get_proposed_experiment("model_weights/generability_10pct/proposed_by_paragraph_2tfpositive_10pct_seed749")

baseline_generability_15pct_run1 = get_baseline_experiment("model_weights/generability_15pct/generability_baseline")
baseline_generability_15pct_run2 = get_baseline_experiment("model_weights/generability_15pct/generability_baseline_run2")
baseline_generability_15pct_run3 = get_baseline_experiment("model_weights/generability_15pct/generability_baseline_run3")
proposed_generability_15pct_run1 = get_proposed_experiment("model_weights/generability_15pct/generability_proposed")
proposed_generability_15pct_run2 = get_proposed_experiment("model_weights/generability_15pct/generability_proposed_run2")
proposed_generability_15pct_run3 = get_proposed_experiment("model_weights/generability_15pct/generability_proposed_run3")

baseline_generability_30pct_run1 = get_baseline_experiment("model_weights/generability_30pct/baseline_by_paragraph_2tfpositive_30pct_seed26")
baseline_generability_30pct_run2 = get_baseline_experiment("model_weights/generability_30pct/baseline_by_paragraph_2tfpositive_30pct_seed420")
baseline_generability_30pct_run3 = get_baseline_experiment("model_weights/generability_30pct/baseline_by_paragraph_2tfpositive_30pct_seed1337")
proposed_generability_30pct_run1 = get_proposed_experiment("model_weights/generability_30pct/proposed_by_paragraph_2tfpositive_30pct_seed26")
proposed_generability_30pct_run2 = get_proposed_experiment("model_weights/generability_30pct/proposed_by_paragraph_2tfpositive_30pct_seed420")
proposed_generability_30pct_run3 = get_proposed_experiment("model_weights/generability_30pct/proposed_by_paragraph_2tfpositive_30pct_seed1337")

paper_proposed_hmc_experiment_run1 = get_proposed_hmc_experiment("model_weights/proposed_multilabel/proposed_hmc_run_420")
paper_proposed_hmc_experiment_run2 = get_proposed_hmc_experiment("model_weights/proposed_multilabel/proposed_hmc_run_26")
paper_proposed_hmc_experiment_run3 = get_proposed_hmc_experiment("model_weights/proposed_multilabel/proposed_hmc_run_666")

xferlearning_sametask_experiment_run1 = get_proposed_hmc_experiment("model_weights/xferlearning/baseline_seed26/top_model")
xferlearning_sametask_experiment_run2 = get_proposed_hmc_experiment("model_weights/xferlearning/baseline_seed420/top_model")
xferlearning_sametask_experiment_run3 = get_proposed_hmc_experiment("model_weights/xferlearning/baseline_seed1337/top_model")


def get_upper_f1_scores_by_experiments(results_array):
    f_scores = np.zeros(len(results_array), dtype=float)
    for run_index, result in enumerate(results_array):
        current_f_score = result[1][1]
        f_scores[run_index] = current_f_score

    return f_scores


def get_lower_f1_scores_by_experiments(results_array):
    f_scores = np.zeros(len(results_array), dtype=float)
    for run_index, result in enumerate(results_array):
        current_f_score = result[1][0]
        f_scores[run_index] = current_f_score

    return f_scores


def print_lowerlevel_results_by_experiment(results_array):
    f_scores = get_lower_f1_scores_by_experiments(results_array)
    for run_index, result in enumerate(f_scores):
        print("Run {}: {}".format(run_index, result))

    print("Average: {:.4f}, Stdev: {:.4f}".format(np.mean(f_scores), np.std(f_scores)))


def print_results_by_experiment(results_array):
    f_scores = get_upper_f1_scores_by_experiments(results_array)
    for run_index, result in enumerate(f_scores):
        print("Run {}: {}".format(run_index, result))

    print("Average: {:.4f}, Stdev: {:.4f}".format(np.mean(f_scores), np.std(f_scores)))


print("~~~ PAPER RESULTS ~~~")

print("Baseline Model")
results = []
results.append(baseline_experiment_run1.model.evaluate_generator(test_loader_1lvl))
results.append(baseline_experiment_run2.model.evaluate_generator(test_loader_1lvl))
results.append(baseline_experiment_run3.model.evaluate_generator(test_loader_1lvl))
print_results_by_experiment(results)
f1_baseline = get_upper_f1_scores_by_experiments(results)

print("Proposed Model")
results = []
results.append(proposed_experiment_run1.model.evaluate_generator(test_loader_2lvl))
results.append(proposed_experiment_run2.model.evaluate_generator(test_loader_2lvl))
results.append(proposed_experiment_run3.model.evaluate_generator(test_loader_2lvl))
print_results_by_experiment(results)
print(" - Lower Level")
print_lowerlevel_results_by_experiment(results)
f1_proposed = get_upper_f1_scores_by_experiments(results)

print("~~~ XFERLEARNING COMPARISON ~~~")

print("XferLearning Same Task Experiment")
results = []
results.append(xferlearning_sametask_experiment_run1.model.evaluate_generator(test_loader_mcbin_2lvl))
results.append(xferlearning_sametask_experiment_run2.model.evaluate_generator(test_loader_mcbin_2lvl))
results.append(xferlearning_sametask_experiment_run3.model.evaluate_generator(test_loader_mcbin_2lvl))
print_results_by_experiment(results)
print(" - Lower Level")
print_lowerlevel_results_by_experiment(results)
f1_xlern = get_upper_f1_scores_by_experiments(results)

print("~~~ MULTI LABEL EXPERIMENT ~~~")

print("HMC Experiment")
results = []
results.append(paper_proposed_hmc_experiment_run1.model.evaluate_generator(test_loader_mcbin_2lvl))
results.append(paper_proposed_hmc_experiment_run2.model.evaluate_generator(test_loader_mcbin_2lvl))
results.append(paper_proposed_hmc_experiment_run3.model.evaluate_generator(test_loader_mcbin_2lvl))
print_results_by_experiment(results)

print("~~~ MODEL GENERABILITY COMPARISON ~~~")

print("Baseline Generability Experiment - 10pct data")
results = []
results.append(baseline_generability_10pct_run1.model.evaluate_generator(test_loader_1lvl))
results.append(baseline_generability_10pct_run2.model.evaluate_generator(test_loader_1lvl))
results.append(baseline_generability_10pct_run3.model.evaluate_generator(test_loader_1lvl))
print_results_by_experiment(results)

print("Proposed Generability Experiment  - 10pct data")
results = []
results.append(proposed_generability_10pct_run1.model.evaluate_generator(test_loader_2lvl))
results.append(proposed_generability_10pct_run2.model.evaluate_generator(test_loader_2lvl))
results.append(proposed_generability_10pct_run3.model.evaluate_generator(test_loader_2lvl))
results.append(proposed_generability_10pct_run4.model.evaluate_generator(test_loader_2lvl))
results.append(proposed_generability_10pct_run5.model.evaluate_generator(test_loader_2lvl))
results.append(proposed_generability_10pct_run6.model.evaluate_generator(test_loader_2lvl))
print_results_by_experiment(results)

print("Baseline Generability Experiment - 15pct data ")
results = []
results.append(baseline_generability_15pct_run1.model.evaluate_generator(test_loader_1lvl))
results.append(baseline_generability_15pct_run2.model.evaluate_generator(test_loader_1lvl))
results.append(baseline_generability_15pct_run3.model.evaluate_generator(test_loader_1lvl))
print_results_by_experiment(results)

print("Proposed Generability Experiment - 15pct data ")
results = []
results.append(proposed_generability_15pct_run1.model.evaluate_generator(test_loader_2lvl))
results.append(proposed_generability_15pct_run2.model.evaluate_generator(test_loader_2lvl))
results.append(proposed_generability_15pct_run3.model.evaluate_generator(test_loader_2lvl))
print_results_by_experiment(results)
f1_generability =  get_upper_f1_scores_by_experiments(results)

print("Baseline Generability Experiment - 30pct data")
results = []
results.append(baseline_generability_30pct_run1.model.evaluate_generator(test_loader_1lvl))
results.append(baseline_generability_30pct_run2.model.evaluate_generator(test_loader_1lvl))
results.append(baseline_generability_30pct_run3.model.evaluate_generator(test_loader_1lvl))
print_results_by_experiment(results)

print("Proposed Generability Experiment  - 30pct data")
results = []
results.append(proposed_generability_30pct_run1.model.evaluate_generator(test_loader_2lvl))
results.append(proposed_generability_30pct_run2.model.evaluate_generator(test_loader_2lvl))
results.append(proposed_generability_30pct_run3.model.evaluate_generator(test_loader_2lvl))
print_results_by_experiment(results)

print("t-test | baseline vs proposed")
print(ttest(f1_proposed,f1_baseline))
print("t-test | xlern vs proposed")
print(ttest(f1_proposed,f1_xlern))
print("t-test | generability_15 vs baseline")
print(ttest(f1_baseline, f1_generability))


