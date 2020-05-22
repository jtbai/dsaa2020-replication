#This scipt compares prediction on a "full dataset" without any "class grooming". It includes a lot of useless sentenses
from torch.utils.data import DataLoader
from scipy.stats import ttest_ind as ttest
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


baseline_experiment_run1 = get_baseline_experiment("official_weights/baseline_sametask/baseline_hc_run_420")
baseline_experiment_run2 = get_baseline_experiment("official_weights/baseline_sametask/baseline_hc_run_1337")
baseline_experiment_run3 = get_baseline_experiment("official_weights/baseline_sametask/baseline_hc_run_101")
proposed_experiment_run1 = get_proposed_experiment("official_weights/proposed_sametask/proposed_hc_run_420")
proposed_experiment_run2 = get_proposed_experiment("official_weights/proposed_sametask/proposed_hc_run_1337")
proposed_experiment_run3 = get_proposed_experiment("official_weights/proposed_sametask/proposed_hc_run_26")

baseline_generability_10pct_run1 = get_baseline_experiment("official_weights/generability_10pct/baseline_by_paragraph_2tfpositive_10pct_seed26")
baseline_generability_10pct_run2 = get_baseline_experiment("official_weights/generability_10pct/baseline_by_paragraph_2tfpositive_10pct_seed420")
baseline_generability_10pct_run3 = get_baseline_experiment("official_weights/generability_10pct/baseline_by_paragraph_2tfpositive_10pct_seed1337")
proposed_generability_10pct_run1 = get_proposed_experiment("official_weights/generability_10pct/proposed_by_paragraph_2tfpositive_10pct_seed26")
proposed_generability_10pct_run2 = get_proposed_experiment("official_weights/generability_10pct/proposed_by_paragraph_2tfpositive_10pct_seed420")
proposed_generability_10pct_run3 = get_proposed_experiment("official_weights/generability_10pct/proposed_by_paragraph_2tfpositive_10pct_seed1337")
proposed_generability_10pct_run4 = get_proposed_experiment("official_weights/generability_10pct/proposed_by_paragraph_2tfpositive_10pct_seed1234")
proposed_generability_10pct_run5 = get_proposed_experiment("official_weights/generability_10pct/proposed_by_paragraph_2tfpositive_10pct_seed7777")
proposed_generability_10pct_run6 = get_proposed_experiment("official_weights/generability_10pct/proposed_by_paragraph_2tfpositive_10pct_seed749")

baseline_generability_15pct_run1 = get_baseline_experiment("official_weights/generability_15pct/generability_baseline")
baseline_generability_15pct_run2 = get_baseline_experiment("official_weights/generability_15pct/generability_baseline_run2")
baseline_generability_15pct_run3 = get_baseline_experiment("official_weights/generability_15pct/generability_baseline_run3")
proposed_generability_15pct_run1 = get_proposed_experiment("official_weights/generability_15pct/generability_proposed")
proposed_generability_15pct_run2 = get_proposed_experiment("official_weights/generability_15pct/generability_proposed_run2")
proposed_generability_15pct_run3 = get_proposed_experiment("official_weights/generability_15pct/generability_proposed_run3")

baseline_generability_30pct_run1 = get_baseline_experiment("official_weights/generability_30pct/baseline_by_paragraph_2tfpositive_30pct_seed26")
baseline_generability_30pct_run2 = get_baseline_experiment("official_weights/generability_30pct/baseline_by_paragraph_2tfpositive_30pct_seed420")
baseline_generability_30pct_run3 = get_baseline_experiment("official_weights/generability_30pct/baseline_by_paragraph_2tfpositive_30pct_seed1337")
proposed_generability_30pct_run1 = get_proposed_experiment("official_weights/generability_30pct/proposed_by_paragraph_2tfpositive_30pct_seed26")
proposed_generability_30pct_run2 = get_proposed_experiment("official_weights/generability_30pct/proposed_by_paragraph_2tfpositive_30pct_seed420")
proposed_generability_30pct_run3 = get_proposed_experiment("official_weights/generability_30pct/proposed_by_paragraph_2tfpositive_30pct_seed1337")

paper_proposed_hmc_experiment_run1 = get_proposed_hmc_experiment("official_weights/proposed_multilabel/proposed_hmc_run_420")
paper_proposed_hmc_experiment_run2 = get_proposed_hmc_experiment("official_weights/proposed_multilabel/proposed_hmc_run_26")
paper_proposed_hmc_experiment_run3 = get_proposed_hmc_experiment("official_weights/proposed_multilabel/proposed_hmc_run_666")

xferlearning_sametask_experiment_run1 = get_proposed_hmc_experiment("official_weights/xferlearning/baseline_seed26/top_model")
xferlearning_sametask_experiment_run2 = get_proposed_hmc_experiment("official_weights/xferlearning/baseline_seed420/top_model")
xferlearning_sametask_experiment_run3 = get_proposed_hmc_experiment("official_weights/xferlearning/baseline_seed1337/top_model")



print("\n\n")
print("~~~ PAPER RESULTS - TABLE VI ~~~")

print("\n")
print("Same Task - Baseline Model")
results = []
results.append(baseline_experiment_run1.model.evaluate_generator(test_loader_1lvl))
results.append(baseline_experiment_run2.model.evaluate_generator(test_loader_1lvl))
results.append(baseline_experiment_run3.model.evaluate_generator(test_loader_1lvl))
print_results_by_experiment(results)
f1_baseline = get_upper_f1_scores_by_experiments(results)
print("\n")
print("Same Task - Proposed Model")
results = []
results.append(proposed_experiment_run1.model.evaluate_generator(test_loader_2lvl))
results.append(proposed_experiment_run2.model.evaluate_generator(test_loader_2lvl))
results.append(proposed_experiment_run3.model.evaluate_generator(test_loader_2lvl))
proposed_model_results = results
print_results_by_experiment(proposed_model_results)
f1_proposed = get_upper_f1_scores_by_experiments(results)
print("\n")
print("Multi Label - Proposed Model")
results = []
results.append(paper_proposed_hmc_experiment_run1.model.evaluate_generator(test_loader_mcbin_2lvl))
results.append(paper_proposed_hmc_experiment_run2.model.evaluate_generator(test_loader_mcbin_2lvl))
results.append(paper_proposed_hmc_experiment_run3.model.evaluate_generator(test_loader_mcbin_2lvl))
print_results_by_experiment(results)

print("\n\n")
print("~~~ MODEL GENERABILITY CAPACITY - TABLE VII ~~~")
print("\n")
print("Baseline Generability Experiment - 10% data")
results = []
results.append(baseline_generability_10pct_run1.model.evaluate_generator(test_loader_1lvl))
results.append(baseline_generability_10pct_run2.model.evaluate_generator(test_loader_1lvl))
results.append(baseline_generability_10pct_run3.model.evaluate_generator(test_loader_1lvl))
print_results_by_experiment(results)
print("\n")
print("Proposed Generability Experiment  - 10% data")
results = []
results.append(proposed_generability_10pct_run1.model.evaluate_generator(test_loader_2lvl))
results.append(proposed_generability_10pct_run2.model.evaluate_generator(test_loader_2lvl))
results.append(proposed_generability_10pct_run3.model.evaluate_generator(test_loader_2lvl))
results.append(proposed_generability_10pct_run4.model.evaluate_generator(test_loader_2lvl))
results.append(proposed_generability_10pct_run5.model.evaluate_generator(test_loader_2lvl))
results.append(proposed_generability_10pct_run6.model.evaluate_generator(test_loader_2lvl))
print_results_by_experiment(results)
print("\n")
print("Baseline Generability Experiment - 15% data ")
results = []
results.append(baseline_generability_15pct_run1.model.evaluate_generator(test_loader_1lvl))
results.append(baseline_generability_15pct_run2.model.evaluate_generator(test_loader_1lvl))
results.append(baseline_generability_15pct_run3.model.evaluate_generator(test_loader_1lvl))
print_results_by_experiment(results)
print("\n")
print("Proposed Generability Experiment - 15% data ")
results = []
results.append(proposed_generability_15pct_run1.model.evaluate_generator(test_loader_2lvl))
results.append(proposed_generability_15pct_run2.model.evaluate_generator(test_loader_2lvl))
results.append(proposed_generability_15pct_run3.model.evaluate_generator(test_loader_2lvl))
print_results_by_experiment(results)
f1_generability =  get_upper_f1_scores_by_experiments(results)
print("\n")
print("Baseline Generability Experiment - 30% data")
results = []
results.append(baseline_generability_30pct_run1.model.evaluate_generator(test_loader_1lvl))
results.append(baseline_generability_30pct_run2.model.evaluate_generator(test_loader_1lvl))
results.append(baseline_generability_30pct_run3.model.evaluate_generator(test_loader_1lvl))
print_results_by_experiment(results)
print("\n")
print("Proposed Generability Experiment  - 30% data")
results = []
results.append(proposed_generability_30pct_run1.model.evaluate_generator(test_loader_2lvl))
results.append(proposed_generability_30pct_run2.model.evaluate_generator(test_loader_2lvl))
results.append(proposed_generability_30pct_run3.model.evaluate_generator(test_loader_2lvl))
print_results_by_experiment(results)

print("\n\n")
print("~~~ TRANSFER LEARNING CAPACITY - TABLE VIII ~~~")

results = []
results.append(xferlearning_sametask_experiment_run1.model.evaluate_generator(test_loader_mcbin_2lvl))
results.append(xferlearning_sametask_experiment_run2.model.evaluate_generator(test_loader_mcbin_2lvl))
results.append(xferlearning_sametask_experiment_run3.model.evaluate_generator(test_loader_mcbin_2lvl))
sequential_transfer_learning_results = results
f1_xlern = get_upper_f1_scores_by_experiments(sequential_transfer_learning_results)

print("\n")
print("Sequential Transfer Learning  - Lower Level")
print_lowerlevel_results_by_experiment(sequential_transfer_learning_results)
print("\n")
print("Simultaneous Transfer Learning (Proposed Model) - Lower Level")
print_lowerlevel_results_by_experiment(proposed_model_results)

print("\n")
print("Sequential Transfer Learning - Upper Level")
print_results_by_experiment(sequential_transfer_learning_results)
print("\n")
print("Simultaneous Transfer Learning (Proposed Model) - Upper Level")
print_results_by_experiment(proposed_model_results)


print("\n\n")
print("~~~ T-Test Scores ~~~")



print("t-test | MLHAN (proposed) vs HAN (baseline) ")
ttest_baseline_vs_proposed = ttest(f1_proposed,f1_baseline)
print("t-statistic : {:.3f} (p-value: {:.6f}) \n".format(ttest_baseline_vs_proposed[0], ttest_baseline_vs_proposed[1]))

print("t-test | Simultaneous (proposed) vs Sequential (baseline) transfer learning")
ttest_baseline_vs_xlearn = ttest(f1_proposed,f1_xlern)
print("t-statistic : {:.3f} (p-value: {:.6f}) \n".format(ttest_baseline_vs_xlearn[0], ttest_baseline_vs_xlearn[1]))

print("t-test | Truncated (15%) train dataset (proposed) vs full dataset (baseline)")
ttest_generability = ttest(f1_baseline, f1_generability)
print("t-statistic : {:.3f} (p-value: {:.6f}) \n".format(ttest_generability[0], ttest_generability[1]))

