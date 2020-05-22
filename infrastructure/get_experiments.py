from poutyne.framework import Experiment
from poutyne.framework.metrics import FBeta
from model import HAN, MLHAN_Binary, MLHAN_MultiLabel
from model.loss import MultiLevelLoss, MultiLevelMultiLabelLoss
from model.metrics import FBetaUpperLevel, FBetaLowerLevel
from model.metrics import FBetaUpperLevelMultiLabel, FBetaLowerLevelMultiLabel


paragraph_hidden_state_size = 20
sentence_hidden_state_size = 10
nb_layers = 1




def get_baseline_experiment(experiment_name):
    generability_baseline_model = HAN(20, 10, 300, 2, nb_layers,.25).eval()
    generability_baseline_experiment = Experiment(experiment_name, generability_baseline_model,monitor_metric="val_fscore_macro", monitor_mode="max",loss_function='cross_entropy', task="classification", epoch_metrics=[FBeta(average='macro')], device=0)
    generability_baseline_experiment.load_checkpoint('best')

    return generability_baseline_experiment


def get_proposed_experiment(experiment_name):
    generability_proposed_model = MLHAN_Binary(20, 10, 300, 2, nb_layers,.25).eval()
    generability_proposed_experiment = Experiment(experiment_name, generability_proposed_model,monitor_metric="val_fscore_macro", monitor_mode="max", loss_function=MultiLevelLoss(),epoch_metrics=[FBetaLowerLevel(average='macro'), FBetaUpperLevel(average='macro')], device=0)
    generability_proposed_experiment.load_checkpoint('best')

    return generability_proposed_experiment


def get_proposed_hmc_experiment(experiment_name):
    generability_proposed_model = MLHAN_MultiLabel(20, 10, 300, [3,2], nb_layers, .25).eval()
    generability_proposed_experiment = Experiment(experiment_name, generability_proposed_model,monitor_metric="val_fscore_macro", monitor_mode="max", loss_function=MultiLevelMultiLabelLoss(),epoch_metrics=[FBetaLowerLevelMultiLabel(average='macro'), FBetaUpperLevelMultiLabel(average='macro')], device=0)
    generability_proposed_experiment.load_checkpoint('best')

    return generability_proposed_experiment