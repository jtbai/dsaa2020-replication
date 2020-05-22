import numpy as np

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