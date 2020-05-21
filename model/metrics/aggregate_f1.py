

def flatten_batched_sentence_predictions(prediction_array):
    predictions = []
    for batch in prediction_array:
        for datapoint in batch:
            predictions.append(np.argmax(datapoint))

    return predictions

def flatten_batch_sentence_ground_truth(ground_truth_array):
    ground_truth = []
    for batch in ground_truth_array:
        for datapoint in batch:
            ground_truth.append(datapoint)

    return ground_truth

def aggregate_sentence_verdict_by_paragraph(paragraph_ids, verdict_array):
    verdict_by_paragraph = {}
    for paragraph_id, verdict in zip(paragraph_ids, verdict_array):
        if paragraph_id not in verdict_by_paragraph:
            verdict_by_paragraph[paragraph_id] = 0
        verdict_by_paragraph[paragraph_id] |= verdict

    return verdict_by_paragraph

from sklearn.metrics import f1_score

def get_f1_for_paragraph_by_sentences(paragraph_ids, model_results, ground_truth ):
    flat_results = flatten_batched_sentence_predictions(model_results)
    flat_ground_truth = flatten_batch_sentence_ground_truth(ground_truth)

    truth_by_paragraph = aggregate_sentence_verdict_by_paragraph(paragraph_ids, flat_ground_truth)
    predict_by_paragraph = aggregate_sentence_verdict_by_paragraph(paragraph_ids, flat_results)

    predictions = []
    truths = []
    for key, prediction in predict_by_paragraph.items():
        truths.append(truth_by_paragraph[key])
        predictions.append(prediction)

    return f1_score(truths, predictions, average="macro")
