import torch
from poutyne.framework.metrics import FBeta
import numpy as np


class FBetaUpperLevel(FBeta):

    def forward(self, y_pred, y_true):

        super().forward(y_pred[-1], y_true[-1])


class FBetaLowerLevel(FBeta):

    def forward(self, y_pred, y_true):

        super().forward(y_pred[0], y_true[0])


class FBetaMultiLabelUpperLevel(FBeta):

    def forward(self, y_pred, y_true):
        y_pred = y_pred[-1]
        y_true = y_true[-1]
        """
        Update the confusion matrix for calculating the F-score.

        Args:
            y_pred : Predictions of the model.
            y_true : A tensor of the gold labels. Can also be a tuple of gold_label and a mask.
        Args:
            y_pred (torch.Tensor): A tensor of predictions of shape (batch_size, ..., num_classes).
            y_true Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                Ground truths. A tensor of the integer class label of shape (batch_size, ...). It must
                be the same shape as the ``y_pred`` tensor without the ``num_classes`` dimension.
                It can also be a tuple with two tensors of the same shape, the first being the
                ground truths and the second being a mask.
        """

        mask = None
        if isinstance(y_true, tuple):
            y_true, mask = y_true

        # Calculate true_positive_sum, true_negative_sum, pred_sum, true_sum
        num_classes = y_pred.size(1)
        if (y_true >= num_classes).any():
            raise ValueError("A gold label passed to FBetaMeasure contains "
                             "an id >= {}, the number of classes.".format(num_classes))

        # It means we call this metric at the first time
        # when `self._true_positive_sum` is None.
        if self._true_positive_sum is None:
            self._true_positive_sum = torch.zeros(num_classes, device=y_pred.device)
            self._true_sum = torch.zeros(num_classes, device=y_pred.device)
            self._pred_sum = torch.zeros(num_classes, device=y_pred.device)
            self._total_sum = torch.zeros(num_classes, device=y_pred.device)

        if mask is None:
            mask = torch.ones_like(y_true)
        mask = mask.to(dtype=torch.bool)
        y_true = y_true.float()

        ### OLD CODE HERE ####

        argmax_y_pred = y_pred > 0.5
        true_positives = (y_true == argmax_y_pred) * mask
        true_positives_bins = y_true[true_positives]

        # Watch it:
        # The total numbers of true positives under all _predicted_ classes are zeros.
        if true_positives_bins.shape[0] == 0:
            true_positive_sum = torch.zeros(num_classes, device=y_pred.device)
        else:
            true_positive_sum = torch.bincount(true_positives_bins.long(), minlength=num_classes).float()

        pred_bins = argmax_y_pred[mask].long()
        # Watch it:
        # When the `mask` is all 0, we will get an _empty_ tensor.
        if pred_bins.shape[0] != 0:
            pred_sum = torch.bincount(pred_bins, minlength=num_classes).float()
        else:
            pred_sum = torch.zeros(num_classes, device=y_pred.device)

        y_true_bins = y_true[mask].long()
        if y_true.shape[0] != 0:
            true_sum = torch.bincount(y_true_bins, minlength=num_classes).float()
        else:
            true_sum = torch.zeros(num_classes, device=y_pred.device)

        ### OLD CODE HERE ####


        self._true_positive_sum += true_positive_sum
        self._pred_sum += pred_sum
        self._true_sum += true_sum
        self._total_sum += mask.sum().to(torch.float)


class FBetaMultiClassAndBinary(FBeta):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.multilabel_F1 = FBetaMultiLabel(*args, **kwargs)

    def forward(self, y_pred, y_true):
        class_1_pred, class_2_pred = y_pred
        class_1_true, class_2_true = y_true

        # F1_multi = self.multilabel_F1(class_1_pred, class_1_true)
        super().forward(class_2_pred, class_2_true)

    def get_metric(self):
        metric_binary = super().get_metric()
        # metric_multiclass = self.multilabel_F1.get_metric()

        # return (metric_binary + metric_multiclass)/2
        return metric_binary


class FBetaMultiLabel(FBeta):

    def forward(self, y_pred, y_true):
        """
        Update the confusion matrix for calculating the F-score.

        Args:
            y_pred : Predictions of the model.
            y_true : A tensor of the gold labels. Can also be a tuple of gold_label and a mask.
        Args:
            y_pred (torch.Tensor): A tensor of predictions of shape (batch_size, ..., num_classes).
            y_true Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                Ground truths. A tensor of the integer class label of shape (batch_size, ...). It must
                be the same shape as the ``y_pred`` tensor without the ``num_classes`` dimension.
                It can also be a tuple with two tensors of the same shape, the first being the
                ground truths and the second being a mask.
        """

        mask = None
        if isinstance(y_true, tuple):
            y_true, mask = y_true

        # Calculate true_positive_sum, true_negative_sum, pred_sum, true_sum
        num_classes = y_pred.size(1)
        if (y_true >= num_classes).any():
            raise ValueError("A gold label passed to FBetaMeasure contains "
                             "an id >= {}, the number of classes.".format(num_classes))

        # It means we call this metric at the first time
        # when `self._true_positive_sum` is None.
        if self._true_positive_sum is None:
            self._true_positive_sum = torch.zeros(num_classes, device=y_pred.device)
            self._true_sum = torch.zeros(num_classes, device=y_pred.device)
            self._pred_sum = torch.zeros(num_classes, device=y_pred.device)
            self._total_sum = torch.zeros(num_classes, device=y_pred.device)

        if mask is None:
            mask = torch.ones_like(y_true)
        mask = mask.to(dtype=torch.bool)
        y_true = y_true.float()

        ### OLD CODE HERE ####

        argmax_y_pred = y_pred > 0.5
        true_positives = (y_true == argmax_y_pred) * mask
        true_positives_bins = y_true[true_positives]

        # Watch it:
        # The total numbers of true positives under all _predicted_ classes are zeros.
        if true_positives_bins.shape[0] == 0:
            true_positive_sum = torch.zeros(num_classes, device=y_pred.device)
        else:
            true_positive_sum = torch.bincount(true_positives_bins.long(), minlength=num_classes).float()

        pred_bins = argmax_y_pred[mask].long()
        # Watch it:
        # When the `mask` is all 0, we will get an _empty_ tensor.
        if pred_bins.shape[0] != 0:
            pred_sum = torch.bincount(pred_bins, minlength=num_classes).float()
        else:
            pred_sum = torch.zeros(num_classes, device=y_pred.device)

        y_true_bins = y_true[mask].long()
        if y_true.shape[0] != 0:
            true_sum = torch.bincount(y_true_bins, minlength=num_classes).float()
        else:
            true_sum = torch.zeros(num_classes, device=y_pred.device)

        ### OLD CODE HERE ####


        self._true_positive_sum += true_positive_sum
        self._pred_sum += pred_sum
        self._true_sum += true_sum
        self._total_sum += mask.sum().to(torch.float)


class UpperLevelAccuracy:

    def __init__(self):
        self.metric = upper_level_accuracy
        self.nb_evaluated = 0
        self.nb_correct = 0

    def get_metric(self):
        the_metric = self.nb_correct/self.nb_evaluated
        self.nb_correct = 0
        self.nb_evaluated = 0
        return the_metric

    def __call__(self,pred_y, y):
        pred_y_numpy = pred_y[-1].cpu().detach().numpy()
        pred_y_numpy_bool = np.argmax(pred_y_numpy, axis=1)
        y_numpy = y[-1].cpu().detach().numpy()
        self.nb_evaluated += len(pred_y_numpy_bool)
        self.nb_correct += sum(pred_y_numpy_bool==y_numpy)


class FBetaUpperLevelMultiLabel(FBeta):

    def forward(self,pred_y, y):
        paragraph_mc, paragraph_bin = pred_y[1]
        paragraph_true_mc, paragraph_true_bin = y[1]

        super().forward(paragraph_bin,paragraph_true_bin)


class FBetaLowerLevelMultiLabel(FBeta):

    def forward(self,pred_y, y):
        sentence_mc, sentence_bin = pred_y[0]
        sentence_true_mc, sentence_true_bin = y[0]

        super().forward(sentence_bin,sentence_true_bin)

def upper_level_accuracy(y_pred, y_true, ignore_index=-100):
    y_pred = y_pred[-1].argmax(1)
    weights = (y_true[-1] != ignore_index).float()
    num_labels = weights.sum()
    acc_pred = ((y_pred[-1] == y_true[-1]).float() * weights).sum() / num_labels

    return acc_pred * 100

