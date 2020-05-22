from poutyne.framework.metrics import FBeta
from torch.nn.functional import cross_entropy, relu_, binary_cross_entropy
from torch.nn import Module, Linear
from torch.nn.init import uniform_
from torch import cat


class MultiLevelLoss(Module):

    def __init__(self):
        super().__init__()
        self.fc_loss_attention = Linear(2, 1).cuda()
        uniform_(self.fc_loss_attention.weight.data)

    def forward(self, y_pred, y_true):

        # l1 = two_level_loss(y_pred, y_true)
        l2 = upper_level_cross_entropy(y_pred, y_true)
        l3 = lower_level_cross_entropy( y_pred, y_true)

        loss = cat((l2.unsqueeze(-1),l3.unsqueeze(-1)))

        combined_loss = relu_(self.fc_loss_attention(loss))
        return combined_loss


class MultiLevelMultiLabelLoss(Module):
    def __init__(self):
        super().__init__()
        self.fc_loss_attention_1 = Linear(2, 1).cuda()
        self.fc_loss_attention_2 = Linear(2, 1).cuda()
        self.fc_loss_attention_12 = Linear(2, 1).cuda()
        uniform_(self.fc_loss_attention_1.weight.data)
        uniform_(self.fc_loss_attention_2.weight.data)
        uniform_(self.fc_loss_attention_12.weight.data)

    def forward(self, y_pred, y_true):
        sentence_mc, sentence_bin = y_pred[0]
        paragraph_mc, paragraph_bin = y_pred[1]

        sentence_true_mc, sentence_true_bin = y_true[0]
        paragraph_true_mc, paragraph_true_bin = y_true[1]

        loss_sentence_bin = cross_entropy(sentence_bin, sentence_true_bin)
        loss_paragraph_bin = cross_entropy(paragraph_bin, paragraph_true_bin)

        loss_sentence_mc = binary_cross_entropy(sentence_mc, sentence_true_mc)
        loss_paragraph_mc = binary_cross_entropy(paragraph_mc, paragraph_true_mc)

        loss_1 = cat((loss_sentence_bin.unsqueeze(-1), loss_sentence_mc.unsqueeze(-1)))
        loss_2 = cat((loss_paragraph_bin.unsqueeze(-1), loss_paragraph_mc.unsqueeze(-1)))

        loss_12 = cat((self.fc_loss_attention_1(loss_1),self.fc_loss_attention_2(loss_2))).squeeze(-1)

        combined_loss = relu_(self.fc_loss_attention_12(loss_12))
        return combined_loss



class LevelSwitchingLoss(Module):
    BINARY = 0
    MULTILABEL = 1
    MIXED = 2

    def __init__(self, classification_type, initial_state=0):
        super().__init__()
        self.lower_classification_type = classification_type[0]
        self.upper_classification_type = classification_type[1]
        self.current_state = initial_state
        self.target_loss = self.state_0
        if initial_state == 1:
            self.target_loss = self.state_1

    def switch_state(self):
        if self.current_state == 0:
            self.current_state = 1
            self.target_loss = self.state_1
        elif self.current_state == 1:
            self.current_state = 0
            self.target_loss = self.state_0

    def forward(self, y_pred, y_true):
        return self.target_loss(y_pred, y_true)

    def state_0(self, y_pred, y_true):
        sentence_mc, sentence_bin = y_pred[0]
        sentence_true_mc, sentence_true_bin = y_true[0]
        if self.lower_classification_type == self.MULTILABEL:
            level_0_loss = binary_cross_entropy(sentence_mc, sentence_true_mc)
        elif self.lower_classification_type == self.BINARY:
            level_0_loss = cross_entropy(sentence_bin, sentence_true_bin)

        return level_0_loss

    def state_1(self, y_pred, y_true):
        paragraph_mc, paragraph_bin = y_pred[1]
        paragraph_true_mc, paragraph_true_bin = y_true[1]
        if self.upper_classification_type == self.MULTILABEL:
            level_1_loss = binary_cross_entropy(paragraph_mc, paragraph_true_mc)
        elif self.upper_classification_type == self.BINARY:
            level_1_loss = cross_entropy(paragraph_bin, paragraph_true_bin)

        return level_1_loss



def upper_level_multilabel_cross_entropy(pred_y, y):
    level_1_pred, level_2_pred = pred_y
    level_1_y , level_2_y = y
    level_2_loss = binary_cross_entropy(level_2_pred, level_2_y)

    return level_2_loss


def lower_level_multilabel_cross_entropy(pred_y, y):
    level_1_pred, level_2_pred = pred_y
    level_1_y, level_2_y = y
    level_1_loss = binary_cross_entropy(level_1_pred, level_1_y)

    return level_1_loss

def upper_level_cross_entropy(pred_y, y):
    level_1_pred, level_2_pred = pred_y
    level_1_y , level_2_y = y
    level_2_loss = cross_entropy(level_2_pred, level_2_y)

    return level_2_loss

def lower_level_cross_entropy(pred_y, y):
    level_1_pred, level_2_pred = pred_y
    level_1_y , level_2_y = y
    level_1_loss = cross_entropy(level_1_pred, level_1_y)

    return level_1_loss

def two_level_loss(pred_y, y):
    level_1_pred, level_2_pred = pred_y
    level_1_y,level_2_y = y
    level_1_loss = cross_entropy(level_1_pred, level_1_y)
    level_2_loss = cross_entropy(level_2_pred, level_2_y)

    return level_1_loss + level_2_loss
