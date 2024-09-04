import torch
from sklearn.metrics import f1_score, recall_score, precision_score, hamming_loss, zero_one_loss, precision_score, roc_auc_score, accuracy_score, balanced_accuracy_score,fbeta_score
from sklearn.preprocessing import MultiLabelBinarizer


def print_eval_metrics(true_prob_l, pred_l, prt=True):
    pred_l = [x>0.5 for x in pred_l]
    #pred_l = torch.argmax(pred_prob_l, dim=1)
    # print('pred:',pred_l)
    # print("true:",true_prob_l)
    recall = recall_score(true_prob_l, pred_l)
    precision = precision_score(true_prob_l, pred_l)
    f1 = f1_score(true_prob_l, pred_l, average='binary')
    beta=2
    f2 = fbeta_score(true_prob_l, pred_l, average='binary', beta=beta)
    bac = balanced_accuracy_score(true_prob_l, pred_l)
    acc = accuracy_score(true_prob_l, pred_l)
    hloss = hamming_loss(true_prob_l, pred_l)
    if prt:
        print("Rec : {:.4f}".format(recall))
        print("Precision : {:.4f}".format(precision))
        print("F1 : {:.4f}".format(f1))
        print("F2 : {:.4f}".format(f2))
        print("BAC : {:.4f}".format(bac))
        print("Acc : {:.4f}".format(acc))
        print("Loss: {:.4f}".format(hloss))
    return hloss, recall, precision ,f1, f2, bac, acc
