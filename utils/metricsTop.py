import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score

__all__ = ['MetricsTop']

class MetricsTop():
    def __init__(self, train_mode):
            self.metrics_dict = {
                'ATTRACTIVENESS': self.__eval_mosei_regression
            }




    def __multiclass_acc(self, y_pred, y_true):
        """
        Compute the multiclass accuracy w.r.t. groundtruth

        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

    def __eval_mosei_regression(self, y_pred, y_true, exclude_zero=False):

        # y_pred = y_pred.cpu().detach().numpy()
        # y_true = y_true.cpu().detach().numpy()
        y_pred = np.array([ y.detach().numpy() for y in y_pred])
        y_true =   np.array([ y.detach().numpy() for y in y_true])
        test_preds_a7 = np.clip(y_pred, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(y_true, a_min=-3., a_max=3.)
        test_preds_a5 = np.clip(y_pred, a_min=-2., a_max=2.)
        test_truth_a5 = np.clip(y_true, a_min=-2., a_max=2.)
        test_preds_a3 = np.clip(y_pred, a_min=-1., a_max=1.)
        test_truth_a3 = np.clip(y_true, a_min=-1., a_max=1.)
        # print((y_pred[1]))

        mae = np.mean(np.absolute(y_pred - y_true)).astype(np.float64) #operands could not be broadcast together with shapes (64,1) (57,1)
        mult_a7 = self.__multiclass_acc(test_preds_a7, test_truth_a7)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)

        non_zeros = np.array([i for i, e in enumerate(y_true) if e != 0])
        non_zeros_binary_truth = (y_true[non_zeros] > 0)
        non_zeros_binary_preds = (y_pred[non_zeros] > 0)

        non_zeros_acc2 = accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)
        non_zeros_f1_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average='weighted')

        binary_truth = (y_true >= 0)
        binary_preds = (y_pred >= 0)
        acc2 = accuracy_score(binary_preds, binary_truth)
        f_score = f1_score(binary_truth, binary_preds, average='weighted')

        y_pred = [i[0] for i in y_pred]
        y_true = [i[0] for i in y_true]
        df_t = pd.DataFrame([y_true, y_pred])
        print(df_t)
        corr = np.corrcoef(y_pred, y_true)[0][1]
        # mult_a5 = 0
        # mult_a7= 0
        eval_results_reg = {
            "Has0_acc_2":  round(acc2, 4),
            "Has0_F1_score": round(f_score, 4),
            "Non0_acc_2":  round(non_zeros_acc2, 4),
            "Non0_F1_score": round(non_zeros_f1_score, 4),
            "Mult_acc_5": round(mult_a5, 4),
            "Mult_acc_7": round(mult_a7, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4),
        }
        return eval_results_reg

    def weighted_accuracy(test_preds_emo, test_truth_emo):
        true_label = (test_truth_emo > 0)
        predicted_label = (test_preds_emo > 0)
        tp = float(np.sum((true_label == 1) & (predicted_label == 1)))
        tn = float(np.sum((true_label == 0) & (predicted_label == 0)))
        p = float(np.sum(true_label == 1))
        n = float(np.sum(true_label == 0))

        return (tp * (n / p) + tn) / (2 * n)


    def getMetics(self, datasetName):
        return self.metrics_dict[datasetName.upper()]

    def multiclass_acc(preds, truths):
        """
        Compute the multiclass accuracy w.r.t. groundtruth

        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label == 1) & (predicted_label == 1)))
    tn = float(np.sum((true_label == 0) & (predicted_label == 0)))
    p = float(np.sum(true_label == 1))
    n = float(np.sum(true_label == 0))

    return (tp * (n / p) + tn) / (2 * n)
def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

def eval_routing( results, truths, results_weighted, truths_rounded     ):


    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()
    test_preds_weighted = results_weighted.view(-1).cpu().detach().numpy()
    test_truth_rounded = truths_rounded.view(-1).cpu().detach().numpy()

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth_rounded, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth_rounded, a_min=-2., a_max=2.)
    MAE_LIST = np.absolute(test_preds - test_truth)
    mae = np.mean(MAE_LIST)
    # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)

    binary_truth = (test_truth >= 0)
    binary_preds = (test_preds >= 0)
    acc2 = accuracy_score(binary_preds, binary_truth)
    f_score = f1_score(binary_truth, binary_preds, average='weighted')

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
    non_zeros_binary_truth = (test_truth[non_zeros] > 0)
    non_zeros_binary_preds = (test_preds[non_zeros] > 0)
    non_zeros_acc2 = accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)
    non_zeros_f1_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average='weighted')


    eval_results_reg = {
        "Has0_acc_2": round(acc2, 4),
        "Has0_F1_score": round(f_score, 4),
        "Non0_acc_2": round(non_zeros_acc2, 4),
        "Non0_F1_score": round(non_zeros_f1_score, 4),
        "Mult_acc_5": round(mult_a5, 4),
        "Mult_acc_7": round(mult_a7, 4),
        "MAE": round(mae, 4),
        "Corr": round(corr, 4),
    }
    return eval_results_reg

