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


        y_pred = np.array([ y.detach().numpy() for y in y_pred])
        y_true =   np.array([ y.detach().numpy() for y in y_true])

        mae = np.mean(np.absolute(y_pred - y_true)).astype(np.float64)  

        non_zeros = np.array([i for i, e in enumerate(y_true) if e != 0])
        binary_truth = (y_true[non_zeros] > 0)
        binary_preds = (y_pred[non_zeros] > 0)

        acc2 = accuracy_score(binary_preds, binary_truth)
        f1_value= f1_score(binary_truth, binary_preds, average='weighted')

        y_pred = [i[0] for i in y_pred]
        y_true = [i[0] for i in y_true]
        df_t = pd.DataFrame([y_true, y_pred])
        print(df_t)
        corr = np.corrcoef(y_pred, y_true)[0][1]

        eval_results_reg = {
            "Acc_2":  round(acc2, 4),
            "F1_score": round(f1_value, 4),
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

