'''
Concrete Evaluate class for multiclass classification metrics (stage 3)
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np


class Evaluate_Accuracy(evaluate):
    data = None

    def _to_numpy(self, arr):
        if hasattr(arr, 'cpu'):
            return arr.cpu().numpy()
        return np.array(arr)

    def evaluate(self):
        print('evaluating performance...')
        true_y = self._to_numpy(self.data['true_y'])
        pred_y = self._to_numpy(self.data['pred_y'])

        accuracy           = accuracy_score(true_y, pred_y)
        precision_macro    = precision_score(true_y, pred_y, average='macro',    zero_division=0)
        precision_weighted = precision_score(true_y, pred_y, average='weighted', zero_division=0)
        recall_macro       = recall_score(true_y,    pred_y, average='macro',    zero_division=0)
        recall_weighted    = recall_score(true_y,    pred_y, average='weighted', zero_division=0)
        f1_macro           = f1_score(true_y,        pred_y, average='macro',    zero_division=0)
        f1_weighted        = f1_score(true_y,        pred_y, average='weighted', zero_division=0)

        return {
            'accuracy':           accuracy,
            'precision_macro':    precision_macro,
            'precision_weighted': precision_weighted,
            'recall_macro':       recall_macro,
            'recall_weighted':    recall_weighted,
            'f1_macro':           f1_macro,
            'f1_weighted':        f1_weighted,
        }

    def print_summary(self):
        results = self.evaluate()
        print('\n' + '='*50)
        print('CLASSIFICATION METRICS SUMMARY')
        print('='*50)
        print(f"Accuracy:           {results['accuracy']:.4f}")
        print(f"Precision (macro):  {results['precision_macro']:.4f}")
        print(f"Precision (wtd):    {results['precision_weighted']:.4f}")
        print(f"Recall    (macro):  {results['recall_macro']:.4f}")
        print(f"Recall    (wtd):    {results['recall_weighted']:.4f}")
        print(f"F1        (macro):  {results['f1_macro']:.4f}")
        print(f"F1        (wtd):    {results['f1_weighted']:.4f}")
        print('='*50)

    def print_detailed_report(self):
        true_y = self._to_numpy(self.data['true_y'])
        pred_y = self._to_numpy(self.data['pred_y'])
        print('\n' + '='*50)
        print('DETAILED CLASSIFICATION REPORT')
        print('='*50)
        print(classification_report(true_y, pred_y, digits=4))
        print('CONFUSION MATRIX:')
        print(confusion_matrix(true_y, pred_y))
        print('='*50)
