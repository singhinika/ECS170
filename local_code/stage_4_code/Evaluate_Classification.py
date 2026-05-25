'''
Evaluation metrics for binary sentiment classification (stage 4)
'''

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
import numpy as np


class Evaluate_Classification(evaluate):
    data = None

    def _arrays(self):
        true_y = np.array(self.data['true_y'])
        pred_y = np.array(self.data['pred_y'])
        return true_y, pred_y

    def evaluate(self):
        true_y, pred_y = self._arrays()
        return {
            'accuracy':           accuracy_score(true_y, pred_y),
            'precision_macro':    precision_score(true_y, pred_y, average='macro',    zero_division=0),
            'precision_weighted': precision_score(true_y, pred_y, average='weighted', zero_division=0),
            'recall_macro':       recall_score(true_y,    pred_y, average='macro',    zero_division=0),
            'recall_weighted':    recall_score(true_y,    pred_y, average='weighted', zero_division=0),
            'f1_macro':           f1_score(true_y,        pred_y, average='macro',    zero_division=0),
            'f1_weighted':        f1_score(true_y,        pred_y, average='weighted', zero_division=0),
        }

    def print_summary(self):
        r = self.evaluate()
        print('\n' + '='*50)
        print('CLASSIFICATION METRICS SUMMARY')
        print('='*50)
        for k, v in r.items():
            print(f'  {k:<25} {v:.4f}')
        print('='*50)

    def print_detailed_report(self):
        true_y, pred_y = self._arrays()
        print('\n' + '='*50)
        print('DETAILED CLASSIFICATION REPORT')
        print('='*50)
        print(classification_report(true_y, pred_y,
                                    target_names=['negative', 'positive'], digits=4))
        print('Confusion Matrix:')
        print(confusion_matrix(true_y, pred_y))
        print('='*50)
