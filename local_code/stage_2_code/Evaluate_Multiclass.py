'''
Concrete Evaluate class for multiclass classification metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np


class Evaluate_Multiclass(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']
        
        # Convert torch tensors to numpy arrays if needed
        if hasattr(true_y, 'cpu'):
            true_y = true_y.cpu().numpy()
        if hasattr(pred_y, 'cpu'):
            pred_y = pred_y.cpu().numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(true_y, pred_y)
        
        # Calculate precision, recall, f1 for different averaging methods
        precision_macro = precision_score(true_y, pred_y, average='macro', zero_division=0)
        precision_micro = precision_score(true_y, pred_y, average='micro', zero_division=0)
        precision_weighted = precision_score(true_y, pred_y, average='weighted', zero_division=0)
        
        recall_macro = recall_score(true_y, pred_y, average='macro', zero_division=0)
        recall_micro = recall_score(true_y, pred_y, average='micro', zero_division=0)
        recall_weighted = recall_score(true_y, pred_y, average='weighted', zero_division=0)
        
        f1_macro = f1_score(true_y, pred_y, average='macro', zero_division=0)
        f1_micro = f1_score(true_y, pred_y, average='micro', zero_division=0)
        f1_weighted = f1_score(true_y, pred_y, average='weighted', zero_division=0)
        
        # Create results dictionary
        results = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'precision_micro': precision_micro,
            'precision_weighted': precision_weighted,
            'recall_macro': recall_macro,
            'recall_micro': recall_micro,
            'recall_weighted': recall_weighted,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'f1_weighted': f1_weighted
        }
        
        return results
    
    def print_detailed_report(self):
        '''Print detailed classification report'''
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']
        
        # Convert torch tensors to numpy arrays if needed
        if hasattr(true_y, 'cpu'):
            true_y = true_y.cpu().numpy()
        if hasattr(pred_y, 'cpu'):
            pred_y = pred_y.cpu().numpy()
        
        print("\n" + "="*50)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*50)
        print(classification_report(true_y, pred_y, digits=4))
        
        print("\nCONFUSION MATRIX:")
        print(confusion_matrix(true_y, pred_y))
        print("="*50)
    
    def print_summary(self):
        '''Print summary of all metrics'''
        results = self.evaluate()
        
        print("\n" + "="*50)
        print("MULTICLASS CLASSIFICATION METRICS SUMMARY")
        print("="*50)
        print(f"Accuracy: {results['accuracy']:.4f}")
        
        print(f"\nPrecision:")
        print(f"  Macro: {results['precision_macro']:.4f}")
        print(f"  Micro: {results['precision_micro']:.4f}")
        print(f"  Weighted: {results['precision_weighted']:.4f}")
        
        print(f"\nRecall:")
        print(f"  Macro: {results['recall_macro']:.4f}")
        print(f"  Micro: {results['recall_micro']:.4f}")
        print(f"  Weighted: {results['recall_weighted']:.4f}")
        
        print(f"\nF1 Score:")
        print(f"  Macro: {results['f1_macro']:.4f}")
        print(f"  Micro: {results['f1_micro']:.4f}")
        print(f"  Weighted: {results['f1_weighted']:.4f}")
        print("="*50)
