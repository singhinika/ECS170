'''
GCN node classification on the Citeseer dataset.
Target: 67%+ test accuracy.

Run from project root:
    python -m script.stage_5_script.script_citeseer
'''

import numpy as np
import torch

from local_code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from local_code.stage_5_code.Method_GCN                         import Method_GCN
from local_code.stage_5_code.Result_Saver                       import Result_Saver
from local_code.stage_5_code.Evaluate_Node_Classification       import Evaluate_Node_Classification
from local_code.stage_5_code.plot_utils                         import plot_learning_curves, print_training_summary

DATASET    = 'citeseer'
MODEL_NAME = 'GCN_Citeseer'
RESULT_DIR = f'./result/stage_5_result/{DATASET}'

if __name__ == '__main__':
    np.random.seed(42)

    # ---- data ----
    data_obj = Dataset_Loader(dName=DATASET, dDescription='Citeseer citation network')
    data_obj.dataset_name = DATASET
    data_obj.dataset_source_folder_path = f'./data/stage_5_data/{DATASET}'
    loaded_data = data_obj.load()

    # L2 row-normalize features (helps Citeseer's sparse/noisy term vectors)
    features = loaded_data['graph']['X']
    row_norms = features.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
    loaded_data['graph']['X'] = features / row_norms

    # set torch seed after data loading so tensor construction doesn't consume RNG state
    torch.manual_seed(42)

    # ---- model ----
    method_obj = Method_GCN(MODEL_NAME, '')
    method_obj.data          = loaded_data
    method_obj.hidden_dim    = 64
    method_obj.num_layers    = 2
    method_obj.dropout       = 0.5
    method_obj.learning_rate = 0.01
    method_obj.weight_decay  = 5e-3
    method_obj.max_epoch     = 200

    # ---- result saver ----
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = RESULT_DIR
    result_obj.result_destination_file_name   = 'prediction_result'
    result_obj.fold_count = 1

    # ---- evaluator ----
    evaluate_obj = Evaluate_Node_Classification('accuracy', '')

    # ---- run ----
    print('************ Start ************')
    learned_result = method_obj.run()

    result_obj.data = learned_result
    result_obj.save()

    evaluate_obj.data = learned_result
    evaluate_obj.print_summary()
    evaluate_obj.print_detailed_report()

    plot_learning_curves(
        method_obj.train_losses,
        method_obj.train_accuracies,
        test_losses=method_obj.test_losses,
        test_accuracies=method_obj.test_accuracies,
        save_dir=RESULT_DIR,
        model_name=MODEL_NAME,
    )
    print_training_summary(method_obj.train_losses, method_obj.train_accuracies, MODEL_NAME)

    metrics = evaluate_obj.evaluate()
    print('************ Overall Performance ************')
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"  F1 (macro):        {metrics['f1_macro']:.4f}")
    print('************ Finish ************')
