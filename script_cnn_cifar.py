'''
Script to train and evaluate CNN on CIFAR-10 object recognition dataset
Run from project root: python -m script.stage_3_script.script_cnn_cifar
'''

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_3_code.plot_utils import plot_convergence_curves, print_training_summary, plot_loss_convergence_detailed

import numpy as np
import torch

if __name__ == '__main__':
    np.random.seed(2)
    torch.manual_seed(2)

    # ---- load data ----
    data_obj = Dataset_Loader('cifar', 'CIFAR-10 colored objects')
    data_obj.dataset_source_folder_path = './data/stage_3_data/'
    data_obj.dataset_source_file_name   = 'CIFAR'
    data_obj.dataset_type               = 'CIFAR'
    loaded_data = data_obj.load()

    # ---- method ----
    method_obj = Method_CNN_CIFAR('CNN-CIFAR', '')
    method_obj.data = loaded_data

    # ---- result saver ----
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = './result/stage_3_result/CNN_CIFAR'
    result_obj.result_destination_file_name   = 'prediction_result'
    result_obj.fold_count = 1

    # ---- evaluate ----
    evaluate_obj = Evaluate_Accuracy('accuracy', '')

    # ---- run ----
    print('************ Start ************')
    learned_result = method_obj.run()

    result_obj.data = learned_result
    result_obj.save()

    evaluate_obj.data = learned_result
    evaluate_obj.print_summary()
    evaluate_obj.print_detailed_report()

    # ---- plots ----
    plot_dir = './result/stage_3_result/CNN_CIFAR'
    plot_convergence_curves(
        method_obj.train_losses,
        method_obj.train_accuracies,
        save_dir=plot_dir,
        model_name='CNN_CIFAR'
    )
    print_training_summary(method_obj.train_losses, method_obj.train_accuracies, 'CNN_CIFAR')
    plot_loss_convergence_detailed(method_obj.train_losses, save_dir=plot_dir, model_name='CNN_CIFAR')

    metrics = evaluate_obj.evaluate()
    print('************ Overall Performance ************')
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1 (macro): {metrics['f1_macro']:.4f}")
    print('************ Finish ************')
