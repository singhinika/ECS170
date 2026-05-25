'''
Stage 4-3: Train RNN for IMDB sentiment classification, evaluate on test set,
           generate learning curves.
Run from project root:
    python -m script.stage_4_script.script_rnn_classification
'''

import numpy as np
import torch

from local_code.stage_4_code.Dataset_Loader_Classification import Dataset_Loader_Classification
from local_code.stage_4_code.Method_RNN_Classification     import Method_RNN_Classification
from local_code.stage_4_code.Result_Saver                  import Result_Saver
from local_code.stage_4_code.Evaluate_Classification       import Evaluate_Classification
from local_code.stage_4_code.plot_utils                    import (plot_learning_curves,
                                                                    print_training_summary)

RESULT_DIR = './result/stage_4_result/RNN_Classification'

if __name__ == '__main__':
    np.random.seed(2)
    torch.manual_seed(2)

    # ---- data ----
    data_obj = Dataset_Loader_Classification('imdb', 'IMDB Movie Reviews')
    data_obj.dataset_source_folder_path = './data/stage_4_data/text_classification'
    loaded_data = data_obj.load()

    # ---- model ----
    method_obj = Method_RNN_Classification('RNN-Classification', '')
    method_obj.data       = loaded_data
    method_obj.vocab_size = len(loaded_data['vocab'])
    method_obj.max_epoch  = 10

    # ---- result saver ----
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = RESULT_DIR
    result_obj.result_destination_file_name   = 'prediction_result'
    result_obj.fold_count = 1

    # ---- evaluator ----
    evaluate_obj = Evaluate_Classification('accuracy', '')

    # ---- run ----
    print('************ Start ************')
    learned_result = method_obj.run()

    result_obj.data = learned_result
    result_obj.save()

    evaluate_obj.data = learned_result
    evaluate_obj.print_summary()
    evaluate_obj.print_detailed_report()

    # ---- learning curves ----
    plot_learning_curves(
        method_obj.train_losses,
        method_obj.train_accuracies,
        save_dir=RESULT_DIR,
        model_name='RNN_Classification',
    )
    print_training_summary(
        method_obj.train_losses,
        method_obj.train_accuracies,
        'RNN_Classification',
    )

    metrics = evaluate_obj.evaluate()
    print('************ Overall Performance ************')
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    print('************ Finish ************')
