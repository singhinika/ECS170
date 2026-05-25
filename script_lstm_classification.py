'''
Stage 4-5: LSTM for IMDB sentiment classification.
Identical setup to the base RNN script; only the recurrent cell changes to LSTM.
Run from project root:
    python -m script.stage_4_script.script_lstm_classification
'''

import numpy as np
import torch

from local_code.stage_4_code.Dataset_Loader_Classification import Dataset_Loader_Classification
from local_code.stage_4_code.Method_RNN_Classification     import Method_RNN_Classification
from local_code.stage_4_code.Result_Saver                  import Result_Saver
from local_code.stage_4_code.Evaluate_Classification       import Evaluate_Classification
from local_code.stage_4_code.plot_utils                    import (plot_learning_curves,
                                                                    print_training_summary)

MODEL_NAME = 'LSTM_Classification'
RESULT_DIR = f'./result/stage_4_result/{MODEL_NAME}'

if __name__ == '__main__':
    np.random.seed(2)
    torch.manual_seed(2)

    # ---- data ----
    data_obj = Dataset_Loader_Classification('imdb', 'IMDB Movie Reviews')
    data_obj.dataset_source_folder_path = './data/stage_4_data/text_classification'
    loaded_data = data_obj.load()

    # ---- model: LSTM instead of RNN ----
    method_obj = Method_RNN_Classification(MODEL_NAME, '')
    method_obj.data       = loaded_data
    method_obj.vocab_size = len(loaded_data['vocab'])
    method_obj.cell_type  = 'LSTM'
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
        test_losses=method_obj.test_losses,
        test_accuracies=method_obj.test_accuracies,
        save_dir=RESULT_DIR,
        model_name=MODEL_NAME,
    )
    print_training_summary(method_obj.train_losses, method_obj.train_accuracies, MODEL_NAME)

    metrics = evaluate_obj.evaluate()
    print('************ Overall Performance ************')
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    print('************ Finish ************')
