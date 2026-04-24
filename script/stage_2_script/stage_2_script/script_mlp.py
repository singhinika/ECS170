from local_code.stage_2_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_2_code.stage_2_code.Evaluate_Multiclass import Evaluate_Multiclass

import numpy as np
import torch

# ---- Multi-Layer Perceptron script ----
if __name__ == "__main__":
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- object initialization section ---------------
    training_data_obj = Dataset_Loader("training_data", "stage 2 training set")
    training_data_obj.dataset_source_folder_path = "/Users/inikasingh/Downloads/davis spring 26/ecs 170/stage_2_data/"
    training_data_obj.dataset_source_file_name = "train.csv"

    testing_data_obj = Dataset_Loader("testing_data", "stage 2 test set")
    testing_data_obj.dataset_source_folder_path = "/Users/inikasingh/Downloads/davis spring 26/ecs 170/stage_2_data/"
    testing_data_obj.dataset_source_file_name = "test.csv"

    training_data = training_data_obj.load()
    testing_data = testing_data_obj.load()

    method_obj = Method_MLP("multi-layer perceptron", "")
    method_obj.data = {
        "train": training_data,
        "test": testing_data,
    }

    result_obj = Result_Saver("saver", "")
    result_obj.result_destination_folder_path = "/Users/inikasingh/Downloads/davis spring 26/ecs 170/ECS170_Spring_2026_Source_Code_Template/result/stage_2_result/MLP"
    result_obj.result_destination_file_name = "prediction_result"

    evaluate_obj = Evaluate_Accuracy("accuracy", "")
    multiclass_eval_obj = Evaluate_Multiclass("multiclass metrics", "")
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print("************ Start ************")

    learned_result = method_obj.run()

    result_obj.data = learned_result
    result_obj.fold_count = 1
    result_obj.save()

    # Basic accuracy evaluation
    evaluate_obj.data = learned_result
    accuracy_score = evaluate_obj.evaluate()

    # Multiclass evaluation
    multiclass_eval_obj.data = learned_result
    multiclass_eval_obj.print_summary()
    multiclass_eval_obj.print_detailed_report()

    print("************ Overall Performance ************")
    print("MLP Accuracy: " + str(accuracy_score))
    print("************ Finish ************")
    # ------------------------------------------------------