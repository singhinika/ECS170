from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Enhanced_Method_MLP import Enhanced_Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_2_code.Evaluate_Multiclass import Evaluate_Multiclass
from local_code.stage_2_code.plot_utils import plot_convergence_curves, plot_loss_convergence_detailed, print_training_summary, plot_multiple_models_comparison

import numpy as np
import torch

# ---- Multi-Layer Perceptron Experiments Script ----
if __name__ == "__main__":
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    
    # Experiment settings
    QUICK_MODE = True  # Set to False for full 500 epochs
    MAX_EPOCHS = 100 if QUICK_MODE else 500
    PRINT_INTERVAL = 25 if QUICK_MODE else 100
    
    print(f"EXPERIMENT MODE: {'QUICK (100 epochs)' if QUICK_MODE else 'FULL (500 epochs)'}")
    print(f"Progress print interval: Every {PRINT_INTERVAL} epochs")
    # ------------------------------------------------------

    # ---- object initialization section ---------------
    training_data_obj = Dataset_Loader("training_data", "stage 2 training set")
    training_data_obj.dataset_source_folder_path = "./data/stage_2_data/"
    training_data_obj.dataset_source_file_name = "train.csv"

    testing_data_obj = Dataset_Loader("testing_data", "stage 2 test set")
    testing_data_obj.dataset_source_folder_path = "./data/stage_2_data/"
    testing_data_obj.dataset_source_file_name = "test.csv"

    training_data = training_data_obj.load()
    testing_data = testing_data_obj.load()

    # Experiment configurations
    experiments = {
        "Baseline": {
            "class": Method_MLP,
            "name": "MLP_Baseline",
            "description": "Original 2-layer architecture (784-128-10) with Adam optimizer"
        },
        "Deep_MLP": {
            "class": Enhanced_Method_MLP,
            "name": "Deep_MLP",
            "description": "4-layer architecture (784-512-256-128-10) with Adam optimizer",
            "architecture": [784, 512, 256, 128, 10],
            "optimizer": "adam",
            "learning_rate": 0.001,
            "dropout_rate": 0.2
        },
        "Deep_MLP_SGD": {
            "class": Enhanced_Method_MLP,
            "name": "Deep_MLP_SGD",
            "description": "4-layer architecture with SGD optimizer and momentum",
            "architecture": [784, 512, 256, 128, 10],
            "optimizer": "sgd",
            "learning_rate": 0.01,
            "momentum": 0.9,
            "dropout_rate": 0.2
        },
        "Deep_MLP_AdamW": {
            "class": Enhanced_Method_MLP,
            "name": "Deep_MLP_AdamW",
            "description": "4-layer architecture with AdamW optimizer and weight decay",
            "architecture": [784, 512, 256, 128, 10],
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "dropout_rate": 0.3
        },
        "Deep_MLP_BN": {
            "class": Enhanced_Method_MLP,
            "name": "Deep_MLP_BN",
            "description": "4-layer architecture with Batch Normalization and dropout",
            "architecture": [784, 512, 256, 128, 10],
            "optimizer": "adam",
            "learning_rate": 0.001,
            "dropout_rate": 0.3,
            "batch_norm": True
        },
        "Wide_MLP": {
            "class": Enhanced_Method_MLP,
            "name": "Wide_MLP",
            "description": "3-layer wide architecture (784-1024-512-10) with Adam",
            "architecture": [784, 1024, 512, 10],
            "optimizer": "adam",
            "learning_rate": 0.001,
            "dropout_rate": 0.2
        }
    }

    # Store results for comparison
    all_results = {}
    models_data = {}

    import time
    
    print("************ Starting MLP Experiments ************")
    print(f"Total experiments to run: {len(experiments)}")
    print("="*60)
    
    # Track overall progress
    start_time = time.time()
    completed_experiments = 0

    # Run each experiment
    for i, (exp_name, config) in enumerate(experiments.items(), 1):
        exp_start_time = time.time()
        print(f"\n{'='*20} EXPERIMENT {i}/{len(experiments)}: {exp_name} {'='*20}")
        print(f"Description: {config['description']}")
        print(f"Progress: {completed_experiments}/{len(experiments)} completed")
        print("-" * 60)
        
        # Show estimated remaining time
        if completed_experiments > 0:
            avg_time_per_exp = (time.time() - start_time) / completed_experiments
            remaining_experiments = len(experiments) - completed_experiments
            eta_minutes = (avg_time_per_exp * remaining_experiments) / 60
            print(f"Estimated time remaining: {eta_minutes:.1f} minutes")
        
        print(f"Starting {exp_name}...")
        
        # Initialize method object
        if exp_name == "Baseline":
            method_obj = config["class"]("multi-layer perceptron", "")
            method_obj.max_epoch = MAX_EPOCHS
        else:
            method_obj = config["class"]("enhanced mlp", "", config)
            method_obj.max_epoch = MAX_EPOCHS
        
        method_obj.data = {
            "train": training_data,
            "test": testing_data,
        }

        print(f"Training {exp_name} for {MAX_EPOCHS} epochs...")
        
        # Train and test
        learned_result = method_obj.run()
        
        exp_time = time.time() - exp_start_time
        print(f"✅ {exp_name} completed in {exp_time:.1f} seconds ({exp_time/60:.1f} minutes)")
        
        # Evaluate
        evaluate_obj = Evaluate_Accuracy("accuracy", "")
        evaluate_obj.data = learned_result
        accuracy_score = evaluate_obj.evaluate()
        
        multiclass_eval_obj = Evaluate_Multiclass("multiclass metrics", "")
        multiclass_eval_obj.data = learned_result
        
        # Store results
        all_results[exp_name] = {
            "accuracy": accuracy_score,
            "config": config,
            "result": learned_result
        }
        
        # Store training data for plotting
        models_data[config["name"]] = {
            "loss": method_obj.train_losses,
            "accuracy": method_obj.train_accuracies
        }
        
        # Print results
        print(f"\n{config['name']} Results:")
        print(f"Test Accuracy: {accuracy_score:.4f}")
        multiclass_eval_obj.print_summary()
        
        # Save plots
        plot_dir = f"./result/stage_2_result/MLP_Experiments"
        
        plot_convergence_curves(
            method_obj.train_losses, 
            method_obj.train_accuracies, 
            save_dir=plot_dir, 
            model_name=config["name"]
        )
        
        plot_loss_convergence_detailed(
            method_obj.train_losses, 
            save_dir=plot_dir, 
            model_name=config["name"]
        )
        
        print_training_summary(method_obj.train_losses, method_obj.train_accuracies, config["name"])
        
        # Update completion counter
        completed_experiments += 1
        print(f"Progress: {completed_experiments}/{len(experiments)} experiments completed")

    # Generate comparison plots
    print("\n" + "="*60)
    print("GENERATING COMPARISON PLOTS")
    print("="*60)
    
    plot_dir = "./result/stage_2_result/MLP_Experiments"
    plot_multiple_models_comparison(models_data, save_dir=plot_dir)

    # Final comparison summary
    print("\n" + "="*60)
    print("FINAL EXPERIMENT COMPARISON")
    print("="*60)
    
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    
    print(f"{'Rank':<5} {'Model':<20} {'Test Accuracy':<15} {'Description'}")
    print("-" * 80)
    
    for rank, (exp_name, results) in enumerate(sorted_results, 1):
        config = results["config"]
        accuracy = results["accuracy"]
        description = config["description"][:40] + "..." if len(config["description"]) > 40 else config["description"]
        print(f"{rank:<5} {config['name']:<20} {accuracy:<15.4f} {description}")
    
    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETED")
    print("="*60)
    # ------------------------------------------------------
