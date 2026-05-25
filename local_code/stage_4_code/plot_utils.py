'''
Plotting utilities for stage 4 learning curves
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_learning_curves(train_losses, train_accuracies,
                         test_losses=None, test_accuracies=None,
                         save_dir=None, model_name='RNN'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, 'b-',  linewidth=2, label='Train Loss')
    if test_losses:
        ax1.plot(epochs, test_losses, 'b--', linewidth=2, label='Test Loss')
    ax1.set_title(f'{model_name} — Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, train_accuracies, 'r-',  linewidth=2, label='Train Accuracy')
    if test_accuracies:
        ax2.plot(epochs, test_accuracies, 'r--', linewidth=2, label='Test Accuracy')
    ax2.set_title(f'{model_name} — Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(f'{model_name} Learning Curves', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'{model_name}_learning_curves.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved learning curves to: {path}')
    plt.close()


def plot_generation_loss(train_losses, save_dir=None, model_name='RNN_Generation'):
    plt.figure(figsize=(9, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss')

    if len(train_losses) > 5:
        w = max(3, len(train_losses) // 10)
        ma = np.convolve(train_losses, np.ones(w) / w, mode='valid')
        plt.plot(range(w, len(train_losses) + 1), ma, 'r--', linewidth=1.5,
                 label=f'Moving Avg (w={w})')

    plt.title(f'{model_name} — Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'{model_name}_loss.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved generation loss plot to: {path}')
    plt.close()


def plot_model_comparison(models_data, save_dir=None, metric='loss'):
    '''
    models_data: {model_name: list_of_values}
    metric: 'loss' or 'accuracy'
    '''
    plt.figure(figsize=(10, 5))
    for name, values in models_data.items():
        plt.plot(range(1, len(values) + 1), values, linewidth=2, label=name)
    plt.title(f'Model Comparison — {metric.capitalize()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'model_comparison_{metric}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved comparison plot to: {path}')
    plt.close()


def print_training_summary(train_losses, train_accuracies, model_name='RNN'):
    print(f'\n{"="*50}')
    print(f'TRAINING SUMMARY — {model_name}')
    print(f'{"="*50}')
    print(f'  Epochs:            {len(train_losses)}')
    print(f'  Initial Loss:      {train_losses[0]:.4f}')
    print(f'  Final Loss:        {train_losses[-1]:.4f}')
    print(f'  Loss Reduction:    {(train_losses[0] - train_losses[-1]) / train_losses[0] * 100:.1f}%')
    print(f'  Initial Accuracy:  {train_accuracies[0]:.4f}')
    print(f'  Final Accuracy:    {train_accuracies[-1]:.4f}')
    best_epoch = int(np.argmax(train_accuracies)) + 1
    print(f'  Best Epoch:        {best_epoch} (Acc: {max(train_accuracies):.4f})')
    print(f'{"="*50}\n')
