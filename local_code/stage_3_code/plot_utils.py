'''
Plotting utilities for learning convergence curves
'''

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import os


def plot_convergence_curves(train_losses, train_accuracies, save_dir=None, model_name="MLP"):
    """
    Plot training loss and accuracy convergence curves
    
    Args:
        train_losses: List of training losses per epoch
        train_accuracies: List of training accuracies per epoch
        save_dir: Directory to save plots (optional)
        model_name: Name for the plot title
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.set_title(f'{model_name} - Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot training accuracy
    ax2.plot(epochs, train_accuracies, 'r-', label='Training Accuracy', linewidth=2)
    ax2.set_title(f'{model_name} - Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plots if directory specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save combined plot
        combined_path = os.path.join(save_dir, f'{model_name}_convergence_curves.png')
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        print(f"Saved convergence curves to: {combined_path}")
        
        # Save individual plots
        # Loss plot
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.title(f'{model_name} - Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        loss_path = os.path.join(save_dir, f'{model_name}_training_loss.png')
        plt.savefig(loss_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Accuracy plot
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_accuracies, 'r-', label='Training Accuracy', linewidth=2)
        plt.title(f'{model_name} - Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend()
        acc_path = os.path.join(save_dir, f'{model_name}_training_accuracy.png')
        plt.savefig(acc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved individual plots to: {loss_path} and {acc_path}")
    
    # Show the plot (disabled for non-interactive mode)
    # # plt.show()  # Disabled for non-interactive mode


def plot_multiple_models_comparison(models_data, save_dir=None):
    """
    Plot comparison between multiple models
    
    Args:
        models_data: Dictionary {model_name: {'loss': losses, 'accuracy': accuracies}}
        save_dir: Directory to save plots (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training losses for all models
    for model_name, data in models_data.items():
        epochs = range(1, len(data['loss']) + 1)
        ax1.plot(epochs, data['loss'], label=model_name, linewidth=2)
    
    ax1.set_title('Training Loss Comparison')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot training accuracies for all models
    for model_name, data in models_data.items():
        epochs = range(1, len(data['accuracy']) + 1)
        ax2.plot(epochs, data['accuracy'], label=model_name, linewidth=2)
    
    ax2.set_title('Training Accuracy Comparison')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        comparison_path = os.path.join(save_dir, 'models_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to: {comparison_path}")
    
    # plt.show()  # Disabled for non-interactive mode


def plot_loss_convergence_detailed(train_losses, save_dir=None, model_name="MLP"):
    """
    Create a detailed loss convergence plot showing gradient descent convergence
    
    Args:
        train_losses: List of training losses per epoch
        save_dir: Directory to save plots (optional)
        model_name: Name for the plot title
    """
    plt.figure(figsize=(12, 8))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Main loss curve
    plt.plot(epochs, train_losses, 'b-', linewidth=2.5, label='Training Loss', alpha=0.8)
    
    # Add gradient descent convergence indicators
    # Calculate moving average to show trend
    window_size = min(20, len(train_losses) // 10)
    if window_size > 1:
        moving_avg = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
        ma_epochs = range(window_size, len(train_losses) + 1)
        plt.plot(ma_epochs, moving_avg, 'r--', linewidth=2, label=f'Moving Average (window={window_size})', alpha=0.7)
    
    # Highlight key convergence points
    # Find where loss reduction rate slows down significantly
    if len(train_losses) > 10:
        loss_changes = np.diff(train_losses)
        # Find point where change becomes small (convergence indicator)
        convergence_threshold = np.std(loss_changes[-len(loss_changes)//2:]) * 0.5
        convergence_points = []
        for i in range(len(loss_changes)):
            if i > 10 and abs(loss_changes[i]) < convergence_threshold:
                convergence_points.append(i + 1)  # +1 because diff shifts index
                break
        
        if convergence_points:
            conv_point = convergence_points[0]
            plt.axvline(x=conv_point, color='green', linestyle=':', linewidth=2, alpha=0.7, 
                       label=f'Convergence Point (Epoch {conv_point})')
            plt.scatter([conv_point], [train_losses[conv_point-1]], color='green', s=100, zorder=5)
    
    # Add annotations for key phases
    if len(train_losses) > 50:
        # Initial rapid learning phase
        plt.annotate('Rapid Learning Phase', 
                    xy=(10, train_losses[9]), 
                    xytext=(20, train_losses[9] * 1.5),
                    arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7),
                    fontsize=10, color='orange')
        
        # Convergence phase
        mid_point = len(train_losses) // 2
        plt.annotate('Gradient Descent Convergence', 
                    xy=(mid_point, train_losses[mid_point-1]), 
                    xytext=(mid_point + 50, train_losses[mid_point-1] * 2),
                    arrowprops=dict(arrowstyle='->', color='purple', alpha=0.7),
                    fontsize=10, color='purple')
    
    # Formatting
    plt.title(f'{model_name} - Loss Convergence with Gradient Descent', fontsize=14, fontweight='bold')
    plt.xlabel('Training Epoch', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Use log scale for better visualization if loss varies widely
    if max(train_losses) / min(train_losses) > 100:
        plt.yscale('log')
        plt.ylabel('Loss Value (log scale)', fontsize=12)
    
    # Add convergence metrics text
    if len(train_losses) > 1:
        initial_loss = train_losses[0]
        final_loss = train_losses[-1]
        reduction_percentage = ((initial_loss - final_loss) / initial_loss) * 100
        
        textstr = f'Initial Loss: {initial_loss:.4f}\nFinal Loss: {final_loss:.4f}\nReduction: {reduction_percentage:.1f}%'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save plot if directory specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        detailed_loss_path = os.path.join(save_dir, f'{model_name}_loss_convergence_detailed.png')
        plt.savefig(detailed_loss_path, dpi=300, bbox_inches='tight')
        print(f"Saved detailed loss convergence plot to: {detailed_loss_path}")
    
    # plt.show()  # Disabled for non-interactive mode


def print_training_summary(train_losses, train_accuracies, model_name="MLP"):
    """
    Print a summary of training performance
    
    Args:
        train_losses: List of training losses
        train_accuracies: List of training accuracies
        model_name: Name of the model
    """
    print(f"\n{'='*50}")
    print(f"TRAINING SUMMARY - {model_name}")
    print(f"{'='*50}")
    print(f"Total Epochs: {len(train_losses)}")
    print(f"Initial Loss: {train_losses[0]:.4f}")
    print(f"Final Loss: {train_losses[-1]:.4f}")
    print(f"Loss Reduction: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.2f}%")
    print(f"Initial Accuracy: {train_accuracies[0]:.4f}")
    print(f"Final Accuracy: {train_accuracies[-1]:.4f}")
    print(f"Accuracy Improvement: {((train_accuracies[-1] - train_accuracies[0]) * 100):.2f}%")
    
    # Find best epoch
    best_epoch = np.argmax(train_accuracies) + 1
    best_accuracy = max(train_accuracies)
    print(f"Best Epoch: {best_epoch} (Accuracy: {best_accuracy:.4f})")
    print(f"{'='*50}\n")
