import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve
import os
from config import VIZ_CONFIG
import numpy as np

sns.set(style=VIZ_CONFIG['style'])
plt.rcParams['figure.figsize'] = VIZ_CONFIG['figure_size']

def setup_plot_directory():
    """Create plots directory if it doesn't exist"""
    if VIZ_CONFIG['save_plots']:
        os.makedirs(VIZ_CONFIG['plot_dir'], exist_ok=True)

def plot_training_curve(losses, model_name='model', save=True):
    """
    Plot training loss curve
    
    Args:
        losses: List of training losses
        model_name: Name of the model for filename
        save: Whether to save the plot
    """
    plt.figure(figsize=VIZ_CONFIG['figure_size'])
    plt.plot(losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training Loss Curve')
    plt.grid(True, alpha=0.3)
    
    if save and VIZ_CONFIG['save_plots']:
        setup_plot_directory()
        plt.savefig(os.path.join(VIZ_CONFIG['plot_dir'], f'{model_name}_training_loss.png'), 
                   dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_confusion_matrix(cm, model_name='model', classes=['Normal', 'Fraud'], save=True):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        model_name: Name of the model for filename
        classes: Class labels
        save: Whether to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Confusion Matrix')
    
    if save and VIZ_CONFIG['save_plots']:
        setup_plot_directory()
        plt.savefig(os.path.join(VIZ_CONFIG['plot_dir'], f'{model_name}_confusion_matrix.png'),
                   dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_precision_recall_curve(y_true, y_prob, model_name='model', save=True):
    """
    Plot precision-recall curve
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        model_name: Name of the model for filename
        save: Whether to save the plot
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    plt.figure(figsize=VIZ_CONFIG['figure_size'])
    plt.plot(recall, precision, 'b-', linewidth=2, label=f'{model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save and VIZ_CONFIG['save_plots']:
        setup_plot_directory()
        plt.savefig(os.path.join(VIZ_CONFIG['plot_dir'], f'{model_name}_precision_recall_curve.png'),
                   dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_model_comparison(results, metric='f1'):
    """
    Plot comparison of different models
    
    Args:
        results: Dictionary of model results
        metric: Metric to compare
    """
    models = list(results.keys())
    scores = [results[model][metric] for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, scores, color=['skyblue', 'lightcoral', 'lightgreen'][:len(models)])
    plt.xlabel('Models')
    plt.ylabel(f'{metric.upper()} Score')
    plt.title(f'Model Comparison - {metric.upper()} Score')
    plt.ylim(0, 1)
    
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    
    if VIZ_CONFIG['save_plots']:
        setup_plot_directory()
        plt.savefig(os.path.join(VIZ_CONFIG['plot_dir'], f'model_comparison_{metric}.png'),
                   dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_multiple_metrics(results):
    """
    Plot multiple metrics for all models
    
    Args:
        results: Dictionary of model results
    """
    metrics = ['accuracy', 'f1', 'auc']
    models = list(results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        scores = [results[model][metric] for model in models]
        bars = axes[i].bar(models, scores, color=['skyblue', 'lightcoral', 'lightgreen'][:len(models)])
        axes[i].set_xlabel('Models')
        axes[i].set_ylabel(f'{metric.upper()} Score')
        axes[i].set_title(f'{metric.upper()} Comparison')
        axes[i].set_ylim(0, 1)
        axes[i].grid(True, alpha=0.3, axis='y')
        
        for bar, score in zip(bars, scores):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if VIZ_CONFIG['save_plots']:
        setup_plot_directory()
        plt.savefig(os.path.join(VIZ_CONFIG['plot_dir'], 'multiple_metrics_comparison.png'),
                   dpi=300, bbox_inches='tight')
    
    plt.show()

def create_summary_plots(model_name, losses, metrics):
    """
    Create a summary of all plots for a model
    
    Args:
        model_name: Name of the model
        losses: Training losses
        metrics: Evaluation metrics
    """
    fig = plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    plt.subplot(2, 3, 3)
    if len(set(metrics['y_true'])) > 1:
        precision, recall, _ = precision_recall_curve(metrics['y_true'], metrics['y_prob'])
        plt.plot(recall, precision, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 4)
    metric_names = ['Accuracy', 'F1', 'AUC']
    metric_values = [metrics['accuracy'], metrics['f1'], metrics['auc']]
    bars = plt.bar(metric_names, metric_values, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.ylabel('Score')
    plt.title('Performance Metrics')
    plt.ylim(0, 1)
    
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.subplot(2, 3, 5)
    unique, counts = np.unique(metrics['y_true'], return_counts=True)
    plt.pie(counts, labels=['Normal', 'Fraud'], autopct='%1.1f%%', startangle=90)
    plt.title('Test Set Distribution')
    
    plt.subplot(2, 3, 6)
    unique_pred, counts_pred = np.unique(metrics['y_pred'], return_counts=True)
    if len(unique_pred) == 2:
        plt.pie(counts_pred, labels=['Normal', 'Fraud'], autopct='%1.1f%%', startangle=90)
    else:
        if unique_pred[0] == 0:
            plt.pie([counts_pred[0], 0], labels=['Normal', 'Fraud'], autopct='%1.1f%%', startangle=90)
        else:
            plt.pie([0, counts_pred[0]], labels=['Normal', 'Fraud'], autopct='%1.1f%%', startangle=90)
    plt.title('Prediction Distribution')
    
    plt.suptitle(f'{model_name} Model Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if VIZ_CONFIG['save_plots']:
        setup_plot_directory()
        plt.savefig(os.path.join(VIZ_CONFIG['plot_dir'], f'{model_name}_summary.png'),
                   dpi=300, bbox_inches='tight')
    
    plt.show()