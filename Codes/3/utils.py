import os
import json
import torch
import numpy as np
from datetime import datetime
from config import FILE_PATHS

def create_directories():
    """Create necessary directories for saving results"""
    for path in FILE_PATHS.values():
        os.makedirs(path, exist_ok=True)

def save_results(results, filename=None):
    """
    Save results to JSON file
    
    Args:
        results: Dictionary of results to save
        filename: Optional filename, defaults to timestamp
    """
    create_directories()
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.json"
    
    serializable_results = {}
    for model_name, metrics in results.items():
        serializable_results[model_name] = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable_results[model_name][key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serializable_results[model_name][key] = float(value)
            else:
                serializable_results[model_name][key] = value
    
    filepath = os.path.join(FILE_PATHS['results_dir'], filename)
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to: {filepath}")

def load_results(filename):
    """
    Load results from JSON file
    
    Args:
        filename: Name of the results file
    
    Returns:
        dict: Loaded results
    """
    filepath = os.path.join(FILE_PATHS['results_dir'], filename)
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results

def print_summary(results):
    """
    Print a summary of all model results
    
    Args:
        results: Dictionary of model results
    """
    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    print(f"{'Model':<15} {'Accuracy':<12} {'F1 Score':<12} {'AUC Score':<12}")
    print("-" * 55)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<15} {metrics['accuracy']:<12.4f} {metrics['f1']:<12.4f} {metrics['auc']:<12.4f}")
    
    print("\n" + "="*40)
    print("BEST PERFORMING MODELS")
    print("="*40)
    
    metrics_to_check = ['accuracy', 'f1', 'auc']
    for metric in metrics_to_check:
        best_model = max(results.items(), key=lambda x: x[1][metric])
        print(f"Best {metric.upper()}: {best_model[0]} ({best_model[1][metric]:.4f})")

def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_device():
    """
    Get the best available device (GPU or CPU)
    
    Returns:
        torch.device: Available device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def count_parameters(model):
    """
    Count the number of trainable parameters in a model
    
    Args:
        model: PyTorch model
    
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_time(seconds):
    """
    Format seconds into readable time format
    
    Args:
        seconds: Time in seconds
    
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{int(minutes)}m {remaining_seconds:.0f}s"
    else:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(remaining_minutes)}m"

def check_data_quality(df):
    """
    Check data quality and print statistics
    
    Args:
        df: Input dataframe
    """
    print("\nDATA QUALITY CHECK")
    print("="*40)
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\nMissing values found:")
        for col, count in missing_values[missing_values > 0].items():
            print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
    else:
        print("No missing values found")
    
    print(f"\nData types:")
    for dtype in df.dtypes.unique():
        cols = df.select_dtypes(include=[dtype]).columns.tolist()
        print(f"  {dtype}: {len(cols)} columns")
    
    if 'isFraud' in df.columns:
        fraud_count = df['isFraud'].sum()
        fraud_rate = fraud_count / len(df) * 100
        print(f"\nFraud distribution:")
        print(f"  Normal transactions: {len(df) - fraud_count} ({100-fraud_rate:.2f}%)")
        print(f"  Fraud transactions: {fraud_count} ({fraud_rate:.2f}%)")
        print(f"  Imbalance ratio: {(len(df) - fraud_count) / fraud_count:.1f}:1")