import time
import torch
from data_loader import prepare_data
from graph_builder import build_transaction_graph, prepare_pytorch_geometric_data
from models import create_model, get_model_config
from trainer import train_model, print_classification_report
from visualizer import (plot_training_curve, plot_confusion_matrix, 
                       plot_precision_recall_curve, plot_model_comparison, 
                       plot_multiple_metrics, create_summary_plots)
from utils import (create_directories, save_results, print_summary, 
                   set_random_seeds, get_device, count_parameters, 
                   format_time, check_data_quality)
from config import DATASET_CONFIG, MODEL_CONFIG, TRAINING_CONFIG

def main():
    """Main execution function"""
    print("="*80)
    print("GRAPH-BASED ANOMALY DETECTION FOR FINANCIAL FRAUD")
    print("="*80)
    
    start_time = time.time()
    set_random_seeds(DATASET_CONFIG['random_state'])
    device = get_device()
    create_directories()
    
    print("\n[STEP 1] Loading and preparing data...")
    df = prepare_data(
        file_path=DATASET_CONFIG['file_path'],
        sample_size=DATASET_CONFIG['sample_size'],
        random_state=DATASET_CONFIG['random_state']
    )
    
    check_data_quality(df)
    
    print("\n[STEP 2] Building transaction graph...")
    G, fraud_nodes = build_transaction_graph(df)
    
    print("\n[STEP 3] Preparing PyTorch Geometric data...")
    data = prepare_pytorch_geometric_data(
        G, fraud_nodes, 
        balance_ratio=DATASET_CONFIG['balance_ratio']
    )
    
    data = data.to(device)
    
    print("\n[STEP 4] Setting up models...")
    model_types = ['gcn', 'gat']
    models = {}
    
    for model_type in model_types:
        config = get_model_config(model_type)
        model = create_model(
            model_type=model_type,
            in_channels=data.num_node_features,
            out_channels=2,
            **config
        ).to(device)
        
        models[model_type.upper()] = model
        print(f"  {model_type.upper()}: {count_parameters(model):,} parameters")
    
    print("\n[STEP 5] Training models...")
    results = {}
    training_times = {}
    
    for name, model in models.items():
        model_start_time = time.time()
        
        trained_model, losses, metrics = train_model(
            model=model,
            data=data,
            model_name=name,
            save_dir='models/'
        )
        
        training_time = time.time() - model_start_time
        training_times[name] = training_time
        
        results[name] = metrics
        
        print_classification_report(metrics, name)
        
        print(f"\nCreating visualizations for {name}...")
        
        plot_training_curve(losses, name)
        plot_confusion_matrix(metrics['confusion_matrix'], name)
        
        if len(set(metrics['y_true'])) > 1: 
            plot_precision_recall_curve(metrics['y_true'], metrics['y_prob'], name)
        
        create_summary_plots(name, losses, metrics)
        
        print(f"Training time for {name}: {format_time(training_time)}")
    
    print("\n[STEP 6] Comparing models...")
    print_summary(results)
    
    plot_model_comparison(results, 'f1')
    plot_multiple_metrics(results)
    
    print("\n[STEP 7] Saving results...")
    
    for name in results:
        results[name]['training_time'] = training_times[name]
    
    save_results(results)
    
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(f"Total execution time: {format_time(total_time)}")
    print(f"Data points processed: {len(df):,}")
    print(f"Graph nodes: {data.num_nodes:,}")
    print(f"Graph edges: {data.num_edges:,}")
    
    best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
    best_f1 = results[best_model_name]['f1']
    print(f"\nBest performing model: {best_model_name}")
    print(f"Best F1 Score: {best_f1:.4f}")
    
    print("\nAll results saved to 'results/' directory")
    print("All plots saved to 'plots/' directory")
    print("All models saved to 'models/' directory")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
        print("\nExecution completed successfully!")
    except Exception as e:
        print(f"\nExecution failed with error: {str(e)}")
        import traceback
        traceback.print_exc()