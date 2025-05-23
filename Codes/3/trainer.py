import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
from models import focal_loss
from config import TRAINING_CONFIG
import os

def train_epoch(model, optimizer, data, use_focal_loss=True):
    """
    Train model for one epoch
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        data: PyTorch Geometric data
        use_focal_loss: Whether to use focal loss
    
    Returns:
        float: Training loss
    """
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    
    if use_focal_loss:
        loss = focal_loss(
            out[data.train_mask], 
            data.y[data.train_mask],
            alpha=TRAINING_CONFIG['focal_loss_alpha'],
            gamma=TRAINING_CONFIG['focal_loss_gamma']
        )
    else:
        n_samples = data.train_mask.sum().item()
        n_fraud = data.y[data.train_mask].sum().item()
        fraud_weight = (n_samples / (2 * n_fraud)) if n_fraud > 0 else 1.0
        normal_weight = (n_samples / (2 * (n_samples - n_fraud))) if n_samples > n_fraud else 1.0
        weights = torch.tensor([normal_weight, fraud_weight], dtype=torch.float)
        
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=weights)
    
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_model(model, data):
    """
    Evaluate model performance
    
    Args:
        model: PyTorch model
        data: PyTorch Geometric data
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr)
        pred = out.argmax(dim=1)
        
        y_true = data.y[data.test_mask].cpu().numpy()
        y_pred = pred[data.test_mask].cpu().numpy()
        prob = F.softmax(out[data.test_mask], dim=1)[:, 1].detach().cpu().numpy()
        
        acc = (y_pred == y_true).mean()
        f1 = f1_score(y_true, y_pred)
        
        if len(np.unique(y_true)) > 1:
            auc_score = roc_auc_score(y_true, prob)
        else:
            auc_score = float('nan')
        
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'auc': auc_score,
            'confusion_matrix': cm,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': prob
        }

def train_model(model, data, model_name, save_dir='models/'):
    """
    Complete training pipeline for a model
    
    Args:
        model: PyTorch model
        data: PyTorch Geometric data
        model_name: Name for saving the model
        save_dir: Directory to save models
    
    Returns:
        tuple: (trained model, training losses, best metrics)
    """
    print(f"\nTraining {model_name} model...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    from models import get_model_config
    model_type = model_name.lower()
    config = get_model_config(model_type)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.get('learning_rate', 0.01),
        weight_decay=config.get('weight_decay', 5e-4)
    )
    
    losses = []
    best_f1 = 0
    patience = TRAINING_CONFIG['patience']
    counter = 0
    best_metrics = {}
    
    for epoch in range(1, TRAINING_CONFIG['max_epochs'] + 1):
        loss = train_epoch(model, optimizer, data, TRAINING_CONFIG['use_focal_loss'])
        losses.append(loss)
        
        if epoch % TRAINING_CONFIG['validation_frequency'] == 0 or epoch == 1:
            metrics = evaluate_model(model, data)
            acc, f1, auc = metrics['accuracy'], metrics['f1'], metrics['auc']
            print(f"Epoch {epoch:03d}: Loss={loss:.4f}, Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                counter = 0
                best_metrics = metrics.copy()
                torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_best_model.pt"))
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    model.load_state_dict(torch.load(os.path.join(save_dir, f"{model_name}_best_model.pt")))
    final_metrics = evaluate_model(model, data)
    
    return model, losses, final_metrics

def print_classification_report(metrics, model_name):
    """
    Print detailed classification report
    
    Args:
        metrics: Dictionary containing evaluation metrics
        model_name: Name of the model
    """
    print(f"\n{model_name} Classification Report:")
    print("="*50)
    print(classification_report(metrics['y_true'], metrics['y_pred']))
    
    print(f"\nDetailed Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"AUC Score: {metrics['auc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(metrics['confusion_matrix'])