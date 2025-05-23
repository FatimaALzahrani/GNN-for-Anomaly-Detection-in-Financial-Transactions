import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("Loading and preparing data...")

# ========================================================
# Phase 1: Efficient Data Loading and Sampling
# ========================================================

def load_and_sample_data(file_path, sample_size=50000, random_state=42):
    df = pd.read_csv(file_path, nrows=sample_size)
    print(f"Loaded sample of {len(df)} transactions")
    
    print(f"Fraud transactions: {df['isFraud'].sum()} ({df['isFraud'].mean()*100:.4f}%)")
    
    return df

# ========================================================
# Phase 2: Focused Feature Engineering for Fraud Detection
# ========================================================

def engineer_features(df):
    
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    
    df['amount_to_oldbalanceOrg_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1) 
    
    df['balanceOrig_change_ratio'] = (df['newbalanceOrig'] - df['oldbalanceOrg']) / (df['oldbalanceOrg'] + 1)
    df['balanceDest_change_ratio'] = (df['newbalanceDest'] - df['oldbalanceDest']) / (df['oldbalanceDest'] + 1)
    
    return df

# ========================================================
# Phase 3: Efficient Graph Construction
# ========================================================

def build_transaction_graph(df):
    print("Building transaction graph...")
    G = nx.DiGraph()
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding edges"):
        sender = row['nameOrig']
        recipient = row['nameDest']
        G.add_node(sender)
        G.add_node(recipient)
        G.add_edge(
            sender, recipient,
            amount=row['amount'],
            isFraud=row['isFraud'],
            errorBalanceOrig=row['errorBalanceOrig'],
            errorBalanceDest=row['errorBalanceDest']
        )
    
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    fraud_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('isFraud') == 1]
    fraud_ratio = len(fraud_edges) / G.number_of_edges()
    print(f"Fraudulent Edges: {len(fraud_edges)} ({fraud_ratio:.4%})")
    
    fraud_nodes = set()
    for u, v in fraud_edges:
        fraud_nodes.add(u)
        fraud_nodes.add(v)
    print(f"Nodes involved in fraud: {len(fraud_nodes)}")
    
    return G, fraud_nodes

# ========================================================
# Phase 4: Convert to PyTorch Geometric Format with Balanced Sampling
# ========================================================

def prepare_pytorch_geometric_data(G, fraud_nodes, balance_ratio=1.0):
    print("Converting to PyTorch Geometric format with balanced sampling...")
    
    accounts = list(G.nodes())
    account_to_id = {acc: idx for idx, acc in enumerate(accounts)}
    
    edge_index = [[], []]
    edge_attr = []
    
    for u, v, data in G.edges(data=True):
        if u in account_to_id and v in account_to_id:
            edge_index[0].append(account_to_id[u])
            edge_index[1].append(account_to_id[v])
            
            edge_attr.append([
                data.get('amount', 0),
                data.get('errorBalanceOrig', 0),
                data.get('errorBalanceDest', 0),
                int(data.get('isFraud', 0))
            ])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    num_nodes = len(accounts)
    x_data = torch.zeros((num_nodes, 4), dtype=torch.float) 
    
    node_features = {}
    for u, v, data in G.edges(data=True):
        if u not in node_features:
            node_features[u] = {'amount': [], 'errorBalanceOrig': [], 'errorBalanceDest': [], 'isFraud': []}
        
        node_features[u]['amount'].append(data.get('amount', 0))
        node_features[u]['errorBalanceOrig'].append(data.get('errorBalanceOrig', 0))
        node_features[u]['errorBalanceDest'].append(data.get('errorBalanceDest', 0))
        node_features[u]['isFraud'].append(int(data.get('isFraud', 0)))
        
        if v not in node_features:
            node_features[v] = {'amount': [], 'errorBalanceOrig': [], 'errorBalanceDest': [], 'isFraud': []}
        
        node_features[v]['amount'].append(data.get('amount', 0))
        node_features[v]['errorBalanceOrig'].append(data.get('errorBalanceOrig', 0))
        node_features[v]['errorBalanceDest'].append(data.get('errorBalanceDest', 0))
        node_features[v]['isFraud'].append(int(data.get('isFraud', 0)))
    
    for node, features in node_features.items():
        if node in account_to_id:
            idx = account_to_id[node]
            if features['amount']: 
                x_data[idx, 0] = np.mean(features['amount'])
                x_data[idx, 1] = np.mean(features['errorBalanceOrig'])
                x_data[idx, 2] = np.mean(features['errorBalanceDest'])
                x_data[idx, 3] = 1 if any(features['isFraud']) else 0
    
    y_data = torch.zeros(num_nodes, dtype=torch.long)
    for node in fraud_nodes:
        if node in account_to_id:
            idx = account_to_id[node]
            y_data[idx] = 1
    
    fraud_indices = (y_data == 1).nonzero(as_tuple=True)[0].numpy()
    normal_indices = (y_data == 0).nonzero(as_tuple=True)[0].numpy()
    
    n_fraud = len(fraud_indices)
    n_normal = len(normal_indices)
    
    n_fraud_train = int(0.7 * n_fraud) 
    n_normal_train = int(n_fraud_train * balance_ratio)
    
    np.random.seed(42)
    fraud_train_idx = np.random.choice(fraud_indices, n_fraud_train, replace=False)
    normal_train_idx = np.random.choice(normal_indices, min(n_normal_train, n_normal), replace=False)
    
    train_idx = np.concatenate([fraud_train_idx, normal_train_idx])
    
    fraud_test_idx = np.setdiff1d(fraud_indices, fraud_train_idx)
    normal_test_idx = np.setdiff1d(normal_indices, normal_train_idx)
    test_idx = np.concatenate([fraud_test_idx, normal_test_idx[:min(len(normal_test_idx), len(fraud_test_idx)*10)]])
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    
    data = Data(x=x_data, edge_index=edge_index, edge_attr=edge_attr, y=y_data)
    data.train_mask = train_mask
    data.test_mask = test_mask
    
    print(f"Data prepared: {data}")
    print(f"Train nodes: {train_mask.sum().item()} (Fraud: {y_data[train_mask].sum().item()})")
    print(f"Test nodes: {test_mask.sum().item()} (Fraud: {y_data[test_mask].sum().item()})")
    
    return data

# ========================================================
# Phase 5: Efficient GNN Models for Imbalanced Data
# ========================================================

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal Loss for handling class imbalance
    """
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)
    loss = alpha * (1 - pt) ** gamma * ce_loss
    return loss.mean()

class ImbalancedGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super(ImbalancedGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class ImbalancedGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2, dropout=0.3):
        super(ImbalancedGAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# ========================================================
# Phase 6: Training and Evaluation Functions
# ========================================================

def train(model, optimizer, data, use_focal_loss=True):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    
    if use_focal_loss:
        loss = focal_loss(out[data.train_mask], data.y[data.train_mask])
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

def evaluate(model, data):
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
        
        return acc, f1, auc_score, cm, y_true, y_pred, prob

# ========================================================
# Phase 7: Visualization Functions
# ========================================================

def plot_training_curve(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()

def plot_confusion_matrix(cm, classes=['Normal', 'Fraud']):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_precision_recall_curve(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig('precision_recall_curve.png')
    plt.close()

# ========================================================
# Phase 8: Main Execution
# ========================================================

def main():
    df = load_and_sample_data('Graph/paysim_full.csv', sample_size=50000)
    
    df = engineer_features(df)
    
    G, fraud_nodes = build_transaction_graph(df)

    data = prepare_pytorch_geometric_data(G, fraud_nodes, balance_ratio=5.0)
    
    models = {
        'GCN': ImbalancedGCN(in_channels=data.num_node_features, hidden_channels=32, out_channels=2),
        'GAT': ImbalancedGAT(in_channels=data.num_node_features, hidden_channels=16, out_channels=2)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        losses = []
        best_f1 = 0
        patience = 10
        counter = 0
        
        for epoch in range(1, 101):
            loss = train(model, optimizer, data, use_focal_loss=True)
            losses.append(loss)
            
            if epoch % 5 == 0 or epoch == 1:
                acc, f1, auc, cm, y_true, y_pred, y_prob = evaluate(model, data)
                print(f"Epoch {epoch:03d}: Loss={loss:.4f}, Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
                
                if f1 > best_f1:
                    best_f1 = f1
                    counter = 0
                    torch.save(model.state_dict(), f"{name}_best_model.pt")
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
        
        model.load_state_dict(torch.load(f"{name}_best_model.pt"))
        acc, f1, auc, cm, y_true, y_pred, y_prob = evaluate(model, data)
        
        results[name] = {
            'accuracy': acc,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
        plot_training_curve(losses)
        
        plot_confusion_matrix(cm)
        
        if len(np.unique(y_true)) > 1:
            plot_precision_recall_curve(y_true, y_prob)
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
    
    print("\nModel Comparison:")
    for name, metrics in results.items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            if metric not in ['confusion_matrix', 'y_true', 'y_pred', 'y_prob']:
                print(f"  {metric}: {value:.4f}")
    
    best_model = max(results.items(), key=lambda x: x[1]['f1'])[0]
    print(f"\nBest model: {best_model} with F1 score: {results[best_model]['f1']:.4f}")

if __name__ == "__main__":
    main()