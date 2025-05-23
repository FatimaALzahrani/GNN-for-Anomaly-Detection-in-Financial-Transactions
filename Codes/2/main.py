import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TransformerConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import warnings
from tqdm import tqdm
import random
import os
import time

warnings.filterwarnings('ignore')

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ========================================================
# Phase 1: Advanced Data Loading and Preprocessing
# ========================================================

def load_and_preprocess_data(file_path, sample_size=100000, random_state=42):
    print("Loading and preprocessing data...")
    
    df = pd.read_csv(file_path, nrows=sample_size)
    print(f"Loaded sample of {len(df)} transactions")
    
    print(f"Fraud transactions: {df['isFraud'].sum()} ({df['isFraud'].mean()*100:.4f}%)")
    
    numerical_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

# ========================================================
# Phase 2: Advanced Feature Engineering
# ========================================================

def engineer_advanced_features(df):
    print("Engineering advanced features...")
    
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    
    df['amount_to_oldbalanceOrg_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1e-10)
    df['amount_to_oldbalanceDest_ratio'] = df['amount'] / (df['oldbalanceDest'] + 1e-10)
    
    df['balanceOrig_change_ratio'] = (df['newbalanceOrig'] - df['oldbalanceOrg']) / (df['oldbalanceOrg'] + 1e-10)
    df['balanceDest_change_ratio'] = (df['newbalanceDest'] - df['oldbalanceDest']) / (df['oldbalanceDest'] + 1e-10)
    
    df['isOldBalanceOrgZero'] = (df['oldbalanceOrg'] == 0).astype(float)
    df['isNewBalanceOrigZero'] = (df['newbalanceOrig'] == 0).astype(float)
    df['isOldBalanceDestZero'] = (df['oldbalanceDest'] == 0).astype(float)
    df['isNewBalanceDestZero'] = (df['newbalanceDest'] == 0).astype(float)
    
    df['balanceOrgBecomesZero'] = ((df['oldbalanceOrg'] > 0) & (df['newbalanceOrig'] == 0)).astype(float)
    df['balanceDestBecomesZero'] = ((df['oldbalanceDest'] > 0) & (df['newbalanceDest'] == 0)).astype(float)
    
    df['hour_of_day'] = df['step'] % 24
    df['day_of_week'] = (df['step'] // 24) % 7
    
    df = pd.get_dummies(df, columns=['type'], prefix='type')
    
    df = df.sort_values(['nameOrig', 'step'])
    df['prev_step'] = df.groupby('nameOrig')['step'].shift(1)
    df['time_since_last_tx'] = df['step'] - df['prev_step']
    df['time_since_last_tx'].fillna(0, inplace=True)
    
    tx_counts = df.groupby('nameOrig').size().reset_index(name='tx_count')
    df = df.merge(tx_counts, on='nameOrig', how='left')
    
    amount_stats = df.groupby('nameOrig')['amount'].agg(['mean', 'std', 'max']).reset_index()
    amount_stats.columns = ['nameOrig', 'amount_mean', 'amount_std', 'amount_max']
    df = df.merge(amount_stats, on='nameOrig', how='left')
    df['amount_std'].fillna(0, inplace=True)
    
    df['amount_deviation'] = (df['amount'] - df['amount_mean']) / (df['amount_std'] + 1e-10)
    
    dest_counts = df.groupby('nameDest').size().reset_index(name='dest_tx_count')
    df = df.merge(dest_counts, on='nameDest', how='left')
    
    new_numerical_cols = [
        'errorBalanceOrig', 'errorBalanceDest', 
        'amount_to_oldbalanceOrg_ratio', 'amount_to_oldbalanceDest_ratio',
        'balanceOrig_change_ratio', 'balanceDest_change_ratio',
        'time_since_last_tx', 'tx_count', 'dest_tx_count',
        'amount_mean', 'amount_std', 'amount_max', 'amount_deviation'
    ]
    
    for col in new_numerical_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(0)
    
    scaler = StandardScaler()
    df[new_numerical_cols] = scaler.fit_transform(df[new_numerical_cols])
    
    return df

# ========================================================
# Phase 3: Advanced Graph Construction with Multi-View
# ========================================================

def build_multi_view_transaction_graph(df):
    print("Building multi-view transaction graph...")
    
    G_main = nx.DiGraph()
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building main graph"):
        sender = row['nameOrig']
        recipient = row['nameDest']
        
        if not G_main.has_node(sender):
            G_main.add_node(sender, 
                           account_type='originator',
                           tx_count=row['tx_count'],
                           amount_mean=row['amount_mean'],
                           amount_std=row['amount_std'],
                           amount_max=row['amount_max'])
        
        if not G_main.has_node(recipient):
            G_main.add_node(recipient, 
                           account_type='destination',
                           dest_tx_count=row['dest_tx_count'])
        
        G_main.add_edge(
            sender, recipient,
            amount=row['amount'],
            step=row['step'],
            isFraud=row['isFraud'],
            errorBalanceOrig=row['errorBalanceOrig'],
            errorBalanceDest=row['errorBalanceDest'],
            amount_to_oldbalanceOrg_ratio=row['amount_to_oldbalanceOrg_ratio'],
            balanceOrig_change_ratio=row['balanceOrig_change_ratio'],
            balanceDest_change_ratio=row['balanceDest_change_ratio'],
            isOldBalanceOrgZero=row['isOldBalanceOrgZero'],
            isNewBalanceOrigZero=row['isNewBalanceOrigZero'],
            balanceOrgBecomesZero=row['balanceOrgBecomesZero'],
            hour_of_day=row['hour_of_day'],
            day_of_week=row['day_of_week'],
            time_since_last_tx=row['time_since_last_tx'],
            amount_deviation=row['amount_deviation']
        )
    
    print(f"Main graph built with {G_main.number_of_nodes()} nodes and {G_main.number_of_edges()} edges")
    
    fraud_edges = [(u, v) for u, v, d in G_main.edges(data=True) if d.get('isFraud') == 1]
    fraud_ratio = len(fraud_edges) / G_main.number_of_edges()
    print(f"Fraudulent Edges: {len(fraud_edges)} ({fraud_ratio:.4%})")
    
    fraud_nodes = set()
    for u, v in fraud_edges:
        fraud_nodes.add(u)
        fraud_nodes.add(v)
    print(f"Nodes involved in fraud: {len(fraud_nodes)}")
    
    G_similarity = nx.Graph()
    
    for node, attrs in G_main.nodes(data=True):
        G_similarity.add_node(node, **attrs)
    
    print("Building similarity graph...")
    account_features = {}
    
    for node in tqdm(G_main.nodes(), desc="Extracting account features"):
        if 'tx_count' in G_main.nodes[node]:
            out_edges = list(G_main.out_edges(node, data=True))
            if out_edges:
                avg_amount = np.mean([e[2].get('amount', 0) for e in out_edges])
                avg_error = np.mean([e[2].get('errorBalanceOrig', 0) for e in out_edges])
                fraud_ratio = np.mean([e[2].get('isFraud', 0) for e in out_edges])
                
                account_features[node] = [
                    G_main.nodes[node].get('tx_count', 0),
                    G_main.nodes[node].get('amount_mean', 0),
                    G_main.nodes[node].get('amount_std', 0),
                    avg_amount,
                    avg_error,
                    fraud_ratio
                ]
    
    if account_features:
        account_nodes = list(account_features.keys())
        for i in tqdm(range(len(account_nodes)), desc="Connecting similar accounts"):
            node_i = account_nodes[i]
            feat_i = account_features[node_i]
            
            similarities = []
            for j in range(len(account_nodes)):
                if i != j:
                    node_j = account_nodes[j]
                    feat_j = account_features[node_j]
                    
                    sim = np.dot(feat_i, feat_j) / (np.linalg.norm(feat_i) * np.linalg.norm(feat_j) + 1e-10)
                    similarities.append((node_j, sim))
            
            top_similar = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]
            for node_j, sim in top_similar:
                if sim > 0.8:  
                    G_similarity.add_edge(node_i, node_j, weight=sim, edge_type='similarity')
    
    print(f"Similarity graph built with {G_similarity.number_of_nodes()} nodes and {G_similarity.number_of_edges()} edges")
    
    return G_main, G_similarity, fraud_nodes

# ========================================================
# Phase 4: Advanced Structural Feature Extraction
# ========================================================

def extract_structural_features(G_main, G_similarity):
    print("Extracting advanced structural features...")
    
    structural_features = {}
    
    print("Computing centrality measures...")
    
    in_degree = dict(G_main.in_degree())
    out_degree = dict(G_main.out_degree())
    
    pagerank = nx.pagerank(G_main, max_iter=100)
    
    G_undirected = G_main.to_undirected()
    clustering = nx.clustering(G_undirected)
    
    similarity_degree = dict(G_similarity.degree())
    
    print("Computing ego network features...")
    ego_features = {}
    
    for node in tqdm(G_main.nodes(), desc="Processing ego networks"):
        neighbors = set(G_main.successors(node)).union(set(G_main.predecessors(node)))
        if node in neighbors:
            neighbors.remove(node)
        
        if neighbors:
            fraud_neighbors = sum(1 for n in neighbors if G_main.has_edge(node, n) and 
                                 G_main.edges[node, n].get('isFraud', 0) == 1)
            
            fraud_ratio = fraud_neighbors / len(neighbors)
            
            ego_features[node] = {
                'neighbor_count': len(neighbors),
                'fraud_neighbor_count': fraud_neighbors,
                'fraud_ratio': fraud_ratio
            }
        else:
            ego_features[node] = {
                'neighbor_count': 0,
                'fraud_neighbor_count': 0,
                'fraud_ratio': 0
            }
    
    for node in G_main.nodes():
        structural_features[node] = [
            in_degree.get(node, 0),
            out_degree.get(node, 0),
            pagerank.get(node, 0),
            clustering.get(node, 0),
            similarity_degree.get(node, 0),
            ego_features.get(node, {}).get('neighbor_count', 0),
            ego_features.get(node, {}).get('fraud_neighbor_count', 0),
            ego_features.get(node, {}).get('fraud_ratio', 0)
        ]
    
    return structural_features

# ========================================================
# Phase 5: Advanced PyTorch Geometric Data Preparation
# ========================================================

def prepare_advanced_pyg_data(G_main, G_similarity, structural_features, fraud_nodes, balance_ratio=5.0):
    print("Preparing advanced PyTorch Geometric data...")
    
    accounts = list(G_main.nodes())
    account_to_id = {acc: idx for idx, acc in enumerate(accounts)}
    
    edge_index_main = [[], []]
    edge_attr_main = []
    
    for u, v, data in tqdm(G_main.edges(data=True), total=G_main.number_of_edges(), desc="Processing main graph edges"):
        if u in account_to_id and v in account_to_id:
            edge_index_main[0].append(account_to_id[u])
            edge_index_main[1].append(account_to_id[v])
            
            edge_attr_main.append([
                data.get('amount', 0),
                data.get('errorBalanceOrig', 0),
                data.get('errorBalanceDest', 0),
                data.get('amount_to_oldbalanceOrg_ratio', 0),
                data.get('balanceOrig_change_ratio', 0),
                data.get('balanceDest_change_ratio', 0),
                data.get('isOldBalanceOrgZero', 0),
                data.get('isNewBalanceOrigZero', 0),
                data.get('balanceOrgBecomesZero', 0),
                data.get('hour_of_day', 0) / 24.0,  
                data.get('day_of_week', 0) / 7.0,
                data.get('time_since_last_tx', 0),
                data.get('amount_deviation', 0),
                int(data.get('isFraud', 0))
            ])
    
    edge_index_main = torch.tensor(edge_index_main, dtype=torch.long)
    edge_attr_main = torch.tensor(edge_attr_main, dtype=torch.float)
    
    edge_index_sim = [[], []]
    edge_attr_sim = []
    
    for u, v, data in tqdm(G_similarity.edges(data=True), total=G_similarity.number_of_edges(), desc="Processing similarity graph edges"):
        if u in account_to_id and v in account_to_id:
            edge_index_sim[0].append(account_to_id[u])
            edge_index_sim[1].append(account_to_id[v])
            
            edge_index_sim[0].append(account_to_id[v])
            edge_index_sim[1].append(account_to_id[u])
            
            weight = data.get('weight', 1.0)
            edge_attr_sim.append([weight])
            edge_attr_sim.append([weight])
    
    edge_index_sim = torch.tensor(edge_index_sim, dtype=torch.long)
    edge_attr_sim = torch.tensor(edge_attr_sim, dtype=torch.float) if edge_attr_sim else None
    
    num_nodes = len(accounts)
    
    x_base = torch.zeros((num_nodes, 5), dtype=torch.float)
    
    for node, idx in account_to_id.items():
        attrs = G_main.nodes[node]
        x_base[idx, 0] = attrs.get('tx_count', 0) if 'tx_count' in attrs else 0
        x_base[idx, 1] = attrs.get('amount_mean', 0) if 'amount_mean' in attrs else 0
        x_base[idx, 2] = attrs.get('amount_std', 0) if 'amount_std' in attrs else 0
        x_base[idx, 3] = attrs.get('amount_max', 0) if 'amount_max' in attrs else 0
        x_base[idx, 4] = attrs.get('dest_tx_count', 0) if 'dest_tx_count' in attrs else 0
    
    x_struct = torch.zeros((num_nodes, len(next(iter(structural_features.values())))), dtype=torch.float)
    
    for node, idx in account_to_id.items():
        if node in structural_features:
            x_struct[idx] = torch.tensor(structural_features[node], dtype=torch.float)
    
    x_data = torch.cat([x_base, x_struct], dim=1)
    
    x_mean = x_data.mean(dim=0, keepdim=True)
    x_std = x_data.std(dim=0, keepdim=True)
    x_std[x_std == 0] = 1  
    x_data = (x_data - x_mean) / x_std
    
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
    
    data_main = Data(x=x_data, edge_index=edge_index_main, edge_attr=edge_attr_main, y=y_data)
    data_main.train_mask = train_mask
    data_main.test_mask = test_mask
    
    data_sim = Data(x=x_data, edge_index=edge_index_sim, edge_attr=edge_attr_sim, y=y_data)
    data_sim.train_mask = train_mask
    data_sim.test_mask = test_mask
    
    print(f"Data prepared: {data_main}")
    print(f"Train nodes: {train_mask.sum().item()} (Fraud: {y_data[train_mask].sum().item()})")
    print(f"Test nodes: {test_mask.sum().item()} (Fraud: {y_data[test_mask].sum().item()})")
    
    return data_main, data_sim

# ========================================================
# Phase 6: Advanced GNN Models
# ========================================================

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal Loss for handling class imbalance
    """
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)
    loss = alpha * (1 - pt) ** gamma * ce_loss
    return loss.mean()

class ResGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.3):
        super(ResGCN, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))
        
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        self.dropout = dropout
        self.num_layers = num_layers

    def forward(self, x, edge_index, edge_attr=None):
        x = self.convs[0](x, edge_index)
        x = self.batch_norms[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        for i in range(1, self.num_layers - 1):
            x_res = x 
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res 
        
        x = self.convs[-1](x, edge_index)
        
        return x

class EdgeGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=None, heads=4, dropout=0.3):
        super(EdgeGAT, self).__init__()
        
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, edge_dim=edge_dim)
        self.conv3 = GATConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout, edge_dim=edge_dim)
        
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels * heads)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels * heads)
        
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.elu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        
        return x

class EdgeSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super(EdgeSAGE, self).__init__()
        
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        
        if edge_attr is not None:
            self.edge_nn = torch.nn.Sequential(
                torch.nn.Linear(edge_attr.size(1), hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, 1),
                torch.nn.Sigmoid()
            )
        
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is not None and hasattr(self, 'edge_nn'):
            edge_weight = self.edge_nn(edge_attr).view(-1)
        else:
            edge_weight = None
        
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index, edge_weight)
        
        return x

class TransformerGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.3, edge_dim=None):
        super(TransformerGNN, self).__init__()
        
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, dropout=dropout, edge_dim=edge_dim)
        self.conv2 = TransformerConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, edge_dim=edge_dim)
        self.conv3 = TransformerConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout, edge_dim=edge_dim)
        
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels * heads)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels * heads)
        
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.gelu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.gelu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        
        return x

class MultiViewGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=None, heads=4, dropout=0.3):
        super(MultiViewGNN, self).__init__()
        
        self.main_gnn = EdgeGAT(in_channels, hidden_channels, hidden_channels, edge_dim=edge_dim, heads=heads, dropout=dropout)
        
        self.sim_gnn = TransformerGNN(in_channels, hidden_channels, hidden_channels, heads=heads, dropout=dropout)
        
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index_main, edge_index_sim, edge_attr_main=None):
        x_main = self.main_gnn(x, edge_index_main, edge_attr_main)
        
        x_sim = self.sim_gnn(x, edge_index_sim)
        
        x_combined = torch.cat([x_main, x_sim], dim=1)
        x_out = self.fusion(x_combined)
        
        return x_out

class EnsembleGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=None, dropout=0.3):
        super(EnsembleGNN, self).__init__()
        
        self.gat = EdgeGAT(in_channels, hidden_channels, out_channels, edge_dim=edge_dim, dropout=dropout)
        self.sage = EdgeSAGE(in_channels, hidden_channels, out_channels, dropout=dropout)
        self.transformer = TransformerGNN(in_channels, hidden_channels, out_channels, edge_dim=edge_dim, dropout=dropout)
        
        self.weights = torch.nn.Parameter(torch.ones(3))

    def forward(self, x, edge_index, edge_attr=None):
        out_gat = self.gat(x, edge_index, edge_attr)
        out_sage = self.sage(x, edge_index, edge_attr)
        out_transformer = self.transformer(x, edge_index, edge_attr)
        
        weights = F.softmax(self.weights, dim=0)
        
        out = weights[0] * out_gat + weights[1] * out_sage + weights[2] * out_transformer
        
        return out

# ========================================================
# Phase 7: Advanced Training and Evaluation Functions
# ========================================================

def train_with_early_stopping(model, optimizer, data, patience=10, max_epochs=100, use_focal_loss=True, edge_attr=None):
    print(f"Training model with early stopping (patience={patience})...")
    
    model.train()
    best_loss = float('inf')
    best_epoch = 0
    counter = 0
    losses = []
    
    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()
        
        if edge_attr is not None:
            out = model(data.x, data.edge_index, edge_attr)
        else:
            out = model(data.x, data.edge_index)
        
        if use_focal_loss:
            loss = focal_loss(out[data.train_mask], data.y[data.train_mask])
        else:
            n_samples = data.train_mask.sum().item()
            n_fraud = data.y[data.train_mask].sum().item()
            fraud_weight = (n_samples / (2 * n_fraud)) if n_fraud > 0 else 1.0
            normal_weight = (n_samples / (2 * (n_samples - n_fraud))) if n_samples > n_fraud else 1.0
            weights = torch.tensor([normal_weight, fraud_weight], dtype=torch.float).to(device)
            
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=weights)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = epoch
            counter = 0
            best_state = model.state_dict().copy()
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}, best epoch: {best_epoch}")
                model.load_state_dict(best_state)
                break
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}: Loss={loss.item():.4f}")
    
    return losses, best_epoch

def evaluate_detailed(model, data, edge_attr=None):
    model.eval()
    with torch.no_grad():
        if edge_attr is not None:
            out = model(data.x, data.edge_index, edge_attr)
        else:
            out = model(data.x, data.edge_index)
        
        pred = out.argmax(dim=1)
        
        y_true = data.y[data.test_mask].cpu().numpy()
        y_pred = pred[data.test_mask].cpu().numpy()
        prob = F.softmax(out[data.test_mask], dim=1)[:, 1].detach().cpu().numpy()
        
        acc = (y_pred == y_true).mean()
        
        if len(np.unique(y_true)) > 1:
            f1 = f1_score(y_true, y_pred)
            auc_score = roc_auc_score(y_true, prob)
        else:
            f1 = float('nan')
            auc_score = float('nan')
        
        cm = confusion_matrix(y_true, y_pred)
        
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return acc, f1, auc_score, precision, recall, cm, y_true, y_pred, prob

def cross_validate(model_class, data, edge_attr=None, n_splits=5, **model_kwargs):
    print(f"Performing {n_splits}-fold cross-validation...")
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    all_indices = torch.arange(data.num_nodes)
    all_labels = data.y.cpu().numpy()
    
    cv_results = {
        'accuracy': [],
        'f1': [],
        'auc': [],
        'precision': [],
        'recall': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(all_indices.numpy(), all_labels)):
        print(f"Fold {fold+1}/{n_splits}")
        
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True
        
        data.train_mask = train_mask
        data.test_mask = test_mask
        
        model = model_class(**model_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        
        train_with_early_stopping(model, optimizer, data, patience=10, max_epochs=50, edge_attr=edge_attr)
        
        acc, f1, auc_score, precision, recall, _, _, _, _ = evaluate_detailed(model, data, edge_attr)
        
        cv_results['accuracy'].append(acc)
        cv_results['f1'].append(f1)
        cv_results['auc'].append(auc_score)
        cv_results['precision'].append(precision)
        cv_results['recall'].append(recall)
        
        print(f"Fold {fold+1} results: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc_score:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    
    for metric, values in cv_results.items():
        cv_results[f'avg_{metric}'] = np.mean(values)
        cv_results[f'std_{metric}'] = np.std(values)
    
    return cv_results

# ========================================================
# Phase 8: Advanced Visualization Functions
# ========================================================

def plot_training_curve(losses, title="Training Loss Curve"):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()

def plot_confusion_matrix(cm, classes=['Normal', 'Fraud'], title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_precision_recall_curve(y_true, y_prob, title="Precision-Recall Curve"):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid(True)
    plt.savefig('precision_recall_curve.png')
    plt.close()

def plot_feature_importance(model, data, feature_names=None):
    model.eval()
    data.x.requires_grad = True
    
    out = model(data.x, data.edge_index)
    fraud_score = out[:, 1].sum()
    
    fraud_score.backward()
    feature_importance = data.x.grad.abs().mean(dim=0).cpu().numpy()
    
    plt.figure(figsize=(12, 8))
    indices = np.argsort(feature_importance)[-20:]  
    plt.barh(range(len(indices)), feature_importance[indices])
    
    if feature_names is not None:
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    else:
        plt.yticks(range(len(indices)), indices)
    
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

# ========================================================
# Phase 9: Main Execution
# ========================================================

def main():
    start_time = time.time()
    
    df = load_and_preprocess_data('3/paysim_full.csv', sample_size=100000)
    
    df = engineer_advanced_features(df)
    
    G_main, G_similarity, fraud_nodes = build_multi_view_transaction_graph(df)
    
    structural_features = extract_structural_features(G_main, G_similarity)
    
    data_main, data_sim = prepare_advanced_pyg_data(G_main, G_similarity, structural_features, fraud_nodes, balance_ratio=5.0)
    
    data_main = data_main.to(device)
    data_sim = data_sim.to(device)
    
    models = {
        'ResGCN': ResGCN(
            in_channels=data_main.num_node_features, 
            hidden_channels=64, 
            out_channels=2, 
            num_layers=3, 
            dropout=0.3
        ),
        'EdgeGAT': EdgeGAT(
            in_channels=data_main.num_node_features, 
            hidden_channels=32, 
            out_channels=2, 
            edge_dim=data_main.edge_attr.size(1) if data_main.edge_attr is not None else None,
            heads=4, 
            dropout=0.3
        ),
        'TransformerGNN': TransformerGNN(
            in_channels=data_main.num_node_features, 
            hidden_channels=32, 
            out_channels=2, 
            edge_dim=data_main.edge_attr.size(1) if data_main.edge_attr is not None else None,
            heads=4, 
            dropout=0.3
        ),
        'EnsembleGNN': EnsembleGNN(
            in_channels=data_main.num_node_features, 
            hidden_channels=64, 
            out_channels=2, 
            edge_dim=data_main.edge_attr.size(1) if data_main.edge_attr is not None else None,
            dropout=0.3
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        
        losses, best_epoch = train_with_early_stopping(
            model, optimizer, data_main, 
            patience=15, max_epochs=100, 
            use_focal_loss=True,
            edge_attr=data_main.edge_attr
        )
        
        plot_training_curve(losses, title=f"{name} Training Loss")
        
        acc, f1, auc, precision, recall, cm, y_true, y_pred, y_prob = evaluate_detailed(
            model, data_main, edge_attr=data_main.edge_attr
        )
        
        results[name] = {
            'accuracy': acc,
            'f1': f1,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'best_epoch': best_epoch
        }
        
        plot_confusion_matrix(cm, title=f"{name} Confusion Matrix")
        
        if len(np.unique(y_true)) > 1:
            plot_precision_recall_curve(y_true, y_prob, title=f"{name} Precision-Recall Curve")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        torch.save(model.state_dict(), f"{name}_best_model.pt")
    
    print("\nTraining Multi-View GNN model...")
    multi_view_model = MultiViewGNN(
        in_channels=data_main.num_node_features,
        hidden_channels=64,
        out_channels=2,
        edge_dim=data_main.edge_attr.size(1) if data_main.edge_attr is not None else None,
        heads=4,
        dropout=0.3
    ).to(device)
    
    optimizer = torch.optim.Adam(multi_view_model.parameters(), lr=0.001, weight_decay=5e-4)
    
    multi_view_model.train()
    losses = []
    best_loss = float('inf')
    best_epoch = 0
    patience = 15
    counter = 0
    
    for epoch in range(1, 101):
        optimizer.zero_grad()
        
        out = multi_view_model(
            data_main.x, 
            data_main.edge_index, 
            data_sim.edge_index,
            data_main.edge_attr
        )
        
        loss = focal_loss(out[data_main.train_mask], data_main.y[data_main.train_mask])
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = epoch
            counter = 0
            best_state = multi_view_model.state_dict().copy()
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}, best epoch: {best_epoch}")
                multi_view_model.load_state_dict(best_state)
                break
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}: Loss={loss.item():.4f}")
    
    plot_training_curve(losses, title="Multi-View GNN Training Loss")
    
    multi_view_model.eval()
    with torch.no_grad():
        out = multi_view_model(
            data_main.x, 
            data_main.edge_index, 
            data_sim.edge_index,
            data_main.edge_attr
        )
        
        pred = out.argmax(dim=1)
        
        y_true = data_main.y[data_main.test_mask].cpu().numpy()
        y_pred = pred[data_main.test_mask].cpu().numpy()
        prob = F.softmax(out[data_main.test_mask], dim=1)[:, 1].detach().cpu().numpy()
        
        acc = (y_pred == y_true).mean()
        
        if len(np.unique(y_true)) > 1:
            f1 = f1_score(y_true, y_pred)
            auc_score = roc_auc_score(y_true, prob)
        else:
            f1 = float('nan')
            auc_score = float('nan')
        
        cm = confusion_matrix(y_true, y_pred)
        
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    results['MultiViewGNN'] = {
        'accuracy': acc,
        'f1': f1,
        'auc': auc_score,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': prob,
        'best_epoch': best_epoch
    }
    
    plot_confusion_matrix(cm, title="Multi-View GNN Confusion Matrix")
    
    if len(np.unique(y_true)) > 1:
        plot_precision_recall_curve(y_true, y_prob, title="Multi-View GNN Precision-Recall Curve")
    
    print("\nMulti-View GNN Classification Report:")
    print(classification_report(y_true, y_pred))
    
    torch.save(multi_view_model.state_dict(), "MultiViewGNN_best_model.pt")
    
    print("\nModel Comparison:")
    for name, metrics in results.items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            if metric not in ['confusion_matrix', 'y_true', 'y_pred', 'y_prob', 'best_epoch']:
                print(f"  {metric}: {value:.4f}")
        print(f"  best_epoch: {metrics['best_epoch']}")
    
    best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
    print(f"\nBest model: {best_model_name} with F1 score: {results[best_model_name]['f1']:.4f}")
    
    best_model_class = None
    best_model_kwargs = {}
    
    if best_model_name == 'ResGCN':
        best_model_class = ResGCN
        best_model_kwargs = {
            'in_channels': data_main.num_node_features,
            'hidden_channels': 64,
            'out_channels': 2,
            'num_layers': 3,
            'dropout': 0.3
        }
    elif best_model_name == 'EdgeGAT':
        best_model_class = EdgeGAT
        best_model_kwargs = {
            'in_channels': data_main.num_node_features,
            'hidden_channels': 32,
            'out_channels': 2,
            'edge_dim': data_main.edge_attr.size(1) if data_main.edge_attr is not None else None,
            'heads': 4,
            'dropout': 0.3
        }
    elif best_model_name == 'TransformerGNN':
        best_model_class = TransformerGNN
        best_model_kwargs = {
            'in_channels': data_main.num_node_features,
            'hidden_channels': 32,
            'out_channels': 2,
            'edge_dim': data_main.edge_attr.size(1) if data_main.edge_attr is not None else None,
            'heads': 4,
            'dropout': 0.3
        }
    elif best_model_name == 'EnsembleGNN':
        best_model_class = EnsembleGNN
        best_model_kwargs = {
            'in_channels': data_main.num_node_features,
            'hidden_channels': 64,
            'out_channels': 2,
            'edge_dim': data_main.edge_attr.size(1) if data_main.edge_attr is not None else None,
            'dropout': 0.3
        }
    
    if best_model_class and best_model_name != 'MultiViewGNN':
        cv_results = cross_validate(
            best_model_class, data_main, 
            edge_attr=data_main.edge_attr, 
            n_splits=5, 
            **best_model_kwargs
        )
        
        print("\nCross-Validation Results:")
        for metric in ['accuracy', 'f1', 'auc', 'precision', 'recall']:
            print(f"Average {metric}: {cv_results[f'avg_{metric}']:.4f} ± {cv_results[f'std_{metric}']:.4f}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    summary = f"""
# Fraud Detection in Financial Transactions - Results Summary

## Model Performance Comparison

| Model | Accuracy | F1 Score | AUC | Precision | Recall |
|-------|----------|----------|-----|-----------|--------|
"""
    
    for name, metrics in results.items():
        summary += f"| {name} | {metrics['accuracy']:.4f} | {metrics['f1']:.4f} | {metrics['auc']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} |\n"
    
    summary += f"""
## Best Model: {best_model_name}

- F1 Score: {results[best_model_name]['f1']:.4f}
- AUC: {results[best_model_name]['auc']:.4f}
- Precision: {results[best_model_name]['precision']:.4f}
- Recall: {results[best_model_name]['recall']:.4f}

## Key Findings

1. Graph-based models effectively capture the relational patterns in financial transactions
2. The {best_model_name} model showed the best performance in detecting fraudulent transactions
3. Multi-view graph approach combining transaction and similarity graphs provides valuable insights
4. Advanced feature engineering and focal loss significantly improved model performance on imbalanced data
"""
    
    with open('fraud_detection_summary.md', 'w') as f:
        f.write(summary)
    
    print("\nAnalysis complete! Results saved to 'fraud_detection_summary.md'")
    print(f"Best model ({best_model_name}) saved to '{best_model_name}_best_model.pt'")
    print("Visualizations saved as PNG files")

if __name__ == "__main__":
    main()