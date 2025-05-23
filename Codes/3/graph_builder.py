import networkx as nx
import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm
from config import DATASET_CONFIG

def build_transaction_graph(df):
    """
    Build transaction graph from dataframe
    
    Args:
        df: Transaction dataframe
    
    Returns:
        tuple: (NetworkX graph, set of fraud nodes)
    """
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

def prepare_pytorch_geometric_data(G, fraud_nodes, balance_ratio=None):
    """
    Convert NetworkX graph to PyTorch Geometric format with balanced sampling
    
    Args:
        G: NetworkX graph
        fraud_nodes: Set of nodes involved in fraud
        balance_ratio: Ratio of normal to fraud samples
    
    Returns:
        torch_geometric.data.Data: PyTorch Geometric data object
    """
    print("Converting to PyTorch Geometric format with balanced sampling...")
    
    balance_ratio = balance_ratio or DATASET_CONFIG['balance_ratio']
    
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
    
    n_fraud_train = int(DATASET_CONFIG.get('train_split', 0.7) * n_fraud)  
    n_normal_train = int(n_fraud_train * balance_ratio) 
    
    np.random.seed(DATASET_CONFIG['random_state'])
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