# GNN for Anomaly Detection in Financial Transactions

This repository implements Graph Neural Networks (GNN) for detecting anomalies and fraudulent activities in financial transaction data. The project leverages the power of Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) to model complex relationships between transactions and identify suspicious patterns that might indicate fraud.

## Overview

Financial fraud detection is a critical application of machine learning that helps protect individuals and institutions from significant financial losses. Traditional methods often struggle with the complex, interconnected nature of financial transactions. This project addresses this challenge by using graph-based deep learning approaches that can effectively capture the relationships between entities in transaction networks.

The implementation compares the performance of two popular GNN architectures:
- **Graph Convolutional Networks (GCN)**: A spectral-based approach that leverages the graph Laplacian
- **Graph Attention Networks (GAT)**: An attention-based approach that dynamically weights node neighborhoods

## Repository Structure

```
GNN-for-Anomaly-Detection-in-Financial-Transactions/
├── Codes/                  # Source code for data processing and model implementation
│   ├── 1/                  # Code module 1
│   ├── 2/                  # Code module 2
│   ├── 3/                  # Code module 3
│   └── EDA.py              # Exploratory Data Analysis script
├── Figures/                # Visualizations and result plots
│   ├── Full Comparision.jpeg             # Comparison of model performance metrics
│   ├── GAT Confusion Matrix.jpeg         # Confusion matrix for GAT model
│   ├── GAT Model Summary.jpeg            # Architecture summary of GAT model
│   ├── GAT Perdection-Recall.jpeg        # Precision-Recall curve for GAT model
│   ├── GAT Training Loss.jpeg            # Training loss curve for GAT model
│   ├── GCN Confusion Matrix.jpeg         # Confusion matrix for GCN model
│   ├── GCN Model Summary.jpeg            # Architecture summary of GCN model
│   ├── GCN Predection-Recall.jpeg        # Precision-Recall curve for GCN model
│   ├── GCN Training Loss.jpeg            # Training loss curve for GCN model
│   ├── Model Comparision F1-Score.jpeg   # F1-Score comparison between models
│   ├── amount_by_transaction_type.png    # Distribution of transaction amounts by type
│   ├── amount_distribution.png           # Overall distribution of transaction amounts
│   ├── amount_fraud_vs_nonfraud.png      # Comparison of amounts in fraudulent vs. legitimate transactions
│   └── balance_analysis.png              # Analysis of account balances
└── Models/                 # Trained model files and model-related code
```

## Requirements

To run the code in this repository, you'll need the following dependencies:

```
pandas
numpy
matplotlib.pyplot
seaborn
plotly
plotly.graph_objects
sklearn
datetime
scipy.stats
torch
torch_geometric
networkx
```

## Dataset

The project uses a financial transaction dataset ((https://www.kaggle.com/datasets/ealaxi/paysim1)[Paysim]). This dataset contains various features of financial transactions, including:

- Transaction amounts
- Transaction types
- Account balances
- Timestamps
- Fraud labels (binary classification: fraudulent or legitimate)

## Exploratory Data Analysis

The repository includes comprehensive exploratory data analysis (EDA) to understand the patterns and distributions in the financial transaction data. The `EDA.py` script performs:

1. Initial data exploration and preview
2. Statistical analysis of transaction amounts
3. Visualization of transaction distributions
4. Analysis of fraudulent vs. legitimate transaction characteristics
5. Feature correlation analysis

## Model Implementation

The implementation includes two Graph Neural Network architectures:

### Graph Convolutional Network (GCN)

The GCN model processes transaction data as a graph where:
- Nodes represent entities (accounts, users)
- Edges represent transactions between entities
- Node features include transaction characteristics and account information

The model uses multiple GCN layers to aggregate information from neighboring nodes, followed by fully connected layers for classification.

### Graph Attention Network (GAT)

The GAT model extends the GCN approach by incorporating attention mechanisms that dynamically weight the importance of different neighbors during the message-passing phase. This allows the model to focus on the most relevant connections for fraud detection.


## Usage

1. Clone the repository:
```
git clone https://github.com/FatimaALzahrani/GNN-for-Anomaly-Detection-in-Financial-Transactions.git
cd GNN-for-Anomaly-Detection-in-Financial-Transactions
```

2. Install the required dependencies:
```
pip install -r requirements.txt  
```

3. Run the exploratory data analysis:
```
python Codes/EDA.py
```

4. Train and evaluate the models:
```
python Codes/train.py  
```

## Future Work

Potential improvements and extensions to this project:

1. Incorporate temporal information to capture evolving fraud patterns
2. Experiment with more complex GNN architectures like GraphSAGE or Graph Transformers
3. Implement ensemble methods combining multiple GNN models
4. Explore semi-supervised and unsupervised approaches for anomaly detection
5. Develop real-time fraud detection capabilities

## License

This project is available for academic and research purposes.
