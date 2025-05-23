import pandas as pd
import numpy as np
from config import DATASET_CONFIG

def load_and_sample_data(file_path=None, sample_size=None, random_state=None):
    """
    Load and sample data from CSV file
    
    Args:
        file_path: Path to the CSV file
        sample_size: Number of rows to sample
        random_state: Random seed for reproducibility
    
    Returns:
        pandas.DataFrame: Sampled dataset
    """
    file_path = file_path or DATASET_CONFIG['file_path']
    sample_size = sample_size or DATASET_CONFIG['sample_size']
    random_state = random_state or DATASET_CONFIG['random_state']
    
    df = pd.read_csv(file_path, nrows=sample_size)
    print(f"Loaded sample of {len(df)} transactions")
    
    print(f"Fraud transactions: {df['isFraud'].sum()} ({df['isFraud'].mean()*100:.4f}%)")
    
    return df

def engineer_features(df):
    """
    Engineer features for fraud detection
    
    Args:
        df: Input dataframe
    
    Returns:
        pandas.DataFrame: Dataframe with engineered features
    """
    
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    
    df['amount_to_oldbalanceOrg_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)  
    
    df['balanceOrig_change_ratio'] = (df['newbalanceOrig'] - df['oldbalanceOrg']) / (df['oldbalanceOrg'] + 1)
    df['balanceDest_change_ratio'] = (df['newbalanceDest'] - df['oldbalanceDest']) / (df['oldbalanceDest'] + 1)
    
    return df

def prepare_data(file_path=None, sample_size=None, random_state=None):
    """
    Complete data preparation pipeline
    
    Args:
        file_path: Path to the CSV file
        sample_size: Number of rows to sample
        random_state: Random seed for reproducibility
    
    Returns:
        pandas.DataFrame: Prepared dataset
    """
    print("Loading and preparing data...")
    
    df = load_and_sample_data(file_path, sample_size, random_state)
    
    df = engineer_features(df)
    
    return df