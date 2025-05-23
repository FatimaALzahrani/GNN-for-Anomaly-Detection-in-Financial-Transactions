import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')
pd.set_option('display.max_columns', None)

# Load the data
df = pd.read_csv('3/PS_20174392719_1491204439457_log.csv')

# 1. Initial Data Exploration
print("1. INITIAL DATA EXPLORATION")
print("-" * 50)
print("\nDataset Preview:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nDescriptive Statistics:")
print(df.describe())

missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values!")

categorical_columns = df.select_dtypes(include=['object']).columns
print("\nUnique Values Distribution for Categorical Columns:")
for col in categorical_columns:
    print(f"{col}: {df[col].nunique()} unique values")
    print(df[col].value_counts().head(10))
    print()

# 2. Fraud Analysis
print("\n2. FRAUD ANALYSIS")
print("-" * 50)

fraud_counts = df['isFraud'].value_counts()
print("\nFraud Distribution:")
print(fraud_counts)
print(f"Fraud Percentage: {fraud_counts[1] / len(df) * 100:.4f}%")

flagged_counts = df['isFlaggedFraud'].value_counts()
print("\nFlagged Transactions Distribution:")
print(flagged_counts)
print(f"Flagged Percentage: {flagged_counts[1] / len(df) * 100:.4f}%")

fraud_vs_flagged = pd.crosstab(df['isFraud'], df['isFlaggedFraud'])
print("\nComparison between Fraud and Flagged Transactions:")
print(fraud_vs_flagged)

# 3. Transaction Types Analysis
print("\n3. TRANSACTION TYPES ANALYSIS")
print("-" * 50)

transaction_counts = df['type'].value_counts()
print("\nTransaction Counts by Type:")
print(transaction_counts)

fraud_by_type = df[df['isFraud'] == 1]['type'].value_counts()
print("\nFraud Distribution by Transaction Type:")
print(fraud_by_type)

fraud_rate_by_type = df.groupby('type')['isFraud'].mean() * 100
print("\nFraud Rate by Transaction Type:")
print(fraud_rate_by_type)

# 4. Amount Analysis
print("\n4. AMOUNT ANALYSIS")
print("-" * 50)

print("\nAmount Statistics:")
print(df['amount'].describe())

print("\nAmount Comparison (Fraud vs Non-Fraud):")
print("Non-Fraud Transactions:")
print(df[df['isFraud'] == 0]['amount'].describe())
print("\nFraud Transactions:")
print(df[df['isFraud'] == 1]['amount'].describe())

print("\nAmount Statistics by Transaction Type:")
for tx_type in df['type'].unique():
    print(f"\nTransaction Type: {tx_type}")
    print(df[df['type'] == tx_type]['amount'].describe())

# 5. Balance Analysis
print("\n5. BALANCE ANALYSIS")
print("-" * 50)

print("\nOriginator Old Balance Statistics:")
print(df['oldbalanceOrg'].describe())
print("\nOriginator New Balance Statistics:")
print(df['newbalanceOrig'].describe())

print("\nDestination Old Balance Statistics:")
print(df['oldbalanceDest'].describe())
print("\nDestination New Balance Statistics:")
print(df['newbalanceDest'].describe())

# Create new features for analysis
df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']

print("\nOriginator Balance Error Statistics:")
print(df['errorBalanceOrig'].describe())
print("\nDestination Balance Error Statistics:")
print(df['errorBalanceDest'].describe())

# 6. Visualizations
print("\n6. VISUALIZATIONS")
print("-" * 50)

# Transaction Types Distribution
plt.figure(figsize=(12, 6))
sns.countplot(x='type', data=df)
plt.title('Transaction Types Distribution', fontsize=15)
plt.xlabel('Transaction Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('transaction_types_distribution.png', bbox_inches='tight')
plt.close()
print("Saved: Transaction Types Distribution")

# Fraud Distribution Pie
plt.figure(figsize=(8, 6))
fraud_counts = df['isFraud'].value_counts()
plt.pie(fraud_counts, labels=['Non-Fraud', 'Fraud'], autopct='%1.2f%%', colors=['#66b3ff', '#ff9999'])
plt.title('Fraud Transaction Percentage', fontsize=15)
plt.savefig('fraud_distribution_pie.png', bbox_inches='tight')
plt.close()
print("Saved: Fraud Transaction Percentage")

# Fraud by Transaction Type
plt.figure(figsize=(12, 6))
sns.countplot(x='type', hue='isFraud', data=df)
plt.title('Fraud Distribution by Transaction Type', fontsize=15)
plt.xlabel('Transaction Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(['Non-Fraud', 'Fraud'])
plt.savefig('fraud_by_transaction_type.png', bbox_inches='tight')
plt.close()
print("Saved: Fraud Distribution by Transaction Type")

# Amount Distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
df['amount'].plot.hist(bins=50, color='skyblue')
plt.title('Amount Distribution', fontsize=12)
plt.xlabel('Amount')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
df[df['amount'] < df['amount'].quantile(0.99)]['amount'].plot.hist(bins=50, color='skyblue')
plt.title('Amount Distribution (Without Outliers)', fontsize=12)
plt.xlabel('Amount')
plt.tight_layout()
plt.savefig('amount_distribution.png', bbox_inches='tight')
plt.close()
print("Saved: Amount Distribution")

# Amount by Transaction Type
plt.figure(figsize=(12, 6))
sns.boxplot(x='type', y='amount', data=df[df['amount'] < df['amount'].quantile(0.99)])
plt.title('Amount Distribution by Transaction Type (Without Outliers)', fontsize=15)
plt.xlabel('Transaction Type')
plt.ylabel('Amount')
plt.xticks(rotation=45)
plt.savefig('amount_by_transaction_type.png', bbox_inches='tight')
plt.close()
print("Saved: Amount Distribution by Transaction Type")

# Amount Comparison Fraud vs Non-Fraud
plt.figure(figsize=(12, 6))
sns.boxplot(x='isFraud', y='amount', data=df[df['amount'] < df['amount'].quantile(0.99)])
plt.title('Amount Comparison (Fraud vs Non-Fraud)', fontsize=15)
plt.xlabel('Is Fraud?')
plt.ylabel('Amount')
plt.xticks(ticks=[0, 1], labels=['Non-Fraud', 'Fraud'])
plt.savefig('amount_fraud_vs_nonfraud.png', bbox_inches='tight')
plt.close()
print("Saved: Amount Comparison (Fraud vs Non-Fraud)")

# Amount and Fraud Relationship
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
fraud_amount = df[df['isFraud'] == 1]['amount']
non_fraud_amount = df[df['isFraud'] == 0]['amount'].sample(n=len(fraud_amount))
combined = pd.DataFrame({'Amount': pd.concat([fraud_amount, non_fraud_amount]), 
                          'Type': ['Fraud'] * len(fraud_amount) + ['Non-Fraud'] * len(non_fraud_amount)})
sns.histplot(data=combined, x='Amount', hue='Type', kde=True, bins=50)
plt.title('Amount Distribution Comparison (Fraud vs Non-Fraud)', fontsize=12)
plt.xlabel('Amount')

plt.subplot(1, 2, 2)
df['amount_bin'] = pd.cut(df['amount'], bins=10)
fraud_rate = df.groupby('amount_bin')['isFraud'].mean() * 100
fraud_rate.plot(kind='bar', color='coral')
plt.title('Fraud Rate by Amount Range', fontsize=12)
plt.xlabel('Amount Range')
plt.ylabel('Fraud Rate %')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('fraud_rate_by_amount.png', bbox_inches='tight')
plt.close()
print("Saved: Fraud Rate by Amount Range")

# Balance Analysis
plt.figure(figsize=(15, 12))

plt.subplot(2, 2, 1)
sns.scatterplot(x='oldbalanceOrg', y='newbalanceOrig', hue='isFraud', data=df.sample(10000), alpha=0.6)
plt.title('Originator Balance Before vs After (Sample)', fontsize=12)
plt.xlabel('Old Balance Orig')
plt.ylabel('New Balance Orig')
plt.legend(['Non-Fraud', 'Fraud'])

plt.subplot(2, 2, 2)
sns.scatterplot(x='oldbalanceDest', y='newbalanceDest', hue='isFraud', data=df.sample(10000), alpha=0.6)
plt.title('Destination Balance Before vs After (Sample)', fontsize=12)
plt.xlabel('Old Balance Dest')
plt.ylabel('New Balance Dest')
plt.legend(['Non-Fraud', 'Fraud'])

plt.subplot(2, 2, 3)
sns.boxplot(x='isFraud', y='errorBalanceOrig', data=df[(df['errorBalanceOrig'] != 0) & (df['errorBalanceOrig'].abs() < df['errorBalanceOrig'].abs().quantile(0.99))])
plt.title('Originator Balance Errors (Fraud vs Non-Fraud)', fontsize=12)
plt.xlabel('Is Fraud?')
plt.ylabel('Balance Error')
plt.xticks(ticks=[0, 1], labels=['Non-Fraud', 'Fraud'])

plt.subplot(2, 2, 4)
sns.boxplot(x='isFraud', y='errorBalanceDest', data=df[(df['errorBalanceDest'] != 0) & (df['errorBalanceDest'].abs() < df['errorBalanceDest'].abs().quantile(0.99))])
plt.title('Destination Balance Errors (Fraud vs Non-Fraud)', fontsize=12)
plt.xlabel('Is Fraud?')
plt.ylabel('Balance Error')
plt.xticks(ticks=[0, 1], labels=['Non-Fraud', 'Fraud'])

plt.tight_layout()
plt.savefig('balance_analysis.png', bbox_inches='tight')
plt.close()
print("Saved: Balance Analysis")

# 7. Time Analysis
print("\n7. TIME ANALYSIS")
print("-" * 50)

print("\nTime Step Statistics:")
print(df['step'].describe())

fraud_by_step = df.groupby('step')['isFraud'].sum().reset_index()
print("\nFraud Distribution Over Time:")
print(fraud_by_step.head(10))

fraud_rate_by_step = df.groupby('step')['isFraud'].mean() * 100
print("\nFraud Rate Over Time (Top 10):")
print(fraud_rate_by_step.sort_values(ascending=False).head(10))

# Time Analysis Plots
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
step_counts = df.groupby('step').size()
step_counts.plot(kind='line', color='blue')
plt.title('Transaction Count Over Time', fontsize=15)
plt.xlabel('Time Step')
plt.ylabel('Transaction Count')

plt.subplot(2, 1, 2)
fraud_by_step.plot(x='step', y='isFraud', kind='line', color='red')
plt.title('Fraud Transactions Over Time', fontsize=15)
plt.xlabel('Time Step')
plt.ylabel('Fraud Count')

plt.tight_layout()
plt.savefig('time_analysis.png', bbox_inches='tight')
plt.close()
print("Saved: Transaction Time Analysis")

# Fraud Rate Over Time
plt.figure(figsize=(15, 6))
fraud_rate_by_step.plot(kind='line', color='purple', marker='o', alpha=0.7)
plt.title('Fraud Rate Over Time', fontsize=15)
plt.xlabel('Time Step')
plt.ylabel('Fraud Rate (%)')
plt.grid(True, alpha=0.3)
plt.savefig('fraud_rate_over_time.png', bbox_inches='tight')
plt.close()
print("Saved: Fraud Rate Over Time")

# 8. Correlation Analysis
print("\n8. CORRELATION ANALYSIS")
print("-" * 50)

# Create additional features
df['diffBalanceOrig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
df['diffBalanceDest'] = df['newbalanceDest'] - df['oldbalanceDest']
df['diffAmount'] = df['amount'] - df['diffBalanceOrig']

# Convert transaction type to dummy variables
df_dummy = pd.get_dummies(df['type'], prefix='type')
df_corr = pd.concat([df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 
                         'isFraud', 'isFlaggedFraud', 'diffBalanceOrig', 'diffBalanceDest', 'diffAmount']], 
                     df_dummy], axis=1)

# Calculate correlation matrix
correlation_matrix = df_corr.corr()
print("\nCorrelations with Fraud:")
print(correlation_matrix['isFraud'].sort_values(ascending=False))

# Correlation Heatmap
plt.figure(figsize=(18, 15))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Variables Correlation Matrix', fontsize=20)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('correlation_matrix.png', bbox_inches='tight')
plt.close()
print("Saved: Variables Correlation Matrix")

# 9. Fraud Patterns Analysis
print("\n9. FRAUD PATTERNS ANALYSIS")
print("-" * 50)

print("\nSample of Fraud Transactions:")
fraud_sample = df[df['isFraud'] == 1].sample(5)
print(fraud_sample)

print("\nFraud Transactions Analysis by Type and Balance:")
fraud_df = df[df['isFraud'] == 1]

for tx_type in fraud_df['type'].unique():
    type_fraud = fraud_df[fraud_df['type'] == tx_type]
    print(f"\nTransaction Type: {tx_type}, Fraud Count: {len(type_fraud)}")
    
    # Balance Analysis for Fraud Transactions
    orig_zero_balance = (type_fraud['oldbalanceOrg'] == 0).sum()
    dest_zero_balance = (type_fraud['oldbalanceDest'] == 0).sum()
    
    print(f"Transactions with zero originator balance: {orig_zero_balance} ({orig_zero_balance/len(type_fraud)*100:.2f}%)")
    print(f"Transactions with zero destination balance: {dest_zero_balance} ({dest_zero_balance/len(type_fraud)*100:.2f}%)")
    
    # Balance Error Analysis for Fraud Transactions
    orig_error = (type_fraud['errorBalanceOrig'] != 0).sum()
    dest_error = (type_fraud['errorBalanceDest'] != 0).sum()
    
    print(f"Transactions with originator balance errors: {orig_error} ({orig_error/len(type_fraud)*100:.2f}%)")
    print(f"Transactions with destination balance errors: {dest_error} ({dest_error/len(type_fraud)*100:.2f}%)")

# 10. Flagged Fraud Analysis
print("\n10. FLAGGED FRAUD ANALYSIS")
print("-" * 50)

print("\nFlagged Transactions Analysis:")
flagged_df = df[df['isFlaggedFraud'] == 1]
print(f"Number of flagged transactions: {len(flagged_df)}")

if len(flagged_df) > 0:
    print("\nDistribution by Transaction Type:")
    print(flagged_df['type'].value_counts())
    
    print("\nComparison with Fraud Transactions:")
    flagged_fraud = flagged_df[flagged_df['isFraud'] == 1]
    print(f"Number of transactions both flagged and fraudulent: {len(flagged_fraud)}")
    print(f"Percentage of fraud transactions detected: {len(flagged_fraud)/len(fraud_df)*100:.2f}%")
    
    print("\nAmount Statistics for Flagged Transactions:")
    print(flagged_df['amount'].describe())

# 11. PCA Analysis
print("\n11. PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("-" * 50)

numerical_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
X = df[numerical_cols].copy()

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Analyze explained variance ratio
explained_variance = pca.explained_variance_ratio_
print("\nExplained Variance Ratio by Component:")
print(explained_variance)
print(f"Cumulative variance explained by first two components: {sum(explained_variance[:2])*100:.2f}%")

# PCA Visualization
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.8, color='skyblue')
plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid', color='red')
plt.title('Explained Variance Ratio by Principal Components', fontsize=15)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True, alpha=0.3)

# Visualize first two components by fraud status
plt.subplot(2, 1, 2)
pca_df = pd.DataFrame(data=X_pca[:, :2], columns=['PC1', 'PC2'])
pca_df['isFraud'] = df['isFraud'].values

# Sample for clearer visualization
fraud_pca = pca_df[pca_df['isFraud'] == 1]
non_fraud_pca = pca_df[pca_df['isFraud'] == 0].sample(n=min(len(fraud_pca)*5, len(pca_df[pca_df['isFraud'] == 0])))
sample_pca = pd.concat([fraud_pca, non_fraud_pca])

colors = {0: 'blue', 1: 'red'}
scatter = plt.scatter(sample_pca['PC1'], sample_pca['PC2'], c=sample_pca['isFraud'].map(colors), alpha=0.5)
plt.title('PCA: First Two Principal Components by Fraud Status', fontsize=15)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(handles=scatter.legend_elements()[0], labels=['Non-Fraud', 'Fraud'])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_analysis.png', bbox_inches='tight')
plt.close()
print("Saved: PCA Analysis")

# 12. Feature Importance for Fraud Detection
print("\n12. FEATURE IMPORTANCE ANALYSIS")
print("-" * 50)

# Calculate feature importance based on correlation with fraud
importance = abs(correlation_matrix['isFraud'])
importance = importance.sort_values(ascending=False)

# Remove isFraud itself
importance = importance[importance.index != 'isFraud']

print("\nFeature Importance for Fraud Detection (based on correlation):")
print(importance.head(10))

# Feature Importance Visualization
plt.figure(figsize=(12, 8))
importance.head(15).plot(kind='bar', color='coral')
plt.title('Feature Importance for Fraud Detection', fontsize=15)
plt.xlabel('Feature')
plt.ylabel('Absolute Correlation with Fraud')
plt.xticks(rotation=90)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', bbox_inches='tight')
plt.close()
print("Saved: Feature Importance Analysis")

# 13. Advanced Transaction Type Analysis
print("\n13. ADVANCED TRANSACTION TYPE ANALYSIS")
print("-" * 50)

print("\nDetailed Analysis by Transaction Type:")
for tx_type in df['type'].unique():
    type_data = df[df['type'] == tx_type]
    type_fraud = type_data[type_data['isFraud'] == 1]
    
    print(f"\nTransaction Type: {tx_type}")
    print(f"Total Count: {len(type_data)}")
    print(f"Fraud Count: {len(type_fraud)}")
    print(f"Fraud Rate: {len(type_fraud)/len(type_data)*100:.4f}%")
    
    print("Amount Statistics:")
    print(f"Mean: {type_data['amount'].mean():.2f}")
    print(f"Median: {type_data['amount'].median():.2f}")
    print(f"Max: {type_data['amount'].max():.2f}")
    
    if len(type_fraud) > 0:
        print("Fraud Amount Statistics:")
        print(f"Mean: {type_fraud['amount'].mean():.2f}")
        print(f"Median: {type_fraud['amount'].median():.2f}")
        print(f"Max: {type_fraud['amount'].max():.2f}")

# 14. Key Insights and Summary
print("\n14. KEY INSIGHTS AND SUMMARY")
print("-" * 50)

print("""
Key Insights from the EDA:
1. Dataset Overview: {0} transactions with {1} fraudulent cases ({2:.4f}%)
2. Fraud mostly occurs in {3} transactions
3. The automated system flagged {4} transactions as fraudulent
4. Highest fraud rate is in transaction type: {5} ({6:.2f}%)
5. Average fraud transaction amount: ${7:.2f}
6. Balance patterns show: {8}
7. Time analysis reveals: {9}
8. Most important features for fraud detection: {10}
""".format(
    len(df), 
    len(df[df['isFraud'] == 1]),
    len(df[df['isFraud'] == 1]) / len(df) * 100,
    fraud_by_type.index[0] if len(fraud_by_type) > 0 else "None",
    len(df[df['isFlaggedFraud'] == 1]),
    fraud_rate_by_type.sort_values(ascending=False).index[0] if len(fraud_rate_by_type) > 0 else "None",
    fraud_rate_by_type.sort_values(ascending=False).iloc[0] if len(fraud_rate_by_type) > 0 else 0,
    df[df['isFraud'] == 1]['amount'].mean(),
    "Zero originator balance is common in fraud" if (fraud_df['oldbalanceOrg'] == 0).mean() > 0.5 else "Various balance patterns",
    "Some time steps have higher fraud rates" if fraud_rate_by_step.max() > fraud_rate_by_step.mean() * 2 else "Relatively consistent fraud rate",
    ", ".join(importance.head(3).index)
))

print("\nEDA COMPLETE!")
print("Total saved visualizations: 11")