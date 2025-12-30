#!/usr/bin/env python3
"""
Script to calculate Top 10 most activated rules per cluster
"""
import pandas as pd
import numpy as np
import sys
sys.path.append('/hdd3/nckh-AIAgent/tyanzuq/DataMining/shop_cluster/src')
from cluster_library import RuleBasedCustomerClusterer

# Load data
print("Loading data...")
rules_df = pd.read_csv('data/processed/rules_apriori_filtered.csv')
cleaned_df = pd.read_csv('data/processed/cleaned_uk_data.csv')
clusters_df = pd.read_csv('data/processed/customer_clusters_from_rules.csv')

print(f"Total customers in clusters: {len(clusters_df)}")

# Initialize clusterer
print("\nInitializing RuleBasedCustomerClusterer...")
clusterer = RuleBasedCustomerClusterer(
    df_clean=cleaned_df,
    customer_col="CustomerID",
    invoice_col="InvoiceNo",
    item_col="Description",
    quantity_col="Quantity"
)

# Build customer-item matrix
print("Building customer-item matrix...")
clusterer.build_customer_item_matrix()
print(f"Customer-item matrix shape: {clusterer.customer_item_bool.shape}")

# Load rules and select Top-K
print("\nPreparing top rules...")
TOP_K = 200
rules_top_k = rules_df.nlargest(TOP_K, 'lift').reset_index(drop=True)

# Ensure str columns exist
if 'antecedents_str' not in rules_top_k.columns:
    if 'antecedents' in rules_top_k.columns:
        rules_top_k['antecedents_str'] = rules_top_k['antecedents'].astype(str)
if 'consequents_str' not in rules_top_k.columns:
    if 'consequents' in rules_top_k.columns:
        rules_top_k['consequents_str'] = rules_top_k['consequents'].astype(str)

# Assign rules directly to clusterer
clusterer.rules_df_ = rules_top_k
print(f"Loaded {len(rules_top_k)} rules")

# Build rule feature matrix
print("\nBuilding rule feature matrix...")
X_rules = clusterer.build_rule_feature_matrix(
    weighting='lift',
    min_antecedent_len=1
)
print(f"Rule feature matrix shape: {X_rules.shape}")

# Merge with clusters
print("\nMerging with cluster assignments...")
customers_list = clusterer.customers_
feature_df = pd.DataFrame(
    X_rules, 
    index=customers_list,
    columns=[f"rule_{i}" for i in range(X_rules.shape[1])]
)
feature_df['CustomerID'] = customers_list
feature_df = feature_df.merge(clusters_df[['CustomerID', 'cluster']], on='CustomerID', how='inner')

print(f"Final feature dataframe shape: {feature_df.shape}")

print("\n" + "="*80)
print("TOP 10 RULES PER CLUSTER (by Mean Weighted Activation)")
print("="*80)

for cluster_id in [0, 1]:
    cluster_data = feature_df[feature_df['cluster'] == cluster_id]
    n_customers = len(cluster_data)
    
    print(f"\n🔍 CLUSTER {cluster_id}: {n_customers} customers ({n_customers/len(feature_df)*100:.1f}%)")
    print("-" * 80)
    
    # Calculate mean activation (only for rule features)
    rule_feature_cols = [col for col in feature_df.columns if col.startswith('rule_')]
    
    rule_means = cluster_data[rule_feature_cols].mean().sort_values(ascending=False)
    top_10 = rule_means.head(10)
    
    for rank, (feature_name, mean_activation) in enumerate(top_10.items(), 1):
        # Extract rule index
        rule_idx = int(feature_name.split('_')[1])
        
        if rule_idx < len(rules_top_k):
            rule_row = rules_top_k.iloc[rule_idx]
            
            # Calculate percentage of customers who activate this rule
            n_active = (cluster_data[feature_name] > 0).sum()
            pct_active = (n_active / n_customers) * 100
            
            # Format rule
            ante_str = rule_row['antecedents_str'] if 'antecedents_str' in rule_row else str(rule_row['antecedents'])
            cons_str = rule_row['consequents_str'] if 'consequents_str' in rule_row else str(rule_row['consequents'])
            
            print(f"\n{rank}. Rule #{rule_idx}")
            print(f"   {ante_str} → {cons_str}")
            print(f"   Mean Weighted Activation: {mean_activation:.2f}")
            print(f"   {n_active}/{n_customers} customers activate ({pct_active:.1f}%)")
            print(f"   Lift={rule_row['lift']:.2f} | Conf={rule_row['confidence']:.1%} | Supp={rule_row['support']:.2%}")

print("\n" + "="*80)
print("✅ Analysis complete!")
