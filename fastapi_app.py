from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json
import sys
sys.path.append('src')
from cluster_library import RuleBasedCustomerClusterer

# Global data storage
data_store = {}

def load_data():
    """Load all processed data files"""
    try:
        # Load customer clusters
        clusters_df = pd.read_csv("data/processed/customer_clusters_from_rules.csv")
        
        # Load rules
        rules_apriori = pd.read_csv("data/processed/rules_apriori_filtered.csv")
        rules_fpgrowth = pd.read_csv("data/processed/rules_fpgrowth_filtered.csv")
        
        # Load cleaned data
        cleaned_df = pd.read_csv("data/processed/cleaned_uk_data.csv")
        
        # Store in global dict
        data_store['clusters'] = clusters_df
        data_store['rules_apriori'] = rules_apriori
        data_store['rules_fpgrowth'] = rules_fpgrowth
        data_store['cleaned'] = cleaned_df
        
        print("✅ Data loaded successfully!")
        print(f"   - Customers: {len(clusters_df)}")
        print(f"   - Apriori Rules: {len(rules_apriori)}")
        print(f"   - FP-Growth Rules: {len(rules_fpgrowth)}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting Shop Cluster Dashboard API...")
    load_data()
    yield
    # Shutdown
    print("👋 Shutting down...")

app = FastAPI(title="Shop Cluster Analysis Dashboard", lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/overview")
async def get_overview():
    """Get overview statistics"""
    clusters_df = data_store.get('clusters')
    rules_apriori = data_store.get('rules_apriori')
    rules_fpgrowth = data_store.get('rules_fpgrowth')
    
    if clusters_df is None:
        return {"error": "Data not loaded"}
    
    # Calculate statistics
    total_customers = len(clusters_df)
    n_clusters = clusters_df['cluster'].nunique()
    
    # Cluster distribution
    cluster_dist = clusters_df['cluster'].value_counts().sort_index().to_dict()
    
    # RFM statistics
    rfm_stats = {}
    if all(col in clusters_df.columns for col in ['Recency', 'Frequency', 'Monetary']):
        rfm_stats = {
            'avg_recency': float(clusters_df['Recency'].mean()),
            'avg_frequency': float(clusters_df['Frequency'].mean()),
            'avg_monetary': float(clusters_df['Monetary'].mean())
        }
    
    return {
        "total_customers": total_customers,
        "n_clusters": n_clusters,
        "cluster_distribution": cluster_dist,
        "total_rules_apriori": len(rules_apriori) if rules_apriori is not None else 0,
        "total_rules_fpgrowth": len(rules_fpgrowth) if rules_fpgrowth is not None else 0,
        "rfm_stats": rfm_stats
    }

@app.get("/api/rules")
async def get_rules(
    algorithm: Optional[str] = "apriori",
    top_n: Optional[int] = 20,
    min_lift: Optional[float] = 1.0,
    sort_by: Optional[str] = "lift"
):
    """Get association rules with filtering"""
    
    # Select algorithm
    if algorithm == "fpgrowth":
        rules_df = data_store.get('rules_fpgrowth')
    else:
        rules_df = data_store.get('rules_apriori')
    
    if rules_df is None:
        return []
    
    # Filter by lift
    filtered = rules_df[rules_df['lift'] >= min_lift].copy()
    
    # Sort
    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=False)
    
    # Get top N
    filtered = filtered.head(top_n)
    
    # Convert to dict
    results = []
    for _, row in filtered.iterrows():
        results.append({
            "antecedents": str(row['antecedents']),
            "consequents": str(row['consequents']),
            "support": float(row['support']),
            "confidence": float(row['confidence']),
            "lift": float(row['lift'])
        })
    
    return results

@app.get("/api/cluster-profile/{cluster_id}")
async def get_cluster_profile(cluster_id: int):
    """Get detailed profile for a specific cluster"""
    clusters_df = data_store.get('clusters')
    
    if clusters_df is None:
        return {"error": "Data not loaded"}
    
    # Filter cluster
    cluster_data = clusters_df[clusters_df['cluster'] == cluster_id]
    
    if len(cluster_data) == 0:
        return {"error": f"Cluster {cluster_id} not found"}
    
    # Calculate statistics
    size = len(cluster_data)
    percentage = (size / len(clusters_df)) * 100
    
    profile = {
        "cluster_id": cluster_id,
        "size": size,
        "percentage": round(percentage, 2)
    }
    
    # RFM statistics if available
    if all(col in cluster_data.columns for col in ['Recency', 'Frequency', 'Monetary']):
        profile.update({
            "avg_recency": float(cluster_data['Recency'].mean()),
            "median_recency": float(cluster_data['Recency'].median()),
            "avg_frequency": float(cluster_data['Frequency'].mean()),
            "median_frequency": float(cluster_data['Frequency'].median()),
            "avg_monetary": float(cluster_data['Monetary'].mean()),
            "median_monetary": float(cluster_data['Monetary'].median())
        })
    
    # Top rule features if available
    rule_cols = [col for col in cluster_data.columns if col.startswith('rule_')]
    if rule_cols:
        top_rules = []
        for col in rule_cols[:10]:  # Top 10 rules
            activation_rate = (cluster_data[col] > 0).mean()
            avg_weight = cluster_data[col].mean()
            top_rules.append({
                "rule": col.replace('rule_', ''),
                "activation_rate": float(activation_rate),
                "avg_weight": float(avg_weight)
            })
        profile['top_rules'] = top_rules
    
    return profile

@app.get("/api/rfm-comparison")
async def get_rfm_comparison():
    """Get RFM comparison across all clusters"""
    clusters_df = data_store.get('clusters')
    
    if clusters_df is None or not all(col in clusters_df.columns for col in ['Recency', 'Frequency', 'Monetary']):
        return {"error": "RFM data not available"}
    
    # Group by cluster
    rfm_by_cluster = clusters_df.groupby('cluster').agg({
        'Recency': ['mean', 'median'],
        'Frequency': ['mean', 'median'],
        'Monetary': ['mean', 'median']
    }).round(2)
    
    results = []
    for cluster_id in sorted(clusters_df['cluster'].unique()):
        cluster_data = clusters_df[clusters_df['cluster'] == cluster_id]
        results.append({
            "cluster_id": int(cluster_id),
            "size": len(cluster_data),
            "recency_mean": float(rfm_by_cluster.loc[cluster_id, ('Recency', 'mean')]),
            "recency_median": float(rfm_by_cluster.loc[cluster_id, ('Recency', 'median')]),
            "frequency_mean": float(rfm_by_cluster.loc[cluster_id, ('Frequency', 'mean')]),
            "frequency_median": float(rfm_by_cluster.loc[cluster_id, ('Frequency', 'median')]),
            "monetary_mean": float(rfm_by_cluster.loc[cluster_id, ('Monetary', 'mean')]),
            "monetary_median": float(rfm_by_cluster.loc[cluster_id, ('Monetary', 'median')])
        })
    
    return results

@app.get("/api/marketing-strategies")
async def get_marketing_strategies():
    """Get marketing strategies for each cluster"""
    
    # Define strategies based on cluster analysis
    # This should be customized based on actual cluster profiling
    strategies = {
        0: {
            "name_en": "Casual Browsers",
            "name_vi": "Khách Hàng Thông Thường",
            "persona": "Khách hàng mua sắm thỉnh thoảng với giá trị đơn hàng thấp, chủ yếu mua các sản phẩm phổ biến đơn lẻ.",
            "strategies": [
                "Gửi email marketing với ưu đãi hấp dẫn để kích hoạt mua hàng",
                "Tạo chương trình khuyến mãi theo mùa để tăng tần suất mua",
                "Đề xuất sản phẩm phổ biến với giá cả phải chăng",
                "Chương trình tích điểm để khuyến khích mua lại"
            ]
        },
        1: {
            "name_en": "VIP Loyalists",
            "name_vi": "Khách Hàng VIP Trung Thành",
            "persona": "Khách hàng có giá trị cao, mua thường xuyên với đơn hàng lớn, thể hiện hành vi mua combo và cross-sell mạnh.",
            "strategies": [
                "Chương trình VIP với ưu đãi độc quyền và dịch vụ cao cấp",
                "Gợi ý bundle sản phẩm cao cấp dựa trên lịch sử mua hàng",
                "Tư vấn cá nhân hóa và chăm sóc khách hàng tận tình",
                "Early access vào sản phẩm mới và sale đặc biệt"
            ]
        }
    }
    
    return strategies

@app.get("/api/pca-data")
async def get_pca_data():
    """Get PCA-transformed data for visualization"""
    clusters_df = data_store.get('clusters')
    
    if clusters_df is None:
        return {"error": "Data not loaded"}
    
    try:
        # Get numeric features (exclude CustomerID and cluster)
        exclude_cols = ['CustomerID', 'cluster']
        feature_cols = [col for col in clusters_df.columns if col not in exclude_cols]
        
        # Get feature matrix
        X = clusters_df[feature_cols].values
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Get cluster labels
        labels = clusters_df['cluster'].values
        
        # Prepare data by cluster
        result = {}
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            result[int(cluster_id)] = {
                'x': X_pca[mask, 0].tolist(),
                'y': X_pca[mask, 1].tolist()
            }
        
        return {
            "clusters": result,
            "explained_variance": pca.explained_variance_ratio_.tolist()
        }
        
    except Exception as e:
        print(f"Error in PCA: {e}")
        return {"error": str(e)}

@app.get("/api/top-rules-by-cluster")
async def get_top_rules_by_cluster(top_n: int = 10):
    """Get top N activated rules for each cluster"""
    clusters_df = data_store.get('clusters')
    rules_df = data_store.get('rules_apriori')
    cleaned_df = data_store.get('cleaned')
    
    if clusters_df is None or rules_df is None or cleaned_df is None:
        return {"error": "Data not loaded"}
    
    try:
        # Initialize clusterer
        clusterer = RuleBasedCustomerClusterer(
            df_clean=cleaned_df,
            customer_col="CustomerID",
            invoice_col="InvoiceNo",
            item_col="Description",
            quantity_col="Quantity"
        )
        
        # Build customer-item matrix
        clusterer.build_customer_item_matrix()
        
        # Prepare top rules
        TOP_K = 200
        rules_top_k = rules_df.nlargest(TOP_K, 'lift').reset_index(drop=True)
        
        # Ensure str columns exist
        if 'antecedents_str' not in rules_top_k.columns:
            rules_top_k['antecedents_str'] = rules_top_k['antecedents'].astype(str)
        if 'consequents_str' not in rules_top_k.columns:
            rules_top_k['consequents_str'] = rules_top_k['consequents'].astype(str)
        
        # Assign rules to clusterer
        clusterer.rules_df_ = rules_top_k
        
        # Build rule feature matrix
        X_rules = clusterer.build_rule_feature_matrix(
            weighting='lift',
            min_antecedent_len=1
        )
        
        # Create feature dataframe
        customers_list = clusterer.customers_
        feature_df = pd.DataFrame(
            X_rules,
            index=customers_list,
            columns=[f"rule_{i}" for i in range(X_rules.shape[1])]
        )
        feature_df['CustomerID'] = customers_list
        feature_df = feature_df.merge(
            clusters_df[['CustomerID', 'cluster']], 
            on='CustomerID', 
            how='inner'
        )
        
        # Calculate top rules for each cluster
        results = {}
        for cluster_id in sorted(clusters_df['cluster'].unique()):
            cluster_data = feature_df[feature_df['cluster'] == cluster_id]
            n_customers = len(cluster_data)
            
            # Get rule columns
            rule_cols = [col for col in feature_df.columns if col.startswith('rule_')]
            
            # Calculate mean activation
            rule_means = cluster_data[rule_cols].mean().sort_values(ascending=False)
            top_rules = rule_means.head(top_n)
            
            rules_list = []
            for rank, (feature_name, mean_activation) in enumerate(top_rules.items(), 1):
                rule_idx = int(feature_name.split('_')[1])
                
                if rule_idx < len(rules_top_k):
                    rule_row = rules_top_k.iloc[rule_idx]
                    
                    # Calculate activation percentage
                    n_active = (cluster_data[feature_name] > 0).sum()
                    pct_active = (n_active / n_customers) * 100
                    
                    rules_list.append({
                        "rank": rank,
                        "rule_index": int(rule_idx),
                        "antecedents": str(rule_row['antecedents_str']),
                        "consequents": str(rule_row['consequents_str']),
                        "mean_activation": round(float(mean_activation), 2),
                        "customers_activated": int(n_active),
                        "pct_activated": round(float(pct_active), 1),
                        "lift": round(float(rule_row['lift']), 2),
                        "confidence": round(float(rule_row['confidence']) * 100, 1),
                        "support": round(float(rule_row['support']) * 100, 2)
                    })
            
            results[str(cluster_id)] = {
                "cluster_id": int(cluster_id),
                "n_customers": int(n_customers),
                "top_rules": rules_list
            }
        
        return results
        
    except Exception as e:
        print(f"Error calculating top rules: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("🌐 Dashboard: http://localhost:8052")
    print("📚 API Docs: http://localhost:8052/docs")
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8052, reload=True)
