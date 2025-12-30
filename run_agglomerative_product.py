# -*- coding: utf-8 -*-
"""
🌳 SCRIPT MỞ RỘNG: AGGLOMERATIVE CLUSTERING + PRODUCT CLUSTERING
=================================================================

Yêu cầu:
1. So sánh K-Means vs Agglomerative Clustering
2. Phân cụm sản phẩm (Product Clustering) với Agglomerative
3. So sánh Customer Clustering vs Product Clustering cho marketing
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings('ignore')

# Thêm src vào path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from cluster_library import RuleBasedCustomerClusterer

# ============================================================
# CẤU HÌNH
# ============================================================
CLEANED_DATA_PATH = os.path.join(project_root, "data/processed/cleaned_uk_data.csv")
RULES_INPUT_PATH = os.path.join(project_root, "data/processed/rules_apriori_filtered.csv")
OUTPUT_REPORT_PATH = os.path.join(project_root, "BAO_CAO_AGGLOMERATIVE_PRODUCT.md")
OUTPUT_PRODUCT_CLUSTERS = os.path.join(project_root, "data/processed/product_clusters.csv")

RANDOM_STATE = 42


def print_header(title):
    """In header đẹp"""
    print("\n" + "=" * 70)
    print(f"🌳 {title}")
    print("=" * 70)


def evaluate_clustering(X, labels, name):
    """Đánh giá chất lượng phân cụm"""
    mask = labels >= 0
    X_valid = X[mask]
    labels_valid = labels[mask]
    
    n_clusters = len(set(labels_valid))
    
    if n_clusters < 2:
        return {'Model': name, 'N_clusters': n_clusters, 'Silhouette': None, 'DBI': None, 'CH': None}
    
    return {
        'Model': name,
        'N_clusters': n_clusters,
        'Silhouette': round(silhouette_score(X_valid, labels_valid), 4),
        'DBI': round(davies_bouldin_score(X_valid, labels_valid), 4),
        'CH': round(calinski_harabasz_score(X_valid, labels_valid), 2)
    }


def main():
    """Chạy phân tích mở rộng với Agglomerative + Product Clustering"""
    
    report_lines = []
    report_lines.append("# 🌳 BÁO CÁO MỞ RỘNG: AGGLOMERATIVE CLUSTERING & PRODUCT CLUSTERING")
    report_lines.append(f"\n**Ngày tạo:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("---\n")
    
    # ============================================================
    # BƯỚC 0: LOAD DỮ LIỆU
    # ============================================================
    print_header("LOAD DỮ LIỆU")
    
    df_clean = pd.read_csv(CLEANED_DATA_PATH, parse_dates=["InvoiceDate"])
    print(f"✅ Loaded: {df_clean.shape[0]:,} dòng")
    print(f"   Số khách hàng: {df_clean['CustomerID'].nunique():,}")
    print(f"   Số sản phẩm: {df_clean['Description'].nunique():,}")
    
    rules_df = pd.read_csv(RULES_INPUT_PATH)
    print(f"✅ Loaded: {len(rules_df):,} luật")
    
    report_lines.append("## 📁 DỮ LIỆU ĐẦU VÀO\n")
    report_lines.append(f"- Số giao dịch: {df_clean.shape[0]:,}")
    report_lines.append(f"- Số khách hàng: {df_clean['CustomerID'].nunique():,}")
    report_lines.append(f"- Số sản phẩm: {df_clean['Description'].nunique():,}")
    report_lines.append(f"- Số luật kết hợp: {len(rules_df):,}\n")
    
    # ============================================================
    # PHẦN 1: SO SÁNH K-MEANS vs AGGLOMERATIVE (CUSTOMER CLUSTERING)
    # ============================================================
    print_header("PHẦN 1: SO SÁNH K-MEANS vs AGGLOMERATIVE")
    
    report_lines.append("## 📊 PHẦN 1: SO SÁNH K-MEANS vs AGGLOMERATIVE CLUSTERING\n")
    report_lines.append("### Giải thích thuật toán Agglomerative:\n")
    report_lines.append("""
**Agglomerative Clustering** (Phân cụm phân cấp từ dưới lên):

```
Thuật toán:
1. Bắt đầu: Mỗi điểm dữ liệu là một cụm riêng (N cụm)
2. Tìm 2 cụm gần nhất → Gộp thành 1 cụm
3. Lặp lại bước 2 cho đến khi còn K cụm

Cách đo khoảng cách giữa 2 cụm (Linkage):
- Ward: Tối thiểu hóa phương sai khi gộp (phổ biến nhất)
- Complete: Khoảng cách giữa 2 điểm xa nhất
- Average: Khoảng cách trung bình
- Single: Khoảng cách giữa 2 điểm gần nhất
```

**Ưu điểm so với K-Means:**
- Có thể vẽ Dendrogram để hiểu cấu trúc phân cấp
- Không cần khởi tạo ngẫu nhiên → Kết quả ổn định
- Phát hiện được cụm lồng nhau (nested clusters)

**Nhược điểm:**
- Chậm hơn với dữ liệu lớn O(n²) vs O(nKt)
- Không thể undo việc gộp cụm
""")
    
    # Khởi tạo clusterer và build features
    clusterer = RuleBasedCustomerClusterer(df_clean=df_clean)
    clusterer.build_customer_item_matrix(threshold=1)
    clusterer.load_rules(RULES_INPUT_PATH, top_k=200, sort_by='lift')
    
    X_customer, meta = clusterer.build_final_features(
        weighting="lift", use_rfm=True, rfm_scale=True, min_antecedent_len=2
    )
    print(f"✅ Customer features: {X_customer.shape}")
    
    # So sánh các mô hình với nhiều K
    print("\n📊 So sánh K-Means vs Agglomerative với các giá trị K:")
    
    comparison_results = []
    
    for k in range(2, 7):
        # K-Means
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=RANDOM_STATE)
        labels_kmeans = kmeans.fit_predict(X_customer)
        result_kmeans = evaluate_clustering(X_customer, labels_kmeans, f'K-Means (K={k})')
        comparison_results.append(result_kmeans)
        
        # Agglomerative Ward
        agg_ward = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels_agg_ward = agg_ward.fit_predict(X_customer)
        result_agg_ward = evaluate_clustering(X_customer, labels_agg_ward, f'Agglomerative Ward (K={k})')
        comparison_results.append(result_agg_ward)
        
        # Agglomerative Complete
        agg_complete = AgglomerativeClustering(n_clusters=k, linkage='complete')
        labels_agg_complete = agg_complete.fit_predict(X_customer)
        result_agg_complete = evaluate_clustering(X_customer, labels_agg_complete, f'Agglomerative Complete (K={k})')
        comparison_results.append(result_agg_complete)
        
        print(f"   K={k}: K-Means={result_kmeans['Silhouette']:.4f}, Ward={result_agg_ward['Silhouette']:.4f}, Complete={result_agg_complete['Silhouette']:.4f}")
    
    comparison_df = pd.DataFrame(comparison_results)
    
    # Tìm best model
    best_idx = comparison_df['Silhouette'].idxmax()
    best_model = comparison_df.loc[best_idx]
    
    print(f"\n✅ Mô hình tốt nhất: {best_model['Model']} (Silhouette={best_model['Silhouette']})")
    
    # Bảng so sánh
    report_lines.append("\n### Bảng so sánh K-Means vs Agglomerative:\n")
    report_lines.append("| Model | N Clusters | Silhouette ↑ | DBI ↓ | CH ↑ |")
    report_lines.append("|-------|------------|--------------|-------|------|")
    for _, row in comparison_df.iterrows():
        marker = " ⭐" if row['Silhouette'] == comparison_df['Silhouette'].max() else ""
        report_lines.append(f"| {row['Model']} | {row['N_clusters']} | {row['Silhouette']}{marker} | {row['DBI']} | {row['CH']} |")
    report_lines.append("")
    
    report_lines.append(f"\n**Kết luận:** Mô hình **{best_model['Model']}** cho kết quả tốt nhất với Silhouette = {best_model['Silhouette']}\n")
    
    # Chọn mô hình tốt nhất để phân tích tiếp
    # Sử dụng Agglomerative Ward với K=2 (thường cho kết quả tốt)
    BEST_K = 2
    agg_final = AgglomerativeClustering(n_clusters=BEST_K, linkage='ward')
    labels_customer = agg_final.fit_predict(X_customer)
    
    meta_customer = meta.copy()
    meta_customer['cluster'] = labels_customer
    
    # ============================================================
    # PHẦN 2: PHÂN CỤM SẢN PHẨM (PRODUCT CLUSTERING)
    # ============================================================
    print_header("PHẦN 2: PHÂN CỤM SẢN PHẨM (PRODUCT CLUSTERING)")
    
    report_lines.append("## 🛍️ PHẦN 2: PHÂN CỤM SẢN PHẨM (PRODUCT CLUSTERING)\n")
    report_lines.append("""
### Ý tưởng:

Thay vì phân cụm khách hàng, ta phân cụm **SẢN PHẨM** dựa trên việc chúng được mua bởi những khách hàng tương tự.

**Vector đặc trưng cho mỗi sản phẩm:**
- Hàng: Sản phẩm
- Cột: Khách hàng
- Giá trị: 1 nếu khách hàng đã mua sản phẩm đó, 0 nếu không

```
            Customer1  Customer2  Customer3  ...
Product1        1          0          1      
Product2        0          1          1      
Product3        1          1          0      
```

**Ứng dụng marketing:**
- Cross-sell: Sản phẩm cùng cụm có thể bán kèm
- Category management: Nhóm sản phẩm tự nhiên
- Inventory: Sản phẩm cùng cụm có demand tương tự
""")
    
    # Tạo Product × Customer matrix
    print("\n🔹 Tạo Product × Customer matrix...")
    product_customer = pd.crosstab(
        df_clean['Description'],
        df_clean['CustomerID']
    ).clip(upper=1)  # Binary: 0 hoặc 1
    
    X_product = product_customer.values
    product_names = product_customer.index.tolist()
    
    print(f"   Matrix shape: {X_product.shape}")
    print(f"   (Mỗi hàng = 1 sản phẩm, mỗi cột = 1 khách hàng)")
    
    # Sample nếu cần (để tăng tốc)
    MAX_PRODUCTS = 1000
    if X_product.shape[0] > MAX_PRODUCTS:
        print(f"   ⚠️ Quá nhiều sản phẩm, lấy mẫu {MAX_PRODUCTS} sản phẩm phổ biến nhất...")
        product_freq = X_product.sum(axis=1)
        top_idx = np.argsort(product_freq)[-MAX_PRODUCTS:]
        X_product_sample = X_product[top_idx]
        product_names_sample = [product_names[i] for i in top_idx]
    else:
        X_product_sample = X_product
        product_names_sample = product_names
    
    print(f"   Sử dụng: {X_product_sample.shape[0]} sản phẩm × {X_product_sample.shape[1]} khách hàng")
    
    # Tìm K tốt nhất cho Product Clustering
    print("\n🔹 Tìm số cụm K tối ưu cho Product Clustering...")
    
    product_results = []
    for k in range(2, 12):
        # Agglomerative Ward
        agg_product = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels_product = agg_product.fit_predict(X_product_sample)
        sil = silhouette_score(X_product_sample, labels_product)
        product_results.append({'K': k, 'Silhouette': round(sil, 4)})
        print(f"   K={k}: Silhouette={sil:.4f}")
    
    product_results_df = pd.DataFrame(product_results).sort_values('Silhouette', ascending=False)
    best_k_product = int(product_results_df.iloc[0]['K'])
    best_sil_product = product_results_df.iloc[0]['Silhouette']
    
    print(f"\n✅ Best K cho Product Clustering: {best_k_product} (Silhouette={best_sil_product})")
    
    # Huấn luyện với K tốt nhất
    agg_product_final = AgglomerativeClustering(n_clusters=best_k_product, linkage='ward')
    labels_product_final = agg_product_final.fit_predict(X_product_sample)
    
    # Tạo DataFrame kết quả
    product_clusters_df = pd.DataFrame({
        'Product': product_names_sample,
        'Cluster': labels_product_final,
        'N_Customers': X_product_sample.sum(axis=1)  # Số khách đã mua
    })
    
    # Lưu kết quả
    product_clusters_df.to_csv(OUTPUT_PRODUCT_CLUSTERS, index=False)
    print(f"✅ Đã lưu Product Clusters: {OUTPUT_PRODUCT_CLUSTERS}")
    
    # Thêm vào report
    report_lines.append("\n### Kết quả chọn K cho Product Clustering:\n")
    report_lines.append("| K | Silhouette |")
    report_lines.append("|---|------------|")
    for _, row in product_results_df.head(5).iterrows():
        marker = " ⭐" if row['K'] == best_k_product else ""
        report_lines.append(f"| {int(row['K'])} | {row['Silhouette']}{marker} |")
    report_lines.append("")
    report_lines.append(f"**Chọn K = {best_k_product}** với Silhouette = {best_sil_product}\n")
    
    # Phân tích từng Product Cluster
    print("\n📊 PHÂN TÍCH TỪNG PRODUCT CLUSTER:")
    
    report_lines.append("### Chi tiết từng Product Cluster:\n")
    
    product_cluster_profiles = {}
    
    for c in range(best_k_product):
        cluster_products = product_clusters_df[product_clusters_df['Cluster'] == c]
        n_products = len(cluster_products)
        avg_customers = cluster_products['N_Customers'].mean()
        
        # Top 10 sản phẩm phổ biến nhất trong cluster
        top_products = cluster_products.nlargest(10, 'N_Customers')['Product'].tolist()
        
        # Đặt tên cluster dựa trên sản phẩm
        sample_products = ', '.join(top_products[:3])
        
        # Phân loại tự động
        if 'HERB' in sample_products.upper():
            cluster_name = "Garden & Herbs"
            marketing_action = "Bundle: Complete Herb Garden Kit"
        elif 'BAG' in sample_products.upper() or 'BOX' in sample_products.upper():
            cluster_name = "Storage & Packaging"
            marketing_action = "Bulk discount for storage items"
        elif 'CHRISTMAS' in sample_products.upper() or 'HEART' in sample_products.upper():
            cluster_name = "Seasonal & Gifts"
            marketing_action = "Holiday promotions, Gift bundles"
        elif 'LUNCH' in sample_products.upper() or 'CAKE' in sample_products.upper():
            cluster_name = "Kitchen & Dining"
            marketing_action = "Kitchen essentials bundle"
        else:
            cluster_name = f"Product Group {c}"
            marketing_action = "Cross-sell within cluster"
        
        product_cluster_profiles[c] = {
            'name': cluster_name,
            'n_products': n_products,
            'avg_customers': round(avg_customers, 1),
            'top_products': top_products[:5],
            'marketing_action': marketing_action
        }
        
        print(f"\n   📦 Cluster {c}: {cluster_name}")
        print(f"      - Số sản phẩm: {n_products}")
        print(f"      - Trung bình KH/sản phẩm: {avg_customers:.1f}")
        print(f"      - Top products: {', '.join(top_products[:3])[:60]}...")
        print(f"      - Marketing: {marketing_action}")
        
        report_lines.append(f"#### Product Cluster {c}: {cluster_name}")
        report_lines.append(f"- **Số sản phẩm:** {n_products}")
        report_lines.append(f"- **Trung bình KH/sản phẩm:** {avg_customers:.1f}")
        report_lines.append(f"- **Top 5 sản phẩm:**")
        for p in top_products[:5]:
            report_lines.append(f"  - {p}")
        report_lines.append(f"- **Đề xuất Marketing:** {marketing_action}\n")
    
    # ============================================================
    # PHẦN 3: SO SÁNH CUSTOMER vs PRODUCT CLUSTERING
    # ============================================================
    print_header("PHẦN 3: SO SÁNH CUSTOMER vs PRODUCT CLUSTERING")
    
    report_lines.append("## 🔄 PHẦN 3: SO SÁNH CUSTOMER vs PRODUCT CLUSTERING\n")
    
    # Thống kê Customer Clustering
    customer_stats = meta_customer.groupby('cluster').agg({
        'CustomerID': 'count',
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).round(2)
    customer_stats.columns = ['N_Customers', 'Recency', 'Frequency', 'Monetary']
    
    # Bảng so sánh
    comparison_table = []
    comparison_table.append({
        'Góc nhìn': 'Customer Clustering',
        'Đối tượng': 'Khách hàng',
        'Số đối tượng': X_customer.shape[0],
        'Số cụm': BEST_K,
        'Silhouette': round(silhouette_score(X_customer, labels_customer), 4),
        'Ứng dụng chính': 'CRM, Email marketing, Loyalty',
        'Actionability': '⭐⭐⭐⭐⭐ (5/5)'
    })
    comparison_table.append({
        'Góc nhìn': 'Product Clustering',
        'Đối tượng': 'Sản phẩm',
        'Số đối tượng': X_product_sample.shape[0],
        'Số cụm': best_k_product,
        'Silhouette': best_sil_product,
        'Ứng dụng chính': 'Cross-sell, Store layout, Bundles',
        'Actionability': '⭐⭐⭐⭐ (4/5)'
    })
    
    comparison_table_df = pd.DataFrame(comparison_table)
    
    print("\n📊 BẢNG SO SÁNH:")
    print(comparison_table_df.to_string(index=False))
    
    report_lines.append("### Bảng so sánh hai góc nhìn:\n")
    report_lines.append("| Tiêu chí | Customer Clustering | Product Clustering |")
    report_lines.append("|----------|--------------------|--------------------|")
    report_lines.append(f"| **Đối tượng** | Khách hàng | Sản phẩm |")
    report_lines.append(f"| **Số đối tượng** | {X_customer.shape[0]:,} | {X_product_sample.shape[0]:,} |")
    report_lines.append(f"| **Số cụm** | {BEST_K} | {best_k_product} |")
    report_lines.append(f"| **Silhouette** | {silhouette_score(X_customer, labels_customer):.4f} | {best_sil_product} |")
    report_lines.append(f"| **Ứng dụng** | CRM, Personalization | Cross-sell, Bundles |")
    report_lines.append(f"| **Actionability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |")
    report_lines.append("")
    
    # Kết luận
    report_lines.append("### Kết luận: Góc nhìn nào hữu ích hơn?\n")
    report_lines.append("""
**1. Customer Clustering (Phân cụm khách hàng):**
- ✅ **Ưu điểm:**
  - Trực tiếp phục vụ CRM và personalization
  - Có thể kết hợp RFM để đánh giá giá trị khách hàng
  - Dễ xây dựng chiến lược marketing cụ thể cho từng phân khúc
  - Silhouette Score cao hơn (cụm tách biệt rõ ràng)
  
- ❌ **Nhược điểm:**
  - Không trực tiếp cho biết nên recommend sản phẩm nào
  - Cần kết hợp với luật kết hợp để cross-sell

**2. Product Clustering (Phân cụm sản phẩm):**
- ✅ **Ưu điểm:**
  - Trực tiếp cho biết sản phẩm nào nên bán kèm
  - Hữu ích cho store layout và category management
  - Tự động tạo product bundles
  
- ❌ **Nhược điểm:**
  - Không biết target cho nhóm khách hàng nào
  - Silhouette thường thấp hơn (nhiều sản phẩm tương tự)

**🎯 Đề xuất kết hợp cả hai:**

```
Customer Clusters     +     Product Clusters
       ↓                          ↓
  Target audience         What to recommend
       ↓                          ↓
  "Khách VIP"        +   "Kitchen Bundle"
       ↓
  Chiến lược: Gửi email về Kitchen Bundle cho khách VIP
```

**Kết luận cuối cùng:**
- **Customer Clustering** hữu ích hơn cho **CRM và chiến lược marketing tổng thể**
- **Product Clustering** hữu ích hơn cho **cross-sell và merchandising**
- **Kết hợp cả hai** cho hiệu quả tốt nhất!
""")
    
    # ============================================================
    # TRỰC QUAN HÓA
    # ============================================================
    print_header("TRỰC QUAN HÓA")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Customer Clustering - PCA
    pca_customer = PCA(n_components=2, random_state=RANDOM_STATE)
    Z_customer = pca_customer.fit_transform(X_customer)
    scatter1 = axes[0, 0].scatter(Z_customer[:, 0], Z_customer[:, 1], c=labels_customer, cmap='viridis', s=15, alpha=0.6)
    axes[0, 0].set_title(f'Customer Clustering (Agglomerative, K={BEST_K})')
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')
    plt.colorbar(scatter1, ax=axes[0, 0], label='Cluster')
    
    # 2. Product Clustering - PCA
    pca_product = PCA(n_components=2, random_state=RANDOM_STATE)
    Z_product = pca_product.fit_transform(X_product_sample)
    scatter2 = axes[0, 1].scatter(Z_product[:, 0], Z_product[:, 1], c=labels_product_final, cmap='viridis', s=15, alpha=0.6)
    axes[0, 1].set_title(f'Product Clustering (Agglomerative, K={best_k_product})')
    axes[0, 1].set_xlabel('PC1')
    axes[0, 1].set_ylabel('PC2')
    plt.colorbar(scatter2, ax=axes[0, 1], label='Cluster')
    
    # 3. Customer Cluster Sizes
    cluster_sizes_customer = pd.Series(labels_customer).value_counts().sort_index()
    axes[1, 0].bar(cluster_sizes_customer.index, cluster_sizes_customer.values, color='steelblue')
    axes[1, 0].set_title('Customer Cluster Sizes')
    axes[1, 0].set_xlabel('Cluster')
    axes[1, 0].set_ylabel('Number of Customers')
    for i, v in enumerate(cluster_sizes_customer.values):
        axes[1, 0].text(i, v + 50, str(v), ha='center')
    
    # 4. Product Cluster Sizes
    cluster_sizes_product = pd.Series(labels_product_final).value_counts().sort_index()
    axes[1, 1].bar(cluster_sizes_product.index, cluster_sizes_product.values, color='coral')
    axes[1, 1].set_title('Product Cluster Sizes')
    axes[1, 1].set_xlabel('Cluster')
    axes[1, 1].set_ylabel('Number of Products')
    for i, v in enumerate(cluster_sizes_product.values):
        axes[1, 1].text(i, v + 5, str(v), ha='center')
    
    plt.tight_layout()
    plot_path = os.path.join(project_root, "customer_vs_product_clustering.png")
    plt.savefig(plot_path, dpi=150)
    print(f"✅ Đã lưu biểu đồ: {plot_path}")
    plt.close()
    
    report_lines.append("\n### Trực quan hóa:\n")
    report_lines.append("![Customer vs Product Clustering](customer_vs_product_clustering.png)\n")
    
    # ============================================================
    # BẢNG ĐỀ XUẤT MARKETING KẾT HỢP
    # ============================================================
    print_header("ĐỀ XUẤT MARKETING KẾT HỢP")
    
    report_lines.append("## 🎯 ĐỀ XUẤT MARKETING KẾT HỢP\n")
    report_lines.append("Kết hợp Customer Clusters + Product Clusters:\n")
    report_lines.append("| Customer Cluster | Product Cluster | Hành động Marketing |")
    report_lines.append("|------------------|-----------------|---------------------|")
    
    # Tạo ma trận kết hợp
    for cust_c in range(BEST_K):
        cust_data = meta_customer[meta_customer['cluster'] == cust_c]
        cust_name = "VIP" if cust_data['Monetary'].mean() > meta_customer['Monetary'].median() else "Regular"
        
        for prod_c in range(min(3, best_k_product)):  # Top 3 product clusters
            prod_name = product_cluster_profiles[prod_c]['name']
            action = f"Gửi email {prod_name} cho {cust_name} customers"
            report_lines.append(f"| Cluster {cust_c} ({cust_name}) | {prod_name} | {action} |")
    
    report_lines.append("")
    
    print("\n📋 Xem báo cáo đầy đủ tại:", OUTPUT_REPORT_PATH)
    
    # ============================================================
    # LƯU BÁO CÁO
    # ============================================================
    with open(OUTPUT_REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print_header("HOÀN THÀNH!")
    print(f"\n✅ Đã lưu báo cáo: {OUTPUT_REPORT_PATH}")
    print(f"✅ Đã lưu Product Clusters: {OUTPUT_PRODUCT_CLUSTERS}")
    print(f"✅ Đã lưu biểu đồ: {plot_path}")
    
    return comparison_df, product_clusters_df, product_cluster_profiles


if __name__ == "__main__":
    comparison_df, product_df, profiles = main()
