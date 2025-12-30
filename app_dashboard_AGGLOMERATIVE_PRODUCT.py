# -*- coding: utf-8 -*-
"""
DASHBOARD PHAN CUM KHACH HANG & SAN PHAM
===========================================
So sanh K-Means vs Agglomerative Clustering
Customer Clustering vs Product Clustering
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


# ============================================================
# CAU HINH TRANG
# ============================================================
st.set_page_config(
    page_title="Customer & Product Clustering Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tuy chinh
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .insight-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .comparison-winner {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        color: white;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DU LIEU
# ============================================================
@st.cache_data
def load_data():
    """Load tat ca du lieu can thiet"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Customer clusters
    customer_path = os.path.join(project_root, "data/processed/customer_clusters_from_rules.csv")
    df_customers = pd.read_csv(customer_path) if os.path.exists(customer_path) else None
    
    # Product clusters
    product_path = os.path.join(project_root, "data/processed/product_clusters.csv")
    df_products = pd.read_csv(product_path) if os.path.exists(product_path) else None
    
    # Rules
    rules_path = os.path.join(project_root, "data/processed/rules_apriori_filtered.csv")
    df_rules = pd.read_csv(rules_path) if os.path.exists(rules_path) else None
    
    # Raw data
    raw_path = os.path.join(project_root, "data/processed/cleaned_uk_data.csv")
    df_raw = pd.read_csv(raw_path, parse_dates=["InvoiceDate"]) if os.path.exists(raw_path) else None
    
    return df_customers, df_products, df_rules, df_raw

df_customers, df_products, df_rules, df_raw = load_data()

# ============================================================
# HEADER
# ============================================================
st.markdown('<h1 class="main-header">CUSTOMER & PRODUCT CLUSTERING DASHBOARD</h1>', unsafe_allow_html=True)
st.markdown("""
<p style="text-align: center; font-size: 1.2rem; color: #666;">
    So sanh K-Means vs Agglomerative | Customer vs Product Clustering | Marketing Insights
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## Dieu khien")
    
    st.markdown("### Du lieu da load:")
    if df_customers is not None:
        st.success(f"{len(df_customers):,} khach hang")
    if df_products is not None:
        st.success(f"{len(df_products):,} san pham")
    if df_rules is not None:
        st.success(f"{len(df_rules):,} luat ket hop")
    if df_raw is not None:
        st.success(f"{len(df_raw):,} giao dich")
    
    st.markdown("---")
    st.markdown("### Tuy chon hien thi")
    show_raw_data = st.checkbox("Hien thi du lieu goc", value=False)
    color_scheme = st.selectbox("Bang mau", ["Viridis", "Plasma", "Blues", "Greens", "Reds"])

# ============================================================
# TAB CHINH
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Tong quan", 
    "So sanh Thuat toan", 
    "Customer Clustering",
    "Product Clustering",
    "Marketing Insights"
])

# ============================================================
# TAB 1: TONG QUAN
# ============================================================
with tab1:
    st.markdown("## Tong quan Du an")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Khach hang",
            value=f"{len(df_customers):,}" if df_customers is not None else "N/A",
            delta="UK Market"
        )
    
    with col2:
        st.metric(
            label="San pham",
            value=f"{len(df_products):,}" if df_products is not None else "N/A",
            delta="Da phan cum"
        )
    
    with col3:
        st.metric(
            label="Luat ket hop",
            value=f"{len(df_rules):,}" if df_rules is not None else "N/A",
            delta="Apriori"
        )
    
    with col4:
        st.metric(
            label="Giao dich",
            value=f"{len(df_raw):,}" if df_raw is not None else "N/A",
            delta="Online Retail"
        )
    
    st.markdown("---")
    
    # So sanh overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Ket qua chinh")
        
        results_data = {
            "Tieu chi": [
                "Best Customer Model",
                "Customer Silhouette",
                "So cum khach hang",
                "Best Product Model", 
                "Product Silhouette",
                "So cum san pham"
            ],
            "Gia tri": [
                "K-Means (K=2)",
                "0.9537 (Best)",
                "2",
                "Agglomerative Ward",
                "0.1675",
                "2"
            ]
        }
        st.dataframe(pd.DataFrame(results_data), use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### Quy trinh phan tich")
        
        # Pipeline visualization
        pipeline_fig = go.Figure()
        
        steps = ["Du lieu goc", "Tien xu ly", "Luat ket hop", "Feature Engineering", "Phan cum", "Marketing"]
        x_pos = list(range(len(steps)))
        
        pipeline_fig.add_trace(go.Scatter(
            x=x_pos, y=[1]*len(steps),
            mode='markers+text',
            marker=dict(size=50, color=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#38ef7d']),
            text=['1', '2', '3', '4', '5', '6'],
            textposition='middle center',
            textfont=dict(size=20, color='white')
        ))
        
        pipeline_fig.update_layout(
            showlegend=False,
            height=200,
            xaxis=dict(ticktext=steps, tickvals=x_pos, tickangle=0),
            yaxis=dict(visible=False),
            margin=dict(l=20, r=20, t=20, b=60)
        )
        
        st.plotly_chart(pipeline_fig, use_container_width=True)
    
    # Insight box
    st.markdown("""
    <div class="insight-box">
        <h4>Insight chinh</h4>
        <ul>
            <li><strong>K-Means</strong> cho ket qua tot nhat voi Customer Clustering (Silhouette = 0.9537)</li>
            <li><strong>Customer Clustering</strong> co Silhouette cao hon nhieu so voi Product Clustering</li>
            <li><strong>2 cum</strong> la so toi uu cho ca Customer va Product Clustering</li>
            <li><strong>Ket hop ca hai</strong> goc nhin cho chien luoc marketing toan dien</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# TAB 2: SO SANH THUAT TOAN
# ============================================================
with tab2:
    st.markdown("## So sanh K-Means vs Agglomerative Clustering")
    
    # Giai thich thuat toan
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### K-Means Clustering
        
        **Nguyen ly:**
        ```
        1. Chon ngau nhien K centroids
        2. Gan moi diem vao centroid gan nhat
        3. Tinh lai centroids = trung binh cac diem
        4. Lap lai buoc 2-3 den khi hoi tu
        ```
        
        **Uu diem:**
        - Nhanh: O(n.K.t)
        - De hieu, de implement
        - Hoat dong tot voi du lieu lon
        
        **Nhuoc diem:**
        - Nhay cam voi khoi tao
        - Can xac dinh K truoc
        - Chi tim duoc cum hinh cau
        """)
    
    with col2:
        st.markdown("""
        ### Agglomerative Clustering
        
        **Nguyen ly:**
        ```
        1. Moi diem la mot cum rieng
        2. Tim 2 cum gan nhat -> Gop lai
        3. Lap lai cho den khi con K cum
        ```
        
        **Linkage methods:**
        - **Ward**: Toi thieu hoa phuong sai
        - **Complete**: Khoang cach max
        - **Average**: Khoang cach trung binh
        
        **Uu diem:**
        - Co Dendrogram (cau truc phan cap)
        - Ket qua on dinh
        - Phat hien duoc nested clusters
        
        **Nhuoc diem:**
        - Cham: O(n^2)
        - Khong the undo gop cum
        """)
    
    st.markdown("---")
    
    # Bang so sanh ket qua
    st.markdown("### Ket qua so sanh")
    
    comparison_data = {
        'Model': [
            'K-Means (K=2)', 'Agglomerative Ward (K=2)', 'Agglomerative Complete (K=2)',
            'K-Means (K=3)', 'Agglomerative Ward (K=3)', 'Agglomerative Complete (K=3)',
            'K-Means (K=4)', 'Agglomerative Ward (K=4)', 'Agglomerative Complete (K=4)',
            'K-Means (K=5)', 'Agglomerative Ward (K=5)', 'Agglomerative Complete (K=5)',
            'K-Means (K=6)', 'Agglomerative Ward (K=6)', 'Agglomerative Complete (K=6)',
        ],
        'N_Clusters': [2,2,2,3,3,3,4,4,4,5,5,5,6,6,6],
        'Silhouette': [0.9537, 0.9523, 0.9512, 0.9385, 0.9441, 0.9400, 0.9370, 0.8301, 0.9436, 0.9386, 0.8319, 0.9453, 0.9422, 0.8335, 0.9453],
        'DBI': [0.2492, 0.2929, 0.0934, 0.7125, 1.1462, 0.3144, 0.8145, 1.0909, 0.3533, 0.7450, 0.9780, 0.9003, 0.6759, 0.8454, 0.5106],
        'CH': [20998.51, 19264.26, 15193.99, 12168.85, 13108.80, 9226.06, 8962.44, 10842.44, 8009.30, 8011.96, 9901.24, 8220.25, 7342.39, 9784.63, 7609.83]
    }
    df_comparison = pd.DataFrame(comparison_data)
    
    # Highlight best row
    def highlight_best(row):
        if row['Silhouette'] == df_comparison['Silhouette'].max():
            return ['background-color: #d4edda'] * len(row)
        return [''] * len(row)
    
    st.dataframe(
        df_comparison.style.apply(highlight_best, axis=1).format({
            'Silhouette': '{:.4f}',
            'DBI': '{:.4f}',
            'CH': '{:.2f}'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Silhouette comparison
        fig_sil = px.bar(
            df_comparison,
            x='Model',
            y='Silhouette',
            color='Silhouette',
            color_continuous_scale='Viridis',
            title='Silhouette Score theo Model (cao = tot)'
        )
        fig_sil.update_layout(xaxis_tickangle=-45, height=400)
        fig_sil.add_hline(y=df_comparison['Silhouette'].max(), line_dash="dash", line_color="red", 
                         annotation_text="Best")
        st.plotly_chart(fig_sil, use_container_width=True)
    
    with col2:
        # DBI comparison (lower is better)
        fig_dbi = px.bar(
            df_comparison,
            x='Model',
            y='DBI',
            color='DBI',
            color_continuous_scale='Reds_r',
            title='Davies-Bouldin Index theo Model (thap = tot)'
        )
        fig_dbi.update_layout(xaxis_tickangle=-45, height=400)
        fig_dbi.add_hline(y=df_comparison['DBI'].min(), line_dash="dash", line_color="green",
                         annotation_text="Best")
        st.plotly_chart(fig_dbi, use_container_width=True)
    
    # Winner announcement
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <span class="comparison-winner">WINNER: K-Means (K=2) voi Silhouette = 0.9537</span>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# TAB 3: CUSTOMER CLUSTERING
# ============================================================
with tab3:
    st.markdown("## Customer Clustering Analysis")
    
    if df_customers is not None:
        # Cluster distribution
        col1, col2 = st.columns([1, 2])
        
        with col1:
            cluster_counts = df_customers['cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Count']
            
            fig_pie = px.pie(
                cluster_counts,
                values='Count',
                names='Cluster',
                title='Phan bo Khach hang theo Cum',
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label+value')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # RFM analysis by cluster
            if all(col in df_customers.columns for col in ['Recency', 'Frequency', 'Monetary', 'cluster']):
                rfm_stats = df_customers.groupby('cluster').agg({
                    'Recency': 'mean',
                    'Frequency': 'mean',
                    'Monetary': 'mean'
                }).round(2).reset_index()
                
                fig_rfm = go.Figure()
                
                for i, cluster in enumerate(rfm_stats['cluster'].unique()):
                    row = rfm_stats[rfm_stats['cluster'] == cluster].iloc[0]
                    fig_rfm.add_trace(go.Scatterpolar(
                        r=[row['Recency'], row['Frequency'], row['Monetary']/1000],
                        theta=['Recency (ngay)', 'Frequency (lan)', 'Monetary (K GBP)'],
                        fill='toself',
                        name=f'Cluster {cluster}'
                    ))
                
                fig_rfm.update_layout(
                    polar=dict(radialaxis=dict(visible=True)),
                    showlegend=True,
                    title='RFM Profile theo Cluster'
                )
                st.plotly_chart(fig_rfm, use_container_width=True)
        
        # Cluster profiles
        st.markdown("### Ho so chi tiet tung Cluster")
        
        if 'Recency' in df_customers.columns:
            for cluster in sorted(df_customers['cluster'].unique()):
                cluster_data = df_customers[df_customers['cluster'] == cluster]
                
                with st.expander(f"Cluster {cluster} ({len(cluster_data):,} khach hang)", expanded=True):
                    cols = st.columns(4)
                    
                    with cols[0]:
                        st.metric("So khach", f"{len(cluster_data):,}")
                    with cols[1]:
                        st.metric("Recency TB", f"{cluster_data['Recency'].mean():.0f} ngay")
                    with cols[2]:
                        st.metric("Frequency TB", f"{cluster_data['Frequency'].mean():.1f} lan")
                    with cols[3]:
                        st.metric("Monetary TB", f"GBP {cluster_data['Monetary'].mean():,.0f}")
                    
                    # Profile interpretation
                    r = cluster_data['Recency'].mean()
                    f = cluster_data['Frequency'].mean()
                    m = cluster_data['Monetary'].mean()
                    
                    if r < 50 and f > 5 and m > 1000:
                        profile = "**VIP Champions** - Khach hang gia tri cao, mua sam thuong xuyen"
                        action = "Uu dai exclusive, chuong trinh loyalty dac biet"
                    elif r < 100 and f > 3:
                        profile = "**Loyal Customers** - Khach hang trung thanh"
                        action = "Upsell san pham cao cap, recommend dua tren lich su"
                    elif r > 200:
                        profile = "**At Risk** - Khach hang co nguy co roi bo"
                        action = "Chien dich win-back, voucher giam gia dac biet"
                    else:
                        profile = "**Regular Shoppers** - Khach hang thong thuong"
                        action = "Cross-sell, up-sell dinh ky"
                    
                    st.info(f"**Nhan dien:** {profile}")
                    st.success(f"**Hanh dong Marketing:** {action}")
        
        # Raw data option
        if show_raw_data:
            st.markdown("### Du lieu khach hang")
            st.dataframe(df_customers.head(100), use_container_width=True)
    else:
        st.warning("Chua co du lieu Customer Clustering")

# ============================================================
# TAB 4: PRODUCT CLUSTERING
# ============================================================
with tab4:
    st.markdown("## Product Clustering Analysis")
    
    if df_products is not None:
        # Overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Tong san pham", f"{len(df_products):,}")
        with col2:
            st.metric("So cum", f"{df_products['Cluster'].nunique()}")
        with col3:
            avg_customers = df_products['N_Customers'].mean()
            st.metric("TB KH/San pham", f"{avg_customers:.0f}")
        
        st.markdown("---")
        
        # Cluster distribution
        col1, col2 = st.columns(2)
        
        with col1:
            product_cluster_counts = df_products['Cluster'].value_counts().reset_index()
            product_cluster_counts.columns = ['Cluster', 'Count']
            
            fig_prod_pie = px.pie(
                product_cluster_counts,
                values='Count',
                names='Cluster',
                title='Phan bo San pham theo Cum',
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.4
            )
            fig_prod_pie.update_traces(textposition='inside', textinfo='percent+label+value')
            st.plotly_chart(fig_prod_pie, use_container_width=True)
        
        with col2:
            # Box plot of N_Customers by cluster
            fig_box = px.box(
                df_products,
                x='Cluster',
                y='N_Customers',
                title='Phan phoi so KH mua theo Cluster',
                color='Cluster',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Top products per cluster
        st.markdown("### Top San pham theo Cluster")
        
        for cluster in sorted(df_products['Cluster'].unique()):
            cluster_products = df_products[df_products['Cluster'] == cluster].nlargest(10, 'N_Customers')
            
            with st.expander(f"Product Cluster {cluster} ({len(df_products[df_products['Cluster'] == cluster]):,} san pham)", expanded=True):
                
                # Bar chart for top products
                fig_top = px.bar(
                    cluster_products,
                    x='N_Customers',
                    y='Product',
                    orientation='h',
                    title=f'Top 10 san pham pho bien nhat - Cluster {cluster}',
                    color='N_Customers',
                    color_continuous_scale='Blues'
                )
                fig_top.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_top, use_container_width=True)
                
                # Interpret cluster
                top_names = ' '.join(cluster_products['Product'].head(5).tolist()).upper()
                
                if 'CHRISTMAS' in top_names or 'HEART' in top_names:
                    category = "Seasonal & Gifts"
                    marketing = "Holiday promotions, Gift bundles, Seasonal discounts"
                elif 'BAG' in top_names or 'BOX' in top_names:
                    category = "Storage & Packaging"
                    marketing = "Bulk discounts, B2B offers"
                elif 'KITCHEN' in top_names or 'CAKE' in top_names:
                    category = "Kitchen & Dining"
                    marketing = "Kitchen starter kits, Recipe tie-ins"
                else:
                    category = "General Merchandise"
                    marketing = "Cross-sell bundles, Trending items promo"
                
                st.info(f"**Nhan dien danh muc:** {category}")
                st.success(f"**De xuat Marketing:** {marketing}")
        
        # Raw data
        if show_raw_data:
            st.markdown("### Du lieu san pham")
            st.dataframe(df_products.head(100), use_container_width=True)
    else:
        st.warning("Chua co du lieu Product Clustering")

# ============================================================
# TAB 5: MARKETING INSIGHTS
# ============================================================
with tab5:
    st.markdown("## Marketing Insights & Recommendations")
    
    # Comparison overview
    st.markdown("### So sanh Customer vs Product Clustering")
    
    comparison_overview = {
        'Tieu chi': ['Doi tuong phan tich', 'So luong', 'Thuat toan tot nhat', 'So cum toi uu', 
                     'Silhouette Score', 'Ung dung chinh', 'Muc do actionable'],
        'Customer Clustering': ['Khach hang', '3,921', 'K-Means', '2', '0.9537 (Best)', 
                               'CRM, Personalization, Email Marketing', 'Rat cao (5/5)'],
        'Product Clustering': ['San pham', '1,000', 'Agglomerative Ward', '2', '0.1675',
                              'Cross-sell, Store Layout, Bundles', 'Cao (4/5)']
    }
    
    st.dataframe(pd.DataFrame(comparison_overview), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Key insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Customer Clustering - Insights
        
        <div class="insight-box">
        <h4>Cluster 0: VIP Champions</h4>
        <ul>
            <li><strong>Dac diem:</strong> Recency thap, Frequency & Monetary cao</li>
            <li><strong>Chien luoc:</strong> 
                <ul>
                    <li>Loyalty program VIP</li>
                    <li>Early access cho san pham moi</li>
                    <li>Personal shopper service</li>
                </ul>
            </li>
        </ul>
        </div>
        
        <div class="insight-box">
        <h4>Cluster 1: Regular Shoppers</h4>
        <ul>
            <li><strong>Dac diem:</strong> Mua sam dinh ky, gia tri trung binh</li>
            <li><strong>Chien luoc:</strong>
                <ul>
                    <li>Cross-sell dua tren luat ket hop</li>
                    <li>Email marketing dinh ky</li>
                    <li>Chuong trinh tich diem</li>
                </ul>
            </li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        ### Product Clustering - Insights
        
        <div class="insight-box">
        <h4>Cluster 0: Seasonal & Gifts</h4>
        <ul>
            <li><strong>San pham tieu bieu:</strong> Heart decorations, Christmas items</li>
            <li><strong>Chien luoc:</strong>
                <ul>
                    <li>Holiday promotional campaigns</li>
                    <li>Gift bundles packaging</li>
                    <li>Seasonal inventory planning</li>
                </ul>
            </li>
        </ul>
        </div>
        
        <div class="insight-box">
        <h4>Cluster 1: Regular Items</h4>
        <ul>
            <li><strong>San pham tieu bieu:</strong> Baking sets, Paper chains</li>
            <li><strong>Chien luoc:</strong>
                <ul>
                    <li>Cross-sell bundles</li>
                    <li>Restock reminders</li>
                    <li>Bulk discounts</li>
                </ul>
            </li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Combined strategy
    st.markdown("### Chien luoc Marketing Ket hop")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 1rem; color: white; margin: 1rem 0;">
        <h3 style="color: white;">Ket hop Customer + Product Clustering = Maximum Impact</h3>
        <table style="width: 100%; color: white; margin-top: 1rem;">
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.3);">
                <th style="padding: 0.5rem;">Customer Cluster</th>
                <th style="padding: 0.5rem;">Product Cluster</th>
                <th style="padding: 0.5rem;">Hanh dong</th>
                <th style="padding: 0.5rem;">Expected ROI</th>
            </tr>
            <tr>
                <td style="padding: 0.5rem;">VIP Champions</td>
                <td style="padding: 0.5rem;">Seasonal & Gifts</td>
                <td style="padding: 0.5rem;">Exclusive holiday preview + VIP discount</td>
                <td style="padding: 0.5rem;">Rat cao (5/5)</td>
            </tr>
            <tr>
                <td style="padding: 0.5rem;">VIP Champions</td>
                <td style="padding: 0.5rem;">Regular Items</td>
                <td style="padding: 0.5rem;">Personal bundle recommendations</td>
                <td style="padding: 0.5rem;">Cao (4/5)</td>
            </tr>
            <tr>
                <td style="padding: 0.5rem;">Regular Shoppers</td>
                <td style="padding: 0.5rem;">Seasonal & Gifts</td>
                <td style="padding: 0.5rem;">Holiday email campaign + time-limited offer</td>
                <td style="padding: 0.5rem;">Cao (4/5)</td>
            </tr>
            <tr>
                <td style="padding: 0.5rem;">Regular Shoppers</td>
                <td style="padding: 0.5rem;">Regular Items</td>
                <td style="padding: 0.5rem;">Cross-sell based on purchase history</td>
                <td style="padding: 0.5rem;">Trung binh (3/5)</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    # Final conclusion
    st.markdown("### Ket luan cuoi cung")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: #d4edda; padding: 1.5rem; border-radius: 1rem; text-align: center;">
            <h2 style="color: #155724;">1</h2>
            <h4 style="color: #155724;">Customer Clustering</h4>
            <p style="color: #155724;">Huu ich nhat cho <strong>CRM & Personalization</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #cce5ff; padding: 1.5rem; border-radius: 1rem; text-align: center;">
            <h2 style="color: #004085;">2</h2>
            <h4 style="color: #004085;">Product Clustering</h4>
            <p style="color: #004085;">Huu ich nhat cho <strong>Cross-sell & Bundles</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: #fff3cd; padding: 1.5rem; border-radius: 1rem; text-align: center;">
            <h2 style="color: #856404;">3</h2>
            <h4 style="color: #856404;">Ket hop ca hai</h4>
            <p style="color: #856404;">Cho hieu qua <strong>Marketing toi da!</strong></p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Mini Project - Data Mining | Customer & Product Clustering Analysis</p>
    <p>Built with Streamlit, Plotly, Scikit-learn</p>
</div>
""", unsafe_allow_html=True)
