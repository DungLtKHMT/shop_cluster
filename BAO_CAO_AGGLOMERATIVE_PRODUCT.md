# 🌳 BÁO CÁO MỞ RỘNG: AGGLOMERATIVE CLUSTERING & PRODUCT CLUSTERING

**Ngày tạo:** 2025-12-30 02:33:51

---

## 📁 DỮ LIỆU ĐẦU VÀO

- Số giao dịch: 485,123
- Số khách hàng: 3,921
- Số sản phẩm: 4,007
- Số luật kết hợp: 1,794

## 📊 PHẦN 1: SO SÁNH K-MEANS vs AGGLOMERATIVE CLUSTERING

### Giải thích thuật toán Agglomerative:


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


### Bảng so sánh K-Means vs Agglomerative:

| Model | N Clusters | Silhouette ↑ | DBI ↓ | CH ↑ |
|-------|------------|--------------|-------|------|
| K-Means (K=2) | 2 | 0.9537 ⭐ | 0.2492 | 20998.51 |
| Agglomerative Ward (K=2) | 2 | 0.9523 | 0.2929 | 19264.26 |
| Agglomerative Complete (K=2) | 2 | 0.9512 | 0.0934 | 15193.99 |
| K-Means (K=3) | 3 | 0.9385 | 0.7125 | 12168.85 |
| Agglomerative Ward (K=3) | 3 | 0.9441 | 1.1462 | 13108.8 |
| Agglomerative Complete (K=3) | 3 | 0.94 | 0.3144 | 9226.06 |
| K-Means (K=4) | 4 | 0.937 | 0.8145 | 8962.44 |
| Agglomerative Ward (K=4) | 4 | 0.8301 | 1.0909 | 10842.44 |
| Agglomerative Complete (K=4) | 4 | 0.9436 | 0.3533 | 8009.3 |
| K-Means (K=5) | 5 | 0.9386 | 0.745 | 8011.96 |
| Agglomerative Ward (K=5) | 5 | 0.8319 | 0.978 | 9901.24 |
| Agglomerative Complete (K=5) | 5 | 0.9453 | 0.9003 | 8220.25 |
| K-Means (K=6) | 6 | 0.9422 | 0.6759 | 7342.39 |
| Agglomerative Ward (K=6) | 6 | 0.8335 | 0.8454 | 9784.63 |
| Agglomerative Complete (K=6) | 6 | 0.9453 | 0.5106 | 7609.83 |


**Kết luận:** Mô hình **K-Means (K=2)** cho kết quả tốt nhất với Silhouette = 0.9537

## 🛍️ PHẦN 2: PHÂN CỤM SẢN PHẨM (PRODUCT CLUSTERING)


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


### Kết quả chọn K cho Product Clustering:

| K | Silhouette |
|---|------------|
| 2 | 0.1675 ⭐ |
| 3 | 0.1459 |
| 4 | 0.0896 |
| 9 | 0.0313 |
| 8 | 0.0288 |

**Chọn K = 2** với Silhouette = 0.1675

### Chi tiết từng Product Cluster:

#### Product Cluster 0: Seasonal & Gifts
- **Số sản phẩm:** 104
- **Trung bình KH/sản phẩm:** 327.0
- **Top 5 sản phẩm:**
  - WHITE HANGING HEART T-LIGHT HOLDER
  - REGENCY CAKESTAND 3 TIER
  - PARTY BUNTING
  - ASSORTED COLOUR BIRD ORNAMENT
  - NATURAL SLATE HEART CHALKBOARD 
- **Đề xuất Marketing:** Holiday promotions, Gift bundles

#### Product Cluster 1: Seasonal & Gifts
- **Số sản phẩm:** 896
- **Trung bình KH/sản phẩm:** 150.7
- **Top 5 sản phẩm:**
  - PAPER CHAIN KIT 50'S CHRISTMAS 
  - BAKING SET 9 PIECE RETROSPOT 
  - REX CASH+CARRY JUMBO SHOPPER
  - PAPER CHAIN KIT VINTAGE CHRISTMAS
  - VINTAGE SNAP CARDS
- **Đề xuất Marketing:** Holiday promotions, Gift bundles

## 🔄 PHẦN 3: SO SÁNH CUSTOMER vs PRODUCT CLUSTERING

### Bảng so sánh hai góc nhìn:

| Tiêu chí | Customer Clustering | Product Clustering |
|----------|--------------------|--------------------|
| **Đối tượng** | Khách hàng | Sản phẩm |
| **Số đối tượng** | 3,921 | 1,000 |
| **Số cụm** | 2 | 2 |
| **Silhouette** | 0.9523 | 0.1675 |
| **Ứng dụng** | CRM, Personalization | Cross-sell, Bundles |
| **Actionability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### Kết luận: Góc nhìn nào hữu ích hơn?


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


### Trực quan hóa:

![Customer vs Product Clustering](customer_vs_product_clustering.png)

## 🎯 ĐỀ XUẤT MARKETING KẾT HỢP

Kết hợp Customer Clusters + Product Clusters:

| Customer Cluster | Product Cluster | Hành động Marketing |
|------------------|-----------------|---------------------|
| Cluster 0 (VIP) | Seasonal & Gifts | Gửi email Seasonal & Gifts cho VIP customers |
| Cluster 0 (VIP) | Seasonal & Gifts | Gửi email Seasonal & Gifts cho VIP customers |
| Cluster 1 (VIP) | Seasonal & Gifts | Gửi email Seasonal & Gifts cho VIP customers |
| Cluster 1 (VIP) | Seasonal & Gifts | Gửi email Seasonal & Gifts cho VIP customers |
