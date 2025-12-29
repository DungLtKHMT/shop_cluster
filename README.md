# 🔄 HƯỚNG DẪN LUỒNG XỬ LÝ VÀ ĐIỀU CHỈNH THAM SỐ

## 📊 TỔNG QUAN PIPELINE

Dự án phân cụm khách hàng dựa trên luật kết hợp bao gồm 6 bước chính:

```
[1] Tiền xử lý dữ liệu
    ↓
[2] Chuẩn bị Basket (giỏ hàng)
    ↓
[3] Khai phá luật kết hợp (Apriori/FP-Growth)
    ↓
[4] Trích xuất đặc trưng từ luật
    ↓
[5] Phân cụm khách hàng (K-Means)
    ↓
[6] Phân tích và Diễn giải kết quả
```

---

## 🔍 CHI TIẾT TỪNG BƯỚC

### **BƯỚC 1: Tiền xử lý dữ liệu**
📁 **Notebook**: `preprocessing_and_eda.ipynb`  
🔧 **Class**: `DataCleaner`

#### **Chức năng:**
- Load dữ liệu từ file CSV gốc
- Làm sạch dữ liệu:
  - Loại bỏ hóa đơn hủy (InvoiceNo bắt đầu bằng 'C')
  - Chỉ giữ khách hàng UK
  - Loại bỏ Quantity ≤ 0 hoặc UnitPrice ≤ 0
  - Bỏ Description bị thiếu
- Tạo cột `TotalPrice = Quantity × UnitPrice`
- Tính RFM (Recency, Frequency, Monetary)

#### **Output:**
- `data/processed/cleaned_uk_data.csv`

#### **Tham số điều chỉnh:**

| Tham số | Vị trí | Mục đích | Gợi ý |
|---------|--------|----------|-------|
| `Country` filter | `clean_data()` | Chọn thị trường | Có thể thử "Germany", "France" hoặc tất cả |
| `snapshot_date` | `compute_rfm()` | Điểm mốc tính Recency | Mặc định: max(InvoiceDate) + 1 ngày |
| Ngưỡng lọc Quantity | `clean_data()` | Loại giao dịch bất thường | Hiện tại: > 0, có thể tăng lên ≥ 2 |

#### **💡 Gợi ý cải thiện:**
- Thử lọc theo `Quantity < 100` để loại các đơn hàng bán buôn quá lớn
- Thử lọc theo `UnitPrice < 1000` để loại outliers về giá
- Phân tích theo mùa (thêm feature tháng, quý)

---

### **BƯỚC 2: Chuẩn bị Basket**
📁 **Notebook**: `basket_preparation.ipynb`  
🔧 **Class**: `BasketPreparer`

#### **Chức năng:**
- Chuyển dữ liệu giao dịch thành ma trận boolean Invoice × Item
- Mỗi dòng = 1 giỏ hàng (InvoiceNo)
- Mỗi cột = 1 sản phẩm (Description)
- Giá trị: True nếu sản phẩm có trong giỏ, False nếu không

#### **Output:**
- `data/processed/basket_bool.parquet`

#### **Tham số điều chỉnh:**

| Tham số | Vị trí | Mục đích | Gợi ý |
|---------|--------|----------|-------|
| `invoice_col` | `__init__()` | Định nghĩa "basket" | Có thể dùng CustomerID thay vì InvoiceNo |
| `min_items` | `create_basket_matrix()` | Lọc basket quá nhỏ | Mặc định: 1, đề xuất: 2-3 |
| `max_items` | `create_basket_matrix()` | Lọc basket quá lớn | Không có, nên thêm ~50 |
| `min_support_item` | Tùy chỉnh | Lọc item xuất hiện ít | Chưa có, nên thêm |

#### **💡 Gợi ý cải thiện:**
- **Lọc basket size**: Chỉ giữ giỏ có 2-50 items để tránh nhiễu
- **Lọc rare items**: Loại items xuất hiện < 0.1% baskets
- **Group items**: Gom nhóm sản phẩm tương tự (ví dụ: "RED MUG", "BLUE MUG" → "MUG")

---

### **BƯỚC 3: Khai phá luật kết hợp**
📁 **Notebook**: `apriori_modelling.ipynb`, `fp_growth_modelling.ipynb`  
🔧 **Class**: `AssociationRulesMiner`, `FPGrowthMiner`

> ⚡ **LƯU Ý QUAN TRỌNG**: Theo yêu cầu đề bài, bạn **CHỈ CẦN CHỌN 1 TRONG 2** thuật toán (Apriori **HOẶC** FP-Growth).  
> 
> **Khuyến nghị: Dùng FP-Growth** vì:
> - ✅ **Nhanh hơn** Apriori (đặc biệt với min_support thấp)
> - ✅ Không sinh candidate items → tiết kiệm bộ nhớ
> - ✅ Kết quả tương đương về chất lượng luật
> - ✅ File output: `rules_fpgrowth_filtered.csv` (đã có sẵn trong dự án)
>
> Nếu muốn so sánh 2 thuật toán → Làm phần **nâng cao** (không bắt buộc)

#### **Chức năng:**
- Tìm tập phổ biến (frequent itemsets)
- Sinh luật kết hợp: Antecedent → Consequent
- Tính support, confidence, lift

#### **Output:**
- `data/processed/rules_apriori_filtered.csv` (nếu dùng Apriori)
- `data/processed/rules_fpgrowth_filtered.csv` ⭐ (nếu dùng FP-Growth - Khuyến nghị)

#### **Tham số điều chỉnh:**

| Tham số | Vị trí | Ảnh hưởng | Gợi ý điều chỉnh |
|---------|--------|-----------|------------------|
| **`min_support`** | `mine_frequent_itemsets()` | **Quan trọng nhất**<br>Càng thấp → nhiều luật hơn<br>Càng cao → ít luật hơn nhưng mạnh hơn | **Baseline**: 0.01 (1%)<br>**Conservative**: 0.02-0.05<br>**Aggressive**: 0.005 |
| **`min_confidence`** | `generate_rules()` | Độ tin cậy tối thiểu của luật<br>Nếu mua A thì % mua B | **Baseline**: 0.3 (30%)<br>**High quality**: 0.5-0.7<br>**Exploratory**: 0.2 |
| **`min_lift`** | `filter_rules()` | Loại luật không có giá trị<br>Lift > 1: A và B có liên quan | **Must have**: > 1.0<br>**Good**: > 1.2<br>**Strong**: > 1.5 |
| `max_len` | `mine_frequent_itemsets()` | Độ dài tối đa của itemset | 2-4 (dễ diễn giải)<br>5-8 (phức tạp hơn) |
| `metric` | `generate_rules()` | Metric để sinh luật | 'confidence', 'lift', 'leverage' |

#### **💡 Gợi ý cải thiện:**

**Scenario 1: Quá ít luật (< 50)**
```python
min_support = 0.005  # Giảm từ 0.01
min_confidence = 0.2  # Giảm từ 0.3
min_lift = 1.0        # Giảm từ 1.2
```

**Scenario 2: Quá nhiều luật (> 1000)**
```python
min_support = 0.02   # Tăng từ 0.01
min_confidence = 0.5 # Tăng từ 0.3
min_lift = 1.5       # Tăng từ 1.2
max_len = 3          # Giới hạn độ dài
```

**Scenario 3: Chất lượng tốt nhất**
```python
min_support = 0.01
min_confidence = 0.4
min_lift = 1.3
# + Lọc theo antecedent_len >= 2 (ít nhất 2 items)
```

---

### **BƯỚC 4: Trích xuất đặc trưng từ luật**
📁 **Notebook**: `clustering_from_rules.ipynb`  
🔧 **Class**: `RuleBasedCustomerClusterer`

#### **Chức năng:**
Biến luật kết hợp thành vector đặc trưng cho khách hàng:

1. **Load Top-K luật** từ file rules
2. **Build Customer × Item matrix** (boolean)
3. **Build Customer × Rule matrix**:
   - Mỗi cột = 1 luật
   - Giá trị = 1 nếu khách mua đủ antecedents của luật, 0 nếu không
   - (Tuỳ chọn) Nhân trọng số theo lift/confidence
4. **Ghép RFM** (nếu dùng)
5. **Chuẩn hóa** (StandardScaler)

#### **Tham số điều chỉnh:**

| Tham số | Vị trí | Ảnh hưởng | Gợi ý |
|---------|--------|-----------|-------|
| **`TOP_K_RULES`** | `load_rules()` | **Số lượng luật dùng làm features**<br>Càng nhiều → nhiều chiều hơn | **Baseline**: 200<br>**Small**: 50-100<br>**Large**: 300-500 |
| **`SORT_RULES_BY`** | `load_rules()` | Tiêu chí chọn luật quan trọng | **'lift'**: Độ liên quan<br>**'confidence'**: Độ tin cậy<br>**'support'**: Độ phổ biến |
| **`WEIGHTING`** | `build_rule_feature_matrix()` | **Quan trọng**<br>Cách tính giá trị feature | **'none'**: 0/1 binary<br>**'lift'**: Nhân lift<br>**'confidence'**: Nhân confidence<br>**'lift_x_conf'**: Lift × Confidence |
| **`MIN_ANTECEDENT_LEN`** | `build_rule_feature_matrix()` | Lọc luật có antecedent quá ngắn | **1**: Tất cả luật<br>**2**: Ít nhất 2 items<br>**3**: Phức tạp hơn |
| **`USE_RFM`** | `build_final_features()` | Có ghép RFM không? | **True**: Baseline + RFM<br>**False**: Chỉ dùng rules |
| **`RFM_SCALE`** | `build_final_features()` | Chuẩn hóa RFM không? | **True**: Khuyến nghị<br>**False**: Không scale |
| **`RULE_SCALE`** | `build_final_features()` | Chuẩn hóa rule features? | **False**: Giữ nguyên 0/1<br>**True**: Scale về [-1, 1] |
| `min_support`<br>`min_confidence`<br>`min_lift` | `load_rules()` | Lọc lần 2 (sau khi đã có rules) | Tuỳ chọn, để None nếu đã lọc tốt ở bước 3 |

#### **💡 Gợi ý cải thiện:**

**Biến thể 1: Baseline (Rule-only Binary)**
```python
TOP_K_RULES = 200
WEIGHTING = "none"           # Binary 0/1
USE_RFM = False
MIN_ANTECEDENT_LEN = 1
```

**Biến thể 2: Rule + RFM**
```python
TOP_K_RULES = 200
WEIGHTING = "none"
USE_RFM = True               # Thêm RFM
RFM_SCALE = True
```

**Biến thể 3: Weighted Rules**
```python
TOP_K_RULES = 200
WEIGHTING = "lift_x_conf"    # Trọng số kép
USE_RFM = False
MIN_ANTECEDENT_LEN = 2       # Chỉ luật phức tạp
```

**Biến thể 4: Full Features (Khuyến nghị)**
```python
TOP_K_RULES = 300
WEIGHTING = "lift"           # Trọng số lift
USE_RFM = True
RFM_SCALE = True
MIN_ANTECEDENT_LEN = 2
RULE_SCALE = False           # Giữ nguyên để diễn giải
```

**So sánh hiệu quả:**
- **Ít luật + không trọng số**: Nhanh, đơn giản, dễ diễn giải
- **Nhiều luật + trọng số**: Chính xác hơn, phức tạp hơn
- **Rule + RFM**: Kết hợp hành vi mua và giá trị khách hàng

---

### **BƯỚC 5: Phân cụm K-Means**
📁 **Notebook**: `clustering_from_rules.ipynb`  
🔧 **Method**: `choose_k_by_silhouette()`, `fit_kmeans()`

#### **Chức năng:**
1. **Chọn K tối ưu**: Thử K từ K_MIN đến K_MAX, tính Silhouette Score
2. **Huấn luyện K-Means**: Fit mô hình với K đã chọn
3. **Gán nhãn cụm**: Mỗi khách hàng được gán vào 1 cụm

#### **Tham số điều chỉnh:**

| Tham số | Vị trí | Ảnh hưởng | Gợi ý |
|---------|--------|-----------|-------|
| **`K_MIN`** | `choose_k_by_silhouette()` | Số cụm tối thiểu thử nghiệm | **2** (tối thiểu) |
| **`K_MAX`** | `choose_k_by_silhouette()` | Số cụm tối đa thử nghiệm | **8-12**<br>(không nên quá nhiều vì khó diễn giải) |
| **`N_CLUSTERS`** | `fit_kmeans()` | **Số cụm cuối cùng**<br>None = tự động chọn theo Silhouette | **None**: Tự động<br>**3-6**: Thủ công (dễ diễn giải) |
| `RANDOM_STATE` | `fit_kmeans()` | Seed để tái tạo kết quả | 42 (cố định) |
| `n_init` | KMeans parameter | Số lần khởi tạo ngẫu nhiên | 'auto' hoặc 10-20 |
| `max_iter` | KMeans parameter | Số vòng lặp tối đa | 300 (mặc định) |

#### **💡 Gợi ý cải thiện:**

**Chọn K theo ngữ cảnh:**
- **K=3-4**: Phân khúc đơn giản (VIP, Trung bình, Thấp)
- **K=5-6**: Phân khúc chi tiết (nhiều chiến lược hơn)
- **K=7+**: Quá phức tạp, khó triển khai marketing

**Phương pháp chọn K:**
1. **Silhouette Score** (đang dùng):
   - Cao nhất (~0.4-0.6): Cụm tách rõ
   - Trung bình (0.2-0.4): Cụm chấp nhận được
   - Thấp (<0.2): Cụm kém

2. **Elbow Method** (có thể thêm):
   ```python
   # Vẽ biểu đồ Inertia (within-cluster sum of squares)
   inertias = []
   for k in range(2, 11):
       km = KMeans(n_clusters=k, random_state=42)
       km.fit(X)
       inertias.append(km.inertia_)
   # Tìm "khuỷu tay" (elbow) trên đồ thị
   ```

3. **Davies-Bouldin Index** (thấp càng tốt):
   ```python
   from sklearn.metrics import davies_bouldin_score
   score = davies_bouldin_score(X, labels)
   ```

**So sánh thuật toán khác (Nâng cao):**

| Thuật toán | Ưu điểm | Nhược điểm | Khi nào dùng |
|------------|---------|------------|--------------|
| **K-Means** | Nhanh, đơn giản, dễ diễn giải | Giả định cụm tròn, nhạy với outliers | **Baseline** (bắt buộc) |
| **Agglomerative** | Không cần chọn K trước, phân cấp | Chậm với dữ liệu lớn | Muốn dendrogram, phân cấp khách hàng |
| **DBSCAN** | Tìm cụm hình dạng bất kỳ, tự động tìm outliers | Khó chọn epsilon, không ổn định với mật độ khác nhau | Dữ liệu có nhiều noise |
| **HDBSCAN** | Cải tiến DBSCAN, tự động hơn | Chậm hơn | Muốn kết quả tốt nhất, không quan tâm tốc độ |

---

### **BƯỚC 6: Phân tích và Diễn giải**
📁 **Notebook**: `clustering_from_rules.ipynb`  
🔧 **Output**: Bảng profiling, trực quan hóa, chiến lược

#### **Chức năng:**
1. **Profiling cụm**: Thống kê đặc điểm từng cụm
2. **Top rules theo cụm**: Luật nào được kích hoạt nhiều nhất
3. **Trực quan hóa 2D**: Giảm chiều PCA/SVD, vẽ scatter plot
4. **Đặt tên và chiến lược**: Gắn nhãn ý nghĩa cho cụm

#### **Tham số điều chỉnh:**

| Tham số | Vị trí | Ảnh hưởng | Gợi ý |
|---------|--------|-----------|-------|
| `PROJECTION_METHOD` | `project_2d()` | Phương pháp giảm chiều | **'pca'**: Tuyến tính<br>**'svd'**: Sparse data |
| `PLOT_2D` | Notebook | Có vẽ scatter plot không | True (khuyến nghị) |
| Top rules to show | Custom | Số luật hiển thị mỗi cụm | 5-10 luật |

#### **💡 Gợi ý phân tích:**

**Bảng Profiling mẫu:**
```
Cluster | Size | Recency | Frequency | Monetary | Top Rules | Tên | Chiến lược
--------|------|---------|-----------|----------|-----------|-----|------------
0       | 450  | 15      | 8         | 1200     | Tea→Mug   | VIP | Chăm sóc riêng, ưu đãi đặc biệt
1       | 820  | 45      | 3         | 300      | Candle    | Casual | Cross-sell, bundle deals
2       | 210  | 180     | 1         | 150      | Gift      | Dormant | Kích hoạt lại, giảm giá mạnh
```

**Các metric đánh giá:**
- **Silhouette Score**: 0.3-0.5 là tốt
- **Cluster size balance**: Không có cụm quá nhỏ (< 5%) hoặc quá lớn (> 70%)
- **RFM variance**: Các cụm có RFM khác biệt rõ rệt

---

## 🎯 CHIẾN LƯỢC ĐIỀU CHỈNH THAM SỐ

### **Kịch bản 1: Cụm không tách rõ (Silhouette < 0.2)**

**Nguyên nhân:**
- Đặc trưng kém phân biệt
- Quá ít features
- Không chuẩn hóa

**Giải pháp:**
1. Tăng `TOP_K_RULES` lên 300-500
2. Dùng `WEIGHTING = "lift"` hoặc `"lift_x_conf"`
3. Bật `USE_RFM = True` và `RFM_SCALE = True`
4. Tăng `MIN_ANTECEDENT_LEN = 2` để lọc luật chất lượng
5. Thử giảm K xuống 3-4

---

### **Kịch bản 2: Quá nhiều cụm nhỏ lẻ**

**Nguyên nhân:**
- K quá lớn
- Dữ liệu có nhiều outliers

**Giải pháp:**
1. Giảm `K_MAX` xuống 6-8
2. Lọc khách hàng có `Frequency < 2` trước khi phân cụm
3. Thử DBSCAN để tự động loại outliers

---

### **Kịch bản 3: Tất cả cụm giống nhau**

**Nguyên nhân:**
- Đặc trưng không đa dạng
- Luật quá phổ biến (support cao)

**Giải pháp:**
1. Giảm `min_support` ở bước 3 để có luật đa dạng hơn
2. Tăng `min_lift` lên 1.5 để chỉ lấy luật mạnh
3. Sắp xếp luật theo `confidence` thay vì `lift`
4. Tăng `TOP_K_RULES` lên 400-500

---

### **Kịch bản 4: Cụm không có ý nghĩa marketing**

**Nguyên nhân:**
- Phân cụm chỉ dựa vào rules, thiếu context giá trị khách hàng

**Giải pháp:**
1. **Bắt buộc** bật `USE_RFM = True`
2. Cân nhắc tăng tỷ trọng RFM:
   ```python
   # Nhân RFM với trọng số lớn hơn
   rfm_values = rfm_values * 2  # Hoặc 3
   ```
3. Thêm đặc trưng khác: Tổng số đơn, Trung bình giá trị đơn

---

## � CÁCH TRÌNH BÀY LỰA CHỌN LUẬT (YÊU CẦU BẮT BUỘC)

### **Mục đích:**
Nhóm phải **giải thích rõ ràng** và **minh chứng bằng số liệu** cách lựa chọn luật kết hợp làm đầu vào cho phân cụm.

### **Nội dung cần trình bày:**

#### **1. Giải thích quyết định lựa chọn**

Template mẫu:

```markdown
### Lựa chọn luật kết hợp cho Feature Engineering

#### 1.1. Nguồn dữ liệu luật
- **File sử dụng**: `rules_apriori_filtered.csv` (hoặc `rules_fpgrowth_filtered.csv`)
- **Tổng số luật ban đầu**: 1,234 luật
- **Thuật toán**: Apriori (hoặc FP-Growth)

#### 1.2. Tiêu chí chọn Top-K luật
- **Top-K**: Chọn 200 luật hàng đầu
- **Lý do chọn K=200**: 
  - Đủ lớn để capture được đa dạng hành vi mua sắm
  - Không quá nhiều tránh overfitting và chiều cao
  - Thử nghiệm với K=100, 200, 300 cho thấy K=200 cho Silhouette score tốt nhất

#### 1.3. Tiêu chí sắp xếp
- **Sắp xếp theo**: `lift` (giảm dần)
- **Lý do**: 
  - Lift đo độ liên quan giữa antecedent và consequent
  - Lift > 1 nghĩa là mua A làm tăng xác suất mua B
  - Ưu tiên luật có lift cao để tạo features phân biệt rõ ràng giữa các nhóm khách hàng

**Alternative**: Có thể sắp xếp theo `confidence` nếu muốn ưu tiên độ tin cậy

#### 1.4. Ngưỡng lọc bổ sung
- **min_support**: 0.01 (1%) - Chỉ giữ luật xuất hiện ít nhất 1% baskets
- **min_confidence**: 0.3 (30%) - Độ tin cậy tối thiểu
- **min_lift**: 1.2 - Chỉ giữ luật có tương quan dương mạnh
- **min_antecedent_len**: 2 - Chỉ giữ luật có ít nhất 2 items trong antecedent (loại luật đơn giản)

#### 1.5. Lý do chọn bộ ngưỡng này
- Support 1%: Đảm bảo luật đủ phổ biến, không phải noise
- Confidence 30%: Cân bằng giữa số lượng và chất lượng luật
- Lift 1.2: Chỉ lấy luật có ý nghĩa thống kê (lift càng cao càng tốt)
- Antecedent ≥ 2: Luật phức tạp hơn giúp phân biệt hành vi mua kèm
```

---

#### **2. Bảng 10 luật tiêu biểu**

**Code để trích xuất:**

```python
# Sau khi load rules
clusterer = RuleBasedCustomerClusterer(df_clean)
clusterer.build_customer_item_matrix()
rules_top = clusterer.load_rules(
    rules_csv_path="data/processed/rules_apriori_filtered.csv",
    top_k=200,
    sort_by="lift",
    min_support=0.01,
    min_confidence=0.3,
    min_lift=1.2
)

# Hiển thị 10 luật tiêu biểu
print("### Top 10 luật được chọn làm đầu vào:")
display_cols = ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']
print(rules_top.head(10)[display_cols].to_markdown(index=True))
```

**Kết quả mẫu:**

| # | Antecedents | Consequents | Support | Confidence | Lift |
|---|-------------|-------------|---------|------------|------|
| 1 | REGENCY CAKESTAND 3 TIER, PINK REGENCY TEACUP AND SAUCER | GREEN REGENCY TEACUP AND SAUCER | 0.0156 | 0.7692 | 15.38 |
| 2 | GREEN REGENCY TEACUP AND SAUCER, ROSES REGENCY TEACUP AND SAUCER | PINK REGENCY TEACUP AND SAUCER | 0.0134 | 0.7500 | 14.42 |
| 3 | JUMBO BAG RED RETROSPOT, LUNCH BAG RED RETROSPOT | CHARLOTTE BAG PINK POLKADOT | 0.0112 | 0.6923 | 12.85 |
| 4 | SET/6 RED SPOTTY PAPER CUPS, SET/6 RED SPOTTY PAPER PLATES | SET/20 RED RETROSPOT PAPER NAPKINS | 0.0145 | 0.8125 | 11.24 |
| 5 | PARTY BUNTING, POPCORN HOLDER | PAPER CHAIN KIT 50'S CHRISTMAS | 0.0098 | 0.6531 | 10.85 |
| 6 | PLASTERS IN TIN CIRCUS PARADE, PLASTERS IN TIN WOODLAND ANIMALS | PLASTERS IN TIN SPACEBOY | 0.0087 | 0.6154 | 9.73 |
| 7 | FELTCRAFT PRINCESS CHARLOTTE DOLL, MINI CAKE STAND 2 TIER | ALARM CLOCK BAKELIKE PINK | 0.0076 | 0.5833 | 8.92 |
| 8 | GARDENERS KNEELING PAD CUP OF TEA, GARDENERS KNEELING PAD KEEP CALM | GARDENERS KNEELING PAD RETROSPOT | 0.0123 | 0.7241 | 8.45 |
| 9 | PACK OF 72 RETROSPOT CAKE CASES, SAVE THE PLANET MUG | RECIPE BOX PANTRY YELLOW DESIGN | 0.0065 | 0.5417 | 7.82 |
| 10 | DOORMAT NEW ENGLAND, WOOD 2 DRAWER CABINET WHITE FINISH | WOOD S/3 CABINET ANT WHITE FINISH | 0.0054 | 0.5000 | 7.14 |

**Nhận xét về chất lượng:**
- **Lift**: Tất cả > 7.0, chứng tỏ tương quan rất mạnh giữa các sản phẩm
- **Confidence**: Dao động 50-81%, đủ tin cậy để làm features
- **Support**: Từ 0.5% đến 1.6%, đảm bảo không quá phổ biến (universal) cũng không quá hiếm (noise)
- **Ý nghĩa kinh doanh**: Các luật phản ánh các nhóm sản phẩm:
  - Nhóm 1-2: Bộ tách trà Regency (khách hàng mua nhiều màu)
  - Nhóm 3-4: Đồ dùng tiệc (Red Retrospot collection)
  - Nhóm 5: Trang trí tiệc
  - Nhóm 6: Băng keo cá nhân (trẻ em)
  - Nhóm 7-10: Đồ gia dụng, trang trí nhà

---

#### **3. Phân tích phân bố luật**

```python
# Thống kê tổng quan
print(f"Số luật sau khi lọc: {len(rules_top)}")
print(f"\nPhân bố Support:")
print(rules_top['support'].describe())
print(f"\nPhân bố Confidence:")
print(rules_top['confidence'].describe())
print(f"\nPhân bố Lift:")
print(rules_top['lift'].describe())

# Visualize
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
rules_top['support'].hist(bins=30, ax=axes[0])
axes[0].set_title('Distribution of Support')
rules_top['confidence'].hist(bins=30, ax=axes[1])
axes[1].set_title('Distribution of Confidence')
rules_top['lift'].hist(bins=30, ax=axes[2])
axes[2].set_title('Distribution of Lift')
plt.tight_layout()
plt.show()
```

**Output mẫu:**
```
Số luật sau khi lọc: 200

Phân bố Support:
count    200.000
mean       0.015
std        0.008
min        0.010
25%        0.011
50%        0.013
75%        0.017
max        0.045

Phân bố Confidence:
count    200.000
mean       0.52
std        0.14
min        0.30
25%        0.42
50%        0.51
75%        0.63
max        0.85

Phân bố Lift:
count    200.000
mean       5.8
std        3.2
min        1.2
25%        3.4
50%        4.9
75%        7.1
max       18.5
```

**Nhận xét:**
- Support tập trung ở 1-2%, phù hợp với long-tail products
- Confidence trung bình 52%, cho thấy luật có độ tin cậy vừa phải
- Lift trung bình 5.8, cho thấy tương quan mạnh (>> 1.0)

---

#### **4. So sánh các phương án lựa chọn**

| Phương án | Top-K | Sort by | min_lift | Số luật cuối | Silhouette | Nhận xét |
|-----------|-------|---------|----------|--------------|------------|----------|
| **A (Baseline)** | 200 | lift | 1.0 | 200 | 0.34 | Baseline tốt |
| **B (Conservative)** | 150 | lift | 1.5 | 150 | 0.37 | ✅ Tốt nhất - chỉ lấy luật mạnh |
| **C (Aggressive)** | 300 | lift | 1.2 | 300 | 0.31 | Quá nhiều features, overfitting |
| **D (Confidence-based)** | 200 | confidence | 1.2 | 200 | 0.33 | Tương đương baseline |

**Kết luận**: Chọn phương án B với Top-150 luật có lift ≥ 1.5

---

### **Template code hoàn chỉnh cho notebook:**

```python
# Cell: Giải thích lựa chọn luật
print("="*80)
print("PHẦN 1: LỰA CHỌN LUẬT KẾT HỢP CHO FEATURE ENGINEERING")
print("="*80)

print("\n### 1. Nguồn dữ liệu")
print(f"- File: {RULES_INPUT_PATH}")
rules_raw = pd.read_csv(RULES_INPUT_PATH)
print(f"- Tổng số luật ban đầu: {len(rules_raw):,}")

print("\n### 2. Tiêu chí lựa chọn")
print(f"- Top-K: {TOP_K_RULES}")
print(f"- Sắp xếp theo: {SORT_RULES_BY}")
print(f"- Ngưỡng: min_support={MIN_SUPPORT}, min_confidence={MIN_CONFIDENCE}, min_lift={MIN_LIFT}")
print(f"- Độ dài antecedent tối thiểu: {MIN_ANTECEDENT_LEN}")

print("\n### 3. Lý do lựa chọn")
print("""
- TOP_K=200: Đủ lớn để capture hành vi đa dạng, không quá nhiều tránh overfitting
- Sort by lift: Ưu tiên luật có tương quan mạnh nhất
- min_lift=1.2: Chỉ giữ luật có ý nghĩa thống kê
- min_antecedent_len=2: Lọc luật phức tạp, loại luật đơn giản A→B
""")

# Cell: Load và hiển thị 10 luật tiêu biểu
clusterer = RuleBasedCustomerClusterer(df_clean)
clusterer.build_customer_item_matrix()
rules_top = clusterer.load_rules(
    rules_csv_path=RULES_INPUT_PATH,
    top_k=TOP_K_RULES,
    sort_by=SORT_RULES_BY,
    min_support=MIN_SUPPORT if 'MIN_SUPPORT' in globals() else None,
    min_confidence=MIN_CONFIDENCE if 'MIN_CONFIDENCE' in globals() else None,
    min_lift=MIN_LIFT if 'MIN_LIFT' in globals() else None,
)

print("\n### 4. Top 10 luật tiêu biểu:")
display_cols = ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']
print(rules_top.head(10)[display_cols].to_markdown(index=True, floatfmt=".4f"))

# Cell: Thống kê phân bố
print("\n### 5. Phân bố các chỉ số:")
print("\nSupport:")
print(rules_top['support'].describe())
print("\nConfidence:")
print(rules_top['confidence'].describe())
print("\nLift:")
print(rules_top['lift'].describe())

# Cell: Visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
rules_top['support'].hist(bins=30, ax=axes[0], edgecolor='black')
axes[0].set_title('Distribution of Support', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Support')
axes[0].axvline(rules_top['support'].mean(), color='red', linestyle='--', label=f"Mean={rules_top['support'].mean():.4f}")
axes[0].legend()

rules_top['confidence'].hist(bins=30, ax=axes[1], edgecolor='black', color='orange')
axes[1].set_title('Distribution of Confidence', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Confidence')
axes[1].axvline(rules_top['confidence'].mean(), color='red', linestyle='--', label=f"Mean={rules_top['confidence'].mean():.4f}")
axes[1].legend()

rules_top['lift'].hist(bins=30, ax=axes[2], edgecolor='black', color='green')
axes[2].set_title('Distribution of Lift', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Lift')
axes[2].axvline(rules_top['lift'].mean(), color='red', linestyle='--', label=f"Mean={rules_top['lift'].mean():.4f}")
axes[2].legend()

plt.tight_layout()
plt.savefig('figures/rules_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Đã hoàn thành phần trình bày lựa chọn luật!")
```

---

## 📈 ROADMAP THỰC HIỆN

### **Phase 1: Baseline (Bắt buộc)**
1. ✅ **Trình bày lựa chọn luật** (template ở trên)
2. ✅ Chạy với tham số mặc định
3. ✅ So sánh 2 biến thể:
   - Rule-only binary
   - Rule + RFM
4. ✅ Chọn K bằng Silhouette
5. ✅ Profiling và đặt tên cụm

### **Phase 2: Optimization**
1. Thử 2-3 cấu hình khác nhau của tham số
2. So sánh bằng bảng tổng hợp
3. Chọn cấu hình tốt nhất

### **Phase 3: Advanced (Nâng cao - Không bắt buộc)**
1. **So sánh Apriori vs FP-Growth**: Nếu chưa làm, thử cả 2 thuật toán và so sánh tốc độ, số lượng luật
2. So sánh với Agglomerative/DBSCAN
3. Thử basket clustering hoặc product clustering
4. Xây dựng Streamlit dashboard

---

## 📝 CHECKLIST KIỂM TRA

### **Bắt buộc:**
- [ ] **✅ Trình bày rõ cách chọn luật**: Top-K, sort_by, ngưỡng lọc, lý do
- [ ] **✅ Bảng 10 luật tiêu biểu**: Có đầy đủ support, confidence, lift
- [ ] **✅ Phân tích phân bố luật**: Histogram của support/confidence/lift
- [ ] Có ít nhất 2 biến thể feature engineering
- [ ] Silhouette score > 0.25
- [ ] Mỗi cụm có ít nhất 5% tổng khách hàng
- [ ] Có bảng profiling đầy đủ (size, RFM, top rules)
- [ ] Mỗi cụm có tên và chiến lược cụ thể
- [ ] Có trực quan hóa 2D
- [ ] Có so sánh các biến thể bằng bảng

### **Nâng cao (điểm tối đa):**
- [ ] Có so sánh thuật toán khác (Agglomerative/DBSCAN)
- [ ] Có dashboard Streamlit
- [ ] Thử basket/product/rule clustering

---

## 🎓 HƯỚNG DẪN THỰC HIỆN CHI TIẾT (STEP-BY-STEP)

Phần này giải thích **từng bước cụ thể** với **thuật ngữ chuyên ngành** và **vị trí trong hệ thống**.

---

### **BƯỚC 1: Chạy Notebook Clustering với Các Biến Thể**

#### **📚 Giải thích thuật ngữ:**

| Thuật ngữ | Giải thích | Ví dụ |
|-----------|------------|-------|
| **Feature Engineering** | Quá trình tạo ra các biến đầu vào (features/đặc trưng) cho mô hình ML từ dữ liệu thô | Từ luật "A→B" tạo ra feature: khách có mua A không? |
| **Biến thể (Variant)** | Các cách khác nhau để xây dựng features, khác nhau về tham số hoặc phương pháp | Biến thể 1: chỉ dùng rules; Biến thể 2: rules + RFM |
| **Top-K Rules** | Lấy K luật tốt nhất (theo lift/confidence) để làm features | Top-200 = lấy 200 luật có lift cao nhất |
| **Weighting** | Phương pháp gán trọng số cho feature thay vì chỉ 0/1 | Binary (0/1) vs Weighted (nhân lift) |
| **RFM** | Recency-Frequency-Monetary: đo lường giá trị khách hàng | R=15 (mua 15 ngày trước), F=8 (8 đơn), M=1200 (tổng chi 1200$) |

#### **📍 Vị trí trong hệ thống:**

- **File chính**: `notebooks/clustering_from_rules.ipynb` (gốc) hoặc `notebooks/runs/clustering_from_rules_run.ipynb` (đã chạy)
- **File config**: `run_papermill.py` (dòng 125-150)
- **Class xử lý**: `src/cluster_library.py` → `RuleBasedCustomerClusterer`
- **Dữ liệu đầu vào**: 
  - `data/processed/cleaned_uk_data.csv`
  - `data/processed/rules_fpgrowth_filtered.csv`
- **Dữ liệu đầu ra**: `data/processed/customer_clusters_from_rules.csv`

#### **🔧 Hướng dẫn thực hiện:**

**Bước 1.1: Tạo notebook mới cho từng biến thể**

```bash
# Tạo bản sao notebook cho các biến thể
cd notebooks
cp clustering_from_rules.ipynb clustering_variant_1_baseline.ipynb
cp clustering_from_rules.ipynb clustering_variant_2_rfm.ipynb
cp clustering_from_rules.ipynb clustering_variant_3_weighted.ipynb
```

**Bước 1.2: Cấu hình từng biến thể**

**Biến thể 1: Baseline (Rule-only Binary)**
```python
# Cell Parameters trong clustering_variant_1_baseline.ipynb
TOP_K_RULES = 200
SORT_RULES_BY = "lift"
WEIGHTING = "none"              # Binary 0/1
MIN_ANTECEDENT_LEN = 1
USE_RFM = False                 # KHÔNG dùng RFM
RFM_SCALE = False
RULE_SCALE = False

K_MIN = 3
K_MAX = 8
N_CLUSTERS = None               # Tự động chọn bằng Silhouette
```

**Biến thể 2: Rules + RFM**
```python
# Cell Parameters trong clustering_variant_2_rfm.ipynb
TOP_K_RULES = 200
SORT_RULES_BY = "lift"
WEIGHTING = "none"
MIN_ANTECEDENT_LEN = 1
USE_RFM = True                  # THÊM RFM
RFM_SCALE = True                # Chuẩn hóa RFM
RULE_SCALE = False

K_MIN = 3
K_MAX = 8
N_CLUSTERS = None
```

**Biến thể 3: Weighted Rules + RFM**
```python
# Cell Parameters trong clustering_variant_3_weighted.ipynb
TOP_K_RULES = 150               # Ít hơn nhưng chọn lọc hơn
SORT_RULES_BY = "lift"
WEIGHTING = "lift"              # Nhân trọng số lift
MIN_ANTECEDENT_LEN = 2          # Chỉ lấy luật phức tạp (≥2 items)
USE_RFM = True
RFM_SCALE = True
RULE_SCALE = False

K_MIN = 3
K_MAX = 8
N_CLUSTERS = None
```

**Bước 1.3: Chạy từng biến thể**

```python
# Chạy trong Jupyter hoặc VS Code
# Mở từng notebook và chạy tất cả cells (Run All)
# Hoặc dùng papermill:

import papermill as pm

variants = [
    ("clustering_variant_1_baseline.ipynb", {"USE_RFM": False, "WEIGHTING": "none"}),
    ("clustering_variant_2_rfm.ipynb", {"USE_RFM": True, "WEIGHTING": "none"}),
    ("clustering_variant_3_weighted.ipynb", {"USE_RFM": True, "WEIGHTING": "lift"}),
]

for nb_name, params in variants:
    pm.execute_notebook(
        f"notebooks/{nb_name}",
        f"notebooks/runs/{nb_name}",
        parameters=params,
        kernel_name="python3"
    )
```

**Bước 1.4: Ghi lại kết quả**

Tạo bảng so sánh trong notebook hoặc file Excel:

| Biến thể | TOP_K | Weighting | USE_RFM | MIN_ANT_LEN | K (chọn) | Silhouette | Inertia | Thời gian | Ghi chú |
|----------|-------|-----------|---------|-------------|----------|------------|---------|-----------|---------|
| 1 - Baseline | 200 | none | False | 1 | 5 | 0.32 | 15432 | 10s | Baseline đơn giản |
| 2 - RFM | 200 | none | True | 1 | 4 | 0.38 | 12890 | 12s | ✅ Tốt hơn baseline |
| 3 - Weighted | 150 | lift | True | 2 | 5 | 0.41 | 11234 | 15s | 🏆 Tốt nhất |

---

### **BƯỚC 2: Phân Tích và Profiling Từng Cụm**

#### **📚 Giải thích thuật ngữ:**

| Thuật ngữ | Giải thích | Mục đích |
|-----------|------------|----------|
| **Cluster Profiling** | Mô tả đặc điểm của từng cụm bằng thống kê tổng hợp | Hiểu "cụm này là ai?" |
| **Centroid** | Điểm trung tâm của cụm (trung bình các feature) | Đại diện cho cụm |
| **Within-cluster variance** | Độ phân tán trong cụm (mức độ đồng nhất) | Cụm càng "chặt" càng tốt |
| **Between-cluster variance** | Độ khác biệt giữa các cụm | Cụm càng "tách rời" càng tốt |
| **Top Rules per Cluster** | Các luật được kích hoạt nhiều nhất trong cụm | Hành vi đặc trưng của cụm |
| **Persona** | Mô tả nhân vật đại diện cho cụm | Ví dụ: "Bà nội trợ thích đồ bếp" |

#### **📍 Vị trí trong hệ thống:**

Thêm vào cuối notebook `clustering_from_rules.ipynb` hoặc tạo notebook mới `clustering_profiling.ipynb`

#### **🔧 Hướng dẫn thực hiện:**

**Bước 2.1: Load kết quả phân cụm**

```python
# Cell: Load dữ liệu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load kết quả
clusters_df = pd.read_csv("data/processed/customer_clusters_from_rules.csv")
rules_df = pd.read_csv("data/processed/rules_fpgrowth_filtered.csv")
df_clean = pd.read_csv("data/processed/cleaned_uk_data.csv")

print(f"Số khách hàng: {len(clusters_df)}")
print(f"Số cụm: {clusters_df['cluster'].nunique()}")
```

**Bước 2.2: Thống kê cơ bản theo cụm**

```python
# Cell: Thống kê tổng quan
profile = clusters_df.groupby('cluster').agg({
    'CustomerID': 'count',        # Số lượng
    'Recency': ['mean', 'median'],
    'Frequency': ['mean', 'median'],
    'Monetary': ['mean', 'median']
}).round(2)

profile.columns = ['Size', 'Recency_Mean', 'Recency_Median', 
                   'Frequency_Mean', 'Frequency_Median',
                   'Monetary_Mean', 'Monetary_Median']
profile['Percentage'] = (profile['Size'] / len(clusters_df) * 100).round(2)

print("="*80)
print("BẢNG PROFILING CƠ BẢN")
print("="*80)
print(profile.to_string())

# Vẽ biểu đồ
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Kích thước cụm
profile['Size'].plot(kind='bar', ax=axes[0,0], color='steelblue')
axes[0,0].set_title('Cluster Size')
axes[0,0].set_ylabel('Number of Customers')

# RFM theo cụm
profile[['Recency_Mean', 'Frequency_Mean']].plot(kind='bar', ax=axes[0,1])
axes[0,1].set_title('Average RFM by Cluster')

profile['Monetary_Mean'].plot(kind='bar', ax=axes[1,0], color='green')
axes[1,0].set_title('Average Monetary by Cluster')

# Tỷ lệ %
profile['Percentage'].plot(kind='pie', ax=axes[1,1], autopct='%1.1f%%')
axes[1,1].set_title('Cluster Distribution')

plt.tight_layout()
plt.savefig('figures/cluster_profiling_basic.png', dpi=150)
plt.show()
```

**Bước 2.3: Tìm Top Rules cho từng cụm**

```python
# Cell: Top Rules per Cluster
from cluster_library import RuleBasedCustomerClusterer

# Rebuild feature matrix để biết rule nào được kích hoạt
clusterer = RuleBasedCustomerClusterer(df_clean)
clusterer.build_customer_item_matrix()
rules_top = clusterer.load_rules("data/processed/rules_fpgrowth_filtered.csv", top_k=200)
X_rules = clusterer.build_rule_feature_matrix(weighting="none", min_antecedent_len=1)

# Tạo DataFrame rule activation
rule_activation = pd.DataFrame(
    X_rules, 
    columns=[f"rule_{i}" for i in range(X_rules.shape[1])],
    index=clusterer.customers_
)
rule_activation['cluster'] = clusters_df.set_index('CustomerID')['cluster']

# Tính tỷ lệ kích hoạt mỗi rule trong từng cụm
print("="*80)
print("TOP 10 LUẬT THEO TỪNG CỤM")
print("="*80)

for cluster_id in sorted(clusters_df['cluster'].unique()):
    cluster_data = rule_activation[rule_activation['cluster'] == cluster_id]
    
    # Tỷ lệ kích hoạt
    activation_rates = cluster_data.drop('cluster', axis=1).mean().sort_values(ascending=False).head(10)
    
    print(f"\n🔹 CLUSTER {cluster_id} (n={len(cluster_data)}):")
    print("-" * 80)
    
    for i, (rule_col, rate) in enumerate(activation_rates.items(), 1):
        rule_idx = int(rule_col.split('_')[1])
        rule_row = rules_top.iloc[rule_idx]
        
        print(f"{i}. [{rate*100:.1f}%] {rule_row['antecedents_str']} → {rule_row['consequents_str']}")
        print(f"   Support: {rule_row['support']:.3f}, Confidence: {rule_row['confidence']:.3f}, Lift: {rule_row['lift']:.2f}")
```

**Bước 2.4: Phân tích sâu hơn**

```python
# Cell: Chi tiết từng cụm
for cluster_id in sorted(clusters_df['cluster'].unique()):
    cluster_customers = clusters_df[clusters_df['cluster'] == cluster_id]
    
    print("\n" + "="*80)
    print(f"PHÂN TÍCH CHI TIẾT CLUSTER {cluster_id}")
    print("="*80)
    
    # 1. Thống kê RFM
    print("\n📊 Thống kê RFM:")
    print(cluster_customers[['Recency', 'Frequency', 'Monetary']].describe())
    
    # 2. Phân bố RFM
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cluster_customers['Recency'].hist(bins=30, ax=axes[0], edgecolor='black')
    axes[0].set_title(f'Cluster {cluster_id} - Recency Distribution')
    cluster_customers['Frequency'].hist(bins=30, ax=axes[1], edgecolor='black')
    axes[1].set_title(f'Cluster {cluster_id} - Frequency Distribution')
    cluster_customers['Monetary'].hist(bins=30, ax=axes[2], edgecolor='black')
    axes[2].set_title(f'Cluster {cluster_id} - Monetary Distribution')
    plt.tight_layout()
    plt.savefig(f'figures/cluster_{cluster_id}_rfm_dist.png', dpi=150)
    plt.show()
    
    # 3. Sản phẩm phổ biến nhất
    customer_ids = cluster_customers['CustomerID'].values
    cluster_transactions = df_clean[df_clean['CustomerID'].isin(customer_ids)]
    top_products = cluster_transactions['Description'].value_counts().head(10)
    
    print(f"\n🛍️ Top 10 sản phẩm được mua nhiều nhất:")
    for i, (product, count) in enumerate(top_products.items(), 1):
        print(f"  {i}. {product}: {count} lần")
```

---

### **BƯỚC 3: Đặt Tên Cụm và Chiến Lược Marketing**

#### **📚 Giải thích thuật ngữ:**

| Thuật ngữ | Giải thích | Ví dụ |
|-----------|------------|-------|
| **Segment Naming** | Đặt tên có ý nghĩa cho cụm thay vì số | Cluster 0 → "VIP Customers" |
| **Persona** | Mô tả ngắn gọn đặc điểm khách hàng cụm | "Khách hàng trung niên, mua sắm thường xuyên đồ gia dụng" |
| **Actionable Insights** | Thông tin có thể hành động được | "Nên gửi email ưu đãi vào cuối tuần" |
| **Marketing Strategy** | Chiến lược tiếp thị cụ thể cho cụm | Bundle promotion, Cross-sell, Retention campaign |
| **Customer Lifetime Value** | Giá trị khách hàng trong toàn bộ vòng đời | CLV = Frequency × Monetary |

#### **📍 Vị trí trong hệ thống:**

Tạo file mới: `notebooks/cluster_interpretation.ipynb` hoặc thêm vào cuối `clustering_from_rules.ipynb`

#### **🔧 Hướng dẫn thực hiện:**

**Bước 3.1: Phân tích và đặt tên**

```python
# Cell: Đặt tên và mô tả cụm
cluster_profiles = {
    0: {
        "name_en": "Casual Shoppers",
        "name_vi": "Khách Hàng Thường",
        "size": 3797,
        "percentage": 96.84,
        "rfm_profile": {
            "recency": 45,
            "frequency": 3,
            "monetary": 300
        },
        "persona": "Khách hàng mua sắm không thường xuyên, giá trị đơn hàng thấp, chủ yếu mua đồ trang trí nhỏ lẻ",
        "top_products": ["CANDLE", "GIFT CARD", "PAPER NAPKINS"],
        "behavior": "Mua theo nhu cầu đột xuất, không có pattern rõ ràng",
        "marketing_strategy": {
            "objective": "Tăng tần suất mua hàng và giá trị đơn hàng",
            "tactics": [
                "Email marketing với bundle deals (mua 2 tặng 1)",
                "Cross-sell: gợi ý sản phẩm liên quan khi checkout",
                "Loyalty program: tích điểm để khuyến khích quay lại",
                "Seasonal campaigns: gửi catalog vào dịp lễ"
            ],
            "expected_outcome": "Tăng Frequency từ 3 lên 5 đơn/năm, tăng Monetary 20%"
        },
        "budget_allocation": "40% (cụm lớn nhất)",
        "kpi": "Conversion rate, Average order value"
    },
    
    1: {
        "name_en": "VIP High-Value Customers",
        "name_vi": "Khách Hàng VIP",
        "size": 124,
        "percentage": 3.16,
        "rfm_profile": {
            "recency": 15,
            "frequency": 20,
            "monetary": 5000
        },
        "persona": "Khách hàng trung thành, mua sắm thường xuyên, giá trị cao, thích bộ sưu tập cao cấp",
        "top_products": ["REGENCY TEACUP SET", "CERAMIC STORAGE JAR", "VINTAGE ALARM CLOCK"],
        "behavior": "Mua theo bộ sưu tập, quan tâm chất lượng hơn giá cả",
        "marketing_strategy": {
            "objective": "Giữ chân và tăng giá trị lifetime",
            "tactics": [
                "VIP treatment: early access to new collections",
                "Personal shopper service: tư vấn 1-1",
                "Exclusive events: private sale, product launch",
                "Premium loyalty tier: cashback 10%, free shipping",
                "Birthday/anniversary gifts"
            ],
            "expected_outcome": "Retention rate 95%+, tăng Monetary 30%"
        },
        "budget_allocation": "60% (ROI cao nhất)",
        "kpi": "Customer retention rate, CLV"
    }
}

# In ra bảng tóm tắt
import json
print("="*80)
print("BẢNG TÓM TẮT PHÂN KHÚC KHÁCH HÀNG")
print("="*80)

for cluster_id, profile in cluster_profiles.items():
    print(f"\n🏷️  CLUSTER {cluster_id}: {profile['name_en']} ({profile['name_vi']})")
    print("-" * 80)
    print(f"📊 Quy mô: {profile['size']} khách ({profile['percentage']:.2f}%)")
    print(f"📈 RFM Profile: R={profile['rfm_profile']['recency']}, F={profile['rfm_profile']['frequency']}, M=${profile['rfm_profile']['monetary']}")
    print(f"👤 Persona: {profile['persona']}")
    print(f"🛍️  Top Products: {', '.join(profile['top_products'][:3])}")
    print(f"\n🎯 Chiến lược Marketing:")
    print(f"   Mục tiêu: {profile['marketing_strategy']['objective']}")
    print(f"   Chiến thuật:")
    for i, tactic in enumerate(profile['marketing_strategy']['tactics'], 1):
        print(f"     {i}. {tactic}")
    print(f"   Kết quả kỳ vọng: {profile['marketing_strategy']['expected_outcome']}")
    print(f"💰 Phân bổ ngân sách: {profile['budget_allocation']}")
    print(f"📊 KPI: {profile['kpi']}")

# Lưu ra file JSON để dùng cho dashboard
with open('data/processed/cluster_profiles.json', 'w', encoding='utf-8') as f:
    json.dump(cluster_profiles, f, ensure_ascii=False, indent=2)
```

**Bước 3.2: Tạo bảng tổng hợp cho báo cáo**

```python
# Cell: Bảng marketing strategy
strategy_table = pd.DataFrame([
    {
        'Cluster': f"{cid}: {p['name_vi']}",
        'Size': f"{p['size']} ({p['percentage']:.1f}%)",
        'RFM': f"R{p['rfm_profile']['recency']}/F{p['rfm_profile']['frequency']}/M${p['rfm_profile']['monetary']}",
        'Persona': p['persona'][:60] + "...",
        'Strategy': p['marketing_strategy']['tactics'][0][:50] + "...",
        'Budget': p['budget_allocation']
    }
    for cid, p in cluster_profiles.items()
])

print("\n" + "="*100)
print("BẢNG CHIẾN LƯỢC MARKETING")
print("="*100)
print(strategy_table.to_markdown(index=False))

# Xuất ra Excel
strategy_table.to_excel('reports/marketing_strategy.xlsx', index=False)
```

---

### **BƯỚC 4: So Sánh Các Thuật Toán Phân Cụm (Nâng Cao)**

#### **📚 Giải thích thuật ngữ:**

| Thuật ngữ | Giải thích |
|-----------|------------|
| **K-Means** | Phân cụm theo khoảng cách Euclidean, giả định cụm hình cầu |
| **Agglomerative (Hierarchical)** | Phân cụm phân cấp từ dưới lên, tạo dendrogram |
| **DBSCAN** | Density-based clustering, tìm cụm theo mật độ, tự động tìm outliers |
| **Silhouette Score** | Đo chất lượng phân cụm, từ -1 đến 1, càng cao càng tốt |
| **Davies-Bouldin Index** | Tỷ lệ within/between cluster variance, càng thấp càng tốt |
| **Calinski-Harabasz** | Tỷ lệ between/within variance, càng cao càng tốt |

#### **📍 Vị trí trong hệ thống:**

Tạo file mới: `notebooks/clustering_comparison.ipynb`

#### **🔧 Hướng dẫn thực hiện:**

```python
# Cell: So sánh thuật toán
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import time

# Load features
X = np.load('data/processed/X_features.npy')  # Lưu từ notebook clustering

results = []

# 1. K-Means với K khác nhau
for k in [3, 4, 5, 6]:
    start = time.time()
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = km.fit_predict(X)
    elapsed = time.time() - start
    
    results.append({
        'Algorithm': f'K-Means (K={k})',
        'N_Clusters': k,
        'Silhouette': silhouette_score(X, labels),
        'Davies-Bouldin': davies_bouldin_score(X, labels),
        'Calinski-Harabasz': calinski_harabasz_score(X, labels),
        'Time (s)': elapsed
    })

# 2. Agglomerative
for k in [3, 4, 5, 6]:
    start = time.time()
    agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = agg.fit_predict(X)
    elapsed = time.time() - start
    
    results.append({
        'Algorithm': f'Agglomerative (K={k})',
        'N_Clusters': k,
        'Silhouette': silhouette_score(X, labels),
        'Davies-Bouldin': davies_bouldin_score(X, labels),
        'Calinski-Harabasz': calinski_harabasz_score(X, labels),
        'Time (s)': elapsed
    })

# 3. DBSCAN với eps khác nhau
for eps in [0.5, 1.0, 1.5]:
    start = time.time()
    db = DBSCAN(eps=eps, min_samples=10)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    elapsed = time.time() - start
    
    if n_clusters > 1:
        results.append({
            'Algorithm': f'DBSCAN (eps={eps})',
            'N_Clusters': n_clusters,
            'Silhouette': silhouette_score(X, labels),
            'Davies-Bouldin': davies_bouldin_score(X, labels),
            'Calinski-Harabasz': calinski_harabasz_score(X, labels),
            'Time (s)': elapsed
        })

# Tạo bảng so sánh
comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.round(3)
comparison_df = comparison_df.sort_values('Silhouette', ascending=False)

print("="*100)
print("BẢNG SO SÁNH CÁC THUẬT TOÁN PHÂN CỤM")
print("="*100)
print(comparison_df.to_markdown(index=False))

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
comparison_df.plot(x='Algorithm', y='Silhouette', kind='bar', ax=axes[0], legend=False)
axes[0].set_title('Silhouette Score (higher is better)')
axes[0].set_ylabel('Score')

comparison_df.plot(x='Algorithm', y='Davies-Bouldin', kind='bar', ax=axes[1], legend=False, color='orange')
axes[1].set_title('Davies-Bouldin Index (lower is better)')

comparison_df.plot(x='Algorithm', y='Time (s)', kind='bar', ax=axes[2], legend=False, color='green')
axes[2].set_title('Execution Time')

plt.tight_layout()
plt.savefig('figures/algorithm_comparison.png', dpi=150)
plt.show()

# Kết luận
best_row = comparison_df.iloc[0]
print(f"\n🏆 Thuật toán tốt nhất: {best_row['Algorithm']}")
print(f"   Silhouette: {best_row['Silhouette']:.3f}")
print(f"   Davies-Bouldin: {best_row['Davies-Bouldin']:.3f}")
print(f"   Số cụm: {best_row['N_Clusters']}")
```

---

### **BƯỚC 5: Xây Dựng Streamlit Dashboard (Nâng Cao)**

#### **📚 Giải thích thuật ngữ:**

| Thuật ngữ | Giải thích |
|-----------|------------|
| **Streamlit** | Framework Python để tạo web app data science nhanh chóng |
| **Dashboard** | Bảng điều khiển hiển thị metrics và visualizations |
| **Interactive Filter** | Bộ lọc tương tác (dropdown, slider) |
| **Real-time Update** | Cập nhật biểu đồ theo selection |

#### **📍 Vị trí trong hệ thống:**

Tạo file mới: `app/dashboard.py`

#### **🔧 Hướng dẫn thực hiện:**

**Bước 5.1: Tạo file dashboard**

```python
# File: app/dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

# Config
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    clusters = pd.read_csv("../data/processed/customer_clusters_from_rules.csv")
    rules = pd.read_csv("../data/processed/rules_fpgrowth_filtered.csv")
    with open("../data/processed/cluster_profiles.json", 'r', encoding='utf-8') as f:
        profiles = json.load(f)
    return clusters, rules, profiles

clusters_df, rules_df, cluster_profiles = load_data()

# Sidebar
st.sidebar.title("🎛️ Bộ Lọc")
selected_cluster = st.sidebar.selectbox(
    "Chọn cụm khách hàng:",
    options=["Tất cả"] + sorted(clusters_df['cluster'].unique().tolist())
)

# Main
st.title("🛍️ Customer Segmentation Dashboard")
st.markdown("Phân khúc khách hàng dựa trên Association Rules")

# Overview
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Tổng khách hàng", f"{len(clusters_df):,}")
with col2:
    st.metric("Số cụm", clusters_df['cluster'].nunique())
with col3:
    avg_monetary = clusters_df['Monetary'].mean()
    st.metric("Avg Monetary", f"${avg_monetary:,.2f}")
with col4:
    avg_freq = clusters_df['Frequency'].mean()
    st.metric("Avg Frequency", f"{avg_freq:.1f}")

# Cluster Distribution
st.subheader("📊 Phân Bố Cụm")
cluster_counts = clusters_df['cluster'].value_counts().sort_index()
fig_dist = px.bar(
    x=cluster_counts.index, 
    y=cluster_counts.values,
    labels={'x': 'Cluster', 'y': 'Number of Customers'},
    title="Cluster Size Distribution"
)
st.plotly_chart(fig_dist, use_container_width=True)

# Cluster Detail
if selected_cluster != "Tất cả":
    st.subheader(f"🔍 Chi Tiết Cluster {selected_cluster}")
    
    # Get profile
    profile = cluster_profiles[str(selected_cluster)]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Tên:** {profile['name_vi']} ({profile['name_en']})")
        st.markdown(f"**Quy mô:** {profile['size']} khách ({profile['percentage']:.2f}%)")
        st.markdown(f"**Persona:** {profile['persona']}")
    
    with col2:
        st.markdown("**RFM Profile:**")
        st.markdown(f"- Recency: {profile['rfm_profile']['recency']} ngày")
        st.markdown(f"- Frequency: {profile['rfm_profile']['frequency']} đơn")
        st.markdown(f"- Monetary: ${profile['rfm_profile']['monetary']:,}")
    
    # Marketing Strategy
    st.markdown("### 🎯 Chiến Lược Marketing")
    st.info(f"**Mục tiêu:** {profile['marketing_strategy']['objective']}")
    
    st.markdown("**Chiến thuật:**")
    for i, tactic in enumerate(profile['marketing_strategy']['tactics'], 1):
        st.markdown(f"{i}. {tactic}")
    
    st.success(f"**Kết quả kỳ vọng:** {profile['marketing_strategy']['expected_outcome']}")
    
    # Top Rules
    st.markdown("### 📋 Top 10 Luật Kết Hợp")
    # Filter rules for this cluster (simplified - actual implementation needs rule activation matrix)
    st.dataframe(rules_df.head(10)[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']])

else:
    # Compare all clusters
    st.subheader("📊 So Sánh Các Cụm")
    
    comparison = clusters_df.groupby('cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Recency', x=comparison['cluster'], y=comparison['Recency']))
    fig.add_trace(go.Bar(name='Frequency', x=comparison['cluster'], y=comparison['Frequency']))
    fig.add_trace(go.Bar(name='Monetary', x=comparison['Monetary']/100, y=comparison['cluster']))
    fig.update_layout(barmode='group', title="RFM Comparison Across Clusters")
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("💡 Dựa trên FP-Growth Association Rules + K-Means Clustering")
```

**Bước 5.2: Chạy dashboard**

```bash
cd app
streamlit run dashboard.py
```

Dashboard sẽ mở tại `http://localhost:8501`

---

## 📝 CHECKLIST HOÀN CHỈNH

### **Bắt buộc:**
- [ ] **✅ Trình bày rõ cách chọn luật**: Top-K, sort_by, ngưỡng lọc, lý do
- [ ] **✅ Bảng 10 luật tiêu biểu**: Có đầy đủ support, confidence, lift
- [ ] **✅ Phân tích phân bố luật**: Histogram của support/confidence/lift
- [ ] **✅ Tạo ≥2 biến thể feature engineering**
- [ ] **✅ So sánh các biến thể**: Bảng tổng hợp với Silhouette score
- [ ] **✅ Profiling từng cụm**: Thống kê RFM + top rules
- [ ] **✅ Đặt tên cụm**: Tiếng Anh + tiếng Việt
- [ ] **✅ Mô tả persona**: 1-2 câu cho mỗi cụm
- [ ] **✅ Chiến lược marketing**: Cụ thể cho từng cụm
- [ ] **✅ Trực quan hóa 2D**: PCA/SVD scatter plot
- [ ] Silhouette score > 0.25
- [ ] Mỗi cụm có ít nhất 5% tổng khách hàng

### **Nâng cao (điểm tối đa):**
- [ ] **✅ So sánh thuật toán**: K-Means vs Agglomerative vs DBSCAN
- [ ] **✅ Dashboard Streamlit**: Interactive visualization
- [ ] Thử basket/product/rule clustering

---

## 🔗 TÀI LIỆU THAM KHẢO

- [K-Means Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Silhouette Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
- [MLxtend Association Rules](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/)
- [RFM Analysis Guide](https://www.optimove.com/resources/learning-center/rfm-segmentation)

---

**Cập nhật lần cuối**: December 29, 2025  
**Tác giả**: AI Assistant for DataMining Project
