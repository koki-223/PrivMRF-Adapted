# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('Agg')
# import seaborn as sns
# from sklearn.isotonic import IsotonicRegression

# # 📌 โหลด Custom DP K-Means จากไฟล์ของคุณ
# from custom_dp_kmeans import KMeans as DPKMeans 

# # ==========================================
# # 1. 🎯 ฟังก์ชัน DP Noisy CDF Adaptive Binning (ของคุณ)
# # ==========================================
# def dp_microbin_cdf_merging(raw_col_data, epsilon, num_micro_bins=2000, target_bins=10):
#     domain_min = np.min(raw_col_data)
#     domain_max = np.max(raw_col_data)
    
#     raw_counts, micro_edges = np.histogram(raw_col_data, bins=num_micro_bins, range=(domain_min, domain_max))
    
#     noise_scale = 1.0 / epsilon if epsilon > 0 else 0
#     noisy_counts = raw_counts + np.random.laplace(loc=0.0, scale=noise_scale, size=num_micro_bins)
    
#     kernel_size = 5
#     kernel = np.ones(kernel_size) / kernel_size
#     smoothed_counts = np.convolve(noisy_counts, kernel, mode='same')
    
#     raw_cdf = np.cumsum(smoothed_counts)
#     iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
#     cdf = iso.fit_transform(np.arange(num_micro_bins), raw_cdf)
    
#     total_noisy_pop = cdf[-1]
#     if total_noisy_pop <= 0: total_noisy_pop = 1.0 

#     target_capacity = total_noisy_pop / target_bins
#     adaptive_edges_raw = [domain_min]
    
#     current_target = target_capacity
#     for i in range(num_micro_bins):
#         if cdf[i] >= current_target:
#             adaptive_edges_raw.append(micro_edges[i+1])
#             current_target += target_capacity
            
#     if adaptive_edges_raw[-1] < domain_max:
#         adaptive_edges_raw.append(domain_max)
        
#     adaptive_edges_raw = np.sort(np.unique(adaptive_edges_raw))
    
#     adaptive_edges = []
#     for i in range(len(adaptive_edges_raw) - 1):
#         adaptive_edges.append((adaptive_edges_raw[i], adaptive_edges_raw[i+1]))
        
#     return adaptive_edges_raw, len(adaptive_edges)

# # ==========================================
# # 2. 📊 สร้างข้อมูลจำลอง (Skewed Data แบบรุนแรง)
# # ==========================================
# np.random.seed(42)
# # ใช้ Log-Normal จำลองความเบ้ขวา (เช่น Area ของ Bean)
# raw_data = np.random.lognormal(mean=0.0, sigma=0.8, size=5000) * 100
# domain_min, domain_max = np.min(raw_data), np.max(raw_data)

# epsilon = 1.0
# num_bins = 10

# # ==========================================
# # 3. 🧮 รัน DP K-Means เพื่อหาขอบตะกร้า
# # ==========================================
# # 🌟 Split Epsilon (มี 1 คอลัมน์ ดังนั้น eps_per_col = epsilon / 1)
# eps_per_col = epsilon / 1.0  
# X_col = raw_data.reshape(-1, 1)
# bounds = ([domain_min], [domain_max])

# dp_kmeans = DPKMeans(epsilon=eps_per_col, bounds=bounds, n_clusters=num_bins, random_state=42)
# dp_kmeans.fit(X_col)

# # วิธีคำนวณขอบตะกร้าของ K-Means คือการหา "จุดกึ่งกลาง" ระหว่าง Cluster Centers (Voronoi Boundaries)
# centers = np.sort(np.unique(dp_kmeans.cluster_centers_.flatten()))
# kmeans_edges = [domain_min]
# for i in range(len(centers)-1):
#     kmeans_edges.append((centers[i] + centers[i+1]) / 2.0)
# kmeans_edges.append(domain_max)

# # ==========================================
# # 4. 🧮 รัน DP Noisy CDF เพื่อหาขอบตะกร้า
# # ==========================================
# cdf_edges, _ = dp_microbin_cdf_merging(raw_data, epsilon=epsilon, num_micro_bins=2000, target_bins=num_bins)

# # ==========================================
# # 5. 🎨 วาดกราฟเปรียบเทียบ (สำหรับส่งอาจารย์)
# # ==========================================
# sns.set_theme(style="white", rc={"axes.edgecolor": "gray"})
# fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# # -------------------
# # รูปที่ 1: DP K-Means
# # -------------------
# sns.histplot(raw_data, bins=80, color='lightgray', edgecolor='none', ax=axes[0])
# for edge in kmeans_edges:
#     axes[0].axvline(x=edge, color='red', linestyle='--', linewidth=2.5, alpha=0.8)

# # วาดจุด Center ของ K-Means ไว้ที่พื้น
# axes[0].plot(centers, np.zeros_like(centers), 'rx', markersize=12, label='Cluster Centers')

# axes[0].set_title('DP K-Means Binning (Distance-based)', fontsize=16, fontweight='bold', pad=15)
# axes[0].set_xlabel('Feature Value (Skewed Distribution)', fontsize=14)
# axes[0].set_ylabel('Density / Count', fontsize=14)
# axes[0].legend(fontsize=12)

# # -------------------
# # รูปที่ 2: DP Noisy CDF
# # -------------------
# sns.histplot(raw_data, bins=80, color='lightgray', edgecolor='none', ax=axes[1])
# for edge in cdf_edges:
#     axes[1].axvline(x=edge, color='green', linestyle='--', linewidth=2.5, alpha=0.8)

# axes[1].set_title('DP Noisy CDF Binning (Equi-depth)', fontsize=16, fontweight='bold', pad=15)
# axes[1].set_xlabel('Feature Value (Skewed Distribution)', fontsize=14)
# axes[1].set_ylabel('Density / Count', fontsize=14)

# plt.suptitle("Comparison of Binning Strategies on Skewed Data (Epsilon = 1.0, Bins = 10)", fontsize=18, y=1.05)
# plt.tight_layout()
# plt.savefig('result_accuracy.png', dpi=300, bbox_inches='tight')
# print("✅ เซฟรูป Accuracy เรียบร้อยแล้ว: result_accuracy.png")






# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns


# np.random.seed(42)
# # ใช้ Log-Normal จำลองความเบ้ขวา (เช่น Area ของ Bean)
# raw_data = np.random.lognormal(mean=0.0, sigma=0.8, size=5000) * 100
# domain_min, domain_max = np.min(raw_data), np.max(raw_data)

# # ==========================================
# # 1. 🎯 ฟังก์ชัน Standard Binning (Equal-Width)
# # ==========================================
# def standard_binning_edges(raw_col_data, target_bins=10):
#     domain_min = np.min(raw_col_data)
#     domain_max = np.max(raw_col_data)
    
#     # หั่นแบบความกว้างเท่ากันเป๊ะๆ
#     edges = np.linspace(domain_min, domain_max, target_bins + 1)
#     return edges

# # ==========================================
# # 2. 🎨 วาดกราฟเปรียบเทียบ (เปรียบเทียบกับภาพเดิม)
# # ==========================================
# # (ใช้ข้อมูล raw_data ชุดเดิมจากข้อที่แล้วเพื่อให้เปรียบเทียบได้)
# standard_edges = standard_binning_edges(raw_data, target_bins=10)

# plt.figure(figsize=(8, 6))
# sns.histplot(raw_data, bins=80, color='lightgray', edgecolor='none')

# for edge in standard_edges:
#     plt.axvline(x=edge, color='blue', linestyle='--', linewidth=2.5, alpha=0.8)

# plt.title('Standard (Equal-Width) Binning', fontsize=16, fontweight='bold', pad=15)
# plt.xlabel('Feature Value (Skewed Distribution)', fontsize=14)
# plt.ylabel('Density / Count', fontsize=14)

# plt.tight_layout()
# plt.savefig('standard_binning_skewed.png', dpi=300, bbox_inches='tight')
# plt.show()






import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.isotonic import IsotonicRegression

# 📌 โหลด Custom DP K-Means จากไฟล์ของคุณ
from custom_dp_kmeans import KMeans as DPKMeans 

# ==========================================
# 1. 🎯 ฟังก์ชัน DP Noisy CDF Adaptive Binning
# ==========================================
def dp_microbin_cdf_merging(raw_col_data, epsilon, num_micro_bins=2000, target_bins=10):
    domain_min = np.min(raw_col_data)
    domain_max = np.max(raw_col_data)
    
    raw_counts, micro_edges = np.histogram(raw_col_data, bins=num_micro_bins, range=(domain_min, domain_max))
    noise_scale = 1.0 / epsilon if epsilon > 0 else 0
    noisy_counts = raw_counts + np.random.laplace(loc=0.0, scale=noise_scale, size=num_micro_bins)
    
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    smoothed_counts = np.convolve(noisy_counts, kernel, mode='same')
    
    raw_cdf = np.cumsum(smoothed_counts)
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
    cdf = iso.fit_transform(np.arange(num_micro_bins), raw_cdf)
    
    total_noisy_pop = cdf[-1]
    if total_noisy_pop <= 0: total_noisy_pop = 1.0 

    target_capacity = total_noisy_pop / target_bins
    adaptive_edges_raw = [domain_min]
    
    current_target = target_capacity
    for i in range(num_micro_bins):
        if cdf[i] >= current_target:
            adaptive_edges_raw.append(micro_edges[i+1])
            current_target += target_capacity
            
    if adaptive_edges_raw[-1] < domain_max:
        adaptive_edges_raw.append(domain_max)
        
    adaptive_edges_raw = np.sort(np.unique(adaptive_edges_raw))
    return adaptive_edges_raw

# ==========================================
# 2. 🎯 ฟังก์ชัน Standard Binning (Equal-Width)
# ==========================================
def standard_binning_edges(raw_col_data, target_bins=10):
    domain_min = np.min(raw_col_data)
    domain_max = np.max(raw_col_data)
    # หั่นความกว้างเท่ากันเป๊ะๆ (ไม่มีเรื่อง Epsilon มาเกี่ยว เพราะอิงตาม Min/Max)
    return np.linspace(domain_min, domain_max, target_bins + 1)

# ==========================================
# 3. 📊 สร้างข้อมูลจำลอง (Skewed Data)
# ==========================================
np.random.seed(42)
# จำลองข้อมูลเบ้ขวา (คล้ายๆ Area/Perimeter ของ Bean Dataset)
raw_data = np.random.lognormal(mean=0.0, sigma=0.8, size=5000) * 100
domain_min, domain_max = np.min(raw_data), np.max(raw_data)

epsilon = 1.0
num_bins = 10

# ==========================================
# 4. 🧮 คำนวณขอบตะกร้าของทั้ง 3 วิธี
# ==========================================
# --- วิธีที่ 1: DP K-Means ---
eps_per_col = epsilon / 1.0  
X_col = raw_data.reshape(-1, 1)
bounds = ([domain_min], [domain_max])

dp_kmeans = DPKMeans(epsilon=eps_per_col, bounds=bounds, n_clusters=num_bins, random_state=42)
dp_kmeans.fit(X_col)
centers = np.sort(np.unique(dp_kmeans.cluster_centers_.flatten()))
kmeans_edges = [domain_min]
for i in range(len(centers)-1):
    kmeans_edges.append((centers[i] + centers[i+1]) / 2.0)
kmeans_edges.append(domain_max)

# --- วิธีที่ 2: DP Noisy CDF ---
cdf_edges = dp_microbin_cdf_merging(raw_data, epsilon=epsilon, num_micro_bins=2000, target_bins=num_bins)

# --- วิธีที่ 3: Standard Binning ---
std_edges = standard_binning_edges(raw_data, target_bins=num_bins)

# ==========================================
# 5. 🎨 วาดกราฟเปรียบเทียบ (1x3 Subplots)
# ==========================================
sns.set_theme(style="white", rc={"axes.edgecolor": "gray"})
# สร้างภาพกว้างพิเศษสำหรับ 3 ช่อง
fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# --- กราฟซ้าย: Standard Binning ---
sns.histplot(raw_data, bins=80, color='lightgray', edgecolor='none', ax=axes[0])
for edge in std_edges:
    axes[0].axvline(x=edge, color='blue', linestyle='--', linewidth=3, alpha=0.8)
axes[0].set_title(f'Standard Binning (Equal-Width)\n[Bins = {num_bins}]', fontsize=18, fontweight='bold', pad=15)
axes[0].set_xlabel('Feature Value (Skewed Distribution)', fontsize=16)
axes[0].set_ylabel('Density / Count', fontsize=16)
axes[0].tick_params(axis='both', which='major', labelsize=14)

# --- กราฟกลาง: DP K-Means ---
sns.histplot(raw_data, bins=80, color='lightgray', edgecolor='none', ax=axes[1])
for edge in kmeans_edges:
    axes[1].axvline(x=edge, color='red', linestyle='--', linewidth=3, alpha=0.8)
axes[1].plot(centers, np.zeros_like(centers), 'rx', markersize=12, label='Cluster Centers')
axes[1].set_title(f'DP K-Means Binning (Distance-based)\n[Epsilon = {epsilon}, Bins = {num_bins}]', fontsize=18, fontweight='bold', pad=15)
axes[1].set_xlabel('Feature Value (Skewed Distribution)', fontsize=16)
axes[1].set_ylabel('Density / Count', fontsize=16)
axes[1].tick_params(axis='both', which='major', labelsize=14)
axes[1].legend(fontsize=14)

# --- กราฟขวา: DP Noisy CDF ---
sns.histplot(raw_data, bins=80, color='lightgray', edgecolor='none', ax=axes[2])
for edge in cdf_edges:
    axes[2].axvline(x=edge, color='green', linestyle='--', linewidth=3, alpha=0.8)
axes[2].set_title(f'DP Noisy CDF Binning (Equi-depth)\n[Epsilon = {epsilon}, Bins = {num_bins}]', fontsize=18, fontweight='bold', pad=15)
axes[2].set_xlabel('Feature Value (Skewed Distribution)', fontsize=16)
axes[2].set_ylabel('Density / Count', fontsize=16)
axes[2].tick_params(axis='both', which='major', labelsize=14)

# ใส่ Title ใหญ่ครอบทั้ง 3 รูป
plt.suptitle(f"Comparison of Binning Strategies on Skewed Data", fontsize=24, fontweight='bold', y=1.05)

plt.tight_layout()
plt.savefig('binning_strategies_comparison_1x3.png', dpi=300, bbox_inches='tight')
print("✅ เซฟรูปเรียบร้อยแล้ว: binning_strategies_comparison_1x3.png")