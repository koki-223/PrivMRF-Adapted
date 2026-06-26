import numpy as np
import cupy as cp
import pandas as pd

class DPPCAModule:
    """
    Step 1: DP-PCA Module (ปรับปรุง: เพิ่ม Standardization เพื่อแก้ปัญหา Scale Dominance)
    """
    def __init__(self, epsilon, delta, variance_threshold, clip_bound, max_components=None):
        self.epsilon = epsilon
        self.delta = delta
        self.variance_threshold = variance_threshold
        self.clip_bound = clip_bound
        self.max_components = max_components
        self.n_components = None
        self.components_ = None
        self.mean_ = None
        self.std_ = None #  เพิ่มตัวแปรเก็บค่า Standard Deviation
        
    def fit_transform(self, X):
        X_np = np.array(X, dtype=np.float32)
        
        # 🌟 1. ขาเข้า: ทำ Standardization (Z-score Scaling)
        self.mean_ = np.mean(X_np, axis=0)
        self.std_ = np.std(X_np, axis=0)
        self.std_[self.std_ == 0] = 1e-6 # ป้องกันการหารด้วยศูนย์
        
        # ปรับสเกลข้อมูล: (X - Mean) / SD
        X_scaled = (X_np - self.mean_) / self.std_
        
        # 2. L2-norm Clipping (ทำบนข้อมูลที่ Scale แล้ว)
        norms = np.linalg.norm(X_scaled, axis=1, keepdims=True)
        clip_factors = np.minimum(1.0, self.clip_bound / norms)
        clip_factors[np.isnan(clip_factors)] = 1.0
        X_clipped = X_scaled * clip_factors
        
        # 3. Covariance Matrix + DP Noise
        cov_matrix = np.dot(X_clipped.T, X_clipped)
        sensitivity = self.clip_bound ** 2
        sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / self.delta))) / self.epsilon
        noise = np.random.normal(loc=0.0, scale=sigma, size=cov_matrix.shape)
        noisy_cov_matrix = cov_matrix + ((noise + noise.T) / 2.0)
        
        # 4. Eigen-Decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(noisy_cov_matrix)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_idx]
        
        clean_eigenvalues = np.maximum(sorted_eigenvalues, 0)
        exp_variance_ratio = clean_eigenvalues / np.sum(clean_eigenvalues)
        cumulative_variance = np.cumsum(exp_variance_ratio)
        
        n_to_reach_threshold = np.argmax(cumulative_variance >= self.variance_threshold) + 1
        self.n_components = min(n_to_reach_threshold, self.max_components) if self.max_components else n_to_reach_threshold
            
        print(f"📊 PCA Selection: เลือกใช้ {self.n_components} แกน (ครอบคลุมข้อมูล {cumulative_variance[self.n_components-1]*100:.2f}%)")
        
        self.components_ = eigenvectors[:, sorted_idx[:self.n_components]]
        X_latent = np.dot(X_clipped, self.components_)
        return X_latent

    def inverse_transform(self, X_latent):
        # 🌟 5. ขาออก: กางข้อมูล Latent Space กลับมา
        X_reconstructed_scaled = np.dot(X_latent, self.components_.T)
        
        # นำค่า SD ไปคูณกลับ แล้วบวก Mean เพื่อคืนสเกลดั้งเดิม (Inverse Z-score)
        X_reconstructed = (X_reconstructed_scaled * self.std_) + self.mean_
        return X_reconstructed



class DPIntelligentBinner:
    """
    Step 2: Quantile-based Binning พร้อม Noise ในระดับ Histogram
    """
    def __init__(self, n_bins, epsilon):
        self.n_bins = n_bins
        self.epsilon = epsilon
        self.bin_edges_ = {}
        self.bin_centers_ = {}

    def fit_transform(self, X_latent):
        """
        สร้าง DP-Binning สำหรับแต่ละแกน PC
        """
        n_cols = X_latent.shape[1]
        X_binned = np.zeros_like(X_latent, dtype=int)
        
        # แบ่ง Epsilon ให้แต่ละแกนเท่าๆ กัน
        epsilon_per_col = self.epsilon / n_cols
        
        for col in range(n_cols):
            data = X_latent[:, col]
            
            # 1. หา Min-Max คร่าวๆ เพื่อสร้างตารางย่อย (Micro-bins) 
            min_val, max_val = np.min(data), np.max(data)
            micro_bins = np.linspace(min_val, max_val, 1000)
            hist, _ = np.histogram(data, bins=micro_bins)
            
            # 2. ใส่ Laplace Noise ลงใน Histogram (DP Equal-Frequency Basis)
            # Sensitivity ของ Histogram นับคน คือ 1
            noise = np.random.laplace(0, 1.0 / epsilon_per_col, size=hist.shape)
            noisy_hist = np.maximum(hist + noise, 0) # ห้ามติดลบ
            
            # 3. คำนวณ Cumulative Sum เพื่อหา Quantile Edges
            cdf = np.cumsum(noisy_hist)
            total_noisy_count = cdf[-1]
            
            target_quantiles = np.linspace(0, total_noisy_count, self.n_bins + 1)
            
            # แมปเป้าหมาย CDF กลับไปหาค่าขอบ Bin
            edges = [min_val]
            for q in target_quantiles[1:-1]:
                idx = np.searchsorted(cdf, q)
                edges.append(micro_bins[idx])
            edges.append(max_val)
            
            self.bin_edges_[col] = np.array(edges)
            
            # คำนวณจุดกึ่งกลาง (Center) ไว้สำหรับตอน Inverse_transform กลับ
            self.bin_centers_[col] = (np.array(edges)[:-1] + np.array(edges)[1:]) / 2.0
            
            # 4. แปลงข้อมูล (Digitize) เป็น Bin IDs
            # ขยับ index ให้เป็น 0 ถึง n_bins-1
            X_binned[:, col] = np.clip(np.digitize(data, self.bin_edges_[col]) - 1, 0, self.n_bins - 1)
            
        return X_binned
    
    def decode_bins(self, X_binned):
        """
        แปลง Bin IDs กลับเป็นค่าตัวแทน (Representative value) ใน Latent space
        """
        X_decoded = np.zeros_like(X_binned, dtype=float)
        for col in range(X_binned.shape[1]):
            centers = self.bin_centers_[col]
            X_decoded[:, col] = centers[X_binned[:, col]]
        return X_decoded


class HybridPrivMRFIntegrator:
    """
    Step 3: ตัวจัดการรวบรวมข้อมูลผสม (Continuous -> PCA -> Bin) + Categorical
    """
    def __init__(self, dp_pca_model, dp_binner_model):
        self.pca = dp_pca_model
        self.binner = dp_binner_model
        self.categorical_cols_ = None
        self.continuous_cols_ = None

    def prepare_data_for_privmrf(self, df, continuous_cols, categorical_cols):
        self.continuous_cols_ = continuous_cols
        self.categorical_cols_ = categorical_cols
        
        # 1. จัดการ Continuous Data
        X_cont = df[continuous_cols].values
        X_latent = self.pca.fit_transform(X_cont)
        X_pca_binned = self.binner.fit_transform(X_latent)
        
        # 2. เตรียม DataFrame ใหม่
        # แปลง PCA bins ให้เป็นชื่อคอลัมน์ใหม่
        df_mixed = pd.DataFrame()
        
        for i in range(X_pca_binned.shape[1]):
            df_mixed[f'PCA_Bin_{i}'] = X_pca_binned[:, i]
            
        # 3. นำ Categorical มารวม
        for col in categorical_cols:
            df_mixed[col] = df[col].values
            
        # DataFrame นี้คือ Categorical ล้วน 100% พร้อมโยนเข้า PrivMRF!
        return df_mixed
        
    def reconstruct_synthetic_data(self, syn_df_from_privmrf):
        """
        เมื่อ PrivMRF สร้างข้อมูลเสร็จ เอามาโยนเข้าฟังก์ชันนี้เพื่อประกอบร่างกลับ
        """
        # 1. ดึง PCA Bins ออกมา
        pca_cols = [c for c in syn_df_from_privmrf.columns if 'PCA_Bin_' in c]
        X_syn_binned = syn_df_from_privmrf[pca_cols].values
        
        # 2. ถอดรหัส Bin -> Latent Space
        X_syn_latent = self.binner.decode_bins(X_syn_binned)
        
        # 3. Inverse PCA กลับไปเป็น Continuous
        X_syn_cont = self.pca.inverse_transform(X_syn_latent)
        
        # 4. ประกอบร่าง
        final_df = pd.DataFrame(X_syn_cont, columns=self.continuous_cols_)
        
        for col in self.categorical_cols_:
            final_df[col] = syn_df_from_privmrf[col].values
            
        return final_df