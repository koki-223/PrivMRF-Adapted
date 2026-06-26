import collections
import collections.abc

# แพตช์ (Patch) เพื่อให้โค้ดเก่า (PrivMRF) ทำงานได้บน Python 3.10+
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping

import matplotlib
matplotlib.use('Agg') 

import warnings
warnings.filterwarnings('ignore')

import os
import csv
import json
import random 
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn import svm

# อย่าลืม !pip install diffprivlib ก่อนรัน
from diffprivlib.models import KMeans as DPKMeans
# from custom_dp_kmeans import KMeans as DPKMeans

# 🌟 เปลี่ยน Import เพื่อใช้งาน Bypass (แก้ปัญหา missing parameter ของ Gaussian)
from exp.evaluate import read_preprocessed_data
from PrivMRF.main import run
from PrivMRF.domain import Domain
import PrivMRF.attribute_hierarchy as ah

# ==========================================
# 🌟 ทางลัด: ฟังก์ชันรัน PrivMRF แบบ Custom Parameter (ป้องกัน Gaussian Crash)
# ==========================================
def custom_run_syn(data_name, exp_name, epsilon, task='TVD'):
    data, domain, attr_hierarchy = read_preprocessed_data(data_name, task)
    
    # ส่ง config ทะลุเข้าไปสั่งการ PrivMRF โดยตรง
    p_config = {
        'data': data_name,
        'delta': 1e-5,  # ป้องกัน Error missing parameter ของ Gaussian
        'theta': 4      # ล็อคขนาดกราฟสูงสุดเพื่อเซฟ Memory
    }
    
    # รันผ่านโมเดลหลัก (Bypass evaluate.py)
    model = run(data, domain, attr_hierarchy, exp_name, epsilon, task, p_config)
    data_list = model.synthetic_data('./out/' + 'PrivMRF_'+ data_name + '_' + exp_name + '.csv')
    
    return data_list

# ==========================================
# 1. 🎯 Core Algorithm: DP K-Means 1D Binning
# ==========================================
def dp_kmeans_1d_binning(raw_col_data, epsilon=1.0, n_clusters=70, public_bounds=None):
    X = raw_col_data.reshape(-1, 1)

    # 1. การกำหนดขอบเขต (Bounds) [กฎเหล็กของ DP]
    if public_bounds is None:
        public_bounds = (np.min(raw_col_data), np.max(raw_col_data))

    domain_min, domain_max = public_bounds

    # 🌟 [แก้ไข 1]: Split Epsilon ภายในฟังก์ชัน (ครึ่งนึงหา Center, ครึ่งนึงหา Counts)
    eps_kmeans = epsilon / 2.0
    eps_counts = epsilon / 2.0

    # 2. รัน DP K-Means
    dp_kmeans = DPKMeans(
        epsilon=eps_kmeans, # ใช้ Epsilon ที่ถูกหารแล้ว
        bounds=public_bounds, 
        n_clusters=n_clusters,
        random_state=42
    )
    dp_kmeans.fit(X)

    # 3. ดึง Center และเรียงลำดับจากซ้ายไปขวา
    original_centers = dp_kmeans.cluster_centers_.flatten()
    sorted_idx = np.argsort(original_centers)
    sorted_centers = original_centers[sorted_idx]

    # 4. การนับจำนวนข้อมูล (จำลองฉีด Noise เพื่อให้ปลอดภัย 100%)
    labels = dp_kmeans.predict(X)
    dp_counts = []
    noise_scale = 1.0 / eps_counts # ใช้ Epsilon ส่วนที่เหลือ

    for idx in sorted_idx:
        true_count = np.sum(labels == idx)
        noisy_count = true_count + np.random.laplace(0, noise_scale)
        # 🌟 [แก้ไข 2]: เอา Clip Zero (max 0) ออก ปล่อยให้ติดลบได้
        dp_counts.append(int(noisy_count))

    # 5. ตีเส้นแบ่งเขตแดน (Midpoints ระหว่าง Centers)
    edges = [domain_min - 1e-5]
    for i in range(len(sorted_centers) - 1):
        midpoint = (sorted_centers[i] + sorted_centers[i+1]) / 2.0
        edges.append(midpoint)
    edges.append(domain_max + 1e-5)

    return sorted_centers, np.array(edges), dp_counts

# ==========================================
# 1.5 🔄 ฟังก์ชัน Inverse Transform (ใช้ค่า Center ของ K-Means)
# ==========================================
def inverse_transform_kmeans(discrete_data, bin_info, num_continuous_cols):
    continuous_data = np.zeros(discrete_data.shape, dtype=float)
    
    for col in range(num_continuous_cols):
        centers = bin_info[col]['centers'] 
        col_indices = discrete_data[:, col].astype(int)
        
        safe_indices = np.clip(col_indices, 0, len(centers) - 1)
        continuous_data[:, col] = centers[safe_indices]
        
    continuous_data[:, -1] = discrete_data[:, -1]
    return continuous_data

# ==========================================
# 2. ไปป์ไลน์หลัก: DP K-Means Binning Baseline
# ==========================================
def run_dp_kmeans_binning_pipeline():
    DATA_NAME = 'avila_combined_df'
    CONTINUOUS_COLS = list(range(10))  # Adjust this range as needed
    LABEL_COL = [-1]
    
    # BIN_LIST = [2,3,4,5,6,7]
    BIN_LIST = [2,3,4,5,6,7]  # เพิ่มจำนวน Bin เพื่อดูผลที่ละเอียดขึ้น
    EPSILON_LIST = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
    
    TRAIN_SAMPLE_LIMIT = 10000 

    RAW_FILE_PATH = f'./data/{DATA_NAME}.csv'

    print("🚀 เริ่มต้นระบบ: DP K-Means Binning Baseline")
    print("-" * 85)

    if not os.path.exists(RAW_FILE_PATH):
        print(f"❌ ไม่พบไฟล์ {RAW_FILE_PATH}")
        return
        
    with open(RAW_FILE_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headings = next(reader)
        raw_data = [row for row in reader if len(row) > 0]

    cont_headings = [headings[i] for i in CONTINUOUS_COLS]
    label_heading = [headings[LABEL_COL[0]]]
    all_headings = cont_headings + label_heading

    cont_data = [[float(row[i]) for i in CONTINUOUS_COLS] for row in raw_data] 
    raw_continuous_np = np.array(cont_data)
    raw_label_np = np.array([[row[LABEL_COL[0]]] for row in raw_data]) 
    
    le_master = LabelEncoder()
    encoded_label_master = le_master.fit_transform(raw_label_np.ravel()).reshape(-1, 1)

    raw_scaled = np.concatenate([raw_continuous_np, encoded_label_master], axis=1)

    print(f"\n🔥 กำลังประเมิน The Absolute TRTR Baseline (Continuous Space)...")
    np.random.seed(42)
    shuffled_idx_real = np.random.permutation(len(raw_scaled))
    shuffled_raw_scaled = raw_scaled[shuffled_idx_real]
    
    fold_size_real = len(shuffled_raw_scaled) // 5
    trtr_fold_acc = []
    trtr_fold_f1 = []
    TARGET_LABEL_INDEX = -1

    for j in range(5):
        start_r, end_r = j * fold_size_real, (j + 1) * fold_size_real
        if j == 4: end_r = len(shuffled_raw_scaled)
        
        X_real_test = shuffled_raw_scaled[start_r:end_r]
        X_real_train_full = np.concatenate([shuffled_raw_scaled[:start_r], shuffled_raw_scaled[end_r:]], axis=0)
        
        if len(X_real_train_full) > TRAIN_SAMPLE_LIMIT:
            np.random.seed(42 + j)
            sample_idx = np.random.choice(len(X_real_train_full), TRAIN_SAMPLE_LIMIT, replace=False)
            X_real_train = X_real_train_full[sample_idx]
        else:
            X_real_train = X_real_train_full

        y_train_trtr = X_real_train[:, TARGET_LABEL_INDEX].astype(int)
        X_train_trtr = np.delete(X_real_train, TARGET_LABEL_INDEX, axis=1)
        
        y_test_trtr = X_real_test[:, TARGET_LABEL_INDEX].astype(int)
        X_test_trtr = np.delete(X_real_test, TARGET_LABEL_INDEX, axis=1)

        clf_trtr = make_pipeline(StandardScaler(), svm.SVC(gamma='auto'))
        clf_trtr.fit(X_train_trtr, y_train_trtr)
        y_pred_trtr = clf_trtr.predict(X_test_trtr) 
        
        trtr_fold_acc.append(accuracy_score(y_test_trtr, y_pred_trtr))
        trtr_fold_f1.append(f1_score(y_test_trtr, y_pred_trtr, average='macro'))

    absolute_trtr_acc = np.mean(trtr_fold_acc)
    absolute_trtr_f1 = np.mean(trtr_fold_f1)
    print(f"✅ Absolute TRTR ยืนพื้นสำเร็จ! | Acc: {absolute_trtr_acc:.4f} | F1: {absolute_trtr_f1:.4f}")
    print("-" * 85)

    if not os.path.exists('./preprocess'): os.mkdir('./preprocess')
    if not os.path.exists('./out'): os.mkdir('./out')
    if not os.path.exists('./result'): os.mkdir('./result')

    summary_results = []

    for NUM_BINS in BIN_LIST:
        print(f"\n" + "🔥"*40)
        print(f"🎯 [ทดสอบ DP K-Means (K) = {NUM_BINS}]")
        print("🔥"*40)

        for TOTAL_EPS in EPSILON_LIST:
            print(f"\n📦 [Epsilon = {TOTAL_EPS}] กำลังทำงาน...")

            # 🌟 [แก้ไข 3]: Split Epsilon
            eps_binning_total = TOTAL_EPS * 0.5
            eps_privmrf = TOTAL_EPS * 0.5
            
            # หาร Epsilon ต่อคอลัมน์
            eps_per_col = eps_binning_total / len(CONTINUOUS_COLS)

            json_domain = {}
            processed_data = np.zeros((len(cont_data), len(all_headings)), dtype=int)
            bin_info = {}

            # --- 1. ทำ DP K-Means Binning ทีละคอลัมน์ ---
            for col in range(len(CONTINUOUS_COLS)):
                col_values = raw_continuous_np[:, col]
                
                centers, edges, _ = dp_kmeans_1d_binning(
                    col_values, 
                    epsilon=eps_per_col, 
                    n_clusters=NUM_BINS
                )
                
                bin_info[col] = {'centers': centers, 'edges': edges, 'num_bins': NUM_BINS}
                json_domain[col] = {'domain': NUM_BINS}
                
                b_idx = np.searchsorted(edges, col_values, side='right') - 1
                b_idx = np.clip(b_idx, 0, NUM_BINS - 1)
                
                processed_data[:, col] = b_idx

            processed_data[:, -1] = encoded_label_master.ravel()
            json_domain[len(CONTINUOUS_COLS)] = {'domain': len(le_master.classes_)}

            with open(f'./preprocess/{DATA_NAME}.csv', 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(all_headings)
                csv.writer(f).writerows(processed_data)
            with open(f'./preprocess/{DATA_NAME}.json', 'w', encoding='utf-8') as f:
                json.dump(json_domain, f)

            domain_obj = Domain(json_domain, list(range(len(all_headings))))
            dummy_hierarchy = ah.get_one_level_hierarchy(domain_obj)
            ah.write_hierarchy(dummy_hierarchy, f'./preprocess/{DATA_NAME}_hierarchy.json')

            exp_name = f"DPKMeans_{NUM_BINS}_Eps_{TOTAL_EPS}"
            
            # --- 2. สร้างข้อมูลสังเคราะห์ (PrivMRF) ---
            # 🌟 เรียกใช้ฟังก์ชัน Custom Bypass ที่เราสร้างไว้แทน
            syn_discrete = custom_run_syn(DATA_NAME, exp_name, epsilon=eps_privmrf, task='TVD')
            syn_discrete = np.array(syn_discrete, dtype=int)
            
            num_cont_cols = len(CONTINUOUS_COLS)
            
            # --- 3. แปลงกลับ (Inverse Transform) ---
            syn_continuous = inverse_transform_kmeans(syn_discrete, bin_info, num_cont_cols)

            # --- 4. ประเมิน TRUE TSTR ---
            np.random.seed(42)
            
            shuffled_idx_real = np.random.permutation(len(raw_scaled))
            shuffled_real_raw = raw_scaled[shuffled_idx_real] 
            
            shuffled_idx_syn = np.random.permutation(len(syn_continuous))
            shuffled_syn_continuous = syn_continuous[shuffled_idx_syn]
            
            fold_size_real = len(shuffled_real_raw) // 5
            fold_size_syn = len(shuffled_syn_continuous) // 5
            
            tstr_fold_acc = []
            tstr_fold_f1 = []
            
            for j in range(5): 
                start_r, end_r = j * fold_size_real, (j + 1) * fold_size_real
                if j == 4: end_r = len(shuffled_real_raw)
                X_real_test_fold = shuffled_real_raw[start_r:end_r]
                
                start_s, end_s = j * fold_size_syn, (j + 1) * fold_size_syn
                X_syn_train_fold_full = np.concatenate([shuffled_syn_continuous[:start_s], shuffled_syn_continuous[end_s:]], axis=0)
                
                if len(X_syn_train_fold_full) > TRAIN_SAMPLE_LIMIT:
                    np.random.seed(42 + j)
                    sample_indices = np.random.choice(len(X_syn_train_fold_full), TRAIN_SAMPLE_LIMIT, replace=False)
                    X_syn_train_fold = X_syn_train_fold_full[sample_indices]
                else:
                    X_syn_train_fold = X_syn_train_fold_full

                y_train = X_syn_train_fold[:, TARGET_LABEL_INDEX].astype(int)
                X_train = np.delete(X_syn_train_fold, TARGET_LABEL_INDEX, axis=1)
                
                y_test = X_real_test_fold[:, TARGET_LABEL_INDEX].astype(int)
                X_test = np.delete(X_real_test_fold, TARGET_LABEL_INDEX, axis=1)
                
                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    continue
                    
                clf_tstr = make_pipeline(StandardScaler(), svm.SVC(gamma='auto'))
                clf_tstr.fit(X_train, y_train)
                y_pred = clf_tstr.predict(X_test)
                
                tstr_fold_acc.append(accuracy_score(y_test, y_pred))
                tstr_fold_f1.append(f1_score(y_test, y_pred, average='macro'))

            tstr_acc_final = np.mean(tstr_fold_acc) if tstr_fold_acc else 0.0
            tstr_f1_final = np.mean(tstr_fold_f1) if tstr_fold_f1 else 0.0

            print(f"   -> [K={NUM_BINS:<2} | Eps={TOTAL_EPS:<3}] TRUE TSTR Acc: {tstr_acc_final:.4f} | TRUE TSTR F1: {tstr_f1_final:.4f}")

            summary_results.append({
                'Num Bins (K)': NUM_BINS,
                'Epsilon': TOTAL_EPS,
                'Downstream Acc (TSTR)': tstr_acc_final,
                'Downstream Macro F1 (TSTR)': tstr_f1_final,
                'Absolute TRTR Acc': absolute_trtr_acc,
                'Absolute TRTR F1': absolute_trtr_f1,
                'Utility Score (Acc)': (tstr_acc_final / absolute_trtr_acc) if absolute_trtr_acc > 0 else 0.0,
                'Utility Score (F1)': (tstr_f1_final / absolute_trtr_f1) if absolute_trtr_f1 > 0 else 0.0
            })

    # ==========================================
    # 🌟 สร้างตารางสรุปจบ
    # ==========================================
    print("\n" + "🔥"*45)
    print("🎯 สรุปผล: DP K-Means Binning (True TSTR vs Absolute TRTR)")
    print("🔥"*45)
    
    summary_df = pd.DataFrame(summary_results)
    summary_df.sort_values(by=['Num Bins (K)', 'Epsilon'], inplace=True)
    
    for col in ['Downstream Acc (TSTR)', 'Downstream Macro F1 (TSTR)', 'Absolute TRTR Acc', 'Absolute TRTR F1', 'Utility Score (Acc)', 'Utility Score (F1)']:
        summary_df[col] = summary_df[col].apply(lambda x: f"{x:.5f}" if isinstance(x, float) else x)
    
    print(summary_df.to_string(index=False))
    print("-" * 85)
    
    summary_df.to_csv('./result/dp_kmeans_binning_true_tstr_results.csv', index=False)
    print("✅ รายงานผลถูกเซฟลง ./result/dp_kmeans_binning_true_tstr_results.csv เรียบร้อยแล้วครับ!")

if __name__ == '__main__':
    run_dp_kmeans_binning_pipeline()