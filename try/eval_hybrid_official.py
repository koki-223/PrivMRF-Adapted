import os
import json
import numpy as np
import pandas as pd
import csv

# ดึงผู้ตรวจข้อสอบออริจินัลและคลาส Domain มาใช้งาน
from exp.evaluate import k_way_marginal
from PrivMRF.domain import Domain

def prepare_fair_evaluation_baseline(data_name, raw_df, continuous_cols, bin_size=16):
    """
    ฟังก์ชันนี้จะสร้างไฟล์ข้อมูลจริงในรูปแบบ Discrete (Bin=16) 
    และเซฟลงโฟลเดอร์ ./preprocess/ เพื่อให้ k_way_marginal มีเป้าหมายที่ถูกต้องในการเปรียบเทียบ
    """
    print(f" [Step 1] กำลังเตรียมข้อมูล Baseline (Bin={bin_size}) ลงโฟลเดอร์ preprocess...")
    
    if not os.path.exists('./preprocess'): os.mkdir('./preprocess')
    
    attr_num = raw_df.shape[1]
    processed_data = np.zeros(raw_df.shape, dtype=int)
    json_domain = {}
    bin_edges_dict = {} # เก็บขอบเขตตะกร้าไว้ใช้หั่นข้อมูล Hybrid
    
    for col_idx, col_name in enumerate(raw_df.columns):
        col_values = raw_df[col_name].values
        if col_name in continuous_cols:
            min_val, max_val = np.min(col_values), np.max(col_values)
            bin_edges = np.linspace(min_val, max_val, bin_size + 1)
            bin_edges_dict[col_name] = bin_edges
            
            # หั่นข้อมูลจริง
            processed_data[:, col_idx] = np.clip(np.digitize(col_values, bin_edges) - 1, 0, bin_size - 1)
            json_domain[col_idx] = {'domain': bin_size}
        else:
            unique_vals = sorted(list(set(col_values)))
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            processed_data[:, col_idx] = [mapping[v] for v in col_values]
            json_domain[col_idx] = {'domain': len(unique_vals)}

    # เซฟข้อมูล Baseline ทับลงไป
    with open(f'./preprocess/{data_name}.csv', 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(raw_df.columns)
        csv.writer(f).writerows(processed_data)
    with open(f'./preprocess/{data_name}.json', 'w', encoding='utf-8') as f:
        json.dump(json_domain, f)
        
    return bin_edges_dict, raw_df.columns

def run_official_evaluation():
    print(" เริ่มต้นระบบประเมินผล Hybrid Data ด้วย k_way_marginal")
    print("-" * 60)
    
    DATA_NAME = 'mock_adult_raw'
    CONTINUOUS_COLS = ['0_age', '2_fnlwgt', '10_capital_gain', '11_capital_loss', '12_hours_per_week']
    EVAL_BIN_SIZE = 16 # ขนาดตะกร้าที่เราใช้เป็นมาตรฐานวัดผล
    
    # 1. โหลดข้อมูล
    raw_df = pd.read_csv(f'./data/{DATA_NAME}.csv')
    syn_df = pd.read_csv(f'./result/{DATA_NAME}_hybrid_synthetic.csv')
    
    # 2. จัดเตรียม Baseline (สร้างไฟล์ลง preprocess) และดึงขอบตะกร้ามา
    bin_edges_dict, col_order = prepare_fair_evaluation_baseline(DATA_NAME, raw_df, CONTINUOUS_COLS, EVAL_BIN_SIZE)
    
    # 3. หั่นข้อมูล Hybrid (Continuous -> Discrete) ด้วยขอบตะกร้าเดียวกัน
    print(" [Step 2] กำลังหั่นข้อมูล Hybrid Synthetic ให้เป็นรูปแบบที่เครื่องตรวจข้อสอบอ่านออก...")
    syn_discrete = np.zeros(syn_df.shape, dtype=int)
    
    for col_idx, col_name in enumerate(col_order):
        if col_name in CONTINUOUS_COLS:
            edges = bin_edges_dict[col_name]
            syn_discrete[:, col_idx] = np.clip(np.digitize(syn_df[col_name].values, edges) - 1, 0, EVAL_BIN_SIZE - 1)
        else:
            # Categorical แปลงชื่อเป็น index 0, 1, 2
            unique_vals = sorted(list(set(raw_df[col_name].values)))
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            syn_discrete[:, col_idx] = [mapping.get(v, 0) for v in syn_df[col_name].values]
            
    # 4. ส่งเข้าเครื่องตรวจข้อสอบ Official!
    print(f"\n [Step 3] ส่งข้อมูลเข้า k_way_marginal (สุ่มตรวจ 300 คู่)")
    print("-" * 60)
    
    # จัดข้อมูลเข้า List ตามที่ฟังก์ชัน k_way_marginal ต้องการ
    dp_data_list = [syn_discrete] 
    ways = [3, 4, 5]
    marginal_num = 300
    
    for k in ways:
        # เรียกใช้ฟังก์ชันจากเปเปอร์
        tvd_list = k_way_marginal(DATA_NAME, dp_data_list, k, marginal_num)
        print(f" {k}-way marginal TVD: {np.mean(tvd_list):.4f}")

if __name__ == '__main__':
    run_official_evaluation()