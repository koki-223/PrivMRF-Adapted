import os
import pandas as pd
import numpy as np
import json

# นำเข้าโมดูล PCA-PrivMRF ที่เราสร้างขึ้น
from try.hybrid_dp_pca import DPPCAModule, DPIntelligentBinner, HybridPrivMRFIntegrator

# นำเข้า PrivMRF ออริจินัล
import PrivMRF
from PrivMRF.domain import Domain

def run_hybrid_pipeline():
    print("เริ่มต้นรัน Hybrid Pipeline: DP-PCA + Intelligent Binning + PrivMRF")
    print("=" * 70)

    # ==========================================
    # 1. การตั้งค่าตัวแปรและ Privacy Budget
    # ==========================================
    DATA_NAME = 'adult'
    RAW_FILE_PATH = f'./data/{DATA_NAME}.csv'
    
    # แยกประเภทคอลัมน์ (อ้างอิงจาก mock data ที่เราเคยสร้าง)
    # สมมติชื่อคอลัมน์เป็น string เพื่อใช้กับ Pandas
    # CONTINUOUS_COLS = ['0_age', '2_fnlwgt', '10_capital_gain', '11_capital_loss', '12_hours_per_week']
    # CATEGORICAL_COLS = [
    #     '1_categorical', '3_categorical', '4_categorical', '5_categorical', 
    #     '6_categorical', '7_categorical', '8_categorical', '9_categorical', 
    #     '13_categorical', '14_categorical'
    # ]

    CONTINUOUS_COLS = ['0', '2', '10', '11', '12']
    CATEGORICAL_COLS = [
        '1', '3', '4', '5', 
        '6', '7', '8', '9', 
        '13', '14'
    ]

    # ทฤษฎี Composition Theorem: แบ่ง Epsilon ให้แต่ละส่วน
    TOTAL_EPSILON = 1.0
    EPSILON_PCA = 0.1      # ใช้ 10% สำหรับหาแกน PCA
    EPSILON_BIN = 0.1      # ใช้ 10% สำหรับหั่นตะกร้า
    EPSILON_MRF = 0.8      # ใช้ 80% ที่เหลือให้ PrivMRF สร้างโครงข่าย
    
    # ตั้งค่า PCA
    N_COMPONENTS = 3       # บีบ 5 คอลัมน์ Continuous ให้เหลือแค่ 3 แกน
    CLIP_BOUND = 5.0             # L2-Norm limit (ปรับลงมาเป็น 5.0 แล้วเพราะทำ Standardize)
    VARIANCE_THRESHOLD = 0.85    # ให้ PCA เลือกแกนจนกว่าข้อมูลจะครอบคลุม 85%
    MAX_COMPONENTS = 4           # บังคับ PCA ไม่ให้เกิน 4 แกน
    N_BINS = 16           # จำนวนตะกร้าบน Latent Space

    # ==========================================
    # 2. โหลดข้อมูลดิบ
    # ==========================================
    if not os.path.exists(RAW_FILE_PATH):
        print(f"ไม่พบไฟล์ {RAW_FILE_PATH}")
        return
    
    df_raw = pd.read_csv(RAW_FILE_PATH)
    print(f"โหลดข้อมูลสำเร็จ: {df_raw.shape[0]} แถว | Continuous: {len(CONTINUOUS_COLS)} | Categorical: {len(CATEGORICAL_COLS)}")

    # ==========================================
    # 3. ใช้งาน DP-PCA Module (Pre-processing)
    # ==========================================
    print("\n [Step 1-2] กำลังประมวลผล DP-PCA และ Bining บน GPU...")
    
    # กำหนดค่าให้ Modules
    pca_module = DPPCAModule(epsilon=EPSILON_PCA, delta=1e-5, variance_threshold=0.85, max_components=4, clip_bound=CLIP_BOUND)
    binner_module = DPIntelligentBinner(n_bins=N_BINS, epsilon=EPSILON_BIN)
    integrator = HybridPrivMRFIntegrator(pca_module, binner_module)

    # รันการแปลงข้อมูล (จะได้ DataFrame ที่ Continuous กลายเป็น PCA_Bin หมดแล้ว)
    df_processed = integrator.prepare_data_for_privmrf(df_raw, CONTINUOUS_COLS, CATEGORICAL_COLS)
    print(f" แปลงข้อมูลเสร็จสิ้น! มิติข้อมูลลดลงเหลือ: {df_processed.shape[1]} คอลัมน์")

    # ==========================================
    # 4. เตรียม Domain ให้ PrivMRF
    # ==========================================
    print("\n [Step 3] กำลังเตรียมโครงสร้าง Domain ให้ PrivMRF...")
    json_domain = {}
    numpy_processed_data = np.zeros(df_processed.shape, dtype=int)
    
    # แปลงจาก Pandas DataFrame เป็น Numpy Array และสร้าง Domain Dictionary
    for col_idx, col_name in enumerate(df_processed.columns):
        col_data = df_processed[col_name].values
        
        # เนื่องจากตอนนี้ทุกอย่างเป็น Categorical / Bins หมดแล้ว
        unique_vals = sorted(list(set(col_data)))
        json_domain[col_idx] = {'domain': len(unique_vals)}
        
        # แมปค่าให้เป็น index 0, 1, 2...
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        for row_idx, val in enumerate(col_data):
            numpy_processed_data[row_idx, col_idx] = mapping[val]

    domain_obj = Domain(json_domain, list(range(df_processed.shape[1])))

    # ==========================================
    # 5. สั่งรัน PrivMRF ออริจินัล
    # ==========================================
    print(f"\n [Step 4] ส่งข้อมูลให้ PrivMRF เทรนโมเดล (Epsilon = {EPSILON_MRF})...")
    
    # สร้าง config ดึงสเปกมาใช้
    config = {'data': 'hybrid_mock', 'print': True}
    
    # เรียกใช้ run ตัวจริงของเปเปอร์
    model = PrivMRF.run(
        data=numpy_processed_data, 
        domain=domain_obj, 
        attr_hierarchy=None, 
        exp_name='hybrid_pca_test', 
        epsilon=EPSILON_MRF, 
        p_config=config
    )

    # สร้างข้อมูลสังเคราะห์ออกมา (เป็นแบบ Discrete)
    print("\n🎲 กำลังสร้างข้อมูลสังเคราะห์ดิบจาก MRF...")
    syn_discrete_data = np.array(model.synthetic_data('./temp/hybrid_temp_out.csv'))

    # ==========================================
    # 6. ประกอบร่างกลับ (Reconstruct to Continuous)
    # ==========================================
    print("\n [Step 5] ทำการ Inverse PCA แปลงกลับเป็นข้อมูล Continuous...")
    
    # 6.1 แมปค่าจาก Numpy (0,1,2..) กลับเป็นค่าดั้งเดิมใน df_processed ก่อน
    df_syn_temp = pd.DataFrame(columns=df_processed.columns)
    for col_idx, col_name in enumerate(df_processed.columns):
        unique_vals = sorted(list(set(df_processed[col_name].values)))
        reverse_mapping = {idx: val for idx, val in enumerate(unique_vals)}
        df_syn_temp[col_name] = [reverse_mapping.get(val, 0) for val in syn_discrete_data[:, col_idx]]

    # 6.2 โยนเข้า Integrator เพื่อกางข้อมูล Latent Space ออกมาเป็น Continuous
    final_synthetic_df = integrator.reconstruct_synthetic_data(df_syn_temp)

    # ==========================================
    # 7. เซฟผลลัพธ์
    # ==========================================
    out_dir = './result'
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    
    out_path = f"{out_dir}/{DATA_NAME}_hybrid_synthetic.csv"
    # สร้างลิสต์รายชื่อคอลัมน์เรียงตามต้นฉบับเป๊ะๆ
    # ORIGINAL_COL_ORDER = [
    #     '0_age', '1_categorical', '2_fnlwgt', '3_categorical', 
    #     '4_categorical', '5_categorical', '6_categorical', '7_categorical', 
    #     '8_categorical', '9_categorical', '10_capital_gain', '11_capital_loss', 
    #     '12_hours_per_week', '13_categorical', '14_categorical'
    # ]

    ORIGINAL_COL_ORDER = [
        '1', '3', '4', '5', 
        '6', '7', '8', '9', 
        '13', '14'
    ]
    
    # 🌟 จัดเรียงคอลัมน์ใหม่
    final_synthetic_df = final_synthetic_df[ORIGINAL_COL_ORDER]
    final_synthetic_df.to_csv(out_path, index=False)
    
    print("\n" + "=" * 70)
    print(f" เสร็จสมบูรณ์! ข้อมูลสังเคราะห์ระดับ High-Fidelity ถูกบันทึกไว้ที่: {out_path}")
    print("ตัวอย่างข้อมูล 5 แถวแรกที่กางกลับมาเป็น Continuous แล้ว:")
    print(final_synthetic_df[CONTINUOUS_COLS].head())

if __name__ == '__main__':
    run_hybrid_pipeline()