import os
import csv
import json

def run_data_prep_only():
    # ==========================================
    # 1. การตั้งค่า (Configuration)
    # ==========================================
    DATA_NAME = 'adult'              # ชื่อไฟล์ CSV ดิบของคุณ (ในโฟลเดอร์ ./data/)
    CONTINUOUS_COLS = [0, 2, 10, 11, 12] # กำหนด Index ของคอลัมน์ที่เป็น Continuous
    BIN_NUM = 8                         # ลองปรับเปลี่ยนจำนวนตะกร้าตรงนี้ได้เลย (เช่น 5, 16, 32, 64)
    
    RAW_FILE_PATH = f'./data/{DATA_NAME}.csv'
    OUT_DISCRETE_FILE = f'./data/{DATA_NAME}_discrete_binned.csv'
    OUT_INFO_FILE = f'./data/{DATA_NAME}_bin_rules.json'

    print(f"🚀 เริ่มต้นกระบวนการ Data Preparation (Binning) สำหรับ: {DATA_NAME}")

    # เช็คว่ามีไฟล์ดิบอยู่จริงไหม
    if not os.path.exists(RAW_FILE_PATH):
        print(f"❌ หาไฟล์ไม่เจอ: {RAW_FILE_PATH}")
        print("โปรดตรวจสอบให้แน่ใจว่าคุณมีไฟล์นี้อยู่ในโฟลเดอร์ ./data/ ครับ")
        return

    # ==========================================
    # 2. อ่านไฟล์และหั่นข้อมูล (Discretization)
    # ==========================================
    with open(RAW_FILE_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headings = next(reader)
        raw_data = [row for row in reader if len(row) > 0]

    attr_num = len(headings)
    bin_info = {}       # จำข้อมูล Continuous
    cat_mapping = {}    # จำข้อมูล Categorical (String -> Int)

    # สร้างโครงตารางข้อมูลผลลัพธ์
    processed_data = [[0]*attr_num for _ in range(len(raw_data))]

    for col in range(attr_num):
        col_values = [row[col] for row in raw_data]
        col_name = headings[col]
        
        # กรณีที่ 1: คอลัมน์เป็น Continuous (ต้องหั่นตะกร้า)
        if col in CONTINUOUS_COLS:
            float_vals = [float(v) for v in col_values]
            min_val = min(float_vals)
            max_val = max(float_vals)
            
            # คำนวณความกว้างของตะกร้า
            bin_size = (max_val - min_val) / BIN_NUM if max_val > min_val else 1.0 
            
            # บันทึกกฎการหั่น
            bin_info[col_name] = {
                'col_index': col,
                'min_value': min_val, 
                'max_value': max_val,
                'bin_size': bin_size,
                'total_bins': BIN_NUM
            }
            
            # แปลงค่าแต่ละแถวให้เป็นเลขตะกร้า
            for row_idx, val in enumerate(float_vals):
                b_idx = int((val - min_val) / bin_size)
                b_idx = min(b_idx, BIN_NUM - 1) # ล็อกไม่ให้หลุดขอบตะกร้าสุดท้าย
                processed_data[row_idx][col] = b_idx

        # กรณีที่ 2: คอลัมน์เป็น Categorical (แปลง String เป็น Int)
        else:
            unique_vals = sorted(list(set(col_values)))
            cat_mapping[col_name] = {val: idx for idx, val in enumerate(unique_vals)}
            
            for row_idx, val in enumerate(col_values):
                processed_data[row_idx][col] = cat_mapping[col_name][val]

    print(f"   - แปลงข้อมูลสำเร็จ! จำนวนคอลัมน์: {attr_num}, ขนาดแถว: {len(raw_data)}")

    # ==========================================
    # 3. บันทึกผลลัพธ์
    # ==========================================
    # เซฟตารางข้อมูลที่หั่นเสร็จแล้ว
    with open(OUT_DISCRETE_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headings)
        writer.writerows(processed_data)
        
    # เซฟกฎการหั่น (เพื่อให้คุณเปิดอ่านทำความเข้าใจได้ง่าย)
    with open(OUT_INFO_FILE, 'w', encoding='utf-8') as f:
        json.dump(
            {'continuous_columns_rules': bin_info, 'categorical_mappings': cat_mapping}, 
            f, indent=4, ensure_ascii=False
        )
        
    print(f"\n🎉 เสร็จเรียบร้อย! คุณสามารถเข้าไปตรวจสอบผลลัพธ์ได้ที่:")
    print(f" 1. ข้อมูลที่ถูกหั่นแล้ว: {OUT_DISCRETE_FILE}")
    print(f" 2. กฎการหั่นและจับคู่:  {OUT_INFO_FILE}")

if __name__ == '__main__':
    run_data_prep_only()