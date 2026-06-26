import pandas as pd
import numpy as np
import os

def generate_custom_dataset():
    # 🌟 แก้ไขตรงนี้: เพิ่ม 0 ไปหนึ่งตัว (จำลองข้อมูล 450,000 แถว หรือ x10)
    N_ROWS = 450000 
    
    print(f"กำลังสร้างข้อมูลจำลองจำนวน {N_ROWS} แถว 15 คอลัมน์...")
    
    # ล็อกค่า Seed เพื่อให้สุ่มออกมากี่ครั้งก็ได้ข้อมูลหน้าตาเหมือนเดิม
    np.random.seed(42)
    
    df = pd.DataFrame()

    # (โค้ดส่วนที่เหลือเหมือนเดิมทุกประการครับ)
    df['0_age'] = np.clip(np.random.normal(loc=38, scale=13, size=N_ROWS), 17, 90).round(2)
    df['2_fnlwgt'] = np.random.lognormal(mean=11.5, sigma=0.8, size=N_ROWS).round(2)
    
    cap_gain = np.random.exponential(scale=5000, size=N_ROWS)
    cap_gain[np.random.rand(N_ROWS) < 0.90] = 0.0 
    df['10_capital_gain'] = cap_gain.round(2)
    
    cap_loss = np.random.exponential(scale=2000, size=N_ROWS)
    cap_loss[np.random.rand(N_ROWS) < 0.95] = 0.0
    df['11_capital_loss'] = cap_loss.round(2)
    
    df['12_hours_per_week'] = np.clip(np.random.normal(loc=40, scale=12, size=N_ROWS), 1, 99).round(2)

    cat_columns = [1, 3, 4, 5, 6, 7, 8, 9, 13, 14]
    for col in cat_columns:
        max_categories = np.random.randint(3, 15) 
        df[f'{col}_categorical'] = np.random.randint(0, max_categories, size=N_ROWS)

    ordered_cols = [
        '0_age', '1_categorical', '2_fnlwgt', '3_categorical', '4_categorical',
        '5_categorical', '6_categorical', '7_categorical', '8_categorical', '9_categorical',
        '10_capital_gain', '11_capital_loss', '12_hours_per_week', '13_categorical', '14_categorical'
    ]
    df = df[ordered_cols]

    out_dir = './data'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    out_file = os.path.join(out_dir, 'mock_adult_raw_450k.csv')
    df.to_csv(out_file, index=False)
    
    print(f"🎉 สร้างไฟล์สำเร็จ! บันทึกไว้ที่: {out_file}")

if __name__ == "__main__":
    generate_custom_dataset()