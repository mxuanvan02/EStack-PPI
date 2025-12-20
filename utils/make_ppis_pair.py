import pandas as pd
import numpy as np
import os
from typing import List, Tuple

# === ĐỊNH NGHĨA TÊN FILE ĐẦU VÀO/ĐẦU RA ===
POSITIVE_FILE = 'data/ppis/positive.txt'
NEGATIVE_FILE = 'data/ppis/negative.txt'
OUTPUT_FILE = 'data/ppis/pairs.tsv'

def load_pairs_and_label(file_path: str, label: int) -> List[Tuple[str, str, int]]:
    """
    Đọc các cặp ID từ một tệp và gán nhãn (Label) tương ứng.
    Giả định mỗi dòng trong tệp có định dạng: id_A id_B (cách nhau bởi khoảng trắng hoặc tab).
    """
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Tách ID, xử lý các trường hợp khoảng trắng (space) hoặc tab
                parts = line.split() 
                if len(parts) != 2:
                    # Thử tách bằng tab nếu tách bằng space không đủ
                    parts = line.split('\t')
                
                if len(parts) == 2:
                    id_a, id_b = parts
                    data.append((id_a.strip(), id_b.strip(), label))
                else:
                    print(f"Bỏ qua dòng không hợp lệ trong {file_path}: {line}")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp {file_path}")
    return data

def merge_and_save_tsv(positive_file: str, negative_file: str, output_file: str):
    """
    Ghép dữ liệu dương (1) và âm (0) thành một tệp TSV.
    """
    print(f"Đang xử lý tệp: {positive_file}")
    # 1. Tải và gán nhãn cho Positive Pairs (Label = 1)
    positive_data = load_pairs_and_label(positive_file, label=1)
    
    print(f"Đang xử lý tệp: {negative_file}")
    # 2. Tải và gán nhãn cho Negative Pairs (Label = 0)
    negative_data = load_pairs_and_label(negative_file, label=0)
    
    if not positive_data and not negative_data:
        print("Không có dữ liệu nào được tải. Dừng xử lý.")
        return

    # 3. Hợp nhất dữ liệu
    all_data = positive_data + negative_data
    
    # 4. Tạo DataFrame để dễ dàng lưu trữ và chuẩn hóa
    df = pd.DataFrame(all_data, columns=['ProteinID_A', 'ProteinID_B', 'Label'])
    
    # Optional: Xáo trộn dữ liệu để tránh thiên lệch khi huấn luyện
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 5. Lưu thành tệp TSV (Tab-Separated Values)
    df.to_csv(output_file, sep='\t', index=False, header=False)
    
    print("\n✅ Quá trình ghép tệp hoàn tất!")
    print(f"Tệp đầu ra đã được lưu tại: {output_file}")
    print(f"Tổng số cặp đã được lưu: {len(df)}")
    print(f"Phân bố nhãn: \n{df['Label'].value_counts()}")

# === THỰC THI CHÍNH ===
if __name__ == "__main__":
    # Lưu ý: Cần đảm bảo hai tệp positive.txt và negative.txt nằm cùng thư mục với script này
    merge_and_save_tsv(POSITIVE_FILE, NEGATIVE_FILE, OUTPUT_FILE)