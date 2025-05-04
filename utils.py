import os
import shutil
import random

# Đường dẫn đến thư mục gốc và thư mục đích
source_folder = '/root/AICITY2024_Track4/dataset/vip_cup_day/images/train'
destination_folder = '/root/AICITY2024_Track4/dataset/data_fake'

# Số lượng file muốn sao chép
num_files_to_copy = 20

# Lấy danh sách tất cả các file trong thư mục nguồn
all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

# Lựa chọn ngẫu nhiên một số file
files_to_copy = random.sample(all_files, min(num_files_to_copy, len(all_files)))

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs(destination_folder, exist_ok=True)

# Sao chép các file được chọn
for file_name in files_to_copy:
    src_path = os.path.join(source_folder, file_name)
    dst_path = os.path.join(destination_folder, file_name)
    shutil.copy2(src_path, dst_path)  # copy2 giữ nguyên metadata của file

print(f"Đã sao chép {len(files_to_copy)} file từ {source_folder} sang {destination_folder}.")