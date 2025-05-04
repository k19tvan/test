import cv2
import json
import os

# --- Cấu hình ---
DETECTIONS_FILE = '/root/AICITY2024_Track4/infer/CO-DETR/work_dirs/infer_fake.bbox.json'  # File JSON chứa danh sách các phát hiện dài
METADATA_FILE = '/root/AICITY2024_Track4/dataset/json_labels/datafake.json'      # File JSON chứa thông tin categories và images
IMAGE_FOLDER = '/root/AICITY2024_Track4/dataset/data_fake' # Thư mục chứa ảnh gốc
OUTPUT_FOLDER = 'output_annotated_images' # Thư mục lưu ảnh đã vẽ bounding box
SCORE_THRESHOLD = 0.3 # Chỉ vẽ box có score >= ngưỡng này
BOX_THICKNESS = 2
TEXT_THICKNESS = 1
FONT_SCALE = 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX

# --- Tải dữ liệu từ JSON ---
try:
    with open(DETECTIONS_FILE, 'r', encoding='utf-8') as f: # Thêm encoding='utf-8' để đảm bảo đọc đúng
        detection_data = json.load(f)
    if not isinstance(detection_data, list):
         print(f"Lỗi: Nội dung file {DETECTIONS_FILE} không phải là một danh sách JSON.")
         exit()
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file detections tại {DETECTIONS_FILE}")
    exit()
except json.JSONDecodeError as e:
    print(f"Lỗi: Không thể giải mã JSON từ {DETECTIONS_FILE}. Lỗi: {e}")
    exit()
except Exception as e:
    print(f"Lỗi không xác định khi tải {DETECTIONS_FILE}: {e}")
    exit()

try:
    with open(METADATA_FILE, 'r', encoding='utf-8') as f: # Thêm encoding='utf-8'
        metadata = json.load(f)
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file metadata tại {METADATA_FILE}")
    exit()
except json.JSONDecodeError as e:
    print(f"Lỗi: Không thể giải mã JSON từ {METADATA_FILE}. Lỗi: {e}")
    exit()
except Exception as e:
    print(f"Lỗi không xác định khi tải {METADATA_FILE}: {e}")
    exit()

# --- Chuẩn bị các mapping ---
# Category ID -> Tên
try:
    category_map = {cat['id']: cat['name'] for cat in metadata['categories']}
except KeyError:
    print("Lỗi: Thiếu key 'categories' hoặc cấu trúc không đúng trong metadata.")
    exit()
except TypeError:
     print("Lỗi: Cấu trúc 'categories' trong metadata không phải là danh sách các dictionary.")
     exit()

# Image ID -> Tên file
try:
    image_id_to_filename = {img['id']: img['file_name'] for img in metadata['images']}
except KeyError:
    print("Lỗi: Thiếu key 'images' hoặc cấu trúc không đúng trong metadata.")
    exit()
except TypeError:
     print("Lỗi: Cấu trúc 'images' trong metadata không phải là danh sách các dictionary.")
     exit()

# Định nghĩa màu cho các loại đối tượng (định dạng BGR)
COLORS = [
    (0, 0, 255),    # 0: Bus (Đỏ)
    (0, 255, 0),    # 1: Bike (Xanh lá)
    (255, 0, 0),    # 2: Car (Xanh dương)
    (0, 255, 255),  # 3: Pedestrian (Vàng)
    (255, 0, 255)   # 4: Truck (Magenta)
]

# --- Tạo thư mục output ---
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"Thư mục output: {OUTPUT_FOLDER}")

# --- Nhóm các phát hiện theo Image ID ---
detections_by_image = {}
for det in detection_data:
    img_id = det.get('image_id')
    if img_id is None:
        print(f"Cảnh báo: Bỏ qua detection thiếu 'image_id': {det}")
        continue

    if img_id not in detections_by_image:
        detections_by_image[img_id] = []
    detections_by_image[img_id].append(det)

# --- Xử lý từng ảnh ---
print(f"Đang xử lý {len(detections_by_image)} ảnh...")

for image_id, detections in detections_by_image.items():
    if image_id not in image_id_to_filename:
        print(f"Cảnh báo: Bỏ qua image_id {image_id} - không tìm thấy trong metadata.")
        continue

    file_name = image_id_to_filename[image_id]
    image_path = os.path.join(IMAGE_FOLDER, file_name)
    output_path = os.path.join(OUTPUT_FOLDER, file_name)

    print(f"Đang xử lý {image_path}...")

    # Kiểm tra file ảnh có tồn tại không
    if not os.path.exists(image_path):
        print(f"  Lỗi: Không tìm thấy file ảnh tại {image_path}. Bỏ qua.")
        continue

    # Tải ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"  Lỗi: Không thể đọc file ảnh {image_path}. Bỏ qua.")
        continue

    # Vẽ bounding box cho ảnh này
    detection_count = 0
    for det in detections:
        score = det.get('score', 0)
        category_id = det.get('category_id')
        bbox = det.get('bbox')

        # Kiểm tra cơ bản
        if category_id is None or bbox is None:
             print(f"  Cảnh báo: Bỏ qua entry không hợp lệ: {det}")
             continue
        if category_id not in category_map:
            print(f"  Cảnh báo: Không rõ category_id {category_id} trong detection: {det}. Bỏ qua.")
            continue
        if not isinstance(bbox, list) or len(bbox) != 4:
            print(f"  Cảnh báo: Định dạng bbox không hợp lệ trong detection: {det}. Bỏ qua.")
            continue

        # Áp dụng ngưỡng score
        if score < SCORE_THRESHOLD:
            continue

        detection_count += 1

        # Lấy tọa độ box và chuyển sang số nguyên
        try:
            x, y, w, h = map(int, bbox)
        except (ValueError, TypeError):
            print(f"  Cảnh báo: Không thể chuyển tọa độ bbox {bbox} sang số nguyên. Bỏ qua.")
            continue

        # Tính toán góc trên-trái và dưới-phải
        pt1 = (x, y)
        pt2 = (x + w, y + h)

        # Lấy tên category và màu
        category_name = category_map.get(category_id, f"ID:{category_id}") # Thêm dự phòng nếu ID không có trong map
        color_index = category_id % len(COLORS) # Lặp lại màu nếu số category nhiều hơn số màu định nghĩa
        color = COLORS[color_index]

        # Vẽ hình chữ nhật
        cv2.rectangle(image, pt1, pt2, color, BOX_THICKNESS)

        # Chuẩn bị text cho nhãn
        label = f"{category_name}: {score:.2f}"

        # Tính kích thước text để vẽ nền
        (text_width, text_height), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, TEXT_THICKNESS)

        # Vẽ nền cho text
        # Đặt text phía trên bbox một chút
        text_origin_bg = (x, y - text_height - baseline - 3) # -3 để có thêm khoảng trống
        text_end_bg = (x + text_width, y - baseline + 2)

        # Điều chỉnh nếu text bị lệch ra khỏi mép trên ảnh
        if text_origin_bg[1] < 0:
             text_origin_bg = (x, y + baseline + 2) # Di chuyển vào trong box, phía dưới góc
             text_end_bg = (x + text_width, y + text_height + baseline + 2)

        # Vẽ hình chữ nhật nền (màu tô đầy) và sau đó vẽ text
        cv2.rectangle(image, text_origin_bg, text_end_bg, color, -1) # độ dày -1 để tô đầy
        # Vẽ text màu đen lên nền màu
        cv2.putText(image, label, (text_origin_bg[0] + 1, text_origin_bg[1] + text_height + 1), FONT, FONT_SCALE, (0,0,0), TEXT_THICKNESS, cv2.LINE_AA)

    # Lưu ảnh đã được chú thích
    try:
        cv2.imwrite(output_path, image)
        print(f"  Đã lưu ảnh vào {output_path} với {detection_count} boxes (ngưỡng={SCORE_THRESHOLD}).")
    except Exception as e:
        print(f"  Lỗi: Không thể ghi file ảnh {output_path}. Lý do: {e}")

print("--- Hoàn thành xử lý ---")