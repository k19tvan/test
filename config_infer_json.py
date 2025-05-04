import os
import json
from PIL import Image

image_folder = "/root/AICITY2024_Track4/dataset/vip_cup_night/images/"
categories = [
    {"id": 0, "name": "Bus"},
    {"id": 1, "name": "Bike"},
    {"id": 2, "name": "Car"},
    {"id": 3, "name": "Pedestrian"},
    {"id": 4, "name": "Truck"},
]

images = []
image_id = 0

for filename in sorted(os.listdir(image_folder)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(image_folder, filename)
        with Image.open(image_path) as img:
            width, height = img.size

        images.append({
            "id": image_id,
            "file_name": filename,
            "height": height,
            "width": width
        })
        image_id += 1

coco_output = {
    "info": {},
    "licenses": {},
    "categories": categories,
    "images": images,
    "annotations": []
}

output_path = "/root/AICITY2024_Track4/dataset/json_labels/data_night.json"
with open(output_path, "w") as f:
    json.dump(coco_output, f, indent=4)

print(f"JSON đã được ghi ra: {output_path}")
