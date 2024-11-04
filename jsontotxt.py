import json
import os

# 이미지 크기 (가로와 세로) 설정
image_width = 512
image_height = 512

# 클래스 인덱스 매핑
class_mapping = {
    'st': 0,
    'aq': 1,
    'fl': 2
}

# 경로 설정
train_images_path = './dataset/train/train_defected_dataset'
train_json_folder = './dataset/train/train_defected_json'
train_labels_path = './dataset/train/labels'  # 라벨 폴더 경로
val_images_path = './dataset/validation/validation_defected_dataset'
val_json_folder = './dataset/validation/validation_defected_json'
val_labels_path = './dataset/validation/labels'  # 라벨 폴더 경로

def convert_to_yolo_format(json_file_path, output_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # 데이터가 단일 객체인 경우 리스트로 변환
    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list):
        raise ValueError(f"JSON file {json_file_path} does not contain a list or single object.")

    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"Item in JSON file {json_file_path} is not a dictionary.")

        image_name = item.get('image_name')
        defect_class = item.get('defect_class')
        top_x = item.get('top_x')
        top_y = item.get('top_y')
        bot_x = item.get('bot_x')
        bot_y = item.get('bot_y')

        if image_name is None or defect_class is None or top_x is None or top_y is None or bot_x is None or bot_y is None:
            raise ValueError(f"Missing required fields in JSON file {json_file_path}.")

        # 클래스 인덱스 변환
        class_index = class_mapping.get(defect_class, -1)  # 클래스 인덱스가 없으면 -1로 설정

        # Bounding Box 변환
        x_center = ((top_x + bot_x) / 2) / image_width
        y_center = ((top_y + bot_y) / 2) / image_height
        width = (bot_x - top_x) / image_width
        height = (bot_y - top_y) / image_height

        # YOLO 형식으로 저장
        output_file = os.path.join(output_path, f'{image_name.replace(".png", ".txt")}')
        with open(output_file, 'w') as file:
            file.write(f"{class_index} {x_center} {y_center} {width} {height}\n")

def process_json_folder(json_folder, labels_path):
    # labels_path가 없으면 생성
    if not os.path.exists(labels_path):
        os.makedirs(labels_path)
    
    for file_name in os.listdir(json_folder):
        if file_name.endswith('.json'):
            json_file_path = os.path.join(json_folder, file_name)
            convert_to_yolo_format(json_file_path, labels_path)

# 학습 데이터 변환
process_json_folder(train_json_folder, train_labels_path)

# 검증 데이터 변환
process_json_folder(val_json_folder, val_labels_path)

print("Conversion completed.")
