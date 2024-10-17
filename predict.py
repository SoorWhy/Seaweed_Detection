from ultralytics import YOLO
import os
import json
import cv2

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append((filename, img))
    return images

def save_json_results(image_id, predictions, output_folder):
    result = {"ImageID": image_id, "PredictionString": ""}
    prediction_strings = []

    for pred in predictions:
        label = pred['class']  # 결함 클래스
        confidence = pred['confidence']
        x_min, y_min, x_max, y_max = pred['bbox']
        prediction_strings.append(f"{label} {confidence} {x_min} {y_min} {x_max} {y_max}")

    if prediction_strings:
        result["PredictionString"] = " ".join(prediction_strings)
    
    output_path = os.path.join(output_folder, f"{image_id}.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)

def save_visualization(image, predictions, output_folder, image_id):
    for pred in predictions:
        x_min, y_min, x_max, y_max = pred['bbox']
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # 빨간색으로 변경
        label = f"{pred['class']}"
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    output_path = os.path.join(output_folder, f"visualization_{image_id}.jpg")
    cv2.imwrite(output_path, image)

def main():
    # 클래스 매핑 딕셔너리
    class_mapping = {0: 'String', 1: 'Aqua', 2: 'Floating'}

    # 모델 로드
    model = YOLO('best.pt')

    # 이미지 경로 설정
    image_folder = './images'  # 예측할 이미지 폴더
    json_output_folder = './results/json'
    visualization_output_folder = './results/visualization'

    # 결과 저장 폴더 생성
    os.makedirs(json_output_folder, exist_ok=True)
    os.makedirs(visualization_output_folder, exist_ok=True)

    # 이미지 불러오기
    images = load_images_from_folder(image_folder)

    for image_id, image in images:
        # 예측 수행
        results = model.predict(image)

        for result in results:
            predictions = []
            # bounding box를 가져오는 방식 수정
            for box in result.boxes:
                bbox = box.xyxy[0]  # (x_min, y_min, x_max, y_max)
                label_index = int(box.cls)
                confidence = float(box.conf)
                # 클래스 인덱스를 문자열로 매핑
                label = class_mapping.get(label_index, 'Unknown')
                predictions.append({
                    'class': label,
                    'confidence': confidence,
                    'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                })

            # JSON 파일로 결과 저장
            save_json_results(image_id.split('.')[0], predictions, json_output_folder)

            # 시각화 이미지 저장
            save_visualization(image, predictions, visualization_output_folder, image_id.split('.')[0])

if __name__ == '__main__':
    main()
