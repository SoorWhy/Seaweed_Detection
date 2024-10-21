import os
import json
import cv2
from ultralytics import YOLO

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append((filename, img))
    return images

def save_all_json_results(all_results, output_path):
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=4)

def save_visualization(image, predictions, output_folder, image_id):
    # 예측된 bounding box가 없는 경우 스킵
    if not predictions:
        return

    for pred in predictions:
        # bounding box 정보가 있는지 확인하고 시각화
        if 'bbox' in pred:
            x_min, y_min, x_max, y_max = pred['bbox']
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # 빨간색으로 변경
            label = f"{pred['defect_class']}"  # 'class' 대신 'defect_class'로 수정
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    output_path = os.path.join(output_folder, f"visualization_{image_id}.jpg")
    cv2.imwrite(output_path, image)

def main():
    # 클래스 매핑 딕셔너리
    class_mapping = {0: 'st', 1: 'aq', 2: 'fl'}

    # 모델 로드
    model = YOLO('best.pt')

    # 이미지 경로 설정
    image_folder = './testset'  # 예측할 이미지 폴더
    json_output_file = './results/predictions.json'
    visualization_output_folder = './results/visualization'

    # 결과 저장 폴더 생성
    os.makedirs(visualization_output_folder, exist_ok=True)

    # 이미지 불러오기
    images = load_images_from_folder(image_folder)

    # 전체 결과를 저장할 리스트
    all_results = []

    for image_id, image in images:
        # 예측 수행
        results = model.predict(image)

        image_result = {"image_name": image_id, "Defect coordinates": []}
        predictions = []

        # 예측이 있을 경우 결과 저장
        if results:
            for result in results:
                for box in result.boxes:
                    bbox = box.xyxy[0]  # (x_min, y_min, x_max, y_max)
                    label_index = int(box.cls)
                    confidence = float(box.conf)
                    # 클래스 인덱스를 문자열로 매핑
                    label = class_mapping.get(label_index, 'Unknown')
                    predictions.append({
                        'defect_class': label,  # 'class'가 아닌 'defect_class'로 저장
                        'top_x': int(bbox[0]),
                        'top_y': int(bbox[1]),
                        'bot_x': int(bbox[2]),
                        'bot_y': int(bbox[3]),
                    })

            # 예측된 좌표가 있으면 추가
            if predictions:
                image_result["Defect coordinates"] = predictions

            # 시각화 이미지 저장
            save_visualization(image, predictions, visualization_output_folder, image_id.split('.')[0])

        # 예측이 없더라도 이미지 이름은 기록
        all_results.append(image_result)

    # 최종 JSON 파일로 결과 저장
    save_all_json_results(all_results, json_output_file)

if __name__ == '__main__':
    main()
