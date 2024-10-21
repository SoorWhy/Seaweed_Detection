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
    # 바운딩 박스가 그려지지 않더라도 모든 이미지 저장
    for pred in predictions:
        x_min, y_min, x_max, y_max = pred['top_x'], pred['top_y'], pred['bot_x'], pred['bot_y']
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2) 
        label = f"{pred['defect_class']}"
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 이미지 저장 (모든 이미지 저장)
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
    defect_count = 0  # 결함 이미지 카운트
    no_defect_count = 0  # 무결함 이미지 카운트

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
                        'defect_class': label,
                        'top_x': int(bbox[0]),
                        'top_y': int(bbox[1]),
                        'bot_x': int(bbox[2]),
                        'bot_y': int(bbox[3])
                    })

            # 예측된 좌표가 있으면 추가
            if predictions:
                image_result["Defect coordinates"] = predictions
                defect_count += 1  # 결함 데이터로 카운트
            else:
                no_defect_count += 1  # 무결함 데이터로 카운트
        else:
            no_defect_count += 1  # 예측 결과가 없으면 무결함으로 카운트

        # 예측이 없더라도 이미지 이름은 기록
        all_results.append(image_result)

        # 시각화 이미지 저장 (바운딩 박스 여부 상관없이 이미지 저장)
        save_visualization(image, predictions, visualization_output_folder, image_id.split('.')[0])

    # 최종 JSON 파일로 결과 저장
    save_all_json_results(all_results, json_output_file)

    # 결함 및 무결함 데이터 수 출력
    print(f"총 결함 데이터 수: {defect_count}")
    print(f"총 무결함 데이터 수: {no_defect_count}")

if __name__ == '__main__':
    main()
