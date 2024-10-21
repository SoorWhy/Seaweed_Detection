# Seaweed_Detection
이 프로젝트는 2024 DATA·AI 분석 경진대회 참가를 위해 제작하였습니다. 사전 학습된 YOLO 모델(best.pt)을 활용하여 김(Seaweed) 객체를 감지하고 분류합니다.  

## 프로젝트에 사용된 기술 및 버전
### | Language
* Python `3.11.7`
* torch `2.4.1+cu121`
* ultralytics `8.2.99`
* opencv-python `4.10.0.84`
### | System
* Windows 10

## 종속성 설치
아래 단계에 따라 환경을 설정하고 모델을 실행합니다. requirements.txt 파일을 사용하여 환경을 설정할 수 있습니다.
* python 3.11.7 가상환경 생성
```python
pip install -r requirements.txt
```

## 디렉터리 구성
```
 ┣ testset
 ┃ ┗ ...(예측 할 이미지)
 ┣ results
 ┃ ┣ predictions.json
 ┃ ┗ visualization
 ┃   ┗ ...(예측 결과 이미지)
 ┣ best.pt
 ┣ requirements.txt
 ┗ predict.py
```
testset 폴더에 예측 할 이미지를 넣고, 결과는 results 폴더에 저장됩니다.
* results 폴더는 최초 실행 시 자동으로 생성됩니다.

## 예측 수행
사전 학습 모델(best.pt)을 활용하여 예측을 수행합니다.
```python
python predict.py
```

## 결과 예시
* results/json
* PredictionString {Label Confidence XMin YMin XMax YMax}  
![json_](https://github.com/user-attachments/assets/9f8c0439-a1b3-49fc-ae3b-555e0de3d068)
* results/visualization  
![visualization_seaweed_01348](https://github.com/user-attachments/assets/fc3dc545-e35c-4cd7-9f8a-e73c258acbdc)
