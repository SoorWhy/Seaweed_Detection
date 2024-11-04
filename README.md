# Seaweed_Detection
이 프로젝트는 2024 DATA·AI 분석 경진대회 참가를 위해 제작하였습니다. 사전 학습된 YOLO 모델(best.pt)을 활용하여 김(Seaweed) 객체를 감지하고 분류합니다.  

## 디렉터리 구성
### | code.ipynb
* 라이브러리 설치 및 모델 학습, 평가 등을 위한 코드
### | jsontotxt.py
* json 형식의 라벨링 데이터를 yolo 학습에 맞게 txt 형식으로 변경하는 코드
### | predict.py
* 테스트셋 평가 코드, results 폴더에 visualization 폴더와 json 파일 생성
### | requirements.txt
* 종속성 설치를 위한 파일
### | train.py
* 지정된 하이퍼파라미터를 사용한 모델 학습 코드
### | trainWB.py
* WandB툴을 활용한 하이퍼파라미터 스윕 학습 코드

## 프로젝트에 사용된 기술 및 버전
### | Language
* Python `3.11.7`
* numpy `1.26.3`
* torch `2.4.1+cu121`
* ultralytics `8.2.99`
* opencv-python `4.10.0.84`
### | System
* Windows 10
* Driver Version: 531.14
* CUDA Version: 12.1

## 종속성 설치
아래 단계에 따라 환경을 설정하고 모델을 실행합니다. requirements.txt 파일을 사용하여 환경을 설정할 수 있습니다.
* python 3.11.7 가상환경 생성
```python
pip install -r requirements.txt
```

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

## 결과
### 리더 보드를 통한 정량적 평가
* 총 13위 중 7위
![리더보드](https://github.com/user-attachments/assets/3668195f-aa72-43e0-b494-25c9ef2e172d)
### 발표 평가를 통한 정성적 평가

## 프로젝트 회고
### 어려웠던 점
* 김의 이물질이라는 아주 미세한 객체 탐지 과제의 어려움
* 학습 시엔 검증 데이터셋에 대한 정밀도, 재현율 등이 좋은 흐름을 보였지만 실제로 테스트셋에 대한 추론 결과로는 과적합이었던 모델
### 배운 점
* yolov8 버전의 Nano ~ Xlarge 다양한 모델의 성능을 비교하면서 우수한 모델 선택 과정 경험
  * yolov8 버전만 고정시켜서 비교하지 말고 과거의 버전이나 얼마전에 나온 11 버전의 성능도 비교 해봤으면 좋았을 것 같음
* 검증 데이터셋의 성능 평가 지표가 이상적인 그래프를 그리더라도 절대적으로 좋은 모델이 아니라는 것을 알게 됨
  * 에포크 100까지 지속적인 loss 감소, 정밀도, 재현율 증가하더라도 실제로 과적합이 일어났고 에포크 10 학습 모델의 성능이 가장 좋았음
