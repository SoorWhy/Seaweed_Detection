from ultralytics import YOLO
import wandb

if __name__ == '__main__':
    # W&B 사용 시 주석 해제
    # # W&B API 키 로그인
    # wandb.login(key='여기에 실제 API 키를 입력하세요.')  # 여기에 실제 API 키를 입력하세요.

    # # W&B 초기화
    # wandb.init(
    #     project="my-awesome-project",
    #     config={
    #         "learning_rate": 0.001,
    #         "architecture": "YOLOv8l",
    #         "dataset": "your_dataset_name",
    #         "epochs": 100,  # 최종 에포크 수 (Sweep에 따라 변경될 수 있음)
    #         "img_size": 1024,
    #         "batch_size": 4,
    #     }
    # )

    # 모델 로드
    model = YOLO('yolov8x.pt')

     # 학습 설정 및 결과 저장
    results = model.train(
        data='./data/data.yaml',
        epochs=10,
        imgsz=1024,
        batch=2,
        device=0,
        lr0=0.001,
        lrf=0.1,
        augment=True,
    )

    # W&B 종료
    # wandb.finish()
