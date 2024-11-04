from ultralytics import YOLO
import wandb

if __name__ == '__main__':
    # W&B API 키 로그인
    wandb.login(key='여기에 실제 API 키를 입력하세요.')  # 여기에 실제 API 키를 입력하세요.

    # W&B 초기화
    wandb.init(
        project="my-awesome-project",
        config={
            "learning_rate": 0.001,
            "architecture": "YOLOv8l",
            "dataset": "your_dataset_name",
            "epochs": 100,  # 이 값은 Sweep에서 조정될 수 있음
            "img_size": 1024,
            "batch_size": 4,
        }
    )

    # W&B에서 설정된 파라미터 가져오기
    config = wandb.config

    # 모델 로드 (best.pt 사용)
    model = YOLO('Yolov8x.pt')

    # 학습 설정 및 결과 저장
    results = model.train(
        data='./data/data.yaml',
        epochs=config.epochs,  # W&B에서 설정된 epochs 사용
        imgsz=config.img_size,  # W&B에서 설정된 img_size 사용
        batch=config.batch_size,  # W&B에서 설정된 batch_size 사용
        device=0,
        lr0=config.learning_rate,  # W&B에서 설정된 learning_rate 사용
        lrf=0.1,
        augment=True,
    )

    # 시각화할 메트릭
    for epoch in range(config.epochs):  # W&B에서 설정된 epochs 사용
        metrics = model.metrics
        wandb.log({
            "epoch": epoch,
            "train/box_loss": results.box_loss[epoch],
            "train/cls_loss": results.cls_loss[epoch],
            "train/dfl_loss": results.dfl_loss[epoch],
            "val/box_loss": results.val.box_loss[epoch],
            "val/cls_loss": results.val.cls_loss[epoch],
            "val/dfl_loss": results.val.dfl_loss[epoch],
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "mAP50": metrics['mAP50'],
            "mAP50-95": metrics['mAP50-95'],
        })

    # W&B 종료
    wandb.finish()
