"""Train a YOLO image classifier with normal/occluded folders."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train normal/occluded classifier")
    parser.add_argument("--data", default="datasets/occlusion_classifier")
    parser.add_argument("--model", default="yolo11n-cls.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--name", default="camera_occlusion_classifier")
    args = parser.parse_args()

    data = Path(args.data)
    for split in ("train", "val"):
        for label in ("normal", "occluded"):
            if not (data / split / label).exists():
                raise SystemExit(
                    f"缺少 {data / split / label}。请按 README 建立 train/val 的 normal 和 occluded 文件夹。"
                )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("缺少 ultralytics，请先执行: python -m pip install -r requirements-yolo.txt") from exc

    model = YOLO(args.model)
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                project="runs/classify", name=args.name)
    print("训练完成。请使用 runs/classify/" + args.name + "/weights/best.pt 进行推理。")


if __name__ == "__main__":
    main()
