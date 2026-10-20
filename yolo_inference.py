"""YOLO camera occlusion inference demo.

Run:
    python yolo_inference.py --source 0 --weights yolo11n.pt

The official COCO model does not contain a camera-occlusion class. It is useful
for validating the YOLO inference pipeline, while a custom model trained with
camera/occlusion labels is required for direct occlusion classification.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from alarm_popup import CameraAlarmPopup


def parse_source(value: str):
    return int(value) if value.isdigit() else value


def image_metrics(frame: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    variance = float(gray.var())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 80, 160)
    edge_ratio = float(np.count_nonzero(edges)) / edges.size
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probabilities = hist / max(float(hist.sum()), 1.0)
    entropy = float(-(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0])).sum())
    return {
        "brightness": brightness,
        "variance": variance,
        "sharpness": sharpness,
        "edge_ratio": edge_ratio,
        "entropy": entropy,
    }


def quality_candidate(metrics: dict[str, float], baseline: dict[str, float] | None) -> tuple[bool, list[str]]:
    """Return a conservative image-quality anomaly candidate."""
    if baseline is None:
        return False, []
    reasons: list[str] = []
    hard_black_or_white = metrics["brightness"] < 15 or metrics["brightness"] > 245
    if hard_black_or_white:
        reasons.append("extreme_brightness")
    # A naturally plain wall has little variance, texture, and edge content.
    # Do not call it occluded merely because those values are low.
    low_information_baseline = (
        baseline["variance"] < 20
        or baseline["edge_ratio"] < 0.01
        or baseline["entropy"] < 2
    )
    if low_information_baseline and not hard_black_or_white:
        return False, []
    if baseline["variance"] > 20 and metrics["variance"] < baseline["variance"] * 0.35:
        reasons.append("low_variance")
    if baseline["sharpness"] > 10 and metrics["sharpness"] < baseline["sharpness"] * 0.20:
        reasons.append("blur")
    if baseline["edge_ratio"] > 0.01 and metrics["edge_ratio"] < baseline["edge_ratio"] * 0.55:
        reasons.append("edge_drop")
    if baseline["entropy"] > 2 and metrics["entropy"] < baseline["entropy"] * 0.65:
        reasons.append("low_entropy")
    return len(reasons) >= 2, reasons


def get_class_names(model) -> dict[int, str]:
    names = model.names
    return {int(k): str(v).lower() for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO camera occlusion inference")
    parser.add_argument("--source", default="0", help="camera index or video path")
    parser.add_argument("--weights", default="yolo11n.pt", help="YOLO weights, e.g. custom_occlusion.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=60, help="normal frames used for the quality baseline")
    parser.add_argument("--abnormal-confirm", type=int, default=5)
    parser.add_argument("--recovery-confirm", type=int, default=10)
    parser.add_argument(
        "--test-alarm",
        action="store_true",
        help="仅测试弹窗：显示5秒后自动关闭，不打开摄像头或加载模型",
    )
    args = parser.parse_args()

    if args.test_alarm:
        alarm_popup = CameraAlarmPopup()
        print("弹窗测试中：应显示‘摄像头正在被遮挡’，5秒后自动关闭。")
        deadline = time.monotonic() + 5.0
        try:
            while time.monotonic() < deadline:
                alarm_popup.set_alarm(True)
                time.sleep(0.05)
        finally:
            alarm_popup.close()
        return

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("缺少 ultralytics，请先执行: python -m pip install -r requirements-yolo.txt") from exc

    model = YOLO(args.weights)
    names = get_class_names(model)
    occlusion_keywords = ("occlusion", "occluded", "tamper", "covered", "blocked", "遮挡")
    occlusion_ids = {idx for idx, name in names.items() if any(word in name for word in occlusion_keywords)}
    if not occlusion_ids:
        print("提示：当前模型没有遮挡类别，将使用图像质量指标判断遮挡；训练自定义 YOLO 模型后可直接识别遮挡类别。")

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频源: {args.source}")

    baseline: dict[str, float] | None = None
    baseline_samples: list[dict[str, float]] = []
    abnormal_count = 0
    recovery_count = 0
    camera_status = 1
    previous_status = camera_status
    recent_status = deque(maxlen=5)
    frame_no = 0
    alarm_popup = CameraAlarmPopup()
    print("开始 YOLO 摄像头推理，按 q 退出。camera_status=1 正常，0 疑似遮挡/异常。")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("视频流读取结束或摄像头断开")
                break
            frame_no += 1
            metrics = image_metrics(frame)
            if frame_no <= args.warmup:
                baseline_samples.append(metrics)
            elif baseline is None and baseline_samples:
                baseline = {key: float(np.median([item[key] for item in baseline_samples])) for key in baseline_samples[0]}

            result = model.predict(source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            boxes = result.boxes
            detected_names: list[str] = []
            custom_occlusion = False
            if boxes is not None and len(boxes):
                for cls_id, conf in zip(boxes.cls.cpu().numpy().astype(int), boxes.conf.cpu().numpy()):
                    label = names.get(int(cls_id), str(cls_id))
                    detected_names.append(f"{label}:{conf:.2f}")
                    custom_occlusion |= int(cls_id) in occlusion_ids

            quality_bad, reasons = quality_candidate(metrics, baseline)
            anomaly_candidate = custom_occlusion or quality_bad
            recent_status.append(int(anomaly_candidate))
            if anomaly_candidate:
                abnormal_count += 1
                recovery_count = 0
            else:
                recovery_count += 1
                abnormal_count = max(0, abnormal_count - 1)
            if camera_status == 1 and abnormal_count >= args.abnormal_confirm:
                camera_status = 0
            elif camera_status == 0 and recovery_count >= args.recovery_confirm:
                camera_status = 1

            alarm_popup.set_alarm(camera_status == 0)
            if camera_status != previous_status:
                print(
                    "报警状态切换："
                    + ("摄像头正在被遮挡" if camera_status == 0 else "摄像头已恢复正常")
                )
                previous_status = camera_status

            color = (0, 0, 255) if camera_status == 0 else (0, 180, 0)
            cv2.putText(frame, f"camera_status: {camera_status}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"quality: {'abnormal' if quality_bad else 'normal'}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"reason: {','.join(reasons) or 'none'}", (15, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"YOLO: {', '.join(detected_names[:4]) or 'none'}", (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            annotated = result.plot()
            cv2.putText(annotated, f"camera_status: {camera_status}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("camera", frame)
            cv2.imshow("yolo_detections", annotated)
            print(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} | frame={frame_no} | "
                f"brightness={metrics['brightness']:.1f} | sharpness={metrics['sharpness']:.1f} | "
                f"quality_bad={int(quality_bad)} | camera_status={camera_status} | reasons={','.join(reasons) or '-'}"
            )
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        alarm_popup.close()


if __name__ == "__main__":
    main()
