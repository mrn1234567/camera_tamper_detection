"""Run the trained normal/occluded classifier with a 5-second alarm."""

from __future__ import annotations

import argparse
import time

import cv2

from alarm_popup import CameraAlarmPopup


def main() -> None:
    parser = argparse.ArgumentParser(description="Camera occlusion classifier inference")
    parser.add_argument("--weights", required=True, help="trained classifier best.pt")
    parser.add_argument("--source", default="0")
    parser.add_argument("--conf", type=float, default=0.75)
    parser.add_argument("--alarm-seconds", type=float, default=5.0)
    parser.add_argument("--recovery-seconds", type=float, default=1.0)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("缺少 ultralytics，请先执行: python -m pip install -r requirements-yolo.txt") from exc

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频源: {args.source}")
    model = YOLO(args.weights)
    popup = CameraAlarmPopup()
    alarm_seconds = 0.0
    recovery_seconds = 0.0
    last_time = time.monotonic()
    status = 1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.monotonic()
            elapsed = min(max(now - last_time, 0.0), 1.0)
            last_time = now
            result = model.predict(frame, imgsz=224, verbose=False)[0]
            top1 = int(result.probs.top1)
            confidence = float(result.probs.top1conf)
            label = str(result.names[top1]).lower()
            occluded = label == "occluded" and confidence >= args.conf

            if occluded:
                alarm_seconds += elapsed
                recovery_seconds = 0.0
            else:
                recovery_seconds += elapsed
                alarm_seconds = 0.0
            if status == 1 and alarm_seconds >= args.alarm_seconds:
                status = 0
            elif status == 0 and recovery_seconds >= args.recovery_seconds:
                status = 1

            popup.set_alarm(status == 0)
            color = (0, 0, 255) if status == 0 else (0, 180, 0)
            cv2.putText(frame, f"{label}:{confidence:.2f}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"camera_status: {status} alarm={alarm_seconds:.1f}s", (12, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.imshow("classifier", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        popup.close()


if __name__ == "__main__":
    main()
