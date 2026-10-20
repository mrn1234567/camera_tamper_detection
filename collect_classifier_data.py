"""Collect labeled frames for the normal/occluded image classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect camera classifier images")
    parser.add_argument("--label", choices=("normal", "occluded"), required=True)
    parser.add_argument("--output", default="datasets/occlusion_classifier")
    parser.add_argument("--source", default="0")
    parser.add_argument("--interval", type=int, default=5, help="save one frame every N loop iterations")
    parser.add_argument("--max-images", type=int, default=0, help="0 means unlimited")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    output = Path(args.output) / args.label
    output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频源: {args.source}")

    saved = 0
    tick = 0
    print("按 S 保存当前帧，按 Q 退出。当前类别：", args.label)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            tick += 1
            cv2.putText(frame, f"label={args.label} saved={saved} | S save, Q quit", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.imshow("collect", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s") and tick % max(args.interval, 1) == 0:
                path = output / f"{args.label}_{saved:06d}.jpg"
                if cv2.imwrite(str(path), frame):
                    saved += 1
                    print("saved", path)
            elif key == ord("q") or (args.max_images and saved >= args.max_images):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"完成：{saved} 张 {args.label} 图片 -> {output}")


if __name__ == "__main__":
    main()
