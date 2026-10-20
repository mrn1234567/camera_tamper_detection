"""Practical camera-cover detector without YOLO or edge detection.

It uses image-quality statistics and a time-based state machine.  A cover is
confirmed only after --confirm-seconds of persistent quality degradation.
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np


class AlarmPopup:
    def __init__(self) -> None:
        self._commands: queue.Queue[bool] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_visible(self, visible: bool) -> None:
        self._commands.put(visible)

    def close(self) -> None:
        self._commands.put(False)
        self._commands.put(None)  # type: ignore[arg-type]

    def _run(self) -> None:
        root = tk.Tk()
        root.withdraw()
        window: tk.Toplevel | None = None

        def poll() -> None:
            nonlocal window
            try:
                while True:
                    command = self._commands.get_nowait()
                    if command is None:
                        if window is not None:
                            window.destroy()
                        root.destroy()
                        return
                    if command and window is None:
                        window = tk.Toplevel(root)
                        window.title("摄像头报警")
                        window.geometry("650x180")
                        window.configure(bg="#a40000")
                        window.resizable(False, False)
                        window.protocol("WM_DELETE_WINDOW", lambda: None)
                        tk.Label(window, text="摄像头被遮挡", fg="white", bg="#a40000",
                                 font=("Microsoft YaHei", 28, "bold")).pack(expand=True, fill="both")
                    elif not command and window is not None:
                        window.destroy()
                        window = None
            except queue.Empty:
                pass
            root.after(100, poll)

        root.after(100, poll)
        root.mainloop()


def features(frame: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([small], [0], None, [64], [0, 256]).ravel()
    hist /= max(float(hist.sum()), 1.0)
    entropy = float(-(hist[hist > 1e-8] * np.log2(hist[hist > 1e-8])).sum())
    # Local variance is texture information; it does not rely on edge maps.
    blocks = [small[y:y + 24, x:x + 32] for y in range(0, 120, 24) for x in range(0, 160, 32)]
    local_var = float(np.median([np.var(b) for b in blocks]))
    return {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "entropy": entropy,
        "lap": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "local_var": local_var,
        "dark": float((gray < 18).mean()),
        "bright": float((gray > 238).mean()),
    }


class QualityTamperDetector:
    def __init__(self, confirm_seconds: float = 5.0, clear_seconds: float = 1.5) -> None:
        self.confirm_seconds = confirm_seconds
        self.clear_seconds = clear_seconds
        self.baseline: dict[str, float] | None = None
        self.prev_gray: np.ndarray | None = None
        self.candidate_since: float | None = None
        self.normal_since: float | None = None
        self.alarm = False

    def _degraded(self, current: dict[str, float]) -> tuple[bool, int]:
        if self.baseline is None:
            return False, 0
        b = self.baseline
        ratios = {
            "entropy": current["entropy"] / max(b["entropy"], 0.1),
            "lap": current["lap"] / max(b["lap"], 1.0),
            "local_var": current["local_var"] / max(b["local_var"], 1.0),
            "std": current["std"] / max(b["std"], 1.0),
        }
        votes = sum((ratios["entropy"] < 0.72, ratios["lap"] < 0.48,
                     ratios["local_var"] < 0.52, ratios["std"] < 0.48))
        extreme = current["dark"] > 0.88 or current["bright"] > 0.88
        return (votes >= 2 or (extreme and votes >= 1)), votes

    def update(self, frame: np.ndarray, now: float | None = None) -> tuple[bool, dict[str, float], str]:
        now = time.monotonic() if now is None else now
        current = features(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = 0.0 if self.prev_gray is None else float(cv2.absdiff(gray, self.prev_gray).mean() / 255.0)
        self.prev_gray = gray

        if self.baseline is None:
            self.baseline = current.copy()
            return False, current, "calibrating"

        degraded, votes = self._degraded(current)
        # A large scene change with healthy texture is normal activity, not a cover.
        if motion > 0.18 and votes < 2:
            degraded = False

        if degraded:
            self.normal_since = None
            if self.candidate_since is None:
                self.candidate_since = now
            if now - self.candidate_since > self.confirm_seconds:
                self.alarm = True
            state = f"candidate {max(0.0, now - self.candidate_since):.1f}s ({votes}/4)"
        else:
            self.candidate_since = None
            if self.alarm:
                if self.normal_since is None:
                    self.normal_since = now
                if now - self.normal_since >= self.clear_seconds:
                    self.alarm = False
            else:
                self.normal_since = None
            # Slowly follow legitimate illumination/scene changes only when healthy.
            if motion < 0.12:
                for key in self.baseline:
                    self.baseline[key] = 0.995 * self.baseline[key] + 0.005 * current[key]
            state = "ALARM" if self.alarm else "normal"
        return self.alarm, current, state


def open_source(source: str) -> cv2.VideoCapture:
    value: int | str = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(value)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频源：{source}")
    return cap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="摄像头编号或视频路径")
    parser.add_argument("--confirm-seconds", type=float, default=5.0)
    parser.add_argument("--clear-seconds", type=float, default=1.5)
    args = parser.parse_args()
    cap = open_source(args.source)
    detector = QualityTamperDetector(args.confirm_seconds, args.clear_seconds)
    popup = AlarmPopup()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            alarm, metric, state = detector.update(frame)
            popup.set_visible(alarm)
            text = f"{state} | entropy={metric['entropy']:.2f} local={metric['local_var']:.1f}"
            cv2.putText(frame, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (0, 0, 255) if alarm else (0, 220, 0), 2)
            cv2.imshow("camera_tamper_no_yolo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        popup.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
