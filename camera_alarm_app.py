"""实时摄像头遮挡报警界面。

运行：python camera_alarm_app.py
按 Q 或关闭窗口退出。
"""
from pathlib import Path
import time
import threading
from collections import deque
import tkinter as tk
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "runs" / "detect" / "runs" / "screen_status_synth" / "weights" / "best.pt"
CAMERA_ID = 0
CONFIDENCE = 0.70
OCCLUDED_MIN_CONF = 0.82
OCCLUDED_MARGIN = 0.15
# 摄像头通常约 30 FPS。遮挡必须在画面稳定后持续约 0.5 秒，
# 才确认报警，避免人/物体经过时的瞬时误报。
ALARM_ON_FRAMES = 15
ALARM_OFF_FRAMES = 20
# 固定屏幕区域：按实际摄像头画面调整。区域外的人和物体不参与判断。
ROI = (0.07, 0.10, 0.93, 0.85)  # left, top, right, bottom
VOTE_WINDOW = 30
ALARM_ON_VOTES = 22
ALARM_OFF_VOTES = 24


class CameraAlarmApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("摄像头状态监测")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.running = True
        self.cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
        self.model = YOLO(str(MODEL_PATH))
        self.popup = None
        self.state_votes = deque(maxlen=VOTE_WINDOW)
        self.last_frame = None

        self.video = tk.Label(root, bg="black")
        self.video.pack(padx=10, pady=10)
        self.status = tk.Label(root, text="正在启动摄像头…", font=("Microsoft YaHei", 12))
        self.status.pack(pady=(0, 10))
        self.update_frame()

    def update_frame(self):
        if not self.running:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.status.config(text="无法读取摄像头，请检查摄像头连接")
            self.root.after(300, self.update_frame)
            return

        h, w = frame.shape[:2]
        left, top, right, bottom = ROI
        x1, y1 = int(w * left), int(h * top)
        x2, y2 = int(w * right), int(h * bottom)
        roi = frame[y1:y2, x1:x2]
        results = self.model.predict(roi, conf=CONFIDENCE, verbose=False, imgsz=640)
        annotated_roi = results[0].plot()
        annotated = frame.copy()
        annotated[y1:y2, x1:x2] = annotated_roi
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 200, 0), 2)
        class_conf = {}
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            name = results[0].names.get(cls_id, str(cls_id))
            conf = float(box.conf[0])
            class_conf[name] = max(class_conf.get(name, 0.0), conf)

        # 只有遮挡是“明确的最高置信类别”才计入投票；
        # 物体经过后出现的低置信/模糊预测会被当作不确定，不报警。
        occluded_conf = class_conf.get("occluded", 0.0)
        normal_conf = class_conf.get("normal", 0.0)
        best_name = max(class_conf, key=class_conf.get, default="unknown")
        occluded = (
            best_name == "occluded"
            and occluded_conf >= OCCLUDED_MIN_CONF
            and occluded_conf - normal_conf >= OCCLUDED_MARGIN
        )

        # 不依赖边缘/清晰度规则，只用分类结果做时间窗口投票。
        self.state_votes.append(occluded)
        occluded_votes = sum(self.state_votes)
        normal_votes = len(self.state_votes) - occluded_votes
        if occluded_votes >= ALARM_ON_VOTES:
            self.status.config(text=f"确认遮挡（{occluded_votes}/{len(self.state_votes)}）", fg="#d93025")
            self.show_alarm()
        elif normal_votes >= ALARM_OFF_VOTES:
            self.status.config(text="摄像头正常", fg="#188038")
            self.hide_alarm()
        else:
            self.status.config(text="正在确认画面状态…", fg="#b06000")

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((900, 650))
        self.last_frame = ImageTk.PhotoImage(image=image)
        self.video.config(image=self.last_frame)
        self.root.after(30, self.update_frame)

    def show_alarm(self):
        if self.popup is not None and self.popup.winfo_exists():
            return
        self.popup = tk.Toplevel(self.root)
        self.popup.title("摄像头报警")
        self.popup.configure(bg="#fff1f0")
        self.popup.attributes("-topmost", True)
        self.popup.resizable(False, False)
        tk.Label(self.popup, text="⚠", font=("Segoe UI Symbol", 42),
                 fg="#d93025", bg="#fff1f0").pack(pady=(12, 0))
        tk.Label(self.popup, text="摄像头正在被遮挡", font=("Microsoft YaHei", 20, "bold"),
                 fg="#b42318", bg="#fff1f0").pack(padx=35, pady=(0, 18))
        self.popup.geometry("360x170")
        self.popup.update_idletasks()
        x = self.root.winfo_x() + self.root.winfo_width() // 2 - 180
        y = self.root.winfo_y() + 80
        self.popup.geometry(f"360x170+{max(x, 0)}+{max(y, 0)}")

    def hide_alarm(self):
        if self.popup is not None and self.popup.winfo_exists():
            self.popup.destroy()
        self.popup = None

    def close(self):
        self.running = False
        self.hide_alarm()
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型文件：{MODEL_PATH}")
    root = tk.Tk()
    CameraAlarmApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
