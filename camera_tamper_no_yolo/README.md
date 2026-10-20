# 无 YOLO、无边缘检测的摄像头遮挡检测

这个版本只使用 OpenCV 图像质量指标和时序状态机：

- 不需要 YOLO、训练数据或 GPU；
- 不使用 Sobel/Canny 等边缘检测；
- 使用熵、灰度方差、局部纹理方差、Laplacian 方差、黑白比例；
- 大幅场景变化但画面仍清晰时不会报警；
- 遮挡超过 5 秒才弹窗，显示“摄像头被遮挡”；
- 画面恢复正常 1.5 秒后自动关闭弹窗；
- 启动后自动建立初始基准，并缓慢适应正常光照变化。

## 运行

```powershell
cd C:\Users\AAA\Documents\ChatGPT\ai边缘算法盒子\camera_tamper_no_yolo
python -m pip install -r requirements.txt
python camera_tamper_no_yolo.py --source 0
```

按 `Q` 退出。也可以读取视频文件：

```powershell
python camera_tamper_no_yolo.py --source C:\path\to\video.mp4
```

参数：

```text
--confirm-seconds 5   遮挡确认时间
--clear-seconds 1.5   恢复正常后关闭报警的时间
```

首次运行请让摄像头正常拍摄约 20～30 秒，让基准稳定下来，再测试遮挡。
