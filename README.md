# Camera Occlusion and Tamper Detection

基于 OpenCV 的实时摄像头遮挡与篡改检测项目（Camera Occlusion and Tamper Detection）。项目当前适配 Windows + VSCode，可直接使用电脑自带摄像头进行调试。

## 项目简介

程序通过比较当前画面与摄像头正常工作时的参考画面，判断摄像头是否出现黑屏、白屏、模糊、遮挡或画面突变。检测结果会在 Notebook 输出中持续打印，便于后续接入 AI 边缘算法盒。

## 当前功能

- 使用 `cv2.VideoCapture(0)` 读取电脑自带摄像头，也可以替换为视频文件路径。
- 使用 MOG2 背景建模，并结合 Sobel 边缘消失率（EDR）和自适应阈值。
- 计算亮度、灰度方差、拉普拉斯清晰度、边缘比例和图像熵等图像质量指标。
- 同时维护固定参考画面和缓慢更新的自适应参考画面，兼顾缓慢遮挡与光照变化。
- 使用连续帧确认和恢复计数，减少人移动、瞬时黑帧造成的误报。
- 输出 `camera_status`：`1` 表示摄像头正常，`0` 表示疑似遮挡或异常。
- 当 `camera_status=0` 时显示置顶弹窗“摄像头正在被遮挡”；连续恢复到 `camera_status=1` 后弹窗自动隐藏。

## 运行环境

- Windows 10/11
- Python 3.10 或更高版本
- VSCode、Jupyter Notebook

安装依赖：

```powershell
python -m pip install opencv-python numpy matplotlib jupyter
```

## 运行步骤

1. 用 VSCode 打开本仓库目录。
2. 打开 `camera_tamper_attack.ipynb`，选择 Python 3.10+ 内核。
3. 按顺序运行代码单元，不要跳过配置、函数定义和主循环单元。
4. 确认配置中为 `cap = cv2.VideoCapture(0)`，然后运行主循环。
5. 关闭检测时，在 OpenCV 画面窗口中按 `q`，或在 VSCode 中中断内核。

## 输出窗口和日志

程序会显示原始画面、背景、前景掩码、背景边缘和当前画面边缘。日志中的关键字段如下：

- `camera status: 1`：当前判断正常。
- `camera status: 0`：当前判断疑似遮挡/异常。
- `edr`、`aedr`：边缘消失率及其平均值。
- `quality flag`：图像质量指标是否异常。
- `tamper validation count`：连续异常确认计数。

## 当前限制

- 如果遮挡物与背景颜色、纹理几乎完全相同，仅靠图像分析可能无法可靠区分。
- 不同摄像头、光照和安装角度可能需要重新调整阈值。
- 摄像头断流属于视频流故障，建议在算法盒中由独立的视频流模块处理。
- 首次启动需要一段正常画面用于建立参考，不宜一开始就遮挡镜头。

## 论文与原始项目

本项目的基础算法参考论文：[Unified Camera Tamper Detection Based on Edge and Object Information](https://doi.org/10.3390/s150510315)。

算法说明、基准结果和后续改进记录见仓库中的 [Report.pdf](Report.pdf)。

## YOLO 推理流程

项目新增 `yolo_inference.py`，用于验证“模型加载 → 摄像头读取 → YOLO 推理 → 遮挡判断 → 状态输出”的完整流程。原有 `camera_tamper_attack.ipynb` 的 OpenCV 规则算法仍然保留，便于对比。

安装 YOLO 依赖：

```powershell
python -m pip install -r requirements-yolo.txt
```

运行官方预训练模型：

```powershell
python yolo_inference.py --source 0 --weights yolo11n.pt
```

说明：官方 COCO 模型不包含“摄像头遮挡”类别，因此它主要用于验证 YOLO 推理链路，并通过亮度、清晰度、边缘比例和图像熵辅助判断画面异常。要让 YOLO 直接识别“遮挡”，需要准备遮挡数据集并训练自定义权重，例如：

```powershell
python yolo_inference.py --source 0 --weights runs/detect/train/weights/best.pt
```

`camera_status=1` 表示正常，`camera_status=0` 表示疑似遮挡或异常。

检测循环还会同步显示 Windows 报警弹窗：遮挡确认后弹出“摄像头正在被遮挡”，恢复确认后自动消失。弹窗由 `alarm_popup.py` 提供，不会阻塞 OpenCV 视频循环。

如果要先单独确认弹窗本身可用，可在项目目录执行：

```powershell
python yolo_inference.py --test-alarm
```

该命令会显示报警弹窗 5 秒并自动关闭；正式检测时，只有连续达到 `--abnormal-confirm`（默认 5）帧异常后才会触发报警。
