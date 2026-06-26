# 软件质量保障项目完成交付

## 项目主题

基于 LSTM 的时序感知模型对抗攻击与鲁棒性分析。

本项目使用真实 UCI HAR Dataset 的 `Inertial Signals` 原始时序数据，训练 PyTorch LSTM 分类模型，并评估 Clean、FGSM、PGD、Attention-FGSM 和时序平滑防御对照结果。

## 重要说明

- `.venv/` 或 `.venv_real/` 不上传 GitHub。虚拟环境和依赖应在本地重新安装。
- `data/` 不上传 GitHub。UCI HAR 是外部公开数据集，复现时按 `DATASET.md` 下载并放置。
- `video_demo_output/` 是录屏演示临时输出，不提交仓库。

## 环境安装

建议使用 Python 3.9。

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

如果安装 CPU 版 PyTorch 较慢，也可以使用国内镜像或 PyTorch 官网命令安装。

## 数据集准备

请参考：

```text
DATASET.md
```

默认数据路径：

```text
E:\软件质量保障\data\UCI HAR Dataset\UCI HAR Dataset
```

也可以运行时使用 `--data-root` 指定路径。

## 复现实验

正式实验命令：

```powershell
python code\har_lstm_pytorch_real_experiment.py --epochs 6 --train-limit 2400 --test-limit 900 --batch-size 128
```

如果数据不在默认路径：

```powershell
python code\har_lstm_pytorch_real_experiment.py --data-root "你的\UCI HAR Dataset\路径" --epochs 6 --train-limit 2400 --test-limit 900 --batch-size 128
```

录屏快速演示：

```powershell
powershell -ExecutionPolicy Bypass -File .\录屏演示_一键运行.ps1
```

## 交付文件

- `项目报告/软件质量保障项目报告.docx`：课程项目报告。
- `结题答辩ppt/软件质量保障完成汇报.pptx`：结题答辩 PPT。
- `video demo/小规模演示视频.mp4`：项目操作演示视频。
- `实验结果.csv`：正式实验结果汇总。
- `figures/attack_results.svg`：实验结果图。
- `code/har_lstm_pytorch_real_experiment.py`：真实 UCI HAR + PyTorch LSTM 实验脚本。
- `code/lstm_uci_har.pt`：本次训练得到的模型 checkpoint。
- `code/results_real.csv`：脚本导出的真实实验结果。

## 正式结果摘要

| 场景 | epsilon | 防御方式 | 准确率(%) | 攻击成功率(%) | 预测变化率(%) |
|---|---:|---|---:|---:|---:|
| Clean | 0.0000 | none | 80.11 | 0.00 | 0.00 |
| FGSM | 0.1200 | none | 60.67 | 39.33 | 21.00 |
| PGD | 0.1200 | none | 59.78 | 40.22 | 22.00 |
| Attention-FGSM | 0.1200 | none | 61.78 | 38.22 | 19.78 |
| FGSM + Smooth Defense | 0.1200 | temporal_smoothing | 59.11 | 40.89 | 22.56 |
| PGD + Smooth Defense | 0.1200 | temporal_smoothing | 57.89 | 42.11 | 23.89 |
