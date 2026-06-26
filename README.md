# 软件质量保障项目完成交付

## 主题

基于 LSTM 的时序感知模型对抗攻击与鲁棒性分析。

## 上交文件

- `软件质量保障项目报告.docx`：上交用课程报告。
- `软件质量保障完成汇报.pptx`：课堂汇报 PPT。
- `实验结果.csv`：真实实验结果汇总。
- `figures/attack_results.svg`：实验结果图。
- `code/har_lstm_pytorch_real_experiment.py`：真实 UCI HAR + PyTorch LSTM 实验脚本。
- `code/lstm_uci_har.pt`：本次训练得到的模型文件。
- `code/results_real.csv`：脚本输出的真实实验结果。
- `答辩问答.md`：答辩前快速复习材料。

## 复现命令

```powershell
C:\Users\Administrator\Documents\Codex\2026-06-18\654111813-qq\.venv_real\Scripts\python.exe `
  E:\软件质量保障\项目完成交付\code\har_lstm_pytorch_real_experiment.py `
  --epochs 6 --train-limit 2400 --test-limit 900 --batch-size 128
```

## 结果摘要

| 场景 | epsilon | 防御方式 | 准确率(%) | 攻击成功率(%) | 预测变化率(%) |
|---|---:|---|---:|---:|---:|
| Clean | 0.0000 | none | 80.11 | 0.00 | 0.00 |
| FGSM | 0.1200 | none | 60.67 | 39.33 | 21.00 |
| PGD | 0.1200 | none | 59.78 | 40.22 | 22.00 |
| Attention-FGSM | 0.1200 | none | 61.78 | 38.22 | 19.78 |
| FGSM + Smooth Defense | 0.1200 | temporal_smoothing | 59.11 | 40.89 | 22.56 |
| PGD + Smooth Defense | 0.1200 | temporal_smoothing | 57.89 | 42.11 | 23.89 |
