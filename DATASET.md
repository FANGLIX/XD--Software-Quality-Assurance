# 数据集说明

本项目使用 UCI HAR Dataset（Human Activity Recognition Using Smartphones）。

由于数据集属于公开外部数据，且体积较大，仓库中不直接上传 `data/` 目录。复现实验时请自行下载并解压数据集。

## 推荐目录结构

脚本默认会自动查找以下路径：

```text
E:\软件质量保障\data\UCI HAR Dataset\UCI HAR Dataset
```

如果你的数据放在其他位置，可以运行脚本时通过 `--data-root` 指定：

```powershell
python code\har_lstm_pytorch_real_experiment.py --data-root "你的\UCI HAR Dataset\路径"
```

## 必要文件

至少需要以下目录存在：

```text
UCI HAR Dataset/
  train/
    Inertial Signals/
    y_train.txt
  test/
    Inertial Signals/
    y_test.txt
```

脚本读取的是 `Inertial Signals` 下的 9 个原始时序信号文件，每个样本被整理成 `128 × 9` 的 LSTM 输入。
