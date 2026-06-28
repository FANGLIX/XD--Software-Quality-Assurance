r"""
真实 UCI HAR + PyTorch LSTM 对抗攻击实验

运行：
    python har_lstm_pytorch_real_experiment.py

可选参数：
    --data-root "E:\软件质量保障\data\UCI HAR Dataset\UCI HAR Dataset"
    --epochs 6
    --train-limit 2400
    --test-limit 900
    --eps 0.12

说明：
    读取 UCI HAR Dataset 的 Inertial Signals 原始 128 步、9 通道时序数据；
    训练 LSTM 分类器；
    评估 Clean / FGSM / PGD / Attention-FGSM / 平滑防御对照；
    输出 results_real.csv、attack_results.svg 和 lstm_uci_har.pt。
"""

import argparse
import csv
import html
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


SIGNAL_FILES = [
    "body_acc_x_{split}.txt",
    "body_acc_y_{split}.txt",
    "body_acc_z_{split}.txt",
    "body_gyro_x_{split}.txt",
    "body_gyro_y_{split}.txt",
    "body_gyro_z_{split}.txt",
    "total_acc_x_{split}.txt",
    "total_acc_y_{split}.txt",
    "total_acc_z_{split}.txt",
]


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, num_layers=1, num_classes=6, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_data_root(user_root):
    candidates = []
    if user_root:
        candidates.append(Path(user_root))
    script_dir = Path(__file__).resolve().parent
    candidates += [
        script_dir.parent / "data" / "UCI HAR Dataset",
        Path(r"E:\软件质量保障\data\UCI HAR Dataset\UCI HAR Dataset"),
        Path(r"E:\软件质量保障\data\UCI HAR Dataset"),
        Path.cwd() / "UCI HAR Dataset",
    ]
    for root in candidates:
        if (root / "train" / "Inertial Signals").exists() and (root / "test" / "Inertial Signals").exists():
            return root
    raise FileNotFoundError(
        "未找到 UCI HAR Dataset。请确认存在 train/test/Inertial Signals 目录，"
        "或通过 --data-root 指定数据集根目录。"
    )


def load_split(root, split):
    signal_dir = root / split / "Inertial Signals"
    channels = []
    for pattern in SIGNAL_FILES:
        path = signal_dir / pattern.format(split=split)
        if not path.exists():
            raise FileNotFoundError(path)
        channels.append(np.loadtxt(path, dtype=np.float32))
    x = np.stack(channels, axis=-1)  # [N, 128, 9]
    y = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64) - 1
    return x, y


def limit_data(x, y, limit, seed):
    if not limit or limit >= len(y):
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=limit, replace=False)
    return x[idx], y[idx]


def standardize(train_x, test_x):
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std = train_x.std(axis=(0, 1), keepdims=True) + 1e-6
    return (train_x - mean) / std, (test_x - mean) / std, mean, std


def make_loader(x, y, batch_size, shuffle):
    tx = torch.tensor(x, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=shuffle)


def train_model(model, train_loader, device, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        print(f"epoch={epoch:02d} loss={total_loss / total:.4f} train_acc={correct / total * 100:.2f}%")


@torch.no_grad()
def eval_accuracy(model, loader, device, transform=None):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if transform is not None:
            x = transform(x)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return correct / total


def fgsm_attack(model, x, y, eps):
    x_adv = x.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    return (x_adv + eps * grad.sign()).detach()


def pgd_attack(model, x, y, eps, steps=7, alpha=None):
    alpha = alpha if alpha is not None else eps / 3
    base = x.detach()
    x_adv = base + torch.empty_like(base).uniform_(-eps, eps)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - base, min=-eps, max=eps)
        x_adv = (base + delta).detach()
    return x_adv


def attention_fgsm_attack(model, x, y, eps, keep_ratio=0.35):
    x_adv = x.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    saliency = grad.abs().mean(dim=2)  # [B, T]
    k = max(1, int(saliency.size(1) * keep_ratio))
    topk_idx = saliency.topk(k, dim=1).indices
    mask = torch.zeros_like(saliency)
    mask.scatter_(1, topk_idx, 1.0)
    mask = mask.unsqueeze(2)
    return (x_adv + eps * grad.sign() * mask).detach()


def temporal_smooth(x, kernel_size=5):
    if kernel_size <= 1:
        return x
    pad = kernel_size // 2
    # [B, T, C] -> [B, C, T]
    y = x.transpose(1, 2)
    y = F.avg_pool1d(F.pad(y, (pad, pad), mode="replicate"), kernel_size=kernel_size, stride=1)
    return y.transpose(1, 2)


def eval_attack(model, loader, device, attack_fn, eps, smooth=False):
    model.eval()
    correct = 0
    total = 0
    changed = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            clean_pred = model(x).argmax(1)
        x_adv = attack_fn(model, x, y, eps)
        if smooth:
            x_adv = temporal_smooth(x_adv)
        with torch.no_grad():
            adv_pred = model(x_adv).argmax(1)
        correct += (adv_pred == y).sum().item()
        changed += (adv_pred != clean_pred).sum().item()
        total += len(y)
    return correct / total, changed / total


def save_results(rows, out_path):
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def esc(text):
    return html.escape(str(text), quote=False)


BAR_COLORS = {
    "Clean": "#2563EB",
    "FGSM": "#DC2626",
    "PGD": "#B91C1C",
    "Attention-FGSM": "#EA580C",
    "FGSM + Smooth Defense": "#059669",
    "PGD + Smooth Defense": "#047857",
}

DISPLAY_LABELS = {
    "FGSM + Smooth Defense": "FGSM+Smooth Defense",
    "PGD + Smooth Defense": "PGD+Smooth Defense",
}


def save_results_svg(rows, out_path, caption):
    left, right, top, bottom = 120, 880, 70, 420
    chart_h = bottom - top
    bar_w = 72
    n = len(rows)
    margin = 72
    span = right - left - 2 * margin
    if n <= 1:
        centers = [(left + right) / 2]
    else:
        centers = [left + margin + i * span / (n - 1) for i in range(n)]

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="920" height="500" viewBox="0 0 920 500">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="40" y="35" font-size="22" font-family="Arial, Microsoft YaHei" font-weight="700">'
        f'{esc("真实 UCI HAR + PyTorch LSTM 对抗实验结果")}</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#374151"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#374151"/>',
    ]
    for pct in range(0, 101, 20):
        y = bottom - pct / 100 * chart_h
        parts.append(f'<line x1="115" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#E5E7EB"/>')
        parts.append(
            f'<text x="75" y="{y + 5:.1f}" font-size="13" font-family="Arial">{esc(f"{pct}%")}</text>'
        )

    for row, cx in zip(rows, centers):
        acc = float(row["accuracy"])
        color = BAR_COLORS.get(row["scenario"], "#6B7280")
        label = DISPLAY_LABELS.get(row["scenario"], row["scenario"])
        bar_h = acc / 100 * chart_h
        x = cx - bar_w / 2
        y = bottom - bar_h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{color}"/>')
        parts.append(
            f'<text x="{cx:.1f}" y="{y - 8:.1f}" font-size="13" text-anchor="middle" '
            f'font-family="Arial">{esc(f"{acc:.2f}%")}</text>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="442" font-size="12" text-anchor="middle" '
            f'font-family="Arial, Microsoft YaHei">{esc(label)}</text>'
        )

    parts.append(
        f'<text x="40" y="470" font-size="14" font-family="Arial, Microsoft YaHei">{esc(caption)}</text>'
    )
    parts.append("</svg>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="", help="UCI HAR Dataset 根目录")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=0.12)
    parser.add_argument("--train-limit", type=int, default=0, help="为课堂复现实验限制训练样本数；设为0使用全部训练集")
    parser.add_argument("--test-limit", type=int, default=900, help="为课堂复现实验限制测试样本数；设为0使用全部测试集")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="", help="结果输出目录；为空时输出到脚本所在目录")
    args = parser.parse_args()

    set_seed(args.seed)
    root = find_data_root(args.data_root)
    print("data_root =", root)

    train_x, train_y = load_split(root, "train")
    test_x, test_y = load_split(root, "test")
    train_x, train_y = limit_data(train_x, train_y, args.train_limit, args.seed)
    test_x, test_y = limit_data(test_x, test_y, args.test_limit, args.seed + 1)
    train_x, test_x, _, _ = standardize(train_x, test_x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)
    print("train shape =", train_x.shape, "test shape =", test_x.shape)

    train_loader = make_loader(train_x, train_y, args.batch_size, shuffle=True)
    test_loader = make_loader(test_x, test_y, args.batch_size, shuffle=False)

    model = LSTMClassifier(hidden_dim=args.hidden_dim).to(device)
    train_model(model, train_loader, device, args.epochs, args.lr)

    clean_acc = eval_accuracy(model, test_loader, device)
    fgsm_acc, fgsm_changed = eval_attack(model, test_loader, device, fgsm_attack, args.eps)
    pgd_acc, pgd_changed = eval_attack(model, test_loader, device, pgd_attack, args.eps)
    att_acc, att_changed = eval_attack(model, test_loader, device, attention_fgsm_attack, args.eps)
    fgsm_s_acc, fgsm_s_changed = eval_attack(model, test_loader, device, fgsm_attack, args.eps, smooth=True)
    pgd_s_acc, pgd_s_changed = eval_attack(model, test_loader, device, pgd_attack, args.eps, smooth=True)

    rows = [
        ("Clean", 0.0, "none", clean_acc, 0.0, 0.0, "真实 UCI HAR Inertial Signals + LSTM"),
        ("FGSM", args.eps, "none", fgsm_acc, 1 - fgsm_acc, fgsm_changed, "单步梯度符号攻击"),
        ("PGD", args.eps, "none", pgd_acc, 1 - pgd_acc, pgd_changed, "多步迭代攻击"),
        ("Attention-FGSM", args.eps, "none", att_acc, 1 - att_acc, att_changed, "只扰动高敏感时间步"),
        ("FGSM + Smooth Defense", args.eps, "temporal_smoothing", fgsm_s_acc, 1 - fgsm_s_acc, fgsm_s_changed, "时序平滑防御对照"),
        ("PGD + Smooth Defense", args.eps, "temporal_smoothing", pgd_s_acc, 1 - pgd_s_acc, pgd_s_changed, "强攻击下防御对照"),
    ]
    dict_rows = [
        {
            "scenario": name,
            "epsilon": f"{eps:.4f}",
            "defense": defense,
            "accuracy": f"{acc * 100:.2f}",
            "attack_success_rate": f"{asr * 100:.2f}",
            "prediction_change_rate": f"{changed * 100:.2f}",
            "note": note,
        }
        for name, eps, defense, acc, asr, changed, note in rows
    ]

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.output_dir).resolve() if args.output_dir else script_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_n = len(train_y)
    test_n = len(test_y)
    device_label = "CUDA" if torch.cuda.is_available() else "CPU"
    train_desc = f"{train_n} 条" if args.train_limit else "全部"
    test_desc = f"{test_n} 条" if args.test_limit else "全部"
    svg_caption = (
        f"指标：测试准确率；训练：{train_desc}真实样本，测试：{test_desc}真实样本，"
        f"{args.epochs} epochs，eps={args.eps}，{device_label}。"
    )

    save_results(dict_rows, out_dir / "results_real.csv")
    svg_paths = [out_dir / "attack_results.svg"]
    figures_dir = script_dir.parent / "figures"
    if figures_dir != out_dir:
        svg_paths.append(figures_dir / "attack_results.svg")
    for svg_path in svg_paths:
        save_results_svg(dict_rows, svg_path, svg_caption)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "clean_accuracy": clean_acc,
    }
    with (out_dir / "lstm_uci_har.pt").open("wb") as f:
        torch.save(checkpoint, f)

    print("\nresults:")
    for row in dict_rows:
        print(row)
    print("\nsaved:", out_dir / "results_real.csv")
    for svg_path in svg_paths:
        print("saved:", svg_path)
    print("saved:", out_dir / "lstm_uci_har.pt")


if __name__ == "__main__":
    main()
