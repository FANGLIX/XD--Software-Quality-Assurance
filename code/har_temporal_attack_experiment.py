import csv
import html
import math
import random
import zipfile
from pathlib import Path


ROOT = Path(r"E:\软件质量保障")
DELIVERABLE = ROOT / "项目完成交付"
FIG_DIR = DELIVERABLE / "figures"
CODE_DIR = DELIVERABLE / "code"


def esc(text):
    return html.escape(str(text), quote=False)


def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CODE_DIR.mkdir(parents=True, exist_ok=True)


def make_prototypes(classes=6, steps=128, channels=9):
    prototypes = []
    for cls in range(classes):
        sample = []
        base_freq = 1 + cls * 0.35
        for t in range(steps):
            row = []
            x = t / steps
            for ch in range(channels):
                phase = cls * 0.55 + ch * 0.31
                amp = 0.45 + 0.04 * ((cls + ch) % 4)
                v = amp * math.sin(2 * math.pi * base_freq * x + phase)
                v += 0.22 * math.cos(2 * math.pi * (base_freq + 0.5) * x + ch * 0.17)
                if (cls + ch) % 3 == 0 and 0.18 + 0.06 * cls < x < 0.33 + 0.04 * cls:
                    v += 0.35
                if (cls + ch) % 4 == 1 and 0.55 < x < 0.72:
                    v -= 0.28
                row.append(v)
            sample.append(row)
        prototypes.append(sample)
    return prototypes


def clone_sample(sample):
    return [row[:] for row in sample]


def smooth(x, radius=2):
    out = []
    for t in range(len(x)):
        lo = max(0, t - radius)
        hi = min(len(x), t + radius + 1)
        row = []
        for ch in range(len(x[0])):
            vals = sorted(x[i][ch] for i in range(lo, hi))
            median = vals[len(vals) // 2]
            mean = sum(vals) / len(vals)
            row.append(0.7 * median + 0.3 * mean)
        out.append(row)
    return out


def make_dataset(prototypes, per_class=45, noise=0.18, seed=7):
    rng = random.Random(seed)
    data = []
    for y, proto in enumerate(prototypes):
        for _ in range(per_class):
            sample = []
            scale = rng.uniform(0.88, 1.12)
            bias = [rng.gauss(0, 0.035) for _ in proto[0]]
            for row in proto:
                sample.append([scale * v + bias[ch] + rng.gauss(0, noise) for ch, v in enumerate(row)])
            data.append((sample, y))
    rng.shuffle(data)
    return data


_ATTENTION_CACHE = {}


def attention_weights(proto):
    key = id(proto)
    if key in _ATTENTION_CACHE:
        return _ATTENTION_CACHE[key]
    weights = []
    for row in proto:
        weights.append(0.2 + sum(v * v for v in row) / len(row))
    total = sum(weights)
    result = [w / total for w in weights]
    _ATTENTION_CACHE[key] = result
    return result


def logits(x, prototypes, use_smoothing=False):
    z = smooth(x) if use_smoothing else x
    values = []
    norm = len(z) * len(z[0])
    for proto in prototypes:
        weights = attention_weights(proto)
        dist = 0.0
        for t, row in enumerate(z):
            for ch, value in enumerate(row):
                d = value - proto[t][ch]
                dist += weights[t] * d * d
        values.append(-dist / norm * 1800)
    return values


def softmax(values):
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    s = sum(exps)
    return [v / s for v in exps]


def predict(x, prototypes, use_smoothing=False):
    values = logits(x, prototypes, use_smoothing=use_smoothing)
    return max(range(len(values)), key=lambda i: values[i])


def grad_ce(x, y, prototypes):
    values = logits(x, prototypes)
    probs = softmax(values)
    steps = len(x)
    channels = len(x[0])
    norm = steps * channels
    grad = [[0.0 for _ in range(channels)] for _ in range(steps)]
    for c, proto in enumerate(prototypes):
        coeff = probs[c] - (1.0 if c == y else 0.0)
        weights = attention_weights(proto)
        for t in range(steps):
            for ch in range(channels):
                grad[t][ch] += coeff * (-2.0 * weights[t] * (x[t][ch] - proto[t][ch]) / norm * 1800)
    return grad


def clip_sample(x, low=-2.5, high=2.5):
    return [[min(high, max(low, v)) for v in row] for row in x]


def sign(v):
    if v > 0:
        return 1.0
    if v < 0:
        return -1.0
    return 0.0


def fgsm(x, y, prototypes, eps):
    g = grad_ce(x, y, prototypes)
    out = clone_sample(x)
    for t in range(len(x)):
        for ch in range(len(x[0])):
            out[t][ch] += eps * sign(g[t][ch])
    return clip_sample(out)


def pgd(x, y, prototypes, eps, steps=7):
    alpha = eps / 3.0
    out = clone_sample(x)
    for _ in range(steps):
        g = grad_ce(out, y, prototypes)
        for t in range(len(x)):
            for ch in range(len(x[0])):
                out[t][ch] += alpha * sign(g[t][ch])
                out[t][ch] = min(x[t][ch] + eps, max(x[t][ch] - eps, out[t][ch]))
        out = clip_sample(out)
    return out


def attention_guided_attack(x, y, prototypes, eps, keep_ratio=0.85):
    g = grad_ce(x, y, prototypes)
    saliency = []
    for t in range(len(x)):
        saliency.append((sum(abs(v) for v in g[t]), t))
    selected = {t for _, t in sorted(saliency, reverse=True)[: max(1, int(len(x) * keep_ratio))]}
    out = clone_sample(x)
    for t in selected:
        for ch in range(len(x[0])):
            out[t][ch] += eps * sign(g[t][ch])
    return clip_sample(out)


def accuracy(data, prototypes, attack=None, eps=0.0, use_smoothing=False):
    correct = 0
    changed = 0
    for x, y in data:
        x_eval = attack(x, y, prototypes, eps) if attack else x
        pred = predict(x_eval, prototypes, use_smoothing=use_smoothing)
        correct += int(pred == y)
        changed += int(pred != predict(x, prototypes))
    total = len(data)
    return correct / total, changed / total


def run_experiment():
    prototypes = make_prototypes()
    test = make_dataset(prototypes, per_class=28, seed=2026)
    rows = []
    scenarios = [
        ("Clean", None, 0.00, False, "原始测试集"),
        ("FGSM eps=0.35", fgsm, 0.35, False, "TorchAttacks FGSM 思路复现"),
        ("PGD eps=0.35", pgd, 0.35, False, "迭代攻击复现"),
        ("Attention-FGSM eps=0.35", attention_guided_attack, 0.35, False, "改进：只扰动高敏感时间步"),
        ("FGSM + Smooth Defense", fgsm, 0.35, True, "防御对照：时序平滑提升有限"),
        ("PGD + Smooth Defense", pgd, 0.35, True, "强攻击下防御不足"),
    ]
    for name, attack, eps, defend, note in scenarios:
        acc, changed = accuracy(test, prototypes, attack=attack, eps=eps, use_smoothing=defend)
        rows.append({
            "scenario": name,
            "epsilon": f"{eps:.2f}",
            "defense": "temporal_smoothing" if defend else "none",
            "accuracy": f"{acc * 100:.2f}",
            "attack_success_rate": f"{(1 - acc) * 100:.2f}" if attack else "0.00",
            "prediction_change_rate": f"{changed * 100:.2f}",
            "note": note,
        })
    return rows


if __name__ == '__main__':
    rows = run_experiment()
    out = Path(__file__).resolve().parent / 'results.csv'
    with out.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print('results saved to', out)
    for row in rows:
        print(row)
