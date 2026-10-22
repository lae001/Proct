"""Стадия augment: генерация дополнительных изображений и сохранение нового датасета.

Каждое изображение обучающей выборки порождает `copies` новых изображений,
к каждому из которых применяется случайная комбинация преобразований:
поворот, сдвиг, масштабирование, горизонтальное отражение, гауссов шум.

Вход:  data/prepared/train.npz
Выход: data/augmented/train.npz (исходные + сгенерированные изображения),
       plots/augment_examples.png
"""
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (ROOT, run, load_params, check_int, check_float, save_json,
                    CLASSES, IMG)

SRC = ROOT / "data" / "prepared" / "train.npz"
OUT = ROOT / "data" / "augmented"


def transform(img, rng, p):
    """Одно случайное преобразование изображения 28x28 с яркостью в [0, 1]."""
    angle = np.deg2rad(rng.uniform(-p["rotation"], p["rotation"]))
    scale = 1.0 + rng.uniform(-p["zoom"], p["zoom"])
    dy, dx = rng.uniform(-p["shift"], p["shift"], size=2)
    # аффинное преобразование вокруг центра: поворот + масштаб, затем сдвиг
    c, s = np.cos(angle), np.sin(angle)
    m = np.array([[c, -s], [s, c]]) / scale
    center = np.array([IMG / 2 - 0.5, IMG / 2 - 0.5])
    offset = center - m @ center - m @ np.array([dy, dx])
    out = ndimage.affine_transform(img, m, offset=offset, order=1,
                                   mode="constant", cval=0.0)
    if p["flip"] and rng.random() < 0.5:
        out = out[:, ::-1]
    if p["noise"] > 0:
        out = out + rng.normal(0.0, p["noise"], out.shape)
    return np.clip(out, 0.0, 1.0)


def save_examples(x, x_new, y, path, n=8):
    fig, axes = plt.subplots(2, n, figsize=(n * 1.3, 3))
    for i in range(n):
        axes[0, i].imshow(x[i], cmap="gray_r")
        axes[0, i].set_title(CLASSES[y[i]], fontsize=7)
        axes[1, i].imshow(x_new[i], cmap="gray_r")
        for ax in axes[:, i]:
            ax.axis("off")
    axes[0, 0].text(-6, 14, "исходное", rotation=90, va="center", fontsize=8)
    axes[1, 0].text(-6, 14, "аугм.", rotation=90, va="center", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    raw = load_params()["augment"]
    p = {
        "copies": check_int(raw["copies"], "augment.copies", 0, 10),
        "rotation": check_float(raw["rotation"], "augment.rotation", 0, 180),
        "shift": check_float(raw["shift"], "augment.shift", 0, 10),
        "zoom": check_float(raw["zoom"], "augment.zoom", 0, 0.5),
        "flip": bool(raw["flip"]),
        "noise": check_float(raw["noise"], "augment.noise", 0, 0.5),
    }
    seed = check_int(raw["seed"], "augment.seed", 0)
    rng = np.random.default_rng(seed)

    data = np.load(SRC)
    x, y = data["x"], data["y"]
    new_x = [x]
    new_y = [y]
    for _ in range(p["copies"]):
        new_x.append(np.stack([transform(img, rng, p) for img in x])
                     .astype(np.float32))
        new_y.append(y)
    x_all = np.concatenate(new_x)
    y_all = np.concatenate(new_y)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "train.npz", x=x_all, y=y_all)
    save_json({"original": int(len(y)), "generated": int(len(y_all) - len(y)),
               "total": int(len(y_all))}, OUT / "stats.json")
    if p["copies"] > 0:
        save_examples(x, new_x[1], y, ROOT / "plots" / "augment_examples.png")
    print(f"исходных: {len(y)}, всего после аугментации: {len(y_all)}")


if __name__ == "__main__":
    run(main)
