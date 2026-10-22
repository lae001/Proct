"""Стадия evaluate: оценка качества модели на тестовой выборке (10000 изображений).

Вход:  models/model.pkl, data/prepared/test.npz
Выход: metrics/eval.json, plots/confusion.csv, plots/per_class.csv,
       plots/confusion.png
"""
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from common import ROOT, run, save_json, CLASSES


def main():
    data = np.load(ROOT / "data" / "prepared" / "test.npz")
    x = data["x"].reshape(len(data["x"]), -1)
    y = data["y"]
    with open(ROOT / "models" / "model.pkl", "rb") as f:
        model = pickle.load(f)

    pred = model.predict(x)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro")
    save_json({"accuracy": round(acc, 4), "f1_macro": round(f1, 4)},
              ROOT / "metrics" / "eval.json")

    plots = ROOT / "plots"
    plots.mkdir(exist_ok=True)
    # пары «истинный класс - предсказанный» для dvc plots (шаблон confusion)
    with open(plots / "confusion.csv", "w", encoding="utf-8", newline="\n") as f:
        f.write("actual,predicted\n")
        for a, b in zip(y, pred):
            f.write(f"{CLASSES[a]},{CLASSES[b]}\n")
    prec, rec, f1s, _ = precision_recall_fscore_support(y, pred)
    with open(plots / "per_class.csv", "w", encoding="utf-8", newline="\n") as f:
        f.write("class,precision,recall,f1\n")
        for i, name in enumerate(CLASSES):
            f.write(f"{name},{prec[i]:.4f},{rec[i]:.4f},{f1s[i]:.4f}\n")

    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(10), CLASSES, fontsize=8)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(plots / "confusion.png", dpi=150)
    plt.close(fig)
    print(f"accuracy = {acc:.4f}, f1_macro = {f1:.4f}")


if __name__ == "__main__":
    run(main)
