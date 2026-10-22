"""Сводка экспериментов DVC: таблица results.csv и графики сравнения.

Запуск из корня репозитория:  python src/compare.py
Читает `dvc exp show --json`, поэтому видит все сохраненные эксперименты.
"""
import json
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).absolute().parent.parent
OUT = ROOT / "reports"
MODELS = ["logreg", "random_forest", "mlp", "svm"]
LABELS = {"logreg": "Логистическая\nрегрессия", "random_forest": "Случайный\nлес",
          "mlp": "Многослойный\nперцептрон", "svm": "SVM (RBF)\n+ PCA"}


def collect():
    dvc = os.environ.get("DVC", "dvc").split()  # можно задать свою команду DVC
    raw = subprocess.run(dvc + ["exp", "show", "--json"], cwd=ROOT,
                         capture_output=True, text=True, encoding="utf-8",
                         check=True).stdout
    rows = []
    for commit in json.loads(raw):
        for exp in commit.get("experiments") or []:
            for rev in exp.get("revs", []):
                d = rev.get("data") or {}
                name = rev.get("name") or exp.get("name")
                if not name:
                    continue
                params = d["params"]["params.yaml"]["data"]
                metrics = {}
                for m in d.get("metrics", {}).values():
                    metrics.update(m.get("data", {}))
                ds = params["train"]["dataset"]
                aug = params["augment"]
                rows.append({
                    "name": name,
                    "noise": aug["noise"], "flip": aug["flip"],
                    "rotation": aug["rotation"], "shift": aug["shift"],
                    "zoom": aug["zoom"], "rev": rev["rev"][:7],
                    "model": params["train"]["model"],
                    "dataset": ds,
                    "copies": params["augment"]["copies"] if ds == "augmented" else 0,
                    "n_train": metrics.get("n_train"),
                    "accuracy": metrics.get("accuracy"),
                    "f1_macro": metrics.get("f1_macro"),
                    "train_time_s": metrics.get("train_time_s"),
                })
    # если эксперимент с тем же именем запускался повторно - берем последний
    uniq = {}
    for r in rows:
        uniq.setdefault(r["name"], r)
    return sorted(uniq.values(), key=lambda r: (r["name"].count("-") > 1, r["copies"],
                                                r["name"], MODELS.index(r["model"])))


def main():
    rows = collect()
    OUT.mkdir(exist_ok=True)
    cols = ["name", "rev", "model", "dataset", "copies", "noise", "flip",
            "rotation", "shift", "zoom", "n_train", "accuracy", "f1_macro",
            "train_time_s"]
    with open(OUT / "results.csv", "w", encoding="utf-8", newline="\n") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    # основная серия: orig-<модель> и aug<N>-<модель>; остальные - абляция
    main_names = {f"orig-{m}" for m in MODELS} | {
        f"aug{c}-{m}" for c in range(1, 11) for m in MODELS}
    get = {(r["model"], r["copies"]): r for r in rows if r["name"] in main_names}
    copies = sorted({c for _, c in get})
    x = np.arange(len(MODELS))

    # 1. точность моделей: исходные данные против аугментации x1
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.38
    for i, (c, lab, col) in enumerate([(0, "исходные данные (10 000)", "#4C72B0"),
                                       (1, "исходные + аугментация (20 000)", "#DD8452")]):
        vals = [get[(m, c)]["accuracy"] for m in MODELS]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=lab, color=col)
        ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=2)
    ax.set_xticks(x, [LABELS[m] for m in MODELS])
    ax.set_ylabel("Accuracy на тестовой выборке")
    lo = min(r["accuracy"] for r in get.values())
    hi = max(r["accuracy"] for r in get.values())
    ax.set_ylim(lo - 0.03, hi + 0.02)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_by_model.png", dpi=150)
    plt.close(fig)

    # 2. зависимость точности от числа аугментированных копий
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in MODELS:
        ys = [get[(m, c)]["accuracy"] for c in copies]
        ax.plot(copies, ys, marker="o", label=LABELS[m].replace("\n", " "))
    ax.set_xticks(copies, [f"{c}\n({10000 * (c + 1):,})".replace(",", " ")
                           for c in copies])
    ax.set_xlabel("Число аугментированных копий (размер обучающей выборки)")
    ax.set_ylabel("Accuracy на тестовой выборке")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_vs_copies.png", dpi=150)
    plt.close(fig)

    # 3. зависимость времени обучения от числа копий
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in MODELS:
        ys = [get[(m, c)]["train_time_s"] for c in copies]
        ax.plot(copies, ys, marker="o", label=LABELS[m].replace("\n", " "))
    ax.set_xticks(copies, [f"{c}\n({10000 * (c + 1):,})".replace(",", " ")
                           for c in copies])
    ax.set_xlabel("Число аугментированных копий (размер обучающей выборки)")
    ax.set_ylabel("Время обучения, с")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "train_time_vs_copies.png", dpi=150)
    plt.close(fig)

    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
