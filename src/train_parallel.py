"""Стадия train_all: параллельное обучение всех моделей.

Каждая модель обучается в отдельном процессе (ProcessPoolExecutor), число
одновременно работающих процессов задается параметром train_all.workers
(1 - последовательно). Каждой модели выделяется одинаковая доля ядер
(ядра / число моделей), поэтому результат не зависит от режима запуска.

Вход:  data/prepared/*.npz или data/augmented/train.npz (train.dataset)
Выход: models/parallel/<модель>.pkl, metrics/parallel.json,
       plots/parallel_timeline.png
"""
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from threadpoolctl import threadpool_limits

from common import (ROOT, run, load_params, check_int, check_choice, save_json,
                    MODELS, DATASETS)
from train import build_model

OUT = ROOT / "models" / "parallel"


def load(dataset):
    folder = "prepared" if dataset == "original" else "augmented"
    train = np.load(ROOT / "data" / folder / "train.npz")
    test = np.load(ROOT / "data" / "prepared" / "test.npz")
    return (train["x"].reshape(len(train["x"]), -1), train["y"],
            test["x"].reshape(len(test["x"]), -1), test["y"])


def fit_one(name, params, seed, dataset, threads, t_zero):
    """Обучение и оценка одной модели (выполняется в отдельном процессе)."""
    start = time.time() - t_zero
    with threadpool_limits(threads):
        x, y, x_test, y_test = load(dataset)
        model = build_model(name, params[name], seed)
        if name == "random_forest":
            model.set_params(n_jobs=threads)
        t0 = time.perf_counter()
        model.fit(x, y)
        train_time = time.perf_counter() - t0
        pred = model.predict(x_test)
    with open(OUT / f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)
    return {"model": name, "pid": os.getpid(),
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "f1_macro": round(f1_score(y_test, pred, average="macro"), 4),
            "train_time_s": round(train_time, 2),
            "start_s": round(start, 2), "end_s": round(time.time() - t_zero, 2)}


def plot_timeline(results, workers, path):
    fig, ax = plt.subplots(figsize=(8, 3))
    for i, m in enumerate(MODELS):
        r = results[m]
        ax.barh(i, r["end_s"] - r["start_s"], left=r["start_s"], color="#4C72B0")
        ax.text(r["end_s"] + 0.3, i, f"{r['end_s'] - r['start_s']:.1f} c",
                va="center", fontsize=8)
    ax.set_yticks(range(len(MODELS)), MODELS)
    ax.invert_yaxis()
    ax.set_xlabel("Время от начала стадии, с")
    ax.set_title(f"Обучение моделей, процессов: {workers}", fontsize=10)
    ax.set_xlim(left=0)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    params = load_params()
    p = params["train"]
    dataset = check_choice(p["dataset"], "train.dataset", DATASETS)
    seed = check_int(p["seed"], "train.seed", 0)
    workers = check_int(params["train_all"]["workers"], "train_all.workers",
                        1, len(MODELS))
    for name in MODELS:      # проверка параметров всех моделей до запуска
        build_model(name, p[name], seed)
    # потоков на модель всегда одинаково (ядра / число моделей): от числа
    # потоков BLAS зависит порядок вычислений, а значит и результат MLP,
    # поэтому метрики не должны зависеть от режима запуска
    threads = max(1, (os.cpu_count() or 1) // len(MODELS))

    OUT.mkdir(parents=True, exist_ok=True)
    t_zero = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fit_one, m, p, seed, dataset, threads, t_zero)
                   for m in MODELS]
        for fut in as_completed(futures):
            r = fut.result()
            results[r["model"]] = r
            print(f"[{r['end_s']:6.1f} c] {r['model']:<14} процесс {r['pid']:<6} "
                  f"обучение {r['train_time_s']:6.1f} c  accuracy {r['accuracy']:.4f}")
    wall = time.time() - t_zero
    total = sum(r["end_s"] - r["start_s"] for r in results.values())

    save_json({"workers": workers, "dataset": dataset,
               "wall_time_s": round(wall, 2), "sum_model_time_s": round(total, 2),
               # сколько моделей в среднем обучалось одновременно
               "concurrency": round(total / wall, 2),
               "models": {m: {k: results[m][k] for k in
                              ("accuracy", "f1_macro", "train_time_s")}
                          for m in MODELS}},
              ROOT / "metrics" / "parallel.json")
    plot_timeline(results, workers, ROOT / "plots" / "parallel_timeline.png")
    print(f"Процессов: {workers}, ядер на процесс: {threads}. Общее время "
          f"{wall:.1f} c, сумма времен моделей {total:.1f} c, "
          f"одновременно в среднем {total / wall:.2f} модели")


if __name__ == "__main__":
    run(main)
