"""Стадия train: обучение выбранной модели на выбранном наборе данных.

Вход:  data/prepared/train.npz или data/augmented/train.npz (train.dataset)
Выход: models/model.pkl, metrics/train.json
"""
import pickle
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import (ROOT, run, load_params, check_int, check_float,
                    check_choice, save_json, MODELS, DATASETS, ParamError)


def build_model(name, p, seed):
    if name == "logreg":
        c = check_float(p["C"], "train.logreg.C", 0, lo_strict=True)
        it = check_int(p["max_iter"], "train.logreg.max_iter", 1)
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=c, max_iter=it))
    if name == "random_forest":
        n = check_int(p["n_estimators"], "train.random_forest.n_estimators", 1, 5000)
        depth = check_int(p["max_depth"], "train.random_forest.max_depth", 0)
        return RandomForestClassifier(n_estimators=n, max_depth=depth or None,
                                      n_jobs=-1, random_state=seed)
    if name == "mlp":
        hidden = p["hidden"]
        if not isinstance(hidden, (list, tuple)) or not hidden:
            raise ParamError("train.mlp.hidden: нужен непустой список размеров слоев")
        hidden = tuple(check_int(h, "train.mlp.hidden[i]", 1, 4096) for h in hidden)
        alpha = check_float(p["alpha"], "train.mlp.alpha", 0)
        it = check_int(p["max_iter"], "train.mlp.max_iter", 1)
        return MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, max_iter=it,
                             early_stopping=True, random_state=seed)
    if name == "svm":
        c = check_float(p["C"], "train.svm.C", 0, lo_strict=True)
        n_pca = check_int(p["pca"], "train.svm.pca", 0, 784)
        steps = [StandardScaler()]
        if n_pca:
            steps.append(PCA(n_components=n_pca, random_state=seed))
        steps.append(SVC(C=c, kernel="rbf", gamma="scale"))
        return make_pipeline(*steps)
    raise AssertionError(name)


def main():
    p = load_params()["train"]
    name = check_choice(p["model"], "train.model", MODELS)
    dataset = check_choice(p["dataset"], "train.dataset", DATASETS)
    seed = check_int(p["seed"], "train.seed", 0)

    folder = "prepared" if dataset == "original" else "augmented"
    data = np.load(ROOT / "data" / folder / "train.npz")
    x = data["x"].reshape(len(data["x"]), -1)
    y = data["y"]

    model = build_model(name, p[name], seed)
    start = time.perf_counter()
    model.fit(x, y)
    elapsed = time.perf_counter() - start

    (ROOT / "models").mkdir(exist_ok=True)
    with open(ROOT / "models" / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    save_json({"model": name, "dataset": dataset, "n_train": int(len(y)),
               "train_time_s": round(elapsed, 2)},
              ROOT / "metrics" / "train.json")
    print(f"{name} на {dataset} ({len(y)} изобр.): {elapsed:.1f} c")


if __name__ == "__main__":
    run(main)
