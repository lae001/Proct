"""Стадия prepare: чтение исходных файлов IDX, выбор подвыборки, нормализация.

Вход:  data/raw/*.gz (Fashion-MNIST в формате IDX)
Выход: data/prepared/train.npz, data/prepared/test.npz
"""
import gzip

import numpy as np
from sklearn.model_selection import train_test_split

from common import ROOT, run, load_params, check_int, save_json, IMG

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "prepared"


def read_idx(path, offset):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=offset)


def load_split(prefix):
    x = read_idx(RAW / f"{prefix}-images-idx3-ubyte.gz", 16)
    y = read_idx(RAW / f"{prefix}-labels-idx1-ubyte.gz", 8)
    x = x.reshape(-1, IMG, IMG)
    assert len(x) == len(y), "число изображений и меток не совпадает"
    return x, y


def main():
    p = load_params()["prepare"]
    n_train = check_int(p["n_train"], "prepare.n_train", 100, 60000)
    seed = check_int(p["seed"], "prepare.seed", 0)

    x_train, y_train = load_split("train")
    x_test, y_test = load_split("t10k")

    if n_train < len(x_train):
        x_train, _, y_train, _ = train_test_split(
            x_train, y_train, train_size=n_train, stratify=y_train,
            random_state=seed)

    OUT.mkdir(parents=True, exist_ok=True)
    # яркость приводится к [0, 1]
    np.savez_compressed(OUT / "train.npz", x=(x_train / 255.0).astype(np.float32),
                        y=y_train)
    np.savez_compressed(OUT / "test.npz", x=(x_test / 255.0).astype(np.float32),
                        y=y_test)
    save_json({"train": int(len(y_train)), "test": int(len(y_test)),
               "train_class_counts": np.bincount(y_train).tolist()},
              OUT / "stats.json")
    print(f"train: {len(y_train)}, test: {len(y_test)}")


if __name__ == "__main__":
    run(main)
