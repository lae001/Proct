"""Общие функции: чтение и проверка параметров, константы датасета."""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
IMG = 28
MODELS = ("logreg", "random_forest", "mlp", "svm")
DATASETS = ("original", "augmented")


class ParamError(ValueError):
    pass


def load_params():
    with open(ROOT / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_int(value, name, lo, hi=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or int(value) != value:
        raise ParamError(f"{name}: ожидается целое число, получено {value!r}")
    value = int(value)
    if value < lo or (hi is not None and value > hi):
        rng = f"[{lo}, {hi}]" if hi is not None else f">= {lo}"
        raise ParamError(f"{name}: значение {value} вне диапазона {rng}")
    return value


def check_float(value, name, lo, hi=None, lo_strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParamError(f"{name}: ожидается число, получено {value!r}")
    value = float(value)
    bad_lo = value <= lo if lo_strict else value < lo
    if bad_lo or (hi is not None and value > hi):
        sign = ">" if lo_strict else ">="
        rng = f"{sign} {lo}" + (f" и <= {hi}" if hi is not None else "")
        raise ParamError(f"{name}: значение {value} должно быть {rng}")
    return value


def check_choice(value, name, choices):
    if value not in choices:
        raise ParamError(f"{name}: допустимые значения {list(choices)}, "
                         f"получено {value!r}")
    return value


def run(main):
    """Запускает стадию; ошибку в параметрах выводит понятным сообщением."""
    try:
        main()
    except ParamError as e:
        print(f"Ошибка в params.yaml: {e}", file=sys.stderr)
        sys.exit(2)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
