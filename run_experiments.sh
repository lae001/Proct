#!/bin/bash
# Серия экспериментов: 4 модели на исходных данных и на аугментированных
# (1, 2 и 3 сгенерированные копии каждого изображения), затем абляция
# преобразований. Каждый запуск сохраняется DVC как отдельный эксперимент
# (git-ссылка refs/exps/...), посмотреть все: dvc exp show
set -e
export PYTHONUTF8=1
MODELS="logreg random_forest mlp svm"
# dvc exp run оставляет переопределенные параметры в рабочей копии,
# поэтому параметры аугментации задаются в каждом запуске явно
AUG="-S augment.rotation=12 -S augment.shift=2 -S augment.zoom=0.1"
AUG="$AUG -S augment.flip=true -S augment.noise=0.04"

# 1. Исходные данные (10 000 изображений)
for m in $MODELS; do
  dvc exp run -n "orig-$m" -S train.dataset=original -S augment.copies=1 \
    $AUG -S train.model=$m
done

# 2. Исходные + аугментированные данные (20 000, 30 000, 40 000 изображений)
for c in 1 2 3; do
  for m in $MODELS; do
    dvc exp run -n "aug$c-$m" -S train.dataset=augmented -S augment.copies=$c \
      $AUG -S train.model=$m
  done
done

# 3. Абляция: какие преобразования полезны (1 копия)
for m in $MODELS; do
  dvc exp run -n "aug1-nonoise-$m" -S train.dataset=augmented -S augment.copies=1 \
    $AUG -S augment.noise=0 -S train.model=$m
  dvc exp run -n "aug1-flip-$m" -S train.dataset=augmented -S augment.copies=1 \
    $AUG -S augment.noise=0 -S augment.rotation=0 -S augment.shift=0 \
    -S augment.zoom=0 -S train.model=$m
  dvc exp run -n "aug1-geom-$m" -S train.dataset=augmented -S augment.copies=1 \
    $AUG -S augment.noise=0 -S augment.flip=false -S train.model=$m
done

dvc exp show --no-pager
python src/compare.py
