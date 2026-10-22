# Fashion-MNIST: версионирование данных и экспериментов (Git + DVC)

Практическая работа № 3 «Инженерия машинного обучения», вариант 3 —
Zalando Store Fashion Image Dataset (Fashion-MNIST, 10 классов одежды, 28×28).

## Структура

```
data/raw/            исходные файлы IDX (под DVC, data/raw.dvc)
data/prepared/       подвыборка 10 000 / тест 10 000, нормализация  (стадия prepare)
data/augmented/      исходные + сгенерированные изображения       (стадия augment)
models/model.pkl     обученная модель                              (стадия train)
metrics/*.json       метрики обучения и качества                   (стадии train, evaluate)
plots/               матрица ошибок, F1 по классам, примеры аугментации
reports/             сводная таблица и графики по всем экспериментам
src/                 код стадий; params.yaml — все параметры; dvc.yaml — пайплайн
```

## Воспроизведение

```bash
python -m venv .venv && . .venv/Scripts/activate   # Linux/macOS: . .venv/bin/activate
pip install -r requirements.txt
dvc pull            # скачать данные и модели из DVC-хранилища
dvc repro           # пересчитать пайплайн (выполнит только изменившиеся стадии)
dvc metrics show
```

## Эксперименты

```bash
bash run_experiments.sh                 # 28 экспериментов: 4 модели × (4 набора + 3 варианта абляции)
dvc exp show                            # таблица всех экспериментов
dvc exp show --only-changed --sort-by metrics/eval.json:accuracy
dvc exp diff aug1-svm orig-svm          # сравнить два эксперимента
dvc exp apply aug1-svm                  # перенести эксперимент в рабочую копию
dvc exp branch aug1-svm exp-aug1-svm    # сохранить эксперимент как git-ветку
python src/compare.py                   # таблица reports/results.csv и графики
```

Отдельный запуск с любыми параметрами:

```bash
dvc exp run -n my-exp -S train.model=mlp -S train.dataset=augmented -S augment.copies=2
```

Навигация по зафиксированной истории (теги `baseline`, `best-original`, `best-augmented`):

```bash
git checkout best-original && dvc checkout   # код, параметры, данные и модель той версии
git checkout main && dvc checkout
```

## Публикация на GitLab

1. Создать пустой проект на gitlab.com (без README).
2. Подключить и отправить код и эксперименты:

```bash
git remote add origin https://gitlab.com/<логин>/fashion-mlops.git
git push -u origin main --tags
git push origin exp-aug1-svm             # ветка, созданная из эксперимента
dvc exp push origin --all-commits        # эксперименты DVC (refs/exps) на GitLab
```

3. Данные и модели хранятся в DVC-хранилище (по умолчанию локальная папка
   `../dvc-storage`). Для совместной работы его можно перенести в облако,
   например S3/Google Drive/SSH: `dvc remote add -d -f storage <url>` и `dvc push`.

## Результаты

Лучшая модель на исходных данных — SVM (RBF) + PCA, accuracy 0.8588 (тег `best-original`).
Лучшая модель в целом — многослойный перцептрон на исходных + 3 аугментированных копиях,
accuracy 0.8661 (тег `best-augmented`, текущее состояние `main`).
Полная таблица — `reports/results.csv`, графики — `reports/*.png`.
