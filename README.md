# Fraud Detection Project

Демо-проект для обнаружения мошеннических транзакций.

Dataset:
https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset/data

## Обучение
1) Положите CSV в `data/dataset.csv`
2) Запустите:
```bash
python src/train.py
```

## Версионирование модели
Модель проверяет соответствие версии из metadata и config.version.
Несоответствие приведёт к ошибке загрузки.

## Воспроизводимость
Версия Python и версия sklearn должны совпадать с метаданными.

⚠ Важно:
metadata.feature_schema используется для проверки входных данных.
Если вы изменили список признаков — необходимо пересобрать модель.


## Инференс (UI)
```bash
streamlit run src/fraud_detection.py
```

## Принцип
Pipeline отвечает за преобразования и устойчивость к новым категориям,
поэтому модель должна принимать данные и не блокировать их по несущественным правилам.
