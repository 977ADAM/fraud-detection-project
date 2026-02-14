# Fraud Detection Project

Демо-проект для обнаружения мошеннических транзакций.

Dataset:
https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset/data

## Обучение
1) Положите CSV в `data/dataset.csv`
2) Запустите:
```bash
python train.py
```

## Инференс (UI)
```bash
streamlit run fraud_detection.py
```

## Принцип
Pipeline отвечает за преобразования и устойчивость к новым категориям,
поэтому модель должна принимать данные и не блокировать их по несущественным правилам.