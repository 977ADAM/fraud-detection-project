import streamlit as st
from src.inference import FraudModel
from src.config import config


try:
    model = FraudModel()
except FileNotFoundError:
    st.error("Модель не найдена")
    st.stop()
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()

st.title('💳 Fraud Detection System')
st.markdown('Модель машинного обучения для обнаружения мошеннических транзакций')
st.caption(f"Model version: {config.version}")
st.divider()

transaction_type = st.selectbox(
    'Тип транзакции', [
        'PAYMENT',
        'TRANSFER',
        'CASH_OUT',
        'DEPOSIT'
        ])

amount = st.number_input('Количество', min_value = 0.0, value = 1000.0)

oldbalanceOrg = st.number_input('Старый баланс (отправитель)', min_value = 0.0, value = 10000.0)
newbalanceOrig = st.number_input('Новый баланс (отправитель)', min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input('Старый баланс (приемник)', min_value = 0.0, value = 0.0)
newbalanceDest = st.number_input('Новый баланс (приемник)', min_value=0.0, value=0.0)


if st.button('Predict'):
    with st.spinner("Выполняется анализ транзакции..."):

        if newbalanceOrig > oldbalanceOrg:
            st.warning("Новый баланс отправителя не может быть больше старого.")
            st.stop()

        if amount > oldbalanceOrg and transaction_type in ["PAYMENT", "TRANSFER", "CASH_OUT"]:
            st.warning("Сумма транзакции превышает баланс отправителя.")
            st.stop()

        input_data = {
            "type": transaction_type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
        }

        result = model.predict(input_data)

        try:
            prediction = result.prediction
        except Exception as e:
            st.error(f"Ошибка предсказания: {e}")
            st.stop()

        proba = result.probability

        if proba is not None:
            st.metric("Вероятность мошенничества", f"{proba:.2%}")

        label = "Мошенничество" if prediction == 1 else "Не мошенничество"
        st.subheader(f'Прогноз: {label}')

        if prediction == 1:
            st.error('Эта транзакция может быть мошеннической.')
        else:
            st.success('Похоже, эта транзакция не является мошенничеством.')
