import streamlit as st
import pandas as pd
import joblib

try:
    model = joblib.load("fraud_detection_pipeline.pkl")
except:
    st.error("Модель не найдена")

st.title('💳 Fraud Detection System')
st.markdown('Модель машинного обучения для обнаружения мошеннических транзакций')
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
newbalanceOrig = st.number_input('New Balance (Отправитель)', min_value = 0.0, value = 9000.0)
oldbalanceDest = st.number_input('Старый баланс (приемник)', min_value = 0.0, value = 0.0)
newbalanceDest = st.number_input('New Balance (приемник)', min_value = 0.0, value = 0.0)


if st.button('Predict'):
    input_data = pd.DataFrame([{
        'type': transaction_type,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
    }])

    prediction = model.predict(input_data)[0]

    proba = model.predict_proba(input_data)[0][1]
    st.metric("Вероятность мошенничества", f"{proba:.2%}")


    st.subheader(f'Прогноз: "{int(prediction)}"')

    if prediction == 1:
        st.error('Эта транзакция может быть мошеннической.')
    else:
        st.success('Похоже, эта транзакция не является мошенничеством.')
