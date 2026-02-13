import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "model" / "fraud_detection_pipeline_v1.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Модель не найдена")
    st.stop()
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()

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

        input_data = pd.DataFrame([[
            transaction_type,
            float(amount),
            float(oldbalanceOrg),
            float(newbalanceOrig),
            float(oldbalanceDest),
            float(newbalanceDest),
        ]], columns=[
            'type',
            'amount',
            'oldbalanceOrg',
            'newbalanceOrig',
            'oldbalanceDest',
            'newbalanceDest',
        ])

        try:
            prediction = int(model.predict(input_data)[0])
        except Exception as e:
            st.error(f"Ошибка предсказания: {e}")
            st.stop()

        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(input_data)[0][1]
                st.metric("Вероятность мошенничества", f"{proba:.2%}")
            except Exception:
                pass


        label = "Мошенничество" if prediction == 1 else "Не мошенничество"
        st.subheader(f'Прогноз: {label}')

        if prediction == 1:
            st.error('Эта транзакция может быть мошеннической.')
        else:
            st.success('Похоже, эта транзакция не является мошенничеством.')
