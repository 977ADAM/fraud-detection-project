import streamlit as st
try:
    from .inference import FraudModel
    from .config import config, ALLOWED_TRANSACTION_TYPES
except ImportError:
    from inference import FraudModel
    from config import config, ALLOWED_TRANSACTION_TYPES


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
    'Тип транзакции',
    ALLOWED_TRANSACTION_TYPES
)

amount = st.number_input('Количество', min_value = 0.0, value = 1000.0)

oldbalanceOrg = st.number_input('Старый баланс (отправитель)', min_value = 0.0, value = 10000.0)
newbalanceOrig = st.number_input('Новый баланс (отправитель)', min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input('Старый баланс (приемник)', min_value = 0.0, value = 0.0)
newbalanceDest = st.number_input('Новый баланс (приемник)', min_value=0.0, value=0.0)


if st.button('Predict'):
    with st.spinner("Выполняется анализ транзакции..."):

        input_data = {
            "type": transaction_type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
        }

        try:
            result = model.predict(input_data)
        except ValueError as e:
            st.warning(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Ошибка предсказания: {e}")
            st.stop()

        prediction = result.prediction
        label = result.label

        proba = result.probability
        shap_values = result.shap_values

        if proba is not None:
            st.metric("Вероятность мошенничества", f"{proba:.2%}")

        st.subheader(f'Прогноз: {label}')

        if prediction == 1:
            st.error('Эта транзакция может быть мошеннической.')
        else:
            st.success('Похоже, эта транзакция не является мошенничеством.')

        if shap_values:
            st.divider()
            st.subheader("🔍 Объяснение модели (SHAP)")

            sorted_items = sorted(
                shap_values.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            for feature, value in sorted_items[:10]:
                direction = "⬆️ увеличивает риск" if value > 0 else "⬇️ снижает риск"
                st.write(f"**{feature}**: {value:.4f} ({direction})")
