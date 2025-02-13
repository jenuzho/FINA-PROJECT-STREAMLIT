import streamlit as st
import joblib
import gzip
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Configuración de la app
st.set_page_config(page_title="Detección de Fraude", page_icon="💰", layout="wide")

# Ruta del modelo
MODEL_PATH = "modelo_RandomForest_optimizado.pkl.gz"

# Cargar el modelo
@st.cache_resource()
def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        st.error(f"⚠️ Error: No se encuentra el modelo en {MODEL_PATH}")
        st.stop()
    with gzip.open(MODEL_PATH, "rb") as f:
        model = joblib.load(f)
    return model

model = cargar_modelo()

# Verificar si el modelo es correcto
if not hasattr(model, "feature_names_in_"):
    st.error("El modelo cargado no tiene información de características. Revisa el entrenamiento del modelo.")
    st.stop()

# Obtener las características que espera el modelo
expected_features = list(model.feature_names_in_)

# Definir las variables que se deben ingresar
input_features = expected_features.copy()

# Sidebar con información
st.sidebar.title("📌 Menú de Navegación")
menu = st.sidebar.radio("Selecciona una opción:", ["Predicción de Fraude", "Información sobre Fraude"])

if menu == "Predicción de Fraude":
    st.title("🔍 Predicción de Fraude en Transacciones Bancarias")
    st.markdown("Ingrese los datos de la transacción para detectar si hay fraude.")
    
    # Formulario de entrada
    with st.form("fraud_form"):
        st.subheader("📊 Datos de la Transacción")
        col1, col2 = st.columns(2)
        
        with col1:
            income = st.number_input("Ingresos", min_value=0.0, max_value=10000000.0, step=100.0, value=5000.0)
            name_email_similarity = st.slider("Similitud entre Nombre y Email", 0.0, 1.0, 0.5, step=0.001)
            prev_address_months_count = st.number_input("Meses en Dirección Anterior", min_value=-1, max_value=240, value=12)
            current_address_months_count = st.number_input("Meses en Dirección Actual", min_value=0, max_value=240, value=12)
            customer_age = st.number_input("Edad del Cliente", min_value=18, max_value=100, value=30)
            intended_balcon_amount = st.number_input("Monto Saldo Previsto", min_value=0.0, max_value=1000000.0, value=10000.0)
        
        with col2:
            velocity_6h = st.number_input("Velocidad Transacción En 6h", min_value=0.0, max_value=10000.0, value=10.0)
            velocity_24h = st.number_input("Velocidad Transacción En 24h", min_value=0.0, max_value=10000.0, value=20.0)
            bank_branch_count_8w = st.number_input("Sucursales Bancarias En 8 Semanas", min_value=0, max_value=50, value=5)
            credit_risk_score = st.number_input("Puntuación de Riesgo Crediticio", min_value=0, max_value=1000, value=300)
            proposed_credit_limit = st.number_input("Límite de Crédito Propuesto", min_value=0.0, max_value=1000000.0, value=5000.0)
            month = st.slider("Mes de la Transacción", min_value=1, max_value=12, value=1)
        
        # Variables binarias
        email_is_free = st.radio("¿Email Gratuito?", ["No", "Sí"], index=0)
        phone_home_valid = st.radio("¿Teléfono Fijo Válido?", ["No", "Sí"], index=0)
        phone_mobile_valid = st.radio("¿Teléfono Móvil Válido?", ["No", "Sí"], index=0)
        has_other_cards = st.radio("¿Tiene Otras Tarjetas?", ["No", "Sí"], index=0)
        foreign_request = st.radio("¿Solicitud Extranjera?", ["No", "Sí"], index=0)
        keep_alive_session = st.number_input("Duración Sesión Activa (min)", min_value=0.0, max_value=1440.0, value=60.0)
        device_distinct_emails_8w = st.number_input("Emails Distintos por Dispositivo en 8 Semanas", min_value=0, max_value=50, value=5)
        
        submit_button = st.form_submit_button("🚀 Predecir")
    
    if submit_button:
        # Convertir las variables binarias a valores numéricos
        binary_mapping = {"No": 0, "Sí": 1}
        email_is_free = binary_mapping[email_is_free]
        phone_home_valid = binary_mapping[phone_home_valid]
        phone_mobile_valid = binary_mapping[phone_mobile_valid]
        has_other_cards = binary_mapping[has_other_cards]
        foreign_request = binary_mapping[foreign_request]
        
        # Crear dataframe con las características ordenadas correctamente
        data_df = pd.DataFrame([[
            income, name_email_similarity, prev_address_months_count, current_address_months_count, customer_age,
            intended_balcon_amount, velocity_6h, velocity_24h, bank_branch_count_8w, credit_risk_score,
            email_is_free, phone_home_valid, phone_mobile_valid, has_other_cards, proposed_credit_limit,
            foreign_request, keep_alive_session, device_distinct_emails_8w, month
        ]], columns=input_features)
        
        # Asegurar que las columnas coincidan con el modelo
        data_df = data_df[expected_features]
        
        # Realizar la predicción
        try:
            prediction = model.predict(data_df)[0]
            resultado = "Fraude" if prediction == 1 else "No Fraude"
            st.success(f"🔮 **Predicción:** {resultado}")
        except Exception as e:
            st.error(f"Error en la predicción: {str(e)}")
