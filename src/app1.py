import joblib
import streamlit as st
import os
import gzip
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Configuración de la aplicación
st.set_page_config(page_title="Fraude Financiero", page_icon="💰", layout="wide")

# Ruta del modelo
RUTA_MODELO = "modelo_RandomForest_optimizado.pkl.gz"

def cargar_modelo_comprimido(ruta):
    """Carga el modelo comprimido con gzip y verifica que sea un RandomForestClassifier."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"El archivo {ruta} no existe. Verifica que está en la carpeta correcta.")
    with gzip.open(ruta, "rb") as f:
        modelo = joblib.load(f)
    
    if not isinstance(modelo, RandomForestClassifier):
        raise ValueError("El modelo cargado no es un RandomForestClassifier.")
    
    return modelo

# Cargar el modelo
try:
    model = cargar_modelo_comprimido(RUTA_MODELO)
except Exception as e:
    st.error(f"Error al cargar el modelo: {str(e)}")
    st.stop()

# Diccionario de clases
class_dict = {"0": "No Fraude", "1": "Fraude"}

# Barra lateral
menu = st.sidebar.radio("📌 Menú de Navegación", ["Predicción de Fraude", "Información sobre Fraudes Financieros"])

if menu == "Predicción de Fraude":
    st.title("🔍 Predicción de Fraude en Transacciones Bancarias")
    
    # Agregar créditos en el menú de predicción
    st.markdown("**Aplicación de predicción creada por JEN UZHO y JORGE PEDROZA**")
    
    with st.form("formulario_prediccion"):  
        st.subheader("📊 Ingrese los Datos de la Transacción")
        col1, col2 = st.columns(2)
        
        # Primera columna con las primeras 10 variables
        with col1:
            income = st.number_input("Ingresos", min_value=0.0, max_value=10000000.0, step=1000.0, value=1000.0)
            name_email_similarity = st.slider("Similitud entre Nombre y Email", 0.0, 1.0, 0.5, step=0.001)
            prev_address_months_count = st.number_input("Meses en Dirección Anterior", min_value=-1, max_value=240, value=12)
            current_address_months_count = st.number_input("Meses en Dirección Actual", min_value=0, max_value=240, value=12)
            customer_age = st.number_input("Edad del Cliente", min_value=18, max_value=100, value=30)
            intended_balcon_amount = st.number_input("Monto Saldo Previsto", min_value=0.0, max_value=1000000.0, value=10000.0)
            velocity_6h = st.number_input("Velocidad Transacción En 6h", min_value=0.0, max_value=10000.0, value=10.0)
            velocity_24h = st.number_input("Velocidad Transacción En 24h", min_value=0.0, max_value=10000.0, value=20.0)
            bank_branch_count_8w = st.number_input("Sucursales Bancarias En 8 Semanas", min_value=0, max_value=50, value=5)
            date_of_birth_distinct_emails_4w = st.number_input("Emails Distintos en 4 Semanas", min_value=0, max_value=50, value=5)

        # Segunda columna con las siguientes 10 variables
        with col2:
            credit_risk_score = st.number_input("Puntuación de Riesgo Crediticio", min_value=0, max_value=1000, value=300)
            email_is_free = st.radio("¿Email Gratuito?", ["No", "Sí"], index=0)
            phone_home_valid = st.radio("¿Teléfono Fijo Válido?", ["No", "Sí"], index=0)
            phone_mobile_valid = st.radio("¿Teléfono Móvil Válido?", ["No", "Sí"], index=0)
            has_other_cards = st.radio("¿Tiene Otras Tarjetas?", ["No", "Sí"], index=0)
            proposed_credit_limit = st.number_input("Límite de Crédito Propuesto", min_value=0.0, max_value=1000000.0, value=5000.0)
            foreign_request = st.radio("¿Solicitud Extranjera?", ["No", "Sí"], index=0)
            keep_alive_session = st.number_input("Duración Sesión Activa (min)", min_value=0.0, max_value=1440.0, value=60.0)
            device_distinct_emails_8w = st.number_input("Emails Distintos por Dispositivo en 8 Semanas", min_value=0, max_value=50, value=5)
            month = st.slider("Mes de la Transacción", min_value=0, max_value=12, value=1)

        submit_button = st.form_submit_button("🚀 Predecir")  

    if submit_button:  
        # Crear el dataframe de entrada con las variables seleccionadas
        data_df = pd.DataFrame([[
            income, name_email_similarity, prev_address_months_count, current_address_months_count, customer_age,
            intended_balcon_amount, velocity_6h, velocity_24h, bank_branch_count_8w, date_of_birth_distinct_emails_4w,
            credit_risk_score, email_is_free == "Sí", phone_home_valid == "Sí", phone_mobile_valid == "Sí", has_other_cards == "Sí",
            proposed_credit_limit, foreign_request == "Sí", keep_alive_session, device_distinct_emails_8w, month
        ]], columns=[
            'income', 'name_email_similarity', 'prev_address_months_count', 'current_address_months_count', 'customer_age',
            'intended_balcon_amount', 'velocity_6h', 'velocity_24h', 'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w',
            'credit_risk_score', 'email_is_free', 'phone_home_valid', 'phone_mobile_valid', 'has_other_cards', 'proposed_credit_limit',
            'foreign_request', 'keep_alive_session', 'device_distinct_emails_8w', 'month'
        ])
        
        # Filtrar las características para que coincidan con el modelo
        try:
            data_df = data_df[model.feature_names_in_]
            prediction = str(model.predict(data_df)[0])
            pred_class = class_dict[prediction]
            st.success(f"🔮 **Predicción:** {pred_class}")
        except Exception as e:
            st.error(f"Error en la predicción: {str(e)}")
