import streamlit as st
import joblib
import gzip
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Configuración de la aplicación
st.set_page_config(page_title="Fraude Financiero", page_icon="💰", layout="wide")

# Ruta del modelo
RUTA_MODELO = "modelo_RandomForest_optimizado.pkl.gz"

def cargar_modelo_comprimido(ruta):
    """Carga el modelo comprimido con gzip."""
    with gzip.open(ruta, "rb") as f:
        modelo = joblib.load(f)
    return modelo

# Cargar el modelo
try:
    model = cargar_modelo_comprimido(RUTA_MODELO)
    st.success("✅ Modelo cargado correctamente.")
except Exception as e:
    st.error(f"Error al cargar el modelo: {str(e)}")
    st.stop()

# Diccionario de clases
class_dict = {"0": "No Fraude", "1": "Fraude"}

# Diccionario de ingresos
income_ranges = {
    "Bajo ($0 - $2,000)": 1000,
    "Medio ($2,000 - $6,000)": 4000,
    "Alto ($6,000 - $15,000)": 10000,
    "Muy Alto (> $15,000)": 20000
}

# Interfaz de usuario
st.title("🔍 Predicción de Fraude en Transacciones Bancarias")
with st.form("formulario_prediccion"):  
    st.subheader("📊 Introducir Datos de la Transacción")
    
    col1, col2 = st.columns(2)
    
    with col1:
        income_category = st.selectbox("Ingresos Mensuales", list(income_ranges.keys()))
        name_email_similarity = st.slider("Similitud entre Nombre y Email", 0.0, 1.0, 0.5, step=0.001)
        customer_age = st.number_input("Edad del Cliente", min_value=18, max_value=100, value=30)
    
    with col2:
        proposed_credit_limit = st.number_input("Límite de Crédito Propuesto", min_value=0.0, max_value=1000000.0, value=5000.0)
        velocity_6h = st.number_input("Velocidad Transacción (6h)", min_value=0.0, max_value=10000.0, value=10.0)
        foreign_request = st.radio("¿Solicitud Extranjera?", ["No", "Sí"], index=0)
    
    submit_button = st.form_submit_button("🚀 Predecir")  

if submit_button:  
    data_df = pd.DataFrame([[
        income_ranges[income_category],
        name_email_similarity,
        customer_age,
        proposed_credit_limit,
        velocity_6h,
        1 if foreign_request == "Sí" else 0
    ]], columns=[
        'income', 'name_email_similarity', 'customer_age',
        'proposed_credit_limit', 'velocity_6h', 'foreign_request'
    ])
    
    try:
        prediction = str(model.predict(data_df)[0])
        pred_class = class_dict[prediction]
        st.success(f"🔮 **Predicción:** {pred_class}")
    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
