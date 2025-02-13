import streamlit as st
import joblib
import gzip
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Configuración de la app
st.set_page_config(page_title="Predicción de Fraude Financiero", page_icon="💰", layout="wide")

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

# Verificar si el modelo es un RandomForestClassifier
if not isinstance(model, RandomForestClassifier):
    st.error("El archivo cargado no es un modelo RandomForest.")
    st.stop()

# Definir perfiles de clientes
perfiles = {
    "Cliente Nuevo y Desconocido": {
        "income": 0.3, "name_email_similarity": 0.9, "customer_age": 25, 
        "velocity_6h": 1000, "credit_risk_score": 200, "proposed_credit_limit": 5000
    },
    "Cliente Recurrente y Estable": {
        "income": 0.7, "name_email_similarity": 0.5, "customer_age": 40, 
        "velocity_6h": 200, "credit_risk_score": 700, "proposed_credit_limit": 20000
    }
}

# Selección de Modo
modo = st.radio("Selecciona un Modo:", ["Modo Automático", "Modo Avanzado"], horizontal=True)

# Selección del perfil
perfil_seleccionado = st.selectbox("Seleccione un perfil de cliente", list(perfiles.keys()))
data = perfiles[perfil_seleccionado]

# Ajuste de parámetros
st.subheader("📊 Ajuste de Parámetros")
col1, col2 = st.columns(2)

with col1:
    income = st.slider("Ingresos", 0.0, 1.0, data["income"], step=0.1)
    name_email_similarity = st.slider("Similitud Nombre-Email", 0.0, 1.0, data["name_email_similarity"], step=0.01)
    customer_age = st.number_input("Edad del Cliente", 18, 100, data["customer_age"])

with col2:
    velocity_6h = st.number_input("Velocidad Transacción en 6h", 0, 10000, data["velocity_6h"])
    credit_risk_score = st.number_input("Puntuación de Riesgo Crediticio", 0, 1000, data["credit_risk_score"])
    proposed_credit_limit = st.number_input("Límite de Crédito Propuesto", 0, 1000000, data["proposed_credit_limit"])

# Preparar datos de entrada
input_data = pd.DataFrame([{
    "income": income, "name_email_similarity": name_email_similarity, "customer_age": customer_age,
    "velocity_6h": velocity_6h, "credit_risk_score": credit_risk_score, "proposed_credit_limit": proposed_credit_limit
}])

# Predicción automática en Modo Automático
if modo == "Modo Automático":
    pred = model.predict(input_data)[0]
    resultado = "🚨 Fraude" if pred == 1 else "✅ No Fraude"
    st.success(f"🔮 **Predicción:** {resultado}")

# Botón de predicción en Modo Avanzado
elif modo == "Modo Avanzado":
    if st.button("🚀 Actualizar Predicción"):
        pred = model.predict(input_data)[0]
        resultado = "🚨 Fraude" if pred == 1 else "✅ No Fraude"
        st.success(f"🔮 **Predicción:** {resultado}")
