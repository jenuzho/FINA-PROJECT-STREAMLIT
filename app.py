import streamlit as st
import joblib
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# Cargar estadísticas de fraude/no fraude
stats_fraud = pd.read_csv("/mnt/data/stats_fraud.csv", index_col=0)
stats_no_fraud = pd.read_csv("/mnt/data/stats_no_fraud.csv", index_col=0)

# Definir perfiles de clientes con explicaciones
perfiles = {
    "Cliente Nuevo y Desconocido": {
        "explicacion": "Este cliente es nuevo en la plataforma y representa mayor riesgo. "
                       "Tiene pocos datos previos y sus transacciones pueden ser inusuales.",
        "datos": {
            "income": 0.3, "name_email_similarity": 0.9, "prev_address_months_count": 5,
            "current_address_months_count": 3, "customer_age": 25, "velocity_6h": 1000,
            "velocity_24h": 3000, "credit_risk_score": 200, "proposed_credit_limit": 5000
        }
    },
    "Cliente Recurrente y Estable": {
        "explicacion": "Cliente con historial sólido y transacciones previsibles.",
        "datos": {
            "income": 0.7, "name_email_similarity": 0.5, "prev_address_months_count": 20,
            "current_address_months_count": 50, "customer_age": 40, "velocity_6h": 200,
            "velocity_24h": 800, "credit_risk_score": 700, "proposed_credit_limit": 20000
        }
    }
}

# Interfaz de usuario
st.title("🔍 Predicción de Fraude Financiero")

# Selección de Modo
modo = st.radio("Selecciona un Modo:", ["Modo Automático", "Modo Avanzado"], horizontal=True)

# Selección del perfil
perfil_seleccionado = st.selectbox("Seleccione un perfil de cliente", list(perfiles.keys()))

# Mostrar la explicación del perfil
st.markdown(f"**ℹ️ Sobre este perfil:** {perfiles[perfil_seleccionado]['explicacion']}")

# Cargar los valores del perfil seleccionado
data = perfiles[perfil_seleccionado]["datos"]

# Ajuste de parámetros
st.subheader("📊 Ajuste de Parámetros")
col1, col2 = st.columns(2)

with col1:
    income = st.slider("Ingresos", 0.0, 1.0, data["income"], step=0.1)
    name_email_similarity = st.slider("Similitud Nombre-Email", 0.0, 1.0, data["name_email_similarity"], step=0.01)
    customer_age = st.number_input("Edad del Cliente", 18, 100, data["customer_age"])
    proposed_credit_limit = st.number_input("Límite de Crédito Propuesto", 0, 1000000, data["proposed_credit_limit"])

with col2:
    velocity_6h = st.number_input("Velocidad Transacción en 6h", 0, 10000, data["velocity_6h"])
    velocity_24h = st.number_input("Velocidad Transacción en 24h", 0, 10000, data["velocity_24h"])
    credit_risk_score = st.number_input("Puntuación de Riesgo Crediticio", 0, 1000, data["credit_risk_score"])

# Preparar datos de entrada
input_data = pd.DataFrame([{
    "income": income, "name_email_similarity": name_email_similarity, "customer_age": customer_age,
    "velocity_6h": velocity_6h, "velocity_24h": velocity_24h, "credit_risk_score": credit_risk_score,
    "proposed_credit_limit": proposed_credit_limit
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

# 📊 **Mostrar Gráfica del Perfil del Cliente**
st.subheader("📊 Comparación con Datos de Fraude/No Fraude")

# Extraer promedios
stats_fraud_mean = stats_fraud.mean()
stats_no_fraud_mean = stats_no_fraud.mean()

# Normalizar datos para comparación (convertir a porcentaje del máximo)
max_values = np.maximum(stats_fraud_mean, stats_no_fraud_mean)
input_normalized = input_data.iloc[0] / max_values
fraud_normalized = stats_fraud_mean / max_values
no_fraud_normalized = stats_no_fraud_mean / max_values

# Crear gráfico de radar
labels = list(input_data.columns)
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

ax.fill(angles, fraud_normalized, color="red", alpha=0.25, label="Fraude Promedio")
ax.fill(angles, no_fraud_normalized, color="green", alpha=0.25, label="No Fraude Promedio")
ax.fill(angles, input_normalized, color="blue", alpha=0.5, label="Usuario Actual")

ax.set_xticks(angles)
ax.set_xticklabels(labels, fontsize=8)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

st.pyplot(fig)
