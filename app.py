import streamlit as st
import joblib
import gzip
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Cargar el modelo
MODEL_PATH = "modelo_RandomForest_optimizado.pkl.gz"
SCALER_PATH = "scaler.pkl.gz"  # Archivo con el scaler usado en el preprocesamiento

def cargar_modelo(ruta):
    with gzip.open(ruta, "rb") as f:
        return joblib.load(f)

def cargar_scaler(ruta):
    with gzip.open(ruta, "rb") as f:
        return joblib.load(f)

# Cargar modelo y scaler
modelo = cargar_modelo(MODEL_PATH)
scaler = cargar_scaler(SCALER_PATH)

# Verificar si el modelo cargado es un RandomForestClassifier
if modelo is None:
    st.error("No se pudo cargar el modelo.")
    st.stop()

# Título
st.title("🔍 Predicción de Fraude Bancario")

# 📌 Ajuste de parámetros
st.header("📊 Ajuste de Parámetros")

# Ingreso con valores realistas
income = st.slider("Ingresos ($)", min_value=500, max_value=100000, value=5000, step=100)

# Otros inputs (solo ejemplos)
name_email_similarity = st.slider("Similitud Nombre-Email", 0.0, 1.0, 0.5, step=0.01)
customer_age = st.slider("Edad del Cliente", 18, 90, 30)
proposed_credit_limit = st.slider("Límite de Crédito Propuesto ($)", 500, 50000, 10000, step=500)
velocity_6h = st.slider("Velocidad de Transacción (6h)", 10, 10000, 100, step=50)

# Crear DataFrame con los valores originales
data_df = pd.DataFrame([[income, name_email_similarity, customer_age, proposed_credit_limit, velocity_6h]], 
                        columns=["income", "name_email_similarity", "customer_age", "proposed_credit_limit", "velocity_6h"])

# Aplicar la transformación de escalado antes de predecir
data_scaled = scaler.transform(data_df)

# Botón para predecir
if st.button("🚀 Predecir Fraude"):
    prediction = modelo.predict(data_scaled)[0]
    resultado = "❌ Fraude" if prediction == 1 else "✅ No Fraude"
    st.success(f"🔮 Predicción: {resultado}")
