import streamlit as st
import joblib
import gzip
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 🚀 **Configuración inicial de la app**
st.set_page_config(page_title="Detección de Fraude Bancario", page_icon="💰", layout="wide")

# 📌 **Diccionario de Perfiles de Cliente**
perfiles_clientes = {
    "Cliente Nuevo": {
        "income": 3000, "name_email_similarity": 0.3, "customer_age": 25,
        "proposed_credit_limit": 5000, "velocity_6h": 200
    },
    "Cliente Recurrente": {
        "income": 7000, "name_email_similarity": 0.6, "customer_age": 40,
        "proposed_credit_limit": 15000, "velocity_6h": 400
    },
    "Cliente VIP": {
        "income": 20000, "name_email_similarity": 0.9, "customer_age": 55,
        "proposed_credit_limit": 50000, "velocity_6h": 800
    }
}

# 📌 **Media y Desviación Estándar para Escalar Variables**
scaler_means = {"income": 5000, "name_email_similarity": 0.5, "customer_age": 35, 
                "proposed_credit_limit": 10000, "velocity_6h": 500}
scaler_stds = {"income": 2000, "name_email_similarity": 0.2, "customer_age": 10, 
               "proposed_credit_limit": 5000, "velocity_6h": 1000}

# 📌 **Función para escalar la entrada**
def scale_input(data):
    for feature in data.columns:
        if feature in scaler_means:
            data[feature] = (data[feature] - scaler_means[feature]) / scaler_stds[feature]
    return data

# 📌 **Función para cargar el modelo**
def cargar_modelo(ruta):
    with gzip.open(ruta, "rb") as f:
        modelo = joblib.load(f)
    return modelo

# 🔥 **Cargar el modelo de detección de fraude**
MODEL_PATH = "modelo_RandomForest_optimizado.pkl.gz"
try:
    model = cargar_modelo(MODEL_PATH)
    st.success("✅ Modelo cargado correctamente.")
except Exception as e:
    st.error(f"Error al cargar el modelo: {str(e)}")
    st.stop()

# 📌 **Interfaz en Streamlit**
st.sidebar.title("📌 Configuración de Cliente")
perfil_seleccionado = st.sidebar.selectbox("Selecciona un perfil", list(perfiles_clientes.keys()))

# 📌 **Mostrar inputs con valores predefinidos según perfil**
st.title("🔍 Introducir Datos de Transacción")
st.subheader("Ajuste de Parámetros")

# Obtener valores del perfil seleccionado
perfil = perfiles_clientes[perfil_seleccionado]

# **Inputs con valores precargados del perfil**
income = st.number_input("Ingresos", min_value=0.0, max_value=50000.0, value=perfil["income"], step=500.0)
name_email_similarity = st.slider("Similitud Nombre-Email", 0.0, 1.0, perfil["name_email_similarity"], step=0.01)
customer_age = st.number_input("Edad del Cliente", min_value=18, max_value=100, value=perfil["customer_age"], step=1)
proposed_credit_limit = st.number_input("Límite de Crédito Propuesto", min_value=0.0, max_value=100000.0, 
                                        value=perfil["proposed_credit_limit"], step=500.0)
velocity_6h = st.number_input("Velocidad de Transacción (6h)", min_value=0.0, max_value=10000.0, 
                              value=perfil["velocity_6h"], step=50.0)

# 📌 **Predicción**
if st.button("🚀 Predecir Fraude"):
    # Crear DataFrame con los datos del usuario
    data_df = pd.DataFrame([[income, name_email_similarity, customer_age, 
                              proposed_credit_limit, velocity_6h]], 
                           columns=["income", "name_email_similarity", "customer_age", 
                                    "proposed_credit_limit", "velocity_6h"])

    # Escalar los datos
    data_scaled = scale_input(data_df)

    # Realizar la predicción
    try:
        prediction = model.predict(data_scaled)[0]
        resultado = "Fraude" if prediction == 1 else "No Fraude"
        st.success(f"🔮 **Predicción:** {resultado}")
    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
