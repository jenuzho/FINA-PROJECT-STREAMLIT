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
except Exception as e:
    st.error(f"Error al cargar el modelo: {str(e)}")
    st.stop()

# Verificar si el modelo es un RandomForestClassifier
if not isinstance(model, RandomForestClassifier):
    st.error("El archivo cargado no es un modelo RandomForest.")
    st.stop()

# Interfaz de usuario
st.title("🔍 Predicción de Fraude Financiero")

# Botón de predicción
if st.button("🚀 Predecir Fraude"):
    try:
        pred = model.predict(pd.DataFrame())[0]  # Se requiere entrada de datos
        resultado = "🚨 Fraude" if pred == 1 else "✅ No Fraude"
        st.success(f"🔮 **Predicción:** {resultado}")
    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
