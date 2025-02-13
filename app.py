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

# Definir perfiles de clientes con explicaciones
perfiles = {
    "Cliente Nuevo y Desconocido": {
        "explicacion": "Este cliente es nuevo en la plataforma y tiene un historial financiero limitado. "
                       "Es más difícil de evaluar y representa un mayor riesgo. Suele tener ingresos bajos y realizar pocas transacciones previas.",
        "datos": {
            "income": 0.3, "name_email_similarity": 0.9, "prev_address_months_count": 5,
            "current_address_months_count": 3, "customer_age": 25, "intended_balcon_amount": 500.0,
            "velocity_6h": 1000, "velocity_24h": 3000, "bank_branch_count_8w": 2,
            "date_of_birth_distinct_emails_4w": 10, "credit_risk_score": 200, "email_is_free": 1,
            "phone_home_valid": 0, "phone_mobile_valid": 1, "has_other_cards": 0,
            "proposed_credit_limit": 5000, "foreign_request": 1, "keep_alive_session": 10,
            "device_distinct_emails_8w": 5, "month": 2
        }
    }
}

# Interfaz de usuario
st.title("🔍 Predicción de Fraude Financiero")

# Selección del perfil
perfil_seleccionado = st.selectbox("Seleccione un perfil de cliente", list(perfiles.keys()))

# Mejor presentación de la descripción del perfil sin icono y con buen espaciado
st.markdown(
    f"""
    <div style="
        background-color: #2b2b2b;
        padding: 12px;
        border-radius: 8px;
        font-size: 16px;
        line-height: 1.6;
        color: white;
        margin-bottom: 15px;
    ">
        <strong>Sobre este perfil:</strong> <br>
        {perfiles[perfil_seleccionado]['explicacion']}
    </div>
    """,
    unsafe_allow_html=True
)

# Cargar los valores del perfil seleccionado
data = perfiles[perfil_seleccionado]["datos"]

# Mostrar los valores y permitir ajustes en los parámetros clave
st.subheader("📊 Ajuste de Parámetros")
col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Ingresos", min_value=0.0, max_value=1.0, step=0.1, value=data["income"])
    name_email_similarity = st.slider("Similitud Nombre-Email", 0.0, 1.0, value=data["name_email_similarity"], step=0.01)
    customer_age = st.number_input("Edad del Cliente", min_value=18, max_value=100, value=data["customer_age"])
    proposed_credit_limit = st.number_input("Límite de Crédito Propuesto", min_value=0, max_value=1000000, value=data["proposed_credit_limit"])

with col2:
    foreign_request = st.radio("¿Solicitud Extranjera?", ["No", "Sí"], index=int(data["foreign_request"]))
    email_is_free = st.radio("¿Email Gratuito?", ["No", "Sí"], index=int(data["email_is_free"]))
    has_other_cards = st.radio("¿Tiene Otras Tarjetas?", ["No", "Sí"], index=int(data["has_other_cards"]))

# Botón de predicción
if st.button("🚀 Predecir Fraude"):
    input_data = pd.DataFrame([{**data, 
                                "income": income, "name_email_similarity": name_email_similarity,
                                "customer_age": customer_age, "proposed_credit_limit": proposed_credit_limit,
                                "foreign_request": int(foreign_request == "Sí"),
                                "email_is_free": int(email_is_free == "Sí"), "has_other_cards": int(has_other_cards == "Sí")}])
    
    try:
        pred = model.predict(input_data)[0]
        resultado = "🚨 Fraude" if pred == 1 else "✅ No Fraude"
        st.success(f"🔮 **Predicción:** {resultado}")
    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
