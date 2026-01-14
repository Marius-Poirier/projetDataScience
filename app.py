import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from pathlib import Path

# --- 0. PATCH CRITIQUE MAC M1/M2 (Empêche le chargement infini) ---
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Désactive GPU Nvidia (au cas où)
try:
    import tensorflow as tf
    # Force l'utilisation du CPU uniquement (évite le blocage Metal/GPU sur Mac)
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="FloodRisk AI - Sully-sur-Loire",
    page_icon="🌊",
    layout="wide"
)

# --- CSS PERSONNALISÉ ---
st.markdown("""
<style>
    .big-font {font-size:20px !important; font-weight: bold;}
    .metric-box {
        background-color: #f0f2f6;
        border-left: 5px solid #2e86c1;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def load_assets():
    targets = ['parc_chateau', 'centre_sully', 'gare_sully', 'caserne_pompiers']
    model_store = {t: {} for t in targets}
    scalers = {}

    # A. Scalers
    try: scalers['xgb'] = joblib.load("boosting/scaler.pkl")
    except: pass
    try: scalers['ridge'] = joblib.load("modele_Ridge/pickle/ridge_lasso_scaler.pkl")
    except: pass
    try: scalers['nn'] = joblib.load("neural_network/tensorflow/scaler.pkl")
    except: pass

    # B. Modèles
    # 1. XGBoost
    for t in targets:
        try: model_store[t]["XGBoost"] = joblib.load(f"boosting/xgboost_{t}.pkl")
        except: pass

    # 2. Random Forest
    rf_names = {
        'parc_chateau': 'RF_ParcChateau.pkl',
        'centre_sully': 'RF_CentreSully.pkl',
        'gare_sully': 'RF_GareSully.pkl',
        'caserne_pompiers': 'RF_CasernePompiers.pkl'
    }
    
    # Vérification du dossier 'randomforest/pickels'
    for t, filename in rf_names.items():
        try:
            path = Path("randomforest/pickels") / filename
            if path.exists():
                model_store[t]["Random Forest"] = joblib.load(path)
            else:
                print(f"Fichier RF manquant: {path}")
        except Exception as e:
            print(f"Erreur chargement RF {t}: {e}")

    # 3. Ridge & Lasso
    for t in targets:
        try:
            model_store[t]["Ridge"] = joblib.load(f"modele_Ridge/pickle/ridge/ridge_{t}.pkl")
            model_store[t]["Lasso"] = joblib.load(f"modele_Ridge/pickle/lasso/lasso_{t}.pkl")
        except: pass

    # C. Neural Net (Optimisé CPU)
    print("Tentative de chargement TensorFlow (Mode CPU)...")
    try:
        from tensorflow.keras.models import load_model
        for t in targets:
            path = f"neural_network/tensorflow/keras_model_{t}.keras"
            if os.path.exists(path):
                try: 
                    # compile=False accélère le chargement
                    model_store[t]["Neural Net"] = load_model(path, compile=False)
                    print(f"Loaded Keras: {t}")
                except Exception as e: 
                    print(f"Erreur loading Keras {t}: {e}")
    except ImportError:
        print("TensorFlow non installé.")
    except Exception as e:
        print(f"Erreur générale TensorFlow: {e}")

    return model_store, scalers, targets

# Chargement initial
with st.spinner("Chargement des modèles (Force CPU)..."):
    model_store, scalers, targets = load_assets()

# Ordre EXACT des colonnes pour la prédiction
feature_names = ['er', 'ks2', 'ks3', 'ks4', 'ks_fp', 'of', 'qmax', 'tm']

# Coordonnées GPS
poi_coords = {
    'parc_chateau': [47.7668, 2.3780],
    'centre_sully': [47.7680, 2.3750],
    'gare_sully': [47.7620, 2.3800],
    'caserne_pompiers': [47.7650, 2.3700]
}

# --- FONCTION DE PRÉDICTION OPTIMISÉE ---
def get_prediction(model_name, target, input_df):
    model = model_store.get(target, {}).get(model_name)
    if model is None: return 0.0
    
    # Scaling adapté
    X_in = input_df.copy()
    
    if model_name == "XGBoost" and 'xgb' in scalers:
        X_in = pd.DataFrame(scalers['xgb'].transform(input_df), columns=feature_names)
    elif (model_name in ["Ridge", "Lasso"]) and 'ridge' in scalers:
        X_in = pd.DataFrame(scalers['ridge'].transform(input_df), columns=feature_names)
    elif model_name == "Neural Net" and 'nn' in scalers:
        X_in = pd.DataFrame(scalers['nn'].transform(input_df), columns=feature_names)
    
    try:
        # Optimisation : appel direct pour Neural Net (évite le .predict() lent la 1ère fois)
        if model_name == "Neural Net":
            # Conversion explicite en tenseur pour éviter les retracing
            input_tensor = tf.convert_to_tensor(X_in.values, dtype=tf.float32)
            pred = model(input_tensor, training=False) # Appel direct plus rapide que .predict()
            val = float(pred.numpy()[0][0])
        else:
            # Modèles Sklearn / XGBoost
            pred = model.predict(X_in)
            val = pred.flatten()[0] if hasattr(pred, "flatten") else float(pred)
            
        return max(0.0, val)
    except Exception as e: 
        print(f"Erreur {model_name}: {e}")
        return 0.0

# --- SIDEBAR : NOUVEAUX SLIDERS (Basés sur votre CSV) ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=50)
st.sidebar.title("Paramètres de Crue")

# Basé sur : qmax=23994, tm=643497 dans votre fichier
with st.sidebar.expander("🌊 Hydraulique (Moteur)", expanded=True):
    qmax = st.sidebar.slider("Débit Qmax (m3/s)", 
                             min_value=2000.0, max_value=40000.0, value=24000.0, step=100.0, 
                             help="Débit de pointe de la crue.")
    
    tm = st.sidebar.slider("Durée Crue Tm (s)", 
                           min_value=10000.0, max_value=800000.0, value=640000.0, step=1000.0, 
                           help="Temps de montée de la crue.")
    
    of = st.sidebar.slider("Surverse (of)", -1.0, 1.0, 0.05, step=0.01)

with st.sidebar.expander("🧱 Digue (Erosion)", expanded=True):
    # Votre fichier indique er ~ 0.0008, donc très faible par défaut
    er = st.sidebar.slider("Taux Erosion (er)", 0.0, 1.0, 0.001, step=0.0001, format="%.4f")

with st.sidebar.expander("🌿 Friction Sol (Strickler)", expanded=False):
    ks_fp = st.sidebar.slider("Ks Plaine (fp)", 10.0, 60.0, 15.0)
    ks2 = st.sidebar.slider("Ks Zone 2", 10.0, 100.0, 26.0)
    ks3 = st.sidebar.slider("Ks Zone 3", 10.0, 100.0, 42.0)
    ks4 = st.sidebar.slider("Ks Zone 4", 10.0, 100.0, 20.0)

input_data = pd.DataFrame([[er, ks2, ks3, ks4, ks_fp, of, qmax, tm]], columns=feature_names)

st.sidebar.markdown("---")
model_choice = st.sidebar.selectbox("🤖 Modèle utilisé", ["XGBoost", "Random Forest", "Neural Net", "Ridge", "Lasso"], index=0)

# --- CORPS PRINCIPAL ---
st.title("🌊 Prédiction de Crues : Sully-sur-Loire")

tab_demo, tab_comparaison, tab_analyse = st.tabs(["🚀 DÉMO LIVE", "🏆 LE TOURNOI", "🔬 ANALYSE CRITIQUE"])

# === TAB 1 : LA DÉMO ===
with tab_demo:
    st.subheader(f"Simulation Temps Réel ({model_choice})")
    
    with st.spinner(f"Simulation en cours ({model_choice})..."):
        preds = {t: get_prediction(model_choice, t, input_data) for t in targets}
    
    col_map, col_kpi = st.columns([2, 1])
    
    with col_map:
        map_df = pd.DataFrame([
            {"lat": v[0], "lon": v[1], "size": (preds[k]+1)*100, "color": [255, 0, 0, 200] if preds[k]>1 else [0, 200, 0, 150]} 
            for k,v in poi_coords.items()
        ])
        st.map(map_df, latitude="lat", longitude="lon", size="size", color="color", zoom=13)

    with col_kpi:
        st.markdown("<div class='big-font'>Niveaux d'eau :</div>", unsafe_allow_html=True)
        for t, val in preds.items():
            color = "red" if val > 1.0 else "green"
            st.markdown(f"""
            <div class='metric-box'>
                <b>{t.replace('_', ' ').title()}</b><br>
                <span style='font-size: 24px; color: {color};'>{val:.2f} m</span>
            </div>
            """, unsafe_allow_html=True)

# === TAB 2 : LE TOURNOI ===
with tab_comparaison:
    st.header("🥊 Comparaison des Modèles")
    if os.path.exists("graph_comparison_5models_final.png"):
        st.image("graph_comparison_5models_final.png", caption="Réponse au Débit (Qmax) - Tous modèles", use_column_width=True)
    else:
        st.warning("Graphique comparatif manquant.")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Le Champion : XGBoost**")
        if os.path.exists("assets/graphs/1_pred_vs_actual_XGBoost.png"):
            st.image("assets/graphs/1_pred_vs_actual_XGBoost.png")
    with col2:
        st.markdown("**Le Challenger : Random Forest**")
        if os.path.exists("assets/graphs/1_pred_vs_actual_RandomForest.png"):
            st.image("assets/graphs/1_pred_vs_actual_RandomForest.png")

# === TAB 3 : ANALYSE CRITIQUE ===
with tab_analyse:
    st.header("🧠 Pourquoi ça marche ?")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Feature Importance")
        if os.path.exists("randomforest/graphs/graph1_global_importance.png"):
            st.image("randomforest/graphs/graph1_global_importance.png", caption="Le modèle a 'compris' que le débit (Qmax) est prioritaire.")
    with c2:
        st.subheader("Sécurité (Erreur vs Débit)")
        if os.path.exists("assets/graphs/2_residuals_vs_qmax_XGBoost.png"):
            st.image("assets/graphs/2_residuals_vs_qmax_XGBoost.png", caption="Le modèle ne diverge pas lors des crues extrêmes.")