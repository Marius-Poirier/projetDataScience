import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
st.write(sys.executable)

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Simulateur Inondation Sully", page_icon="🌊", layout="wide")

# --- TITRE ET INTRODUCTION ---
st.title("🌊 Prédiction de Crue - Sully-sur-Loire")
st.markdown("""
Cette application utilise des modèles **XGBoost** pour prédire instantanément la hauteur d'eau 
en différents points stratégiques de la ville, remplaçant le simulateur hydraulique Telemac.
""")

# --- CHARGEMENT DES MODÈLES ---
@st.cache_resource  # Pour ne pas recharger les modèles à chaque clic
def load_assets():
    targets = ['parc_chateau', 'centre_sully', 'gare_sully', 'caserne_pompiers']
    models = {}
    # Résolution robuste des chemins : charger depuis le dossier "boosting"
    base_dir = Path(__file__).parent
    models_dir = base_dir / "boosting"

    for t in targets:
        model_path = models_dir / f"xgboost_{t}.pkl"
        if not model_path.exists():
            st.error(f"❌ Modèle introuvable : {model_path}")
            st.stop()
        models[t] = joblib.load(str(model_path))
    
    # Retourner uniquement les modèles et les cibles
    return models, targets

try:
    models, targets = load_assets()
    st.success("✅ Modèles chargés avec succès.")
except FileNotFoundError:
    st.error("❌ Erreur : Fichiers .pkl introuvables. Vérifiez qu'ils sont dans le dossier.")
    st.stop()

# --- SIDEBAR : PARAMÈTRES D'ENTRÉE ---
st.sidebar.header("⚙️ Paramètres de la Crue")

# Valeurs par défaut basées sur ton CSV
def user_input_features():
    # Débit (Le facteur le plus important)
    qmax = st.sidebar.slider("Débit de pointe (Qmax)", 3000.0, 25000.0, 5500.0, step=100.0, key="qmax")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Paramètres Physiques")
    
    # Les autres variables (Strickler, etc.)
    ks_fp = st.sidebar.slider("Rugosité Lit Majeur (Ks_fp)", 5.0, 20.0, 15.0, key="ks_fp")
    ks2   = st.sidebar.slider("Rugosité Zone 2 (Ks2)", 18.0, 38.0, 28.0, key="ks2")
    ks3   = st.sidebar.slider("Rugosité Zone 3 (Ks3)", 27.0, 47.0, 37.0, key="ks3")
    ks4   = st.sidebar.slider("Rugosité Zone 4 (Ks4)", 18.0, 38.0, 28.0, key="ks4")
    er    = st.sidebar.slider("Érosion (er)", 0.0, 1.0, 0.5, key="er")
    tm    = st.sidebar.slider("Durée Montée (tm)", 2000.0, 25000.0, 14000.0, key="tm")
    of    = st.sidebar.slider("Facteur Ouvrage (of)", -0.2, 0.2, 0.0, key="of")

    # Création du DataFrame avec LES MÊMES NOMS DE COLONNES que l'entraînement
    data = {
        'er': er,
        'ks2': ks2,
        'ks3': ks3,
        'ks4': ks4,
        'ks_fp': ks_fp,
        'of': of,
        'qmax': qmax,
        'tm': tm
    }
    return pd.DataFrame(data, index=[0])

# Appel unique de la fonction user_input_features()
input_df = user_input_features()

# --- PRÉDICTIONS ---
st.subheader("📊 Résultats de la Simulation IA")

# Utilisation des données d'entrée pour les prédictions
results = {}
for t in targets:
    pred = models[t].predict(input_df)[0]
    # On évite les valeurs négatives (physiquement impossible)
    results[t] = max(0.0, pred)

# --- AFFICHAGE ---
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Hauteurs d'eau prédites (m)")
    for t in targets:
        val = results[t]
        
        # Code couleur pour l'alerte
        if val < 0.5:
            color = "green" # Sûr
        elif val < 1.5:
            color = "orange" # Risque
        else:
            color = "red" # Danger
            
        st.metric(label=t.replace("_", " ").title(), value=f"{val:.2f} m", delta_color="off")

with col2:
    # Graphique en barres
    st.markdown("### Comparaison Visuelle")
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Couleurs des barres
    colors = ['#1f77b4', '#1f77b4', '#ff7f0e', '#2ca02c'] # Bleu, Bleu, Orange, Vert
    
    bars = ax.bar([t.replace("_", "\n") for t in targets], results.values(), color=['blue', 'blue', 'orange', 'green'])
    
    ax.set_ylabel("Hauteur d'eau (m)")
    ax.set_ylim(0, 6) # Echelle fixe pour bien voir l'évolution
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval:.2f}m", ha='center', va='bottom', fontweight='bold')

    st.pyplot(fig)

# --- DEBUG (Optionnel - à retirer pour la fin) ---
with st.expander("Voir les données brutes envoyées au modèle"):
    st.write(input_df)