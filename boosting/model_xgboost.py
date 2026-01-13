import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib # Pour sauvegarder le modèle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# --- 1. CHARGEMENT ET PRÉPARATION (Protocole Commun) ---
print("Chargement des données...")
# Adapte le chemin si ton fichier csv n'est pas dans le même dossier
df = pd.read_csv("training_matrix_sully.csv")

# Définition des entrées (X) et sorties (Y)
features = ['er', 'ks2', 'ks3', 'ks4', 'ks_fp', 'of', 'qmax', 'tm']
targets = ['parc_chateau', 'centre_sully', 'gare_sully', 'caserne_pompiers']

X = df[features]
y = df[targets]

# SPLIT TRAIN/TEST (Le même pour tout le monde !)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SCALING (Optionnel pour XGBoost mais conseillé pour rester cohérent avec l'équipe)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# On remet en DataFrame pour garder les noms de colonnes (utile pour l'importance des variables)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=features)

# --- 2. ENTRAÎNEMENT DU BOOSTING (BOUCLE SUR LES 4 LIEUX) ---
models = {}
scores = {}

print("\n--- DÉBUT DE L'ENTRAÎNEMENT XGBOOST ---")

for lieu in targets:
    print(f"🔄 Entraînement pour : {lieu}...")
    
    # Création du modèle XGBoost
    # n_estimators = nombre d'arbres (100 est standard, tu peux monter à 500)
    # learning_rate = vitesse d'apprentissage (0.1 est standard)
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    
    # Entraînement
    model.fit(X_train_scaled, y_train[lieu])
    
    # Prédiction sur le Test set (jamais vu par le modèle)
    y_pred = model.predict(X_test_scaled)
    
    # Évaluation
    r2 = r2_score(y_test[lieu], y_pred)
    rmse = root_mean_squared_error(y_test[lieu], y_pred)
    
    print(f"   ✅ R2 Score : {r2:.4f}")
    print(f"   📉 RMSE (m) : {rmse:.4f}")
    
    # Sauvegarde dans le dictionnaire
    models[lieu] = model
    scores[lieu] = r2
    
    # Sauvegarde du fichier physique pour l'App (IMPORTANT pour Jeudi !)
    joblib.dump(model, f"xgboost_{lieu}.pkl")


# --- 3. ANALYSE : IMPORTANCE DES VARIABLES ---
print("\n--- ANALYSE DE L'IMPORTANCE DES VARIABLES ---")
for lieu_analyse in targets:
    print(f"📊 Génération du graphique d'importance pour {lieu_analyse}...")
    try:
        xgb.plot_importance(
            models[lieu_analyse],
            importance_type='weight',
            title=f"Facteurs d'impact - {lieu_analyse}"
        )
        plt.tight_layout()
        plt.show()
    except ValueError:
        print(f"   ℹ️ Importance XGBoost vide pour {lieu_analyse}. Essai par permutation...")
        r = permutation_importance(
            models[lieu_analyse],
            X_test_scaled,
            y_test[lieu_analyse],
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        importances = pd.Series(r.importances_mean, index=X_test_scaled.columns)
        if (importances.abs().sum() == 0):
            print(f"   ⚠️ Aucune importance détectée pour {lieu_analyse} (cible probablement constante).")
        else:
            importances.sort_values().plot(
                kind="barh",
                title=f"Permutation importance - {lieu_analyse}",
                figsize=(10, 6)
            )
            plt.tight_layout()
            plt.show()

print("\n--- TERMINÉ ---")
print("Tes modèles .pkl sont sauvegardés. Envoie-les à la personne qui fait l'App !")