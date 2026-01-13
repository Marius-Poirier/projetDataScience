import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

# --- 1. CHARGEMENT ET PRÉPARATION ---
print("Chargement des données...")
df = pd.read_csv("training_matrix_sully.csv")

features = ['er', 'ks2', 'ks3', 'ks4', 'ks_fp', 'of', 'qmax', 'tm']
targets = ['parc_chateau', 'centre_sully', 'gare_sully', 'caserne_pompiers']

X = df[features]
y = df[targets]

# On garde un test set fixe pour la vérification finale (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)

# --- 2. CONFIGURATION DE LA CROSS-VALIDATION COMMUNE ---
# C'est cet objet kf que tes camarades doivent copier/coller
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- 3. ENTRAÎNEMENT ET VALIDATION CROISÉE ---
models = {}

print("\n--- DÉBUT DE L'ÉVALUATION (CROSS-VALIDATION) ---")

for lieu in targets:
    print(f"\n🔄 Analyse pour : {lieu}")
    
    # Modèle avec tes hyperparamètres
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    
    # Lancement de la Cross-Validation sur le Train Set
    cv_results = cross_validate(
        model, X_train_scaled, y_train[lieu], 
        cv=kf, 
        scoring=['r2', 'neg_root_mean_squared_error'],
        n_jobs=-1
    )
    
    # Calcul des moyennes
    r2_cv = cv_results['test_r2'].mean()
    rmse_cv = -cv_results['test_neg_root_mean_squared_error'].mean()
    std_r2 = cv_results['test_r2'].std()

    print(f"   📊 Score R2 moyen (CV) : {r2_cv:.4f} (+/- {std_r2 * 2:.4f})")
    print(f"   📉 RMSE moyen (CV)     : {rmse_cv:.4f} m")

    # --- ENTRAÎNEMENT FINAL ---
    # On entraîne sur TOUT le train set pour la sauvegarde
    model.fit(X_train_scaled, y_train[lieu])
    
    # Test rapide sur le Test Set (indépendant de la CV)
    final_pred = model.predict(X_test_scaled)
    final_r2 = r2_score(y_test[lieu], final_pred)
    
    print(f"   ✅ Vérification Test Set : R2 = {final_r2:.4f}")
    
    # Sauvegarde
    models[lieu] = model
    joblib.dump(model, f"xgboost_{lieu}.pkl")

# --- 4. RÉSUMÉ POUR LA COMPARAISON ---
print("\n" + "="*30)
print("TABLEAU DE COMPARAISON (À donner à l'équipe)")
print("="*30)
for lieu in targets:
    # On reprend les scores calculés plus haut
    print(f"{lieu.upper():<20} | R2: {scores_r2_cv_display[lieu]:.4f} | RMSE: {scores_rmse_cv_display[lieu]:.4f}")

print("\nFichiers .pkl générés. N'oublie pas de donner ton scaler aussi s'ils en ont besoin !")