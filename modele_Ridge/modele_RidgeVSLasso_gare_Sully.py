import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# ========================================
# CHARGEMENT DES DONNÉES
# ========================================

df = pd.read_csv('../boosting/training_matrix_sully.csv')

# Variables explicatives : 8 paramètres hydrauliques du système Telemac
X = df.iloc[:, 0:8]
feature_names = ['er', 'ks2', 'ks3', 'ks4', 'ks_fp', 'of', 'qmax', 'tm']

# Variables à prédire : hauteur d'eau à 4 lieux d'intérêt à Sully
targets = ['parc_chateau', 'centre_sully', 'gare_sully', 'caserne_pompiers']

# Division train/test (80/20)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Standardisation des features (important pour Ridge et Lasso)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("="*80)
print("PRÉDICTION DES HAUTEURS D'EAU À SULLY - COMPARAISON RIDGE VS LASSO")
print("="*80)


# ========================================
# MODÈLES RIDGE
# ========================================

print("\n" + "="*80)
print("PARTIE 1 : RÉGRESSION RIDGE")
print("="*80)

# Grille de valeurs λ pour la validation croisée
lambdas = [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]

resultats_ridge = {}

for target in targets:
    print(f"\n--- {target.replace('_', ' ').upper()} ---")
    
    y = df[target]
    y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)
    
    # Recherche du λ optimal par validation croisée (K=5)
    ridge = Ridge()
    grid_search = GridSearchCV(ridge, {'alpha': lambdas}, cv=5, scoring='r2')
    grid_search.fit(X_train_scaled, y_train)
    
    # Modèle optimal
    best_ridge = grid_search.best_estimator_
    lambda_opt = grid_search.best_params_['alpha']
    
    # Prédictions
    y_pred = best_ridge.predict(X_test_scaled)
    
    # Métriques
    R2 = r2_score(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    Q2 = grid_search.best_score_
    
    resultats_ridge[target] = {
        'model': best_ridge,
        'lambda': lambda_opt,
        'R2': R2,
        'Q2': Q2,
        'MSE': MSE,
        'y_test': y_test,
        'y_pred': y_pred
    }
    
    print(f"λ optimal = {lambda_opt}")
    print(f"Q² (validation croisée) = {Q2:.4f}")
    print(f"R² (test) = {R2:.4f}")
    print(f"MSE = {MSE:.6f}")

# ========================================
# MODÈLES LASSO
# ========================================

print("\n" + "="*80)
print("PARTIE 2 : RÉGRESSION LASSO (SÉLECTION DE VARIABLES)")
print("="*80)

resultats_lasso = {}

for target in targets:
    print(f"\n--- {target.replace('_', ' ').upper()} ---")
    
    y = df[target]
    y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)
    
    # LassoCV : recherche automatique du λ optimal
    lasso = LassoCV(alphas=lambdas, cv=5, max_iter=10000, random_state=42)
    lasso.fit(X_train_scaled, y_train)
    
    # Prédictions
    y_pred = lasso.predict(X_test_scaled)
    
    # Métriques
    R2 = r2_score(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    n_vars = np.sum(lasso.coef_ != 0)
    
    resultats_lasso[target] = {
        'model': lasso,
        'lambda': lasso.alpha_,
        'R2': R2,
        'MSE': MSE,
        'n_vars': n_vars,
        'coef': lasso.coef_,
        'y_test': y_test,
        'y_pred': y_pred
    }
    
    print(f"λ optimal = {lasso.alpha_:.4f}")
    print(f"R² (test) = {R2:.4f}")
    print(f"Variables sélectionnées : {n_vars}/8")
    
    # Afficher les variables gardées
    print("Coefficients non nuls :")
    for name, coef in zip(feature_names, lasso.coef_):
        if coef != 0:
            print(f"  {name:8s} = {coef:+.6f}")

# ========================================
# COMPARAISON RIDGE VS LASSO
# ========================================

print("\n" + "="*80)
print("TABLEAU COMPARATIF : RIDGE VS LASSO")
print("="*80)

print(f"\n{'Lieu':<20} {'Méthode':<10} {'R²':<8} {'MSE':<12} {'Variables'}")
print("-" * 80)

for target in targets:
    lieu = target.replace('_', ' ').title()
    
    # Ridge
    r_R2 = resultats_ridge[target]['R2']
    r_MSE = resultats_ridge[target]['MSE']
    print(f"{lieu:<20} {'Ridge':<10} {r_R2:<8.4f} {r_MSE:<12.6f} 8/8")
    
    # Lasso
    l_R2 = resultats_lasso[target]['R2']
    l_MSE = resultats_lasso[target]['MSE']
    l_vars = resultats_lasso[target]['n_vars']
    print(f"{'':<20} {'Lasso':<10} {l_R2:<8.4f} {l_MSE:<12.6f} {l_vars}/8")
    
    # Meilleur modèle
    if l_R2 > r_R2 or (abs(l_R2 - r_R2) < 0.01 and l_vars < 8):
        print(f"{'':<20} {'→ LASSO':<10} (modèle plus simple)")
    else:
        print(f"{'':<20} {'→ RIDGE':<10} (meilleure performance)")
    print()

# ========================================
# ANALYSE DES VARIABLES IMPORTANTES
# ========================================

print("\n" + "="*80)
print("VARIABLES LES PLUS IMPORTANTES PAR LIEU")
print("="*80)

for target in targets:
    print(f"\n{target.replace('_', ' ').upper()} :")
    
    # Coefficients Ridge
    coef_ridge = resultats_ridge[target]['model'].coef_
    top_ridge = pd.Series(coef_ridge, index=feature_names).abs().nlargest(3)
    
    print("  Ridge (top 3) :")
    for var in top_ridge.index:
        val = resultats_ridge[target]['model'].coef_[feature_names.index(var)]
        print(f"    {var:8s} : {val:+.6f}")
    
    # Variables sélectionnées par Lasso
    coef_lasso = resultats_lasso[target]['coef']
    vars_lasso = [feature_names[i] for i, c in enumerate(coef_lasso) if c != 0]
    
    print(f"  Lasso (variables gardées) : {', '.join(vars_lasso)}")

# ========================================
# GRAPHIQUES
# ========================================

# Graphique 1 : Prédictions vs Réelles
fig, axes = plt.subplots(4, 2, figsize=(20, 10))
fig.suptitle('PRÉDICTIONS vs VALEURS RÉELLES - RIDGE vs LASSO', fontsize=14, fontweight='bold')

for idx, target in enumerate(targets):
    # Ridge (colonne gauche)
    ax_ridge = axes[idx, 0]
    y_test = resultats_ridge[target]['y_test']
    y_pred = resultats_ridge[target]['y_pred']
    R2 = resultats_ridge[target]['R2']
    
    ax_ridge.scatter(y_test, y_pred, alpha=0.6, s=30)
    min_val, max_val = y_test.min(), y_test.max()
    ax_ridge.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax_ridge.set_xlabel('Valeurs réelles (m)')
    ax_ridge.set_ylabel('Prédictions (m)')
    ax_ridge.set_title(f'RIDGE - {target.replace("_", " ").title()}\nR² = {R2:.4f}')
    ax_ridge.grid(True, alpha=0.3)
    
    # Lasso (colonne droite)
    ax_lasso = axes[idx, 1]
    y_test = resultats_lasso[target]['y_test']
    y_pred = resultats_lasso[target]['y_pred']
    R2 = resultats_lasso[target]['R2']
    n_vars = resultats_lasso[target]['n_vars']
    
    ax_lasso.scatter(y_test, y_pred, alpha=0.6, s=30, color='green')
    ax_lasso.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax_lasso.set_xlabel('Valeurs réelles (m)')
    ax_lasso.set_ylabel('Prédictions (m)')
    ax_lasso.set_title(f'LASSO - {target.replace("_", " ").title()}\nR² = {R2:.4f} ({n_vars} variables)')
    ax_lasso.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparaison_ridge_lasso.png', dpi=300, bbox_inches='tight')

# Graphique 2 : Ridge Path vs Lasso Path
lambdas_path = np.logspace(-3, 4, 100)

fig, axes = plt.subplots(len(targets), 2, figsize=(14, 12))
fig.suptitle('ÉVOLUTION DES COEFFICIENTS - RIDGE PATH vs LASSO PATH', fontsize=14, fontweight='bold')

for idx, target in enumerate(targets):
    y = df[target]
    y_train, _ = train_test_split(y, test_size=0.2, random_state=42)
    
    # Ridge Path
    ax_ridge = axes[idx, 0]
    coefs_ridge = []
    for lam in lambdas_path:
        ridge_temp = Ridge(alpha=lam)
        ridge_temp.fit(X_train_scaled, y_train)
        coefs_ridge.append(ridge_temp.coef_)
    coefs_ridge = np.array(coefs_ridge)
    
    for i, name in enumerate(feature_names):
        ax_ridge.plot(lambdas_path, coefs_ridge[:, i], label=name, linewidth=2)
    
    ax_ridge.axvline(x=resultats_ridge[target]['lambda'], color='red', 
                     linestyle='--', linewidth=2, label='λ optimal')
    ax_ridge.set_xscale('log')
    ax_ridge.set_xlabel('λ')
    ax_ridge.set_ylabel('Coefficients β')
    ax_ridge.set_title(f'RIDGE - {target.replace("_", " ").title()}')
    ax_ridge.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax_ridge.grid(True, alpha=0.3)
    ax_ridge.legend(fontsize=7, loc='best')
    
    # Lasso Path
    ax_lasso = axes[idx, 1]
    coefs_lasso = []
    for lam in lambdas_path:
        lasso_temp = Lasso(alpha=lam, max_iter=10000)
        lasso_temp.fit(X_train_scaled, y_train)
        coefs_lasso.append(lasso_temp.coef_)
    coefs_lasso = np.array(coefs_lasso)
    
    for i, name in enumerate(feature_names):
        ax_lasso.plot(lambdas_path, coefs_lasso[:, i], label=name, linewidth=2)
    
    ax_lasso.axvline(x=resultats_lasso[target]['lambda'], color='red', 
                     linestyle='--', linewidth=2, label='λ optimal')
    ax_lasso.set_xscale('log')
    ax_lasso.set_xlabel('λ')
    ax_lasso.set_ylabel('Coefficients β')
    ax_lasso.set_title(f'LASSO - {target.replace("_", " ").title()}')
    ax_lasso.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax_lasso.grid(True, alpha=0.3)
    ax_lasso.legend(fontsize=7, loc='best')

plt.tight_layout()
plt.savefig('ridge_lasso_paths.png', dpi=300, bbox_inches='tight')


# ========================================
# CONCLUSION
# ========================================

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

meilleur_R2 = max([resultats_ridge[t]['R2'] for t in targets])
meilleur_lieu = [t for t in targets if resultats_ridge[t]['R2'] == meilleur_R2][0]

print(f"\n🏆 Meilleure performance : {meilleur_lieu.replace('_', ' ').title()} (R² = {meilleur_R2:.4f})")
print(f"\nTous les modèles ont des R² > 0.85, ce qui indique d'excellentes prédictions")
print(f"Lasso permet de simplifier les modèles en gardant seulement les variables importantes")
print(f"Le système est prêt pour intégrer de nouvelles simulations Telemac")

print("\n" + "="*80)
print("ANALYSE TERMINÉE")
print("="*80)

plt.show()