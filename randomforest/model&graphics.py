import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# --- 1. SETUP & LOADING ---
print("Loading data...")
# Load the clean matrix we created earlier
df = pd.read_csv('training_matrix_sully.csv')

input_cols = ['er','ks2','ks3','ks4','ks_fp','of','qmax','tm']
target_cols = ['parc_chateau', 'centre_sully', 'gare_sully', 'caserne_pompiers']

X = df[input_cols]
y = df[target_cols]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. TRAINING THE MODEL ---
print("Training Random Forest (Multi-Output)...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'sully_flood_model_final.pkl')
print("Model saved as 'sully_flood_model_final.pkl'")
print(f"Global Accuracy (R²): {model.score(X_test, y_test):.4f}")

# --- 3. GRAPHICS GENERATION ---

# GRAPH A: GLOBAL FEATURE IMPORTANCE (The "Big Picture")
plt.figure(figsize=(10, 6))
importances = pd.Series(model.feature_importances_, index=input_cols).sort_values(ascending=False)
importances.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Global Factor: What drives flooding in Sully?")
plt.ylabel("Importance Score")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('graph1_global_importance.png')
print("Saved graph1_global_importance.png")
plt.show()

# GRAPH B: ACCURACY CHECK (2x2 Grid)
predictions = model.predict(X_test)
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for i, loc in enumerate(target_cols):
    ax = axes[i]
    y_real = y_test.iloc[:, i]
    y_pred = predictions[:, i]
    
    ax.scatter(y_real, y_pred, alpha=0.5, color='purple')
    # Perfect diagonal line
    m_max = max(y_real.max(), y_pred.max())
    ax.plot([0, m_max], [0, m_max], 'r--', lw=2, label='Perfect')
    
    ax.set_title(f"Accuracy: {loc}")
    ax.set_xlabel("Real (m)")
    ax.set_ylabel("Predicted (m)")
    ax.grid(True)

plt.tight_layout()
plt.savefig('graph2_accuracy_grid.png')
print("Saved graph2_accuracy_grid.png")
plt.show()

# GRAPH C: PHYSICS / VULNERABILITY (Qmax vs Water Level)
plt.figure(figsize=(12, 7))
colors = ['teal', 'orange', 'green', 'red']
for i, loc in enumerate(target_cols):
    plt.scatter(df['qmax'], df[loc], alpha=0.4, s=15, label=loc, color=colors[i])

plt.title("Vulnerability Analysis: River Flow vs. Water Levels")
plt.xlabel("River Flow Qmax (m3/s)")
plt.ylabel("Water Height (m)")
plt.legend(markerscale=3)
plt.grid(True, alpha=0.5)
plt.savefig('graph3_physics_curves.png')
print("Saved graph3_physics_curves.png")
plt.show()

# GRAPH D: IMPORTANCE PER LOCATION (Comparison)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, loc in enumerate(target_cols):
    # Train a temporary local model just to extract specific importance
    local_model = RandomForestRegressor(n_estimators=50, random_state=42)
    local_model.fit(X, df[loc])
    
    local_imp = pd.Series(local_model.feature_importances_, index=input_cols).sort_values(ascending=False)
    
    ax = axes[i]
    local_imp.plot(kind='bar', color='lightgreen', edgecolor='black', ax=ax)
    ax.set_title(f"Drivers for: {loc}")
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--')

plt.tight_layout()
plt.savefig('graph4_location_importances.png')
print("Saved graph4_location_importances.png")
plt.show()

print("\nAll Done.")