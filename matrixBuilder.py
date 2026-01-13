import pandas as pd
import os
import re

# --- 1. CONFIGURATION ---
data_folder = 'data'
output_file = 'training_matrix_sully.csv'

# COORDONNEES VALIDEES PAR R (Ligne = Index R, Colonne = Index R)
# On ne parle plus de X/Y pour ne pas s'embrouiller.
targets = {
    'parc_chateau':     (27, 50),
    'centre_sully':     (18, 42),
    'gare_sully':       (16, 28),
    'caserne_pompiers': (12, 11)
}

# --- 2. PROCESSING ---
print(f"Scanning '{data_folder}' for CSV files...")
files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
print(f"Found {len(files)} files. Building matrix...")

dataset = []
input_names = ['er','ks2','ks3','ks4','ks_fp','of','qmax','tm']
target_names = list(targets.keys())

for filename in files:
    try:
        # A. EXTRACT INPUTS (From Filename)
        name_clean = filename.replace('.csv', '')
        if '=' not in name_clean: continue
        _, values_str = name_clean.split('=')
        
        val_parts = values_str.split(',')
        val_parts[-1] = re.match(r'^-?\d+(\.\d+)?', val_parts[-1]).group(0)
        inputs = [float(v) for v in val_parts]

        # B. EXTRACT OUTPUTS (From Grid Content)
        file_path = os.path.join(data_folder, filename)
        
        # index_col=0 car la première colonne de tes CSV contient les numéros de ligne (1, 2, 3...)
        df = pd.read_csv(file_path, index_col=0) 
        
        outputs = []
        for name, (ligne_r, col_r) in targets.items():
            
            row_idx = ligne_r - 1
            col_idx = col_r - 1
            
            # Extraction simple
            val = df.iloc[row_idx, col_idx]
            outputs.append(val)

        # C. Combine
        dataset.append(inputs + outputs)

    except Exception as e:
        print(f"Skipping {filename}: {e}")
        continue

# --- 3. SAVE RESULT ---
final_df = pd.DataFrame(dataset, columns=input_names + target_names)

# Vérification finale
print("\n--- MATRIX REPORT ---")
print(f"Total Rows: {len(final_df)}")
print("\nMax Water Levels found (Doit être > 0 partout !):")
print(final_df[target_names].max())

final_df.to_csv(output_file, index=False)
print(f"\nSuccess! Matrix saved to '{output_file}'")