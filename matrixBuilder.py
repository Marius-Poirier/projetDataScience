import pandas as pd
import os
import re

# --- 1. CONFIGURATION ---
data_folder = 'data'
output_file = 'training_matrix_sully.csv'

# Your verified coordinates (Cartesian: X=0..63, Y=0..63 where 0 is BOTTOM)
# Format: 'Name': (Column_X, Row_Y)
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
        # Remove extension and split by '='
        name_clean = filename.replace('.csv', '')
        if '=' not in name_clean: continue
        _, values_str = name_clean.split('=')
        
        # Clean the suffix (e.g. "_maxH_sully")
        val_parts = values_str.split(',')
        val_parts[-1] = re.match(r'^-?\d+(\.\d+)?', val_parts[-1]).group(0)
        
        # Convert to floats
        inputs = [float(v) for v in val_parts]

        # B. EXTRACT OUTPUTS (From Grid Content)
        file_path = os.path.join(data_folder, filename)
        df = pd.read_csv(file_path, index_col=0) # Read grid (64x64)
        
        outputs = []
        for name, (x, y) in targets.items():
            # IMPORTANT: FLIP Y
            # Your Y=50 is North (Top), Y=11 is South (Bottom).
            # DataFrame Row 0 is Top, Row 63 is Bottom.
            # So: Matrix_Row = 63 - Coordinate_Y
            row_idx = 63 - y 
            col_idx = x
            
            # Extract value
            val = df.iloc[row_idx, col_idx]
            outputs.append(val)

        # C. Combine
        dataset.append(inputs + outputs)

    except Exception as e:
        print(f"Skipping {filename}: {e}")
        continue

# --- 3. SAVE RESULT ---
final_df = pd.DataFrame(dataset, columns=input_names + target_names)

# Sanity Check: Show me if we actually found water
print("\n--- MATRIX REPORT ---")
print(f"Total Rows: {len(final_df)}")
print("\nMax Water Levels found (check if these are > 0):")
print(final_df[target_names].max())

final_df.to_csv(output_file, index=False)
print(f"\nSuccess! Matrix saved to '{output_file}'")