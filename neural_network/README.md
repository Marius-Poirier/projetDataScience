# Neural Network Model - Flood Prediction

This folder contains the MLP (Multi-Layer Perceptron) implementation for the Sully flood prediction project.

## Requirements
- Python 3
- Libraries: `pandas`, `sklearn`, `joblib`
- Dataset: `../dataset_final_Sully.csv` (must be generated first)

## Usage
Run the scripts using the project's virtual environment:

### Scikit-Learn Version (Standard)
```bash
../.venv/bin/python mlp_model.py
```

### TensorFlow/Keras Version (Alternative)
```bash
../.venv/bin/python keras_model.py
```

## Model Details
- **Architecture**: 2 hidden layers (100 neurons each).
- **Solver**: Adam.
- **Protocol**: Standard strict split (80/20, random_state=42).
- **Files**:
    - `mlp_model.py`: Original Scikit-Learn implementation.
    - `keras_model.py`: TensorFlow/Keras implementation.
