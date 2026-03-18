from pathlib import Path

project = Path(r"C:\Users\utkar\chemblitz_project")
(project / "README.md").write_text(
"""# ChemBLitz — CDK2 pIC50 Scientific Predictor

A scientific Streamlit web app to predict **CDK2 inhibitory potency (pIC50)** from **SMILES**, with validation and interpretability features.
here is the link of cleaned data : -https://drive.google.com/file/d/1pOgZVHG7BfrcXE7ZmHJNM9CMnsBNlCQa/view?usp=drive_link
## What this app provides

### 1) Prediction (QSAR)
- Input: **SMILES**
- Output: **Predicted pIC50** (higher = stronger inhibition)
- Also shows **Predicted IC50 (nM)** converted from pIC50

### 2) Scientific validation (important)
- **Uncertainty proxy:** standard deviation across RandomForest trees  
- **Applicability domain:** Tanimoto similarity to the training dataset  
- **Nearest-neighbor evidence:** shows top similar dataset compounds and their experimental pIC50

### 3) PhysChem + ADMET-like heuristics (rule-based)
- RDKit descriptors: MolWt, cLogP, TPSA, HBD/HBA, Rotatable Bonds, Rings, QED
- Lipinski Ro5 violations + Veber rule check
- A final summary + improvement suggestions (e.g., reduce LogP, reduce flexibility, etc.)

### 4) Dataset dashboard
- pIC50 distribution
- log10(IC50 nM) distribution
- measurement count distribution (`n_measurements`)

---

## Methodology (high level)

- Features: **Morgan fingerprints** (radius=2, 2048 bits)
- Model: **RandomForest regression**
- Target: **CDK2** (Cyclin-dependent kinase 2)
- Endpoint: IC50 → pIC50 conversion

---

## How to run locally

### Install dependencies
```bash
pip install -r requirements.txt