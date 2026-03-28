# ChemBLitz — CDK2 Pharmacological Diagnostic Suite

Live App: https://cdk2-ai-drug-discovery-ffr8hm9thsadgagahfhjy6.streamlit.app/

ChemBLitz is a professional decision-support dashboard for **CDK2 inhibitor discovery**. It combines a **pIC50 QSAR predictor** with **chemical asset validation** (applicability domain, nearest-neighbor evidence, PAINS alerts, scaffold context, and developability flags) to help prioritize compounds for synthesis and experimental testing.

---

## What this tool does

### 1) Precision QSAR prediction (CDK2)
- Predicts **pIC50** (and converts to **IC50 nM**) for a query SMILES.
- Reports an **uncertainty proxy (σ)** from tree-to-tree disagreement in the RandomForest model.

### 2) Chemical asset validation (evidence + risk)
- **Applicability Domain (AD):** Tanimoto similarity vs the evidence subset.
- **Nearest-neighbor evidence:** Top-K similar compounds with experimental pIC50/IC50 (when available).
- **PAINS screening:** flags common assay-interference motifs.
- **Murcko scaffold context:** scaffold frequency + activity distribution + top exemplars.
- **Developability flags:** Ro5/Veber-style heuristics + property risk indicators.

### 3) Library triage (CSV)
Real-world discovery work involves scoring libraries/series, not just single molecules.
- Upload a CSV with a `smiles` column (optional `id`).
- Exports a results table + **pass_triage** flags for fast prioritization.

---

## How to interpret results (quick guide)

- **pIC50:** +1 pIC50 ≈ 10× stronger potency.
- **IC50 (nM):** computed as `10^(9 - pIC50)`.
- **σ (uncertainty):** model disagreement signal (not a calibrated confidence interval).
- **Applicability Domain:** low similarity indicates extrapolation risk.
  - max sim ≥ 0.50: in-domain  
  - 0.30–0.50: borderline  
  - < 0.30: out-of-domain risk  

---

## How to run locally

### Requirements
- Python 3.x
- RDKit available in your environment

### Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Repository structure
- `app.py` — Streamlit dashboard
- `cdk2_pic50_clean.parquet` — curated CDK2 dataset used for evidence + validation
- `cdk2_rf_final_all_data.joblib` — trained RandomForest model (downloaded/loaded by the app)
- `requirements.txt` — dependencies

---

## Disclaimer
This is a **research decision-support tool** intended for hypothesis generation and prioritization. Predictions require experimental validation and are not intended for clinical or safety-critical decisions.
