from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Optional

import gdown
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Draw, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold

# =========================
# 1. SCIENTIFIC UI CONFIG
# =========================
st.set_page_config(page_title="ChemBLitz Pro | CDK2 Diagnostic", layout="wide")

st.markdown("""
<style>
    html, body, [class*="css"], .stMarkdown, p, li, div, span {
        font-family: "Times New Roman", Times, serif !important;
        font-size: 18px !important;
        color: #1a202c;
    }
    [data-testid="stMetricValue"] {
        font-size: 30px !important;
        font-weight: 700 !important;
        color: #2c5282 !important;
    }
    .priority-high { color: #2f855a; font-weight: bold; }
    .priority-med { color: #c05621; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# =========================
# 2. ASSET MANAGEMENT
# =========================
MODEL_PATH = Path("cdk2_rf_final.joblib")
DATA_PATH = Path("cdk2_pic50_clean.parquet")
DRIVE_ID = "1pOgZVHG7BfrcXE7ZmHJNM9CMnsBNlCQa"

@st.cache_resource
def load_all_assets():
    if not MODEL_PATH.exists():
        with st.spinner("Downloading 142MB Model from Cloud Storage..."):
            gdown.download(id=DRIVE_ID, output=str(MODEL_PATH), quiet=False)
    
    if not DATA_PATH.exists():
        st.error("Critical Error: 'cdk2_pic50_clean.parquet' missing from repository.")
        st.stop()
        
    try:
        model = joblib.load(MODEL_PATH)
        data = pd.read_parquet(DATA_PATH)
        return model, data
    except Exception as e:
        st.error(f"Asset Corruption Detected: {e}. Please Reboot the App.")
        st.stop()

@st.cache_resource
def get_pains_filter():
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)

# =========================
# 3. ANALYTICAL FUNCTIONS
# =========================
def check_pains(mol, catalog):
    return [e.GetDescription() for e in catalog.GetMatches(mol)]

def rf_predict(model, mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    X = arr.reshape(1, -1)
    preds = np.array([t.predict(X)[0] for t in model.estimators_])
    return float(preds.mean()), float(preds.std(ddof=1)), fp

def calculate_le(pic50, mol):
    n_heavy = mol.GetNumHeavyAtoms()
    return (1.37 * pic50 / n_heavy) if n_heavy > 0 else 0.0

# =========================
# 4. MAIN INTERFACE
# =========================
st.title("🧪 CDK2 Pharmacological Diagnostic Suite")
st.caption("Professional QSAR Predictive Engine | University Research Standard")

model, df = load_all_assets()
pains_catalog = get_pains_filter()

t1, t2 = st.tabs(["🧬 Diagnostic Lead", "📖 Interpretation Guide"])

with t1:
    c_in, c_res = st.columns([1, 1.5], gap="large")
    with c_in:
        st.subheader("I. Query Input")
        q_smiles = st.text_input("Target SMILES", value="Cc1cc(Nc2nc(N)nc(N3CCCCC3)n2)no1")
        execute = st.button("Run Diagnostic", type="primary", use_container_width=True)

    with c_res:
        if execute and q_smiles:
            mol = Chem.MolFromSmiles(q_smiles)
            if mol:
                p_m, p_s, fp = rf_predict(model, mol)
                le = calculate_le(p_m, mol)
                pains = check_pains(mol, pains_catalog)
                
                with st.container(border=True):
                    st.markdown("### Molecular Potency & Efficiency")
                    m = st.columns(4)
                    m[0].metric("Pred pIC50", f"{p_m:.2f}")
                    m[1].metric("IC50 (nM)", f"{10**(9-p_m):.1f}")
                    m[2].metric("σ (Conf.)", f"{p_s:.3f}")
                    m[3].metric("LE", f"{le:.2f}")

                with st.container(border=True):
                    st.markdown("### Structural Risk Assessment")
                    if pains:
                        st.error(f"⚠️ PAINS Detected: {', '.join(pains)}")
                    else:
                        st.success("✅ Clean Lead: No PAINS Structural Alerts.")
                    
                    st.markdown(f"**Murcko Scaffold:** `{Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))}`")
                    st.write(f"**MW:** {Descriptors.MolWt(mol):.1f} | **LogP:** {Descriptors.MolLogP(mol):.2f} | **TPSA:** {rdMolDescriptors.CalcTPSA(mol):.1f}")
                    st.image(Draw.MolToImage(mol, size=(400, 250)), use_container_width=True)
            else:
                st.error("Invalid SMILES format.")

with t2:
    st.subheader("How to Interpret Your Diagnostic")
    st.markdown("""
    - **Ligand Efficiency (LE):** Professional leads target **LE > 0.3**.
    - **Confidence (σ):** Low values (< 0.4) indicate the model is very familiar with your structure.
    - **PAINS:** Flags "Pan-Assay Interference" risks—essential for avoiding false positives in the lab.
    """)
