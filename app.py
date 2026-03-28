from __future__ import annotations
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple
import gdown
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Draw, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# =========================
# 1. SCIENTIFIC UI CONFIG
# =========================
st.set_page_config(page_title="ChemBLitz Professional | CDK2 Suite", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Times+New+Roman&display=swap');
    html, body, [class*="css"], .stMarkdown, p, li, div, span {
        font-family: 'Times New Roman', Times, serif !important;
        font-size: 20px !important;
    }
    .stMetric label { font-size: 22px !important; font-weight: bold !important; color: #1f77b4 !important; }
    .stMetric div { font-size: 32px !important; font-weight: 500 !important; }
    .stButton>button { height: 3em; font-size: 20px !important; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# =========================
# 2. ASSET LOADER
# =========================
PROJECT = Path(__file__).resolve().parent
DATA_PATH = PROJECT / "cdk2_pic50_clean.parquet"
MODEL_PATH = PROJECT / "cdk2_rf_final_all_data.joblib"
MODEL_DRIVE_FILE_ID = "1pOgZVHG7BfrcXE7ZmHJNM9CMnsBNlCQa"

@st.cache_resource
def get_pains_filter():
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)

@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists():
        gdown.download(id=MODEL_DRIVE_FILE_ID, output=str(MODEL_PATH), quiet=False)
    return joblib.load(MODEL_PATH), pd.read_parquet(DATA_PATH)

# =========================
# 3. DIAGNOSTIC FUNCTIONS
# =========================
def calculate_le(pic50: float, mol: Chem.Mol) -> float:
    # Ligand Efficiency = (1.37 / HeavyAtomCount) * pIC50
    n_heavy = mol.GetNumHeavyAtoms()
    return (1.37 / n_heavy) * pic50 if n_heavy > 0 else 0.0

def rf_predict(model, mol: Chem.Mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    X = arr.reshape(1, -1)
    preds = np.array([t.predict(X)[0] for t in model.estimators_])
    return float(preds.mean()), float(preds.std())

# =========================
# 4. DASHBOARD
# =========================
st.title("🧪 CDK2 Pharmacological Diagnostic Suite")
st.markdown("#### Precision QSAR Prediction & Chemical Asset Validation")
st.divider()

model, df = load_assets()
pains_catalog = get_pains_filter()

tab1, tab2 = st.tabs(["🧬 Diagnostic Lead", "📊 Methodology"])

with tab1:
    col_input, col_results = st.columns([1, 1.4], gap="large")
    with col_input:
        st.subheader("I. Query Definition")
        target_smiles = st.text_input("Input Target SMILES:", value="Cc1cc(Nc2nc(N)nc(N3CCCCC3)n2)no1")
        execute = st.button("Execute Diagnostic Analysis", type="primary", use_container_width=True)

    with col_results:
        if execute and target_smiles:
            mol = Chem.MolFromSmiles(target_smiles)
            if mol:
                p_mean, p_std = rf_predict(model, mol)
                le = calculate_le(p_mean, mol)
                pains = pains_catalog.GetMatches(mol)
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                
                with st.container(border=True):
                    st.markdown("### A. Affinity & Efficiency")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Pred. pIC50", f"{p_mean:.2f}")
                    m2.metric("Ligand Efficiency", f"{le:.2f}")
                    m3.metric("Conf. (σ)", f"±{p_std:.3f}")
                
                with st.container(border=True):
                    st.markdown("### B. Structural Alerts")
                    if pains:
                        st.error(f"⚠️ PAINS Hits: {', '.join([e.GetDescription() for e in pains])}")
                    else:
                        st.success("✅ Clean Lead: No PAINS Structural Alerts Found.")
                    st.markdown(f"**Murcko Scaffold:** `{Chem.MolToSmiles(scaffold)}`")
            else:
                st.error("Invalid SMILES format.")
