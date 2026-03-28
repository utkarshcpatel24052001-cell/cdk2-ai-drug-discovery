from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import gdown
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Draw, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold

# =========================
# 1. SCIENTIFIC UI CONFIG
# =========================
st.set_page_config(page_title="ChemBLitz Pro | CDK2 Diagnostic", layout="wide")

# Professional LIMS Styling: Balanced 18px Times New Roman
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
    [data-testid="stMetricLabel"] {
        font-size: 15px !important;
        font-weight: bold !important;
        text-transform: uppercase;
    }
    .stTabs [data-baseweb="tab"] { 
        font-size: 20px !important; 
        font-weight: 600 !important;
    }
    .priority-high { color: #2f855a; font-weight: bold; }
    .priority-med { color: #c05621; font-weight: bold; }
    .priority-low { color: #718096; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# =========================
# 2. PATHS & ASSETS
# =========================
PROJECT = Path(__file__).resolve().parent
DATA_PATH = PROJECT / "cdk2_pic50_clean.parquet"
MODEL_PATH = PROJECT / "cdk2_rf_final_all_data.joblib"
MODEL_DRIVE_FILE_ID = "1pOgZVHG7BfrcXE7ZmHJNM9CMnsBNlCQa"

FP_RADIUS = 2
FP_NBITS = 2048

# =========================
# 3. CORE ANALYTICAL ENGINES
# =========================
def download_model(file_id: str, dest: Path):
    gdown.download(id=file_id, output=str(dest), quiet=False)

@st.cache_resource
def get_pains_filter():
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)

@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists():
        download_model(MODEL_DRIVE_FILE_ID, MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    data = pd.read_parquet(DATA_PATH)
    data["pic50"] = pd.to_numeric(data["pic50"], errors="coerce")
    data["n_measurements"] = pd.to_numeric(data["n_measurements"], errors="coerce").fillna(0).astype(int)
    return model, data

def mol_to_png(mol: Chem.Mol):
    img = Draw.MolToImage(mol, size=(500, 350))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def keep_largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags: return mol
    return sorted(frags, key=lambda m: m.GetNumHeavyAtoms(), reverse=True)[0]

def rf_predict(model, mol: Chem.Mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)
    arr = np.zeros((FP_NBITS,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    X = arr.reshape(1, -1)
    preds = np.array([t.predict(X)[0] for t in model.estimators_], dtype=float)
    return float(preds.mean()), float(preds.std(ddof=1)), fp

def calculate_le(pic50: float, mol: Chem.Mol) -> float:
    n_heavy = mol.GetNumHeavyAtoms()
    return (1.37 * pic50 / n_heavy) if n_heavy > 0 else 0.0

def check_pains(mol: Chem.Mol, catalog: FilterCatalog) -> list[str]:
    entries = catalog.GetMatches(mol)
    return [e.GetDescription() for e in entries]

@st.cache_resource
def build_dataset_fps(df: pd.DataFrame):
    fps, idx = [], []
    for i, s in enumerate(df["smiles"].astype(str).tolist()):
        m = Chem.MolFromSmiles(s)
        if m:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(m, FP_RADIUS, nBits=FP_NBITS))
            idx.append(i)
    return fps, np.array(idx, dtype=int)

# =========================
# 4. DECISION LOGIC
# =========================
def make_decision_summary(pred_p, pred_s, max_sim, pains, le, mw, clogp):
    rationale, next_steps = [], []
    rationale.append(f"Predicted Potency: {pred_p:.2f} pIC50 (~{10**(9-pred_p):.1f} nM).")
    rationale.append(f"Model Reliability: {'High' if pred_s <= 0.4 else 'Low' if pred_s > 0.7 else 'Moderate'}.")
    
    if le >= 0.3: rationale.append(f"Ligand Efficiency: {le:.2f} (Target reached).")
    if pains: next_steps.append("⚠️ PAINS alert: High risk of non-specific binding.")
    if max_sim < 0.35: next_steps.append("⚠️ Novel Chemotype: Domain extrapolation detected.")
    
    score = (2 if pred_p >= 7 else 1 if pred_p >= 6 else 0) - (2 if pains else 0)
    priority = "High" if score >= 2 else "Medium" if score >= 1 else "Low"
    return priority, rationale, next_steps

# =========================
# 5. UI LAYOUT
# =========================
st.title("🧪 CDK2 Pharmacological Diagnostic Suite")
st.caption("Industrial-grade Predictive Engine for Cyclin-Dependent Kinase 2 (CDK2) Research")

model, df = load_assets()
pains_catalog = get_pains_filter()

with st.sidebar:
    st.header("LIMS Settings")
    mode = st.radio("Operating Mode", ["Single Molecule", "Batch Scoring"])
    st.divider()
    p_range = st.slider("pIC50 Inclusion", float(df.pic50.min()), float(df.pic50.max()), (6.0, float(df.pic50.max())))
    df_f = df[(df.pic50 >= p_range[0]) & (df.pic50 <= p_range[1])].copy()

t1, t2, t3 = st.tabs(["🧬 Diagnostic Lead", "📂 Batch Analysis", "📖 Methodology"])

with t1:
    c_in, c_res = st.columns([1, 1.5], gap="large")
    with c_in:
        st.subheader("I. Query Input")
        q_smiles = st.text_input("Target SMILES", value="Cc1cc(Nc2nc(N)nc(N3CCCCC3)n2)no1")
        execute = st.button("Execute Diagnostic Analysis", type="primary", use_container_width=True)
        st.info(f"Available Evidence Base: {len(df_f)} Compounds")

    with c_res:
        if execute and q_smiles:
            mol = Chem.MolFromSmiles(q_smiles)
            if mol:
                mol = keep_largest_fragment(mol)
                p_m, p_s, fp = rf_predict(model, mol)
                le = calculate_le(p_m, mol)
                pains = check_pains(mol, pains_catalog)
                
                fps, _ = build_dataset_fps(df_f)
                sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, fps))
                max_s = sims.max() if len(sims) > 0 else 0.0

                priority, rationale, next_steps = make_decision_summary(p_m, p_s, max_s, pains, le, Descriptors.MolWt(mol), Descriptors.MolLogP(mol))

                with st.container(border=True):
                    st.markdown(f"### Diagnostic Summary: <span class='priority-{priority.lower()}'>{priority} Priority</span>", unsafe_allow_html=True)
                    for r in rationale: st.write(f"• {r}")
                    for ns in next_steps: st.write(f"• {ns}")

                st.markdown("### Molecular Profile")
                m_row = st.columns(4)
                m_row[0].metric("Pred pIC50", f"{p_m:.2f}")
                m_row[1].metric("IC50 (nM)", f"{10**(9-p_m):.1f}")
                m_row[2].metric("σ (Conf.)", f"{p_s:.3f}")
                m_row[3].metric("LE", f"{le:.2f}")

                with st.expander("Detailed Chemical Intelligence"):
                    st.image(mol_to_png(mol), use_container_width=True)
                    st.markdown(f"**Murcko Scaffold:** `{Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))}`")
                    st.write(f"**MolWt:** {Descriptors.MolWt(mol):.1f} | **cLogP:** {Descriptors.MolLogP(mol):.2f} | **TPSA:** {rdMolDescriptors.CalcTPSA(mol):.1f}")
            else:
                st.error("Invalid SMILES input.")

with t3:
    st.subheader("How to Interpret These Results")
    st.markdown("""
    * **pIC50**: The logarithmic measure of potency. Every +1 increase represents a 10-fold increase in binding strength.
    * **Ligand Efficiency (LE)**: Normalizes potency by the number of heavy atoms. **LE > 0.3** is the industry standard for a promising lead.
    * **Model Confidence (σ)**: The standard deviation across the Random Forest trees. **σ < 0.4** indicates high model agreement.
    * **PAINS**: Alerts for structural motifs associated with non-specific assay interference.
    """)
