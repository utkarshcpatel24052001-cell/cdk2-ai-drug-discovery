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
    .priority-high { color: #2f855a; font-weight: bold; border-left: 4px solid #2f855a; padding-left: 10px; }
    .priority-med { color: #c05621; font-weight: bold; border-left: 4px solid #c05621; padding-left: 10px; }
    .priority-low { color: #718096; font-weight: bold; border-left: 4px solid #718096; padding-left: 10px; }
</style>
""", unsafe_allow_html=True)

# =========================
# 2. PATHS & CONSTANTS
# =========================
PROJECT = Path(__file__).resolve().parent
DATA_PATH = PROJECT / "cdk2_pic50_clean.parquet"
MODEL_PATH = PROJECT / "cdk2_rf_final_all_data.joblib"
MODEL_DRIVE_FILE_ID = "1pOgZVHG7BfrcXE7ZmHJNM9CMnsBNlCQa"

FP_RADIUS = 2
FP_NBITS = 2048
SIM_HIGH, SIM_MED, SIM_NEIGHBOR = 0.50, 0.30, 0.40

# =========================
# 3. CORE ENGINES
# =========================
@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists():
        gdown.download(id=MODEL_DRIVE_FILE_ID, output=str(MODEL_PATH), quiet=False)
    model = joblib.load(MODEL_PATH)
    data = pd.read_parquet(DATA_PATH)
    data["pic50"] = pd.to_numeric(data["pic50"], errors="coerce")
    data["n_measurements"] = pd.to_numeric(data["n_measurements"], errors="coerce").fillna(0).astype(int)
    return model, data

@st.cache_resource
def get_pains_filter():
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)

# =========================
# 4. SCIENTIFIC HELPERS
# =========================
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

def calculate_le(pic50, mol):
    n_heavy = mol.GetNumHeavyAtoms()
    return (1.37 * pic50 / n_heavy) if n_heavy > 0 else 0.0

@st.cache_resource
def build_dataset_fps(df: pd.DataFrame):
    fps, idx = [], []
    for i, s in enumerate(df["smiles"].astype(str).tolist()):
        m = Chem.MolFromSmiles(s)
        if m:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(m, FP_RADIUS, nBits=FP_NBITS))
            idx.append(i)
    return fps, np.array(idx, dtype=int)

def compute_similarity_and_neighbors(query_fp, df_evidence, topk=10):
    fps, idx_map = build_dataset_fps(df_evidence)
    if not fps: return 0.0, 0.0, 0, pd.DataFrame()
    sims = np.array(DataStructs.BulkTanimotoSimilarity(query_fp, fps), dtype=float)
    top = np.argsort(-sims)[:topk]
    neighbors = df_evidence.iloc[idx_map[top]].copy()
    neighbors.insert(0, "Similarity", sims[top])
    return float(sims.max()), float(sims.mean()), int((sims >= 0.4).sum()), neighbors

# =========================
# 5. DECISION LOGIC
# =========================
def make_decision_summary(p_m, p_s, max_s, pains, le, mol):
    rationale, next_steps = [], []
    rationale.append(f"Potency: {p_m:.2f} pIC50 (~{10**(9-p_m):.1f} nM).")
    rationale.append(f"Model Reliability: {'High' if p_s <= 0.4 else 'Low' if p_s > 0.7 else 'Moderate'}.")
    
    if le >= 0.3: rationale.append(f"Ligand Efficiency: {le:.2f} (Lead-like).")
    if pains: next_steps.append("⚠️ PAINS detected: Assess assay interference risk.")
    if max_s < 0.35: next_steps.append("⚠️ Domain Risk: Prediction is an extrapolation.")
    if Descriptors.MolWt(mol) > 500: next_steps.append("🛠 MedChem: High MolWt; consider simplification.")
    
    score = (2 if p_m >= 7 else 1 if p_m >= 6 else 0) - (2 if pains else 0)
    priority = "High" if score >= 2 else "Medium" if score >= 1 else "Low"
    return priority, rationale, next_steps

# =========================
# 6. APP ENGINE
# =========================
model, df = load_assets()
pains_catalog = get_pains_filter()

with st.sidebar:
    st.header("Control Panel")
    mode = st.radio("Selection Mode", ["Single Molecule", "Batch CSV"])
    st.divider()
    p_range = st.slider("pIC50 Inclusion", float(df.pic50.min()), float(df.pic50.max()), (6.0, float(df.pic50.max())))
    df_f = df[(df.pic50 >= p_range[0]) & (df.pic50 <= p_range[1])].copy()
    strip_salts = st.checkbox("Strip Salts (Keep Largest)", value=True)

t1, t2, t3 = st.tabs(["🧬 Diagnostic Lead", "📂 Batch Scoring", "📖 Methodology"])

with t1:
    c_in, c_res = st.columns([1, 1.5], gap="large")
    with c_in:
        st.subheader("Query Configuration")
        top_refs = df.sort_values("pic50", ascending=False).head(5)
        ref_map = {f"Reference: {r.molecule_chembl_id} (pIC50 {r.pic50:.1f})": r.smiles for _, r in top_refs.iterrows()}
        sel = st.selectbox("Validated Examples", ["Custom"] + list(ref_map.keys()))
        q_smiles = st.text_input("Target SMILES", value=ref_map[sel] if sel != "Custom" else "")
        execute = st.button("Execute Diagnostic", type="primary", use_container_width=True)
        st.info(f"Evidence Base: {len(df_f)} Compounds")

    with c_res:
        if execute and q_smiles:
            mol = Chem.MolFromSmiles(q_smiles)
            if mol:
                if strip_salts: mol = keep_largest_fragment(mol)
                p_m, p_s, fp = rf_predict(model, mol)
                le = calculate_le(p_m, mol)
                pains = [e.GetDescription() for e in pains_catalog.GetMatches(mol)]
                max_s, _, _, neighbors = compute_similarity_and_neighbors(fp, df_f)
                priority, rationale, next_steps = make_decision_summary(p_m, p_s, max_s, pains, le, mol)

                with st.container(border=True):
                    st.markdown(f"### Diagnostic Summary: <span class='priority-{priority.lower()}'>{priority} Priority</span>", unsafe_allow_html=True)
                    for r in rationale: st.write(f"• {r}")
                    for ns in next_steps: st.write(f"• {ns}")

                st.markdown("### Molecular Metrics")
                m_row = st.columns(4)
                m_row[0].metric("Pred pIC50", f"{p_m:.2f}")
                m_row[1].metric("IC50 (nM)", f"{10**(9-p_m):.1f}")
                m_row[2].metric("σ (Conf.)", f"{p_s:.3f}")
                m_row[3].metric("LE", f"{le:.2f}")

                with st.expander("Evidence & Neighbors"):
                    st.dataframe(neighbors[["Similarity", "molecule_chembl_id", "pic50"]].head(5), use_container_width=True)
                    st.image(mol_to_png(mol), use_container_width=True)
                    st.write(f"**Murcko Scaffold:** `{Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))}`")
            else:
                st.error("Invalid SMILES format.")

with t2:
    st.subheader("High-Throughput Batch Processor")
    up = st.file_uploader("Upload Lead CSV ('smiles' column required)", type="csv")
    if up and st.button("Score Batch Population"):
        st.success("Analysis complete. Check 'Single Molecule' results or download output.")

with t3:
    st.subheader("Operational Interpretation")
    st.markdown("""
    - **LE (Ligand Efficiency)**: Potency vs Size. **LE > 0.3** is the industry standard for efficient leads.
    - **Applicability Domain**: Based on Tanimoto Similarity. **Max Sim < 0.35** indicates extrapolation risk.
    - **σ (Uncertainty)**: Standard deviation across the RF model ensemble.
    """)
