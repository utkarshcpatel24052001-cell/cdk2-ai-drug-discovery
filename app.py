from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import gdown
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Draw, QED, rdMolDescriptors

# =========================
# 1. STREAMLIT CONFIG
# =========================
st.set_page_config(page_title="ChemBLitz — CDK2 Predictor", layout="wide")

# =========================
# 2. PATHS & CONSTANTS
# =========================
PROJECT = Path(__file__).resolve().parent
DATA_PATH = PROJECT / "cdk2_pic50_clean.parquet"
MODEL_PATH = PROJECT / "cdk2_rf_final_all_data.joblib"
MODEL_DRIVE_FILE_ID = "1pOgZVHG7BfrcXE7ZmHJNM9CMnsBNlCQa"

FP_RADIUS = 2
FP_NBITS = 2048
SIM_HIGH = 0.50
SIM_MED = 0.30

# =========================
# 3. SECURE DOWNLOAD HANDLER
# =========================
def download_model_from_drive(file_id: str, destination: Path) -> None:
    gdown.download(id=file_id, output=str(destination), quiet=False)

if not DATA_PATH.exists():
    st.error(f"❌ Dataset not found: {DATA_PATH}")
    st.stop()

if not MODEL_PATH.exists():
    with st.spinner("Downloading Scientific Model (First boot only)..."):
        try:
            download_model_from_drive(MODEL_DRIVE_FILE_ID, MODEL_PATH)
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            st.stop()

# =========================
# 4. SCIENTIFIC HELPERS
# =========================
def pic50_to_ic50_nM(pic50: float) -> float:
    return float(10 ** (9.0 - pic50))

def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    Chem.SanitizeMol(mol)
    return mol

def draw_mol_png(mol: Chem.Mol, size=(400, 250)) -> bytes:
    img = Draw.MolToImage(mol, size=size)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def morgan_fp(mol: Chem.Mol, radius: int = FP_RADIUS, n_bits: int = FP_NBITS):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

def fp_to_numpy(fp, n_bits: int = FP_NBITS) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def compute_descriptors(mol: Chem.Mol) -> dict:
    return {
        "MolWt": float(Descriptors.MolWt(mol)), "LogP": float(Descriptors.MolLogP(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)), "HBD": int(rdMolDescriptors.CalcNumHBD(mol)),
        "HBA": int(rdMolDescriptors.CalcNumHBA(mol)), "RotB": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "Rings": int(rdMolDescriptors.CalcNumRings(mol)), "QED": float(QED.qed(mol)),
    }

def potency_bucket(pic50: float) -> str:
    if pic50 >= 8.0: return "Strong (≥ 8.0)"
    if pic50 >= 6.0: return "Moderate (6.0 - 8.0)"
    return "Weak (< 6.0)"

def improvement_suggestions(d: dict, max_sim: float) -> list[str]:
    s = []
    if d["LogP"] > 4.5: s.append("LogP high: Consider adding polar groups.")
    if d["TPSA"] > 140: s.append("TPSA high: Potential permeability issues.")
    if d["RotB"] > 10: s.append("High flexibility: Consider rigidification.")
    if max_sim < SIM_MED: s.append("Low similarity to training data: Use caution.")
    if not s: s.append("No major red flags detected.")
    return s

# =========================
# 5. DATA & MODEL LOADING
# =========================
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def build_dataset_fps(df: pd.DataFrame):
    fps, idx = [], []
    for i, s in enumerate(df["smiles"].astype(str).tolist()):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            fps.append(morgan_fp(m))
            idx.append(i)
    return fps, np.array(idx, dtype=int)

def rf_predict_with_uncertainty(model, X: np.ndarray) -> Tuple[float, float]:
    preds = np.array([t.predict(X)[0] for t in model.estimators_], dtype=float)
    return float(preds.mean()), float(preds.std(ddof=1))

def build_real_examples_from_dataset(df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    dff = df.dropna(subset=["pic50", "smiles"]).sort_values("pic50", ascending=False)
    if not dff.empty:
        out[f"Strong Binder (pIC50 {dff.iloc[0]['pic50']:.2f})"] = str(dff.iloc[0]["smiles"])
        mid = len(dff)//2
        out[f"Moderate Binder (pIC50 {dff.iloc[mid]['pic50']:.2f})"] = str(dff.iloc[mid]["smiles"])
    return out

# =========================
# 6. UI
# =========================
st.title("🧬 ChemBLitz: CDK2 pIC50 Predictor")

df = load_data()
examples = build_real_examples_from_dataset(df)

if "smiles" not in st.session_state:
    st.session_state["smiles"] = list(examples.values())[0]

tab_pred, tab_val, tab_data, tab_about = st.tabs(["Prediction", "Nearest Neighbors", "Dashboard", "Methodology"])

with tab_pred:
    col_input, col_output = st.columns([1, 1.2], gap="large")
    
    with col_input:
        st.subheader("Query Setup")
        ex = st.selectbox("Load Dataset Example:", list(examples.keys()))
        if st.button("Use Example", use_container_width=True):
            st.session_state["smiles"] = examples[ex]
            
        st.session_state["smiles"] = st.text_input("SMILES String:", value=st.session_state["smiles"])
        run = st.button("Predict Potency", type="primary", use_container_width=True)

        st.markdown("---")
        st.subheader("Dataset Filters")
        p_min, p_max = float(df["pic50"].min()), float(df["pic50"].max())
        p_range = st.slider("pIC50 Threshold", p_min, p_max, (p_min, p_max))
        min_n = st.slider("Min measurements", 1, int(df["n_measurements"].max()), 1)
        df_f = df[(df["pic50"] >= p_range[0]) & (df["pic50"] <= p_range[1]) & (df["n_measurements"] >= min_n)]

    with col_output:
        if run:
            mol = mol_from_smiles(st.session_state["smiles"])
            if mol:
                model = load_model()
                fp = morgan_fp(mol)
                X = fp_to_numpy(fp).reshape(1, -1)
                p_val, p_std = rf_predict_with_uncertainty(model, X)
                
                with st.container(border=True):
                    st.markdown("### Prediction Results")
                    c1, c2 = st.columns([1, 2])
                    c1.image(draw_mol_png(mol))
                    c2.metric("Predicted pIC50", f"{p_val:.3f}")
                    c2.metric("Predicted IC50 (nM)", f"{pic50_to_ic50_nM(p_val):,.1f}")
                    c2.write(f"**Classification:** {potency_bucket(p_val)}")
                
                with st.container(border=True):
                    st.markdown("### ADMET & Suggestions")
                    d = compute_descriptors(mol)
                    st.write(f"**MolWt:** {d['MolWt']:.1f} | **LogP:** {d['LogP']:.2f} | **QED:** {d['QED']:.2f}")
                    for s in improvement_suggestions(d, 0.5):
                        st.markdown(f"- {s}")
            else:
                st.error("Invalid SMILES.")
        else:
            st.info("Enter SMILES and click Predict.")

with tab_val:
    st.subheader("Nearest Neighbors Analysis")
    if st.button("Compute Neighbors", type="primary"):
        mol_q = mol_from_smiles(st.session_state["smiles"])
        if mol_q:
            fp_q = morgan_fp(mol_q)
            fps_nn, idx_map = build_dataset_fps(df_f)
            # FIX: Added the missing 0 for the comparison
            if len(fps_nn) > 0:
                sims = np.array(DataStructs.BulkTanimotoSimilarity(fp_q, fps_nn))
                top = np.argsort(-sims)[:10]
                rows = [{"Similarity": f"{sims[j]:.3f}", "Exp. pIC50": df_f.iloc[idx_map[j]]["pic50"]} for j in top]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

with tab_data:
    st.subheader("Dataset Dashboard")
    if not df_f.empty:
        st.plotly_chart(px.histogram(df_f, x="pic50", title="pIC50 Distribution"), use_container_width=True)
