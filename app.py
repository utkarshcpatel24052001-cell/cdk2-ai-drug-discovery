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
st.set_page_config(page_title="ChemBLitz — CDK2 Predictor", layout="wide", initial_sidebar_state="collapsed")

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
    with st.spinner("Downloading 142MB Model from secure storage (First boot only)..."):
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

def canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True)

def inchikey(mol: Chem.Mol) -> str:
    try: return Chem.inchi.MolToInchiKey(mol)
    except Exception: return "N/A"

def morgan_fp(mol: Chem.Mol, radius: int = FP_RADIUS, n_bits: int = FP_NBITS):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

def fp_to_numpy(fp, n_bits: int = FP_NBITS) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def draw_mol_png(mol: Chem.Mol, size=(400, 250)) -> bytes:
    img = Draw.MolToImage(mol, size=size)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def compute_descriptors(mol: Chem.Mol) -> dict:
    return {
        "MolWt": float(Descriptors.MolWt(mol)), "LogP": float(Descriptors.MolLogP(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)), "HBD": int(rdMolDescriptors.CalcNumHBD(mol)),
        "HBA": int(rdMolDescriptors.CalcNumHBA(mol)), "RotB": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "Rings": int(rdMolDescriptors.CalcNumRings(mol)), "QED": float(QED.qed(mol)),
    }

def lipinski_violations(d: dict) -> int:
    return sum([d["MolWt"] > 500, d["LogP"] > 5, d["HBD"] > 5, d["HBA"] > 10])

def veber_pass(d: dict) -> bool:
    return (d["TPSA"] <= 140.0) and (d["RotB"] <= 10)

def potency_bucket(pic50: float) -> str:
    if pic50 >= 8.0: return "Strong (≥ 8.0)"
    if pic50 >= 6.0: return "Moderate (6.0 - 8.0)"
    return "Weak (< 6.0)"

def confidence_label(pred_std: float, max_sim: float) -> str:
    if (max_sim >= SIM_HIGH) and (pred_std <= 0.35): return "High"
    if (max_sim >= SIM_MED) and (pred_std <= 0.60): return "Medium"
    return "Low"

def improvement_suggestions(d: dict, max_sim: float) -> list[str]:
    s = []
    if d["LogP"] > 4.5: s.append("LogP is high: Consider adding polarity (HBA/HBD) or removing hydrophobic groups.")
    if d["TPSA"] > 140: s.append("TPSA is high: May reduce cell permeability. Consider masking polar groups.")
    if d["RotB"] > 10: s.append("High flexibility: Reduce rotatable bonds via rigidification to improve binding entropy.")
    if d["MolWt"] > 500: s.append("MolWt > 500: Consider scaffold simplification to maintain drug-likeness.")
    if max_sim < SIM_MED: s.append("Out-of-Domain Risk: Low similarity to known CDK2 data. Treat prediction with caution.")
    if not s: s.append("No major ADMET red flags detected. Next focus on kinase selectivity and metabolic stability.")
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
    dff = df.dropna(subset=["pic50", "smiles"]).copy()
    
    top = dff.sort_values("pic50", ascending=False).head(1)
    if len(top): out[f"Strong Binder (pIC50 {top.iloc[0]['pic50']:.2f})"] = str(top.iloc[0]["smiles"])
    
    med_pic = float(dff["pic50"].median())
    med_row = dff.iloc[(dff["pic50"] - med_pic).abs().argsort()[:1]]
    if len(med_row): out[f"Moderate Binder (pIC50 ~{med_pic:.2f})"] = str(med_row.iloc[0]["smiles"])
    
    low = dff.sort_values("pic50", ascending=True).head(1)
    if len(low): out[f"Weak Binder (pIC50 {low.iloc[0]['pic50']:.2f})"] = str(low.iloc[0]["smiles"])
    
    return out

# =========================
# 6. USER INTERFACE
# =========================
st.title("🧬 ChemBLitz: CDK2 pIC50 Scientific Predictor")
st.markdown("A professional cheminformatics pipeline for predicting Cyclin-dependent kinase 2 (CDK2) inhibition, integrating RandomForest modeling, Applicability Domain assessment, and ADMET heuristics.")

df = load_data()
examples = build_real_examples_from_dataset(df)

if "smiles" not in st.session_state:
    st.session_state["smiles"] = examples.get(list(examples.keys())[0], "CCO")

tab_pred, tab_val, tab_data, tab_about = st.tabs(["Target Prediction", "Nearest Neighbors", "Dataset Dashboard", "Methodology"])

with tab_pred:
    c_left, c_right = st.columns([1.0, 1.2], gap="large")
    
    with c_left:
        st.subheader("Query Setup")
        
        # Example Loader
        ex = st.selectbox("Load known CDK2 ligand:", list(examples.keys()), index=0)
        if st.button("Use selected example", use_container_width=True):
            st.session_state["smiles"] = examples[ex]
            
        st.session_state["smiles"] = st.text_input("Enter SMILES string:", value=st.session_state["smiles"])
        
        run = st.button("Predict Potency", type="primary", use_container_width=True)

        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.subheader("Dataset Filters")
        st.caption("These filters control the Nearest Neighbors and Dashboard tabs.")
        
        pic50_min, pic50_max = float(df["pic50"].min()), float(df["pic50"].max())
        pic50_range = st.slider("pIC50 Threshold", pic50_min, pic50_max, (pic50_min, pic50_max))
        min_n = st.slider("Minimum assay measurements", 1, int(df["n_measurements"].max()), 1)
        
        df_f = df[(df["pic50"] >= pic50_range[0]) & (df["pic50"] <= pic50_range[1]) & (df["n_measurements"] >= min_n)]
        st.info(f"Filtered compounds available: {len(df_f)} / {len(df)}")

    with c_right:
        if not run:
            st.info("👈 Enter a SMILES string and click **Predict Potency** to generate a comprehensive molecular report.")
        else:
            mol = mol_from_smiles(st.session_state["smiles"])
            if mol is None:
                st.error("Invalid SMILES structure. Please verify your input.")
                st.stop()
            
            with st.spinner("Executing RandomForest Model..."):
                model = load_model()
                fp = morgan_fp(mol)
                X = fp_to_numpy(fp).reshape(1, -1)
                pred_pic50, pred_std = rf_predict_with_uncertainty(model, X)
                
                fps_all, _ = build_dataset_fps(df)
                sims_all = np.array(DataStructs.BulkTanimotoSimilarity(fp, fps_all), dtype=float) if len(fps_all) else np.array([])
                max_sim = float(sims_all.max()) if len(sims_all) else 0.0

            # --- CARD 1: Prediction Overview ---
            with st.container(border=True):
                colA, colB = st.columns([1, 2])
                with colA:
                    st.image(draw_mol_png(mol), use_container_width=True)
                with colB:
                    st.markdown("### Primary Prediction")
                    m1, m2 = st.columns(2)
                    m1.metric("Predicted pIC50", f"{pred_pic50:.3f}", help="Higher is more potent")
                    m2.metric("Predicted IC50 (nM)", f"{pic50_to_ic50_nM(pred_pic50):,.1f}", help="Lower is more potent")
                    
                    st.markdown(f"**Classification:** `{potency_bucket(pred_pic50)}`")
                    
                    # Fix: Match by exact string to prevent InChIKey hashing failures
                    local_match = df[df["smiles"] == st.session_state["smiles"]]
                    if local_match.empty and "inchikey" in df.columns:
                        local_match = df[df["inchikey"].astype(str) == inchikey(mol)]
                        
                    chembl_id = str(local_match.iloc[0].get("molecule_chembl_id", "")).strip() if len(local_match) > 0 else None
                    if chembl_id:
                        st.success(f"Verified Training Compound: **{chembl_id}**")
                    else:
                        st.caption("Compound not found in the training dataset (Novel).")

            # --- CARD 2: Confidence & Domain ---
            with st.container(border=True):
                st.markdown("### Applicability Domain & Confidence")
                c1, c2, c3 = st.columns(3)
                c1.metric("Model Uncertainty (σ)", f"{pred_std:.3f}")
                c2.metric("Max Tanimoto Sim.", f"{max_sim:.3f}")
                c3.metric("Overall Confidence", confidence_label(pred_std, max_sim))
                
                if max_sim >= SIM_HIGH: 
                    st.success("High structural overlap with training data. Prediction is reliable.")
                elif max_sim >= SIM_MED: 
                    st.warning("Moderate structural overlap. Borderline applicability domain.")
                else: 
                    st.error("Low structural overlap (Novel Chemotype). Treat prediction with high caution.")

            # --- CARD 3: ADMET Profile ---
            with st.container(border=True):
                st.markdown("### Lead-Likeness & ADMET Profile")
                d = compute_descriptors(mol)
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Mol. Weight", f"{d['MolWt']:.1f}")
                d2.metric("cLogP", f"{d['LogP']:.2f}")
                d3.metric("TPSA", f"{d['TPSA']:.1f}")
                d4.metric("QED Score", f"{d['QED']:.2f}")

                ro5 = lipinski_violations(d)
                vp = "Pass" if veber_pass(d) else "Fail"
                st.markdown(f"**Lipinski Violations:** `{ro5}` | **Veber Rule:** `{vp}`")
                
                with st.expander("View MedChem Improvement Suggestions"):
                    for s in improvement_suggestions(d, max_sim):
                        st.markdown(f"- {s}")

with tab_val:
    st.subheader("Nearest Neighbors Analysis")
    st.caption("Identifies the most structurally similar compounds based on your Dataset Filters.")
    mol_q = mol_from_smiles(st.session_state.get("smiles", "CCO"))
    if mol_q and st.button("Compute Structural Neighbors", type="primary"):
        with st.spinner("Querying dataset..."):
            fp_q = morgan_fp(mol_q)
            fps_nn, idx_map = build_dataset_fps(df_f)
            
            if len(fps_nn) >
