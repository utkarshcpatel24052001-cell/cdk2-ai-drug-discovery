from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Draw, QED, rdMolDescriptors

# =========================
# 1. STREAMLIT CONFIG (MUST BE FIRST)
# =========================
st.set_page_config(page_title="ChemBLitz — CDK2 Scientific Predictor", layout="wide")

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
# 3. DOWNLOAD HANDLER & FILE CHECKS
# =========================
def download_model_from_drive(file_id: str, destination: Path) -> None:
    session = requests.Session()
    url = "https://drive.google.com/uc?export=download"
    params = {"id": file_id}
    
    resp = session.get(url, params=params, stream=True, timeout=60)
    
    # Handle Google's virus scan warning for large files
    token = None
    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break
    if not token:
        m = re.search(r"confirm=([0-9A-Za-z_]+)", resp.text)
        if m:
            token = m.group(1)
            
    if token:
        params["confirm"] = token
        resp = session.get(url, params=params, stream=True, timeout=60)
        
    resp.raise_for_status()
    
    with open(destination, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

if not DATA_PATH.exists():
    st.error(f"❌ Dataset not found: {DATA_PATH}")
    st.stop()

if not MODEL_PATH.exists():
    with st.spinner("Downloading 142MB model from Google Drive (this takes ~15 seconds on first boot)..."):
        try:
            download_model_from_drive(MODEL_DRIVE_FILE_ID, MODEL_PATH)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download model from Google Drive: {e}")
            st.stop()

# =========================
# 4. HELPERS
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

def draw_mol_png(mol: Chem.Mol, size=(440, 300)) -> bytes:
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
    if pic50 >= 8.0: return "Strong (≥ 8)"
    if pic50 >= 6.0: return "Moderate (6–8)"
    return "Weak (< 6)"

def confidence_label(pred_std: float, max_sim: float) -> str:
    if (max_sim >= SIM_HIGH) and (pred_std <= 0.35): return "High"
    if (max_sim >= SIM_MED) and (pred_std <= 0.60): return "Medium"
    return "Low"

def improvement_suggestions(d: dict, max_sim: float) -> list[str]:
    s = []
    if d["LogP"] > 4.5: s.append("LogP high → consider adding polarity (HBA/HBD) or removing hydrophobic groups.")
    if d["TPSA"] > 140: s.append("TPSA high → may reduce permeability; consider masking polar groups.")
    if d["RotB"] > 10: s.append("Too flexible → reduce rotatable bonds via rigidification.")
    if d["MolWt"] > 500: s.append("MolWt high → consider scaffold simplification.")
    if max_sim < SIM_MED: s.append("Out-of-domain risk → low similarity to known CDK2 data; treat cautiously.")
    if not s: s.append("No major red flags → next focus on selectivity and metabolic stability.")
    return s

@st.cache_data(show_spinner=False)
def chembl_pref_name_from_chembl_id(chembl_id: str) -> Optional[str]:
    try:
        r = requests.get(f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json", timeout=10)
        return str(r.json().get("pref_name", "")) or None if r.status_code == 200 else None
    except Exception: return None

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
    dff["pic50"] = pd.to_numeric(dff["pic50"], errors="coerce")
    dff = dff.dropna(subset=["pic50"])
    
    top = dff.sort_values("pic50", ascending=False).head(1)
    if len(top): out[f"Top binder (pIC50 {top.iloc[0]['pic50']:.2f})"] = str(top.iloc[0]["smiles"])
    
    med_pic = float(dff["pic50"].median())
    med_row = dff.iloc[(dff["pic50"] - med_pic).abs().argsort()[:1]]
    if len(med_row): out[f"Median binder (pIC50 ~{med_pic:.2f})"] = str(med_row.iloc[0]["smiles"])
    
    low = dff.sort_values("pic50", ascending=True).head(1)
    if len(low): out[f"Weak binder (pIC50 {low.iloc[0]['pic50']:.2f})"] = str(low.iloc[0]["smiles"])
    
    if not out: out["Dataset example"] = str(df.iloc[0]["smiles"])
    return out

@st.cache_resource
def get_df_and_examples():
    df_local = load_data()
    return df_local, build_real_examples_from_dataset(df_local)

# =========================
# 6. USER INTERFACE
# =========================
st.title("ChemBLitz — CDK2 pIC50 Scientific Predictor")
st.caption("CDK2-only. RandomForest on Morgan fingerprints. Model dynamically loaded.")

df, examples = get_df_and_examples()

if "smiles" not in st.session_state:
    st.session_state["smiles"] = str(df.iloc[0]["smiles"]) if "smiles" in df.columns else "CCO"

tab_pred, tab_val, tab_data, tab_about = st.tabs(["Predict", "Validate (Similarity)", "Dataset Dashboard", "Methodology"])

with tab_pred:
    c_left, c_right = st.columns([1.05, 0.95], gap="large")
    with c_left:
        st.subheader("Input")
        ex = st.selectbox("Load CDK2 dataset example", list(examples.keys()), index=0)
        if st.button("Use selected example"): st.session_state["smiles"] = examples[ex]
        st.session_state["smiles"] = st.text_input("SMILES", value=st.session_state["smiles"])
        run = st.button("Predict", type="primary")

        st.markdown("---")
        st.subheader("Dataset filters")
        pic50_min, pic50_max = float(df["pic50"].min()), float(df["pic50"].max())
        pic50_range = st.slider("pIC50 range", pic50_min, pic50_max, (pic50_min, pic50_max))
        min_n = st.slider("Min measurements", 1, int(df["n_measurements"].max()), 1)
        df_f = df[(df["pic50"] >= pic50_range[0]) & (df["pic50"] <= pic50_range[1]) & (df["n_measurements"] >= min_n)]
        st.info(f"Filtered dataset size: {len(df_f)} / {len(df)}")

    with c_right:
        st.subheader("Output")
        if not run:
            st.info("Click **Predict** to generate outputs.")
        else:
            mol = mol_from_smiles(st.session_state["smiles"])
            if mol is None:
                st.error("Invalid SMILES.")
                st.stop()
            
            with st.spinner("Loading model into memory…"):
                model = load_model()
            
            st.image(draw_mol_png(mol), caption="Input structure (RDKit 2D)")
            
            ik = inchikey(mol)
            local_match = df[df["inchikey"].astype(str) == ik] if "inchikey" in df.columns else pd.DataFrame()
            chembl_id = str(local_match.iloc[0].get("molecule_chembl_id", "")).strip() if len(local_match) > 0 else None
            
            if chembl_id: st.success(f"Found in dataset: {chembl_id}")
            else: st.warning("Not found in dataset.")

            fp = morgan_fp(mol)
            X = fp_to_numpy(fp).reshape(1, -1)
            pred_pic50, pred_std = rf_predict_with_uncertainty(model, X)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted pIC50", f"{pred_pic50:.3f}")
            m2.metric("Predicted IC50 (nM)", f"{pic50_to_ic50_nM(pred_pic50):,.2f}")
            m3.metric("Uncertainty (σ)", f"{pred_std:.3f}")
            m4.metric("Potency class", potency_bucket(pred_pic50))

            st.markdown("### Applicability domain")
            if st.button("Compute similarity", key="sim_button"):
                with st.spinner("Computing dataset similarity…"):
                    fps_all, _ = build_dataset_fps(df)
                    sims_all = np.array(DataStructs.BulkTanimotoSimilarity(fp, fps_all), dtype=float) if len(fps_all) else np.array([])
                    max_sim = float(sims_all.max()) if len(sims_all) else 0.0
                    if max_sim >= SIM_HIGH: st.success(f"In-domain (similarity: {max_sim:.3f})")
                    else: st.warning(f"Borderline/Out-of-domain (similarity: {max_sim:.3f})")

            st.markdown("### PhysChem & ADMET")
            d = compute_descriptors(mol)
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("MolWt", f"{d['MolWt']:.1f}")
            d2.metric("cLogP", f"{d['LogP']:.2f}")
            d3.metric("TPSA", f"{d['TPSA']:.1f}")
            d4.metric("QED", f"{d['QED']:.2f}")

            for s in improvement_suggestions(d, 0.0):
                st.write(f"- {s}")

with tab_val:
    st.subheader("Nearest Neighbors")
    mol_q = mol_from_smiles(st.session_state["smiles"])
    if mol_q and st.button("Compute nearest neighbors", type="primary"):
        with st.spinner("Computing similarity…"):
            fp_q = morgan_fp(mol_q)
            fps_nn, idx_map = build_dataset_fps(df)
            sims = np.array(DataStructs.BulkTanimotoSimilarity(fp_q, fps_nn), dtype=float)
            top = np.argsort(-sims)[:10]
            
            rows = []
            for j in top:
                row = df.iloc[int(idx_map[j])]
                rows.append({
                    "Tanimoto": float(sims[j]),
                    "pic50": float(row.get("pic50", np.nan)),
                    "smiles": str(row.get("smiles", ""))
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

with tab_data:
    st.subheader("Dataset Dashboard")
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.histogram(df, x="pic50", title="pIC50 Distribution"), use_container_width=True)
    with c2: st.plotly_chart(px.histogram(df, x="n_measurements", title="Measurement Counts"), use_container_width=True)

with tab_about:
    st.markdown("""
    **Target:** Cyclin-dependent kinase 2 (CDK2)  
    **Model:** RandomForest regression on Morgan fingerprints (Radius=2, 2048 bits)  
    **Uncertainty proxy:** standard deviation across RF trees (σ)
    """)
