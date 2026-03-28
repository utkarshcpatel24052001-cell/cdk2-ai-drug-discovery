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
from rdkit.Chem import AllChem, Descriptors, Draw, QED, rdMolDescriptors, Scaffolds
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# =========================
# 1. SCIENTIFIC UI CONFIG
# =========================
st.set_page_config(page_title="ChemBLitz Professional | CDK2 Suite", layout="wide")

# Injecting CSS for professional "Scientist" aesthetics and 20px font size
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Times+New+Roman&display=swap');
    html, body, [class*="css"], .stMarkdown, p, li, div {
        font-family: 'Times New Roman', serif !important;
        font-size: 20px !important;
    }
    .stMetric label { font-size: 18px !important; font-weight: bold !important; }
    .stMetric div { font-size: 28px !important; }
    .stAlert { border-radius: 0px; border-left: 5px solid #ff4b4b; }
    </style>
    """, unsafe_allow_html=True)

# =========================
# 2. PATHS & ASSETS
# =========================
PROJECT = Path(__file__).resolve().parent
DATA_PATH = PROJECT / "cdk2_pic50_clean.parquet"
MODEL_PATH = PROJECT / "cdk2_rf_final_all_data.joblib"
MODEL_DRIVE_FILE_ID = "1pOgZVHG7BfrcXE7ZmHJNM9CMnsBNlCQa"

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
    return model, data

def calculate_ligand_efficiency(pic50: float, mol: Chem.Mol) -> float:
    # LE = (1.37 / HeavyAtomCount) * pIC50
    n_heavy = mol.GetNumHeavyAtoms()
    return (1.37 / n_heavy) * pic50 if n_heavy > 0 else 0.0

def check_pains(mol: Chem.Mol, catalog: FilterCatalog) -> list[str]:
    entries = catalog.GetMatches(mol)
    return [e.GetDescription() for e in entries]

# =========================
# 4. DATA PROCESSING
# =========================
def mol_to_png(mol: Chem.Mol):
    img = Draw.MolToImage(mol, size=(450, 300))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def rf_predict(model, mol: Chem.Mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    X = arr.reshape(1, -1)
    preds = np.array([t.predict(X)[0] for t in model.estimators_])
    return float(preds.mean()), float(preds.std())

# =========================
# 5. PROFESSIONAL DASHBOARD
# =========================
st.title("🧪 CDK2 Pharmacological Diagnostic Suite")
st.markdown("### Advanced QSAR Prediction & Chemical Asset Validation")

try:
    model, df = load_assets()
    pains_catalog = get_pains_filter()
except Exception as e:
    st.error(f"System Error in Asset Loading: {e}")
    st.stop()

# Interactive Filter Block
st.sidebar.header("🎚️ Global Dataset Filters")
p_min, p_max = float(df["pic50"].min()), float(df["pic50"].max())
pic_range = st.sidebar.slider("pIC50 Selection", p_min, p_max, (p_min, p_max))
df_f = df[(df["pic50"] >= pic_range[0]) & (df["pic50"] <= pic_range[1])]

tab1, tab2, tab3 = st.tabs(["🧬 Diagnostic Lead", "📊 Subset Analytics", "📖 Documentation"])

with tab1:
    col_input, col_results = st.columns([1, 1.3], gap="large")
    
    with col_input:
        st.subheader("I. Query Definition")
        ex_options = {"Custom Input": ""}
        top_binders = df.sort_values("pic50", ascending=False).head(5)
        for _, r in top_binders.iterrows():
            ex_options[f"Ref: {r['molecule_chembl_id']} (pIC50 {r['pic50']:.1f})"] = r['smiles']
        
        selected_ex = st.selectbox("Reference Ligands", list(ex_options.keys()))
        input_smiles = st.text_input("Target SMILES", value=ex_options[selected_ex] if selected_ex != "Custom Input" else "")
        
        analyze = st.button("Execute Full Diagnostic", type="primary", use_container_width=True)

    with col_results:
        if analyze and input_smiles:
            mol = Chem.MolFromSmiles(input_smiles)
            if not mol:
                st.error("Structure Error: Invalid SMILES format.")
            else:
                p_mean, p_std = rf_predict(model, mol)
                le_score = calculate_ligand_efficiency(p_mean, mol)
                pains_hits = check_pains(mol, pains_catalog)
                scaffold = Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
                
                # --- Result Panel 1: Potency & Efficiency ---
                with st.container(border=True):
                    st.markdown("#### Section A: Binding Affinity & Efficiency")
                    cA, cB = st.columns([1, 2])
                    cA.image(mol_to_png(mol), caption="Query Topology")
                    with cB:
                        m1, m2 = st.columns(2)
                        m1.metric("Pred. pIC50", f"{p_mean:.2f}")
                        m2.metric("Ligand Efficiency", f"{le_score:.2f}", help="LE > 0.3 is target for lead discovery.")
                        st.info(f"Uncertainty (σ): ±{p_std:.3f}")

                # --- Result Panel 2: Safety & Structural Integrity ---
                with st.container(border=True):
                    st.markdown("#### Section B: Structural Alert Analysis")
                    if pains_hits:
                        st.warning(f"⚠️ PAINS Alerts Detected: {', '.join(pains_hits)}")
                    else:
                        st.success("✅ No PAINS Structural Alerts (High-Quality Lead)")
                    
                    st.markdown(f"**Murcko Scaffold:** `{Chem.MolToSmiles(scaffold)}`")

                # --- Result Panel 3: ADMET Descriptor Profile ---
                with st.container(border=True):
                    st.markdown("#### Section C: Physiochemical Optimization")
                    d = {
                        "MW": Descriptors.MolWt(mol),
                        "LogP": Descriptors.MolLogP(mol),
                        "HBD": rdMolDescriptors.CalcNumHBD(mol),
                        "TPSA": rdMolDescriptors.CalcTPSA(mol)
                    }
                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("MolWt", f"{d['MW']:.1f}")
                    d2.metric("LogP", f"{d['LogP']:.2f}")
                    d3.metric("HBD", d['HBD'])
                    d4.metric("TPSA", f"{d['TPSA']:.1f}")
        else:
            st.info("System Ready. Define Query and Execute Diagnostic.")

with tab2:
    st.subheader("Global Chemical Space Dashboard")
    st.plotly_chart(px.scatter(df_f, x="pic50", y="ic50_nM", size="n_measurements", hover_name="molecule_chembl_id", title="Structure-Activity Relationship (Filtered Subset)"), use_container_width=True)

with tab3:
    st.markdown("""
    ### Technical Specification
    - **Architecture:** RandomForest Regression (2048-bit Morgan Fingerprints, r=2).
    - **Safety:** PAINS (Pan-Assay Interference) filters enabled.
    - **Optimization:** Metrics include Ligand Efficiency (LE) for normalization.
    """)
