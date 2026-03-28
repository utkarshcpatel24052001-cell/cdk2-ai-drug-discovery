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

# Injecting CSS for professional "Scientist" aesthetics and 20px Times New Roman font
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
    .stTab { font-size: 22px !important; }
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
    # LE = (1.37 / HeavyAtomCount) * pIC50 (Standard MedChem Normalization)
    n_heavy = mol.GetNumHeavyAtoms()
    return (1.37 / n_heavy) * pic50 if n_heavy > 0 else 0.0

def check_pains(mol: Chem.Mol, catalog: FilterCatalog) -> list[str]:
    entries = catalog.GetMatches(mol)
    return [e.GetDescription() for e in entries]

# =========================
# 4. DATA PROCESSING
# =========================
def mol_to_png(mol: Chem.Mol):
    img = Draw.MolToImage(mol, size=(600, 400))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def rf_predict(model, mol: Chem.Mol):
    # Morgan Fingerprint generation (Radius 2, 2048-bit) 
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    X = arr.reshape(1, -1)
    # Uncertainty proxy: standard deviation across trees 
    preds = np.array([t.predict(X)[0] for t in model.estimators_])
    return float(preds.mean()), float(preds.std())

# =========================
# 5. PROFESSIONAL DASHBOARD
# =========================
st.title("🧪 CDK2 Pharmacological Diagnostic Suite")
st.markdown("#### Operational Intelligence for Cyclin-Dependent Kinase 2 Inhibitor Discovery")
st.divider()

try:
    model, df = load_assets()
    pains_catalog = get_pains_filter()
except Exception as e:
    st.error(f"Critical System Failure: {e}")
    st.stop()

# Initialize SMILES state
if "query_smiles" not in st.session_state:
    st.session_state.query_smiles = ""

tab1, tab2, tab3 = st.tabs(["🧬 Lead Diagnostic", "📊 Chemical Space", "📖 Methodology"])

with tab1:
    col_input, col_results = st.columns([1, 1.4], gap="large")
    
    with col_input:
        st.subheader("I. Query Definition")
        # Extract top binders for professional reference selection 
        top_binders = df.sort_values("pic50", ascending=False).head(5)
        refs = {f"Reference: {r['molecule_chembl_id']} (pIC50 {r['pic50']:.1f})": r['smiles'] for _, r in top_binders.iterrows()}
        
        selected_ref = st.selectbox("Select Validated Reference Ligand:", ["None"] + list(refs.keys()))
        if selected_ref != "None":
            st.session_state.query_smiles = refs[selected_ref]
        
        target_smiles = st.text_input("Input Query SMILES:", value=st.session_state.query_smiles)
        
        execute = st.button("Execute Diagnostic Analysis", type="primary", use_container_width=True)

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.subheader("II. Dataset Boundaries")
        p_min, p_max = float(df["pic50"].min()), float(df["pic50"].max())
        pic_range = st.slider("pIC50 Inclusion Range", p_min, p_max, (p_min, p_max))
        df_f = df[(df["pic50"] >= pic_range[0]) & (df["pic50"] <= pic_range[1])]
        st.info(f"Sub-population Size: {len(df_f)} Compounds")

    with col_results:
        if execute and target_smiles:
            mol = Chem.MolFromSmiles(target_smiles)
            if not mol:
                st.error("Invalid Structure: SMILES parsing failed.")
            else:
                # Core Calculations 
                p_mean, p_std = rf_predict(model, mol)
                le_score = calculate_ligand_efficiency(p_mean, mol)
                pains_hits = check_pains(mol, pains_catalog)
                # Corrected Scaffolds module call 
                scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold_mol)
                
                # --- Result Panel A: Binding & Potency ---
                with st.container(border=True):
                    st.markdown("### A. Affinity & Efficiency Profile")
                    cA, cB = st.columns([1, 1.5])
                    cA.image(mol_to_png(mol), caption="Structural Topology")
                    with cB:
                        m1, m2 = st.columns(2)
                        m1.metric("Predicted pIC50", f"{p_mean:.2f}")
                        m2.metric("Ligand Efficiency", f"{le_score:.2f}")
                        st.caption(f"**Confidence Interval (σ):** ±{p_std:.3f}")
                        st.caption(f"**Predicted IC50:** {10**(9-p_mean):.1f} nM")

                # --- Result Panel B: Structural Alerts ---
                with st.container(border=True):
                    st.markdown("### B. Structural Integrity Analysis")
                    if pains_hits:
                        st.error(f"⚠️ **PAINS ALERTS DETECTED:** {', '.join(pains_hits)}")
                    else:
                        st.success("✅ **Screening Clearance:** No PAINS Structural Alerts Identified.")
                    
                    st.markdown(f"**Murcko Scaffold Identifier:**")
                    st.code(scaffold_smiles, language="text")

                # --- Result Panel C: ADMET Heuristics ---
                with st.container(border=True):
                    st.markdown("### C. Physiochemical Descriptors")
                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("Mol. Weight", f"{Descriptors.MolWt(mol):.1f}")
                    d2.metric("cLogP", f"{Descriptors.MolLogP(mol):.2f}")
                    d3.metric("TPSA", f"{rdMolDescriptors.CalcTPSA(mol):.1f}")
                    d4.metric("QED Score", f"{QED.qed(mol):.2f}")
        else:
            st.info("System Ready. Please define query parameters and execute diagnostic.")

with tab2:
    st.subheader("Global Structure-Activity Landscape")
    sar_fig = px.scatter(df_f, x="pic50", y="ic50_nM", size="n_measurements", 
                         hover_name="molecule_chembl_id", template="plotly_white",
                         labels={"pic50": "Experimental pIC50", "ic50_nM": "IC50 (nM)"})
    st.plotly_chart(sar_fig, use_container_width=True)

with tab3:
    st.markdown(f"""
    ### Scientific Methodology & System Specifications 
    - **Engine:** RandomForest Regression trained on 2048-bit Morgan Fingerprints (Radius 2).
    - **Data Source:** Curated CDK2 inhibitory population from ChEMBL. 
    - **Validation:** PAINS (Pan-Assay Interference Compounds) filter applied via RDKit `FilterCatalog`.
    - **Efficiency Metrics:** Ligand Efficiency (LE) normalized by heavy atom count.
    - **Environment:** Streamlit Cloud Deployment with RDKit-C++ binaries.
    """)
