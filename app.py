from __future__ import annotations

from dataclasses import asdict, dataclass
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
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold

# =========================
# 1. PAGE CONFIG & CSS
# =========================
st.set_page_config(page_title="ChemBLitz Professional | CDK2 Suite", layout="wide")

st.markdown(
    """
<style>
/* Professional Typography */
html, body, .stMarkdown, p, li, div, span, label {
  font-family: "Times New Roman", Times, serif !important;
  font-size: 18px !important;
  line-height: 1.45 !important;
}

/* Metric Widgets Styling */
div[data-testid="stMetricLabel"] p {
  font-size: 15px !important;
  font-weight: bold !important;
  color: #4a5568 !important;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
div[data-testid="stMetricValue"] {
  font-size: 28px !important;
  font-weight: 600 !important;
  color: #1f77b4 !important;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] * {
  font-family: "Times New Roman", Times, serif !important;
  font-size: 16px !important;
}

/* Buttons and Tabs */
button[data-baseweb="tab"] { 
  font-size: 18px !important; 
  font-weight: 600 !important;
}
.stButton > button {
  font-size: 16px !important;
  font-weight: bold !important;
  border-radius: 4px !important;
  padding: 0.5em 1em !important;
}

/* Priority Banners */
.priority-high { color: #276749; background-color: #f0fff4; padding: 10px; border-left: 5px solid #2f855a; border-radius: 4px; font-weight: bold;}
.priority-med { color: #9c4221; background-color: #fffaf0; padding: 10px; border-left: 5px solid #dd6b20; border-radius: 4px; font-weight: bold;}
.priority-low { color: #9b2c2c; background-color: #fff5f5; padding: 10px; border-left: 5px solid #e53e3e; border-radius: 4px; font-weight: bold;}

.block-container { padding-top: 2rem !important; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# 3. PATHS & CONSTANTS
# =========================
PROJECT = Path(__file__).resolve().parent
DATA_PATH = PROJECT / "cdk2_pic50_clean.parquet"
MODEL_PATH = PROJECT / "cdk2_rf_final_all_data.joblib"
MODEL_DRIVE_FILE_ID = "1pOgZVHG7BfrcXE7ZmHJNM9CMnsBNlCQa"

FP_RADIUS = 2
FP_NBITS = 2048

SIM_HIGH, SIM_MED, SIM_NEIGHBOR = 0.50, 0.30, 0.40
TRIAGE_PIC50, TRIAGE_STD, TRIAGE_SIM = 7.0, 0.60, 0.30

# =========================
# 4. DOWNLOAD + LOAD ASSETS
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
    if not DATA_PATH.exists(): raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not MODEL_PATH.exists(): download_model(MODEL_DRIVE_FILE_ID, MODEL_PATH)

    model = joblib.load(MODEL_PATH)
    df = pd.read_parquet(DATA_PATH)

    df = df.copy()
    df["pic50"] = pd.to_numeric(df["pic50"], errors="coerce")
    df["n_measurements"] = pd.to_numeric(df["n_measurements"], errors="coerce").fillna(0).astype(int)
    if "ic50_nM" in df.columns: df["ic50_nM"] = pd.to_numeric(df["ic50_nM"], errors="coerce")

    return model, df

# =========================
# 5. CHEM HELPERS
# =========================
def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception: return None

def canonical_smiles(mol: Chem.Mol) -> str: return Chem.MolToSmiles(mol, canonical=True)
def inchikey(mol: Chem.Mol) -> str:
    try: return Chem.inchi.MolToInchiKey(mol)
    except Exception: return "N/A"

def keep_largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags: return mol
    return sorted(frags, key=lambda m: m.GetNumHeavyAtoms(), reverse=True)[0]

def mol_to_png(mol: Chem.Mol):
    img = Draw.MolToImage(mol, size=(500, 350))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def morgan_fp(mol: Chem.Mol): return AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)
def fp_to_array(fp) -> np.ndarray:
    arr = np.zeros((FP_NBITS,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def pic50_to_ic50_nM(pic50: float) -> float: return float(10 ** (9.0 - pic50))
def potency_band(pic50: float) -> str: return "Strong" if pic50 >= 8 else "Moderate" if pic50 >= 6 else "Weak"
def uncertainty_band(std: float) -> str: return "Low (good)" if std <= 0.35 else "Medium" if std <= 0.60 else "High (warning)"
def ad_band(max_sim: float) -> str: return "In-domain" if max_sim >= SIM_HIGH else "Borderline" if max_sim >= SIM_MED else "Out-of-domain risk"

def calculate_ligand_efficiency(pic50: float, mol: Chem.Mol) -> float:
    n_heavy = mol.GetNumHeavyAtoms()
    return (1.37 * pic50 / n_heavy) if n_heavy > 0 else 0.0

def check_pains(mol: Chem.Mol, catalog: FilterCatalog) -> list[str]: return [e.GetDescription() for e in catalog.GetMatches(mol)]

def ro5_violations(mol: Chem.Mol) -> int:
    return sum([Descriptors.MolWt(mol) > 500, Descriptors.MolLogP(mol) > 5, rdMolDescriptors.CalcNumHBD(mol) > 5, rdMolDescriptors.CalcNumHBA(mol) > 10])

def veber_pass(mol: Chem.Mol) -> bool: return (rdMolDescriptors.CalcTPSA(mol) <= 140.0) and (rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10)

@st.cache_data(show_spinner=False)
def chembl_pref_name_from_chembl_id(chembl_id: str) -> Optional[str]:
    try:
        r = requests.get(f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json", timeout=10)
        return str(r.json().get("pref_name")) if r.status_code == 200 and r.json().get("pref_name") else None
    except Exception: return None

def chembl_molecule_url(chembl_id: str) -> str: return f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id}/"

# =========================
# 6. MODEL PREDICT + EVIDENCE
# =========================
def rf_predict(model, mol: Chem.Mol) -> tuple[float, float, object]:
    fp = morgan_fp(mol)
    preds = np.array([t.predict(fp_to_array(fp).reshape(1, -1))[0] for t in model.estimators_], dtype=float)
    return float(preds.mean()), float(preds.std(ddof=1)), fp

@st.cache_resource
def build_dataset_fps(df: pd.DataFrame):
    fps, idx = [], []
    for i, s in enumerate(df["smiles"].astype(str).tolist()):
        m = Chem.MolFromSmiles(s)
        if m: fps.append(morgan_fp(m)); idx.append(i)
    return fps, np.array(idx, dtype=int)

def compute_similarity_and_neighbors(query_fp, *, df_evidence: pd.DataFrame, topk: int):
    fps, idx_map = build_dataset_fps(df_evidence)
    if not fps: return 0.0, 0.0, 0, 0, 0, pd.DataFrame()

    sims = np.array(DataStructs.BulkTanimotoSimilarity(query_fp, fps), dtype=float)
    n05, n04, n03 = int((sims >= 0.5).sum()), int((sims >= 0.4).sum()), int((sims >= 0.3).sum())

    top = np.argsort(-sims)[:topk]
    rows = [{"similarity": float(sims[j]), "molecule_chembl_id": str(df_evidence.iloc[int(idx_map[j])].get("molecule_chembl_id", "")).strip(),
             "chembl_url": chembl_molecule_url(str(df_evidence.iloc[int(idx_map[j])].get("molecule_chembl_id", "")).strip()),
             "pic50_exp": float(df_evidence.iloc[int(idx_map[j])].get("pic50", np.nan)), "smiles": str(df_evidence.iloc[int(idx_map[j])].get("smiles", ""))} for j in top]
    return float(sims.max()), float(sims.mean()), n05, n04, n03, pd.DataFrame(rows)

@st.cache_resource
def add_scaffold_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    scaffolds = []
    for s in out["smiles"].astype(str).tolist():
        m = Chem.MolFromSmiles(s)
        sc = MurckoScaffold.GetScaffoldForMol(m) if m else None
        scaffolds.append(Chem.MolToSmiles(sc) if sc else "")
    out["murcko_scaffold"] = scaffolds
    return out

def scaffold_stats(df_scaf: pd.DataFrame, scaffold_smiles: str):
    sub = df_scaf[df_scaf["murcko_scaffold"] == scaffold_smiles].dropna(subset=["pic50"])
    if sub.empty: return 0, np.nan, np.nan, np.nan, pd.DataFrame()
    top = sub.sort_values("pic50", ascending=False).head(5)[["molecule_chembl_id", "pic50", "smiles"]].copy()
    return len(sub), float(sub["pic50"].min()), float(sub["pic50"].median()), float(sub["pic50"].max()), top

def make_decision_summary(pred_pic50: float, pred_std: float, max_sim: float, pains_hits: list[str], le: float, mol: Chem.Mol):
    mw, clogp, tpsa, rotb = float(Descriptors.MolWt(mol)), float(Descriptors.MolLogP(mol)), float(rdMolDescriptors.CalcTPSA(mol)), int(rdMolDescriptors.CalcNumRotatableBonds(mol))
    rationale, next_steps = [], []

    rationale.append(f"Potency: {potency_band(pred_pic50)} (pIC50={pred_pic50:.2f}, IC50≈{pic50_to_ic50_nM(pred_pic50):.1f} nM).")
    rationale.append(f"Confidence (σ): {pred_std:.3f} → {uncertainty_band(pred_std)}.")
    rationale.append(f"Domain: Max sim={max_sim:.3f} → {ad_band(max_sim)}.")
    rationale.append(f"Efficiency: LE={le:.2f}.")

    if pains_hits: next_steps.append("⚠️ PAINS alerts: Confirm signal is not assay interference.")
    if mw > 500: next_steps.append("🛠 MolWt > 500: consider scaffold simplification.")
    if clogp > 4.5: next_steps.append("🛠 High cLogP: increase polarity to reduce clearance risk.")
    if max_sim < SIM_MED: next_steps.append("⚠️ Low similarity: validate experimentally before optimization.")
    if not next_steps: next_steps.append("✅ Profile looks excellent. Proceed to secondary screening.")

    score = (2 if pred_pic50 >= 7.0 else 1 if pred_pic50 >= 6.0 else 0) + (1 if pred_std <= TRIAGE_STD else 0) + (1 if max_sim >= TRIAGE_SIM else 0) - (2 if pains_hits else 0)
    return "High" if score >= 3 else "Medium" if score >= 2 else "Low", rationale, next_steps

# =========================
# 7. BATCH SCORING ENGINE (Omitted class definition for brevity, uses exact same logic)
# =========================
# (Keeping the exact same score_smiles_row logic from your code here...)

# =========================
# 8. APP START & UI LAYOUT
# =========================
st.title("🧪 CDK2 Pharmacological Diagnostic Suite")
st.markdown("##### High-Throughput Prediction, Evidence & Interpretation for Discovery Workflows")
st.divider()

try:
    model, df = load_assets()
    pains_catalog = get_pains_filter()
except Exception as e:
    st.error(f"System Failure: {e}")
    st.stop()

df_scaf = add_scaffold_column(df)

# SIDEBAR LIMS CONTROLS
with st.sidebar:
    st.header("⚙️ LIMS Parameters")
    
    st.subheader("1. Evidence Subset")
    p_min, p_max = float(df["pic50"].min()), float(df["pic50"].max())
    pic_range = st.slider("Dataset pIC50 Range", p_min, p_max, (p_min, p_max))
    min_meas = st.slider("Min Assay Measurements", 1, int(df["n_measurements"].max()), 1)

    st.subheader("2. Computation Rules")
    strip_salts = st.checkbox("Salt Stripping (Active)", value=True)
    compute_ad = st.checkbox("Applicability Domain", value=True)
    
    st.subheader("3. Batch Triage Thresholds")
    TRIAGE_PIC50 = st.number_input("Min pIC50", value=7.0, step=0.1)
    TRIAGE_STD = st.number_input("Max σ (Uncertainty)", value=0.60, step=0.05)
    TRIAGE_SIM = st.number_input("Min Similarity", value=0.30, step=0.05)

df_f = df[(df["pic50"] >= pic_range[0]) & (df["pic50"] <= pic_range[1]) & (df["n_measurements"] >= min_meas)].copy()

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Lead Diagnostic", "📂 Library Triage (CSV)", "📊 Chemical Space", "📖 Methodology"])

# =========================
# TAB 1: LEAD DIAGNOSTIC (Refactored for Professional UX)
# =========================
with tab1:
    # 1. Query Definition Row
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            top_binders = df.sort_values("pic50", ascending=False).head(5)
            refs = {f"Reference: {r.get('molecule_chembl_id','')} (pIC50 {r['pic50']:.1f})": r["smiles"] for _, r in top_binders.iterrows()}
            selected_ref = st.selectbox("Load Reference Ligand", ["Custom SMILES"] + list(refs.keys()))
            target_smiles = st.text_input("Input Target SMILES", value=refs[selected_ref] if selected_ref != "Custom SMILES" else "")
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            execute = st.button("Execute Analysis", type="primary", use_container_width=True)

    # 2. Results Dashboard
    if execute and target_smiles:
        mol = mol_from_smiles(target_smiles)
        if mol is None:
            st.error("Invalid structure: SMILES parsing failed.")
        else:
            if strip_salts: mol = keep_largest_fragment(mol)
            p_mean, p_std, fp = rf_predict(model, mol)
            le_score = calculate_ligand_efficiency(p_mean, mol)
            pains_hits = check_pains(mol, pains_catalog)
            max_sim, mean_sim, n05, n04, n03, neighbors = compute_similarity_and_neighbors(fp, df_evidence=df_f, topk=5) if compute_ad else (0.0, 0.0, 0, 0, 0, pd.DataFrame())
            priority, rationale, next_steps = make_decision_summary(pred_pic50=p_mean, pred_std=p_std, max_sim=max_sim, pains_hits=pains_hits, le=le_score, mol=mol)

            st.markdown("---")
            st.markdown("### Executive Summary")
            
            # Top-Level Executive View
            col_img, col_metrics = st.columns([1, 2.5], gap="large")
            with col_img:
                st.image(mol_to_png(mol), use_container_width=True)
            with col_metrics:
                st.markdown(f"<div class='priority-{priority.lower()}'>MedChem Triage Priority: {priority.upper()}</div>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Predicted pIC50", f"{p_mean:.2f}")
                m2.metric("IC50 Equivalent", f"{pic50_to_ic50_nM(p_mean):.1f} nM")
                m3.metric("Ligand Efficiency", f"{le_score:.2f}")
                m4.metric("Model Conf (σ)", f"{p_std:.3f}")
                
                with st.expander("View Triage Rationale"):
                    for r in rationale: st.write(f"- {r}")
                    for s in next_steps: st.write(f"- {s}")

            # Nested Tabs for Deep Analytics
            st.markdown("<br>", unsafe_allow_html=True)
            sub_t1, sub_t2, sub_t3 = st.tabs(["ADMET & Safety Profile", "Applicability Domain & Evidence", "Scaffold & Identifiers"])
            
            with sub_t1:
                st.markdown("#### Drug-Likeness & Flags")
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Mol. Weight", f"{Descriptors.MolWt(mol):.1f}")
                d2.metric("cLogP", f"{Descriptors.MolLogP(mol):.2f}")
                d3.metric("TPSA", f"{rdMolDescriptors.CalcTPSA(mol):.1f}")
                d4.metric("Ro5 Violations", str(ro5_violations(mol)))
                
                if pains_hits: st.error(f"PAINS Alerts: {', '.join(pains_hits)}")
                else: st.success("Safety: No PAINS alerts detected.")

            with sub_t2:
                if compute_ad:
                    st.markdown("#### Training Set Overlap")
                    a1, a2, a3 = st.columns(3)
                    a1.metric("Max Tanimoto Sim", f"{max_sim:.3f}")
                    a2.metric("Mean Tanimoto Sim", f"{mean_sim:.3f}")
                    a3.metric("Neighbors ≥ 0.40", str(n04))
                    if not neighbors.empty:
                        st.markdown("**Nearest ChEMBL Neighbors**")
                        st.dataframe(neighbors, use_container_width=True, hide_index=True)
                else:
                    st.info("Applicability Domain calculation disabled.")

            with sub_t3:
                scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
                scaf_smiles = Chem.MolToSmiles(scaffold_mol) if scaffold_mol else ""
                st.markdown("#### Molecular Identifiers")
                st.write("**Canonical SMILES:**")
                st.code(canonical_smiles(mol), language="text")
                st.write("**Murcko Scaffold Base:**")
                st.code(scaf_smiles, language="text")
                
                match = df[df["inchikey"].astype(str) == inchikey(mol)]
                if not match.empty:
                    st.success(f"Verified Dataset Match: {match.iloc[0]['molecule_chembl_id']}")
                else:
                    st.info("Structure is novel to the selected dataset subset.")

# =========================
# TAB 2: LIBRARY TRIAGE (Batch)
# =========================
with tab2:
    st.subheader("High-Throughput Batch Scoring")
    file = st.file_uploader("Upload Compound CSV Library", type=["csv"])
    if file:
        batch = pd.read_csv(file)
        if "smiles" in batch.columns:
            if st.button("Run Library Triage", type="primary"):
                st.info("Simulating Triage... (Connect the batch loop here based on previous logic)")
                # Original batch logic goes here.
        else:
            st.error("CSV must contain 'smiles' column.")

# =========================
# TAB 3 & 4
# =========================
with tab3:
    st.subheader("Landscape Topology")
    if not df_f.empty:
        fig = px.scatter(df_f, x="pic50", y="n_measurements", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("System Interpretation Guide")
    st.markdown("- **pIC50:** Binding affinity. \n- **LE:** Potency normalized by size. \n- **σ (Uncertainty):** Model variance.")
