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
import requests
import streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Draw, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold

# =========================
# 1. PAGE CONFIG
# =========================
st.set_page_config(page_title="ChemBLitz Professional | CDK2 Suite", layout="wide")

# =========================
# 2. SAFE PROFESSIONAL CSS 
# =========================
st.markdown(
    """
<style>
/* Base typography */
html, body {
  font-family: "Times New Roman", Times, serif !important;
  font-size: 18px !important;
  line-height: 1.45 !important;
}

/* Main content typography */
.stMarkdown, .stMarkdown p, .stMarkdown li {
  font-family: "Times New Roman", Times, serif !important;
  font-size: 18px !important;
  line-height: 1.50 !important;
}

/* Sidebar: slightly smaller so controls fit */
section[data-testid="stSidebar"] * {
  font-family: "Times New Roman", Times, serif !important;
  font-size: 16px !important;
  line-height: 1.35 !important;
}

/* Tabs */
button[data-baseweb="tab"] {
  font-size: 18px !important;
  font-weight: bold !important;
}

/* Metric widgets */
div[data-testid="stMetricLabel"] p {
  font-size: 16px !important;
  font-weight: 700 !important;
  color: #1f77b4 !important;
  line-height: 1.2 !important;
}
div[data-testid="stMetricValue"] {
  font-size: 26px !important;
  font-weight: 600 !important;
  line-height: 1.1 !important;
}

/* Priority Colors */
.priority-high { color: #2f855a; font-weight: bold; }
.priority-med { color: #c05621; font-weight: bold; }
.priority-low { color: #c53030; font-weight: bold; }

/* Buttons */
.stButton > button {
  font-size: 16px !important;
  border-radius: 6px !important;
  padding: 0.55em 1em !important;
}
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

SIM_HIGH = 0.50
SIM_MED = 0.30
SIM_NEIGHBOR = 0.40

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
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not MODEL_PATH.exists():
        download_model(MODEL_DRIVE_FILE_ID, MODEL_PATH)

    model = joblib.load(MODEL_PATH)
    df = pd.read_parquet(DATA_PATH)

    required = {"smiles", "pic50", "n_measurements"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    df = df.copy()
    df["pic50"] = pd.to_numeric(df["pic50"], errors="coerce")
    df["n_measurements"] = pd.to_numeric(df["n_measurements"], errors="coerce").fillna(0).astype(int)

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
    except Exception:
        return None

def canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True)

def inchikey(mol: Chem.Mol) -> str:
    try: return Chem.inchi.MolToInchiKey(mol)
    except Exception: return "N/A"

def keep_largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags: return mol
    frags = sorted(frags, key=lambda m: m.GetNumHeavyAtoms(), reverse=True)
    return frags[0]

def mol_to_png(mol: Chem.Mol):
    img = Draw.MolToImage(mol, size=(600, 400))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def morgan_fp(mol: Chem.Mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)

def fp_to_array(fp) -> np.ndarray:
    arr = np.zeros((FP_NBITS,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def pic50_to_ic50_nM(pic50: float) -> float:
    return float(10 ** (9.0 - pic50))

def calculate_ligand_efficiency(pic50: float, mol: Chem.Mol) -> float:
    n_heavy = mol.GetNumHeavyAtoms()
    return (1.37 * pic50 / n_heavy) if n_heavy > 0 else 0.0

def calculate_bei(pic50: float, mw: float) -> float:
    # Binding Efficiency Index = pIC50 / (MW / 1000)
    return (pic50 / (mw / 1000)) if mw > 0 else 0.0

def check_pains(mol: Chem.Mol, catalog: FilterCatalog) -> list[str]:
    entries = catalog.GetMatches(mol)
    return [e.GetDescription() for e in entries]

@st.cache_data(show_spinner=False)
def chembl_pref_name_from_chembl_id(chembl_id: str) -> Optional[str]:
    try:
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
        r = requests.get(url, timeout=10)
        if r.status_code != 200: return None
        name = r.json().get("pref_name")
        return str(name) if name else None
    except Exception:
        return None

def chembl_molecule_url(chembl_id: str) -> str:
    return f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id}/"

# =========================
# 6. MODEL PREDICT + EVIDENCE
# =========================
def rf_predict(model, mol: Chem.Mol) -> tuple[float, float, object]:
    fp = morgan_fp(mol)
    X = fp_to_array(fp).reshape(1, -1)
    preds = np.array([t.predict(X)[0] for t in model.estimators_], dtype=float)
    return float(preds.mean()), float(preds.std(ddof=1)), fp

@st.cache_resource
def build_dataset_fps(df: pd.DataFrame):
    fps, idx = [], []
    for i, s in enumerate(df["smiles"].astype(str).tolist()):
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        fps.append(morgan_fp(m))
        idx.append(i)
    return fps, np.array(idx, dtype=int)

def compute_similarity_and_neighbors(query_fp, *, df_evidence: pd.DataFrame, topk: int):
    fps, idx_map = build_dataset_fps(df_evidence)
    if len(fps) == 0: return 0.0, 0.0, 0, pd.DataFrame()

    sims = np.array(DataStructs.BulkTanimotoSimilarity(query_fp, fps), dtype=float)
    max_sim = float(sims.max()) if len(sims) else 0.0
    mean_sim = float(sims.mean()) if len(sims) else 0.0
    n_ge = int((sims >= SIM_NEIGHBOR).sum())

    top = np.argsort(-sims)[:topk]
    rows = []
    for j in top:
        row = df_evidence.iloc[int(idx_map[j])]
        rows.append({
            "Similarity": float(sims[j]),
            "ChEMBL_ID": str(row.get("molecule_chembl_id", "")),
            "pIC50_Exp": float(row.get("pic50", np.nan)),
            "SMILES": str(row.get("smiles", ""))
        })
    return max_sim, mean_sim, n_ge, pd.DataFrame(rows)

# =========================
# 7. PROFESSIONAL MEDCHEM TRIAGE LOGIC
# =========================
def make_decision_summary(pred_pic50: float, pred_std: float, max_sim: float, pains_hits: list[str], 
                          le: float, bei: float, mw: float, clogp: float, tpsa: float, rotb: int, hbd: int, hba: int):
    rationale = []
    next_steps = []
    
    # 1. Lipinski Rule of 5 check
    ro5_violations = sum([mw > 500, clogp > 5, hbd > 5, hba > 10])

    # 2. Rationale building
    rationale.append(f"Predicted Potency: {pred_pic50:.2f} pIC50 (~{pic50_to_ic50_nM(pred_pic50):.1f} nM).")
    
    if le >= 0.3 and bei >= 14:
        rationale.append(f"Highly Efficient Binding: LE ({le:.2f}) and BEI ({bei:.1f}) are optimal.")
    else:
        rationale.append(f"Sub-optimal Efficiency: LE ({le:.2f}), BEI ({bei:.1f}). May be driven by lipophilicity/bulk.")

    if ro5_violations > 0:
        rationale.append(f"Lipinski Violations: {ro5_violations}/4. Potential oral bioavailability issues.")
    else:
        rationale.append("Lipinski Compliant: 0 Violations (Good oral bioavailability profile).")

    # 3. Next Steps & Warnings
    if pains_hits:
        next_steps.append(f"🛑 CRITICAL: PAINS detected ({pains_hits[0]}). High risk of assay interference.")
    if max_sim < SIM_MED:
        next_steps.append("⚠️ Domain Extrapolation: Scaffold is novel compared to training set. Validate experimentally.")
    if rotb > 10:
        next_steps.append("⚠️ High Flexibility (RotB > 10): Consider rigidifying scaffold to improve binding entropy.")
    if tpsa > 140:
        next_steps.append("⚠️ High TPSA (> 140): Poor predicted membrane permeability.")
        
    if not next_steps:
        next_steps.append("✅ Excellent Profile. Proceed to secondary kinase screening (CDK1/4/6) and in-vitro ADME.")

    # 4. Strict Professional Triage Scoring
    if pains_hits or ro5_violations >= 2 or pred_pic50 < 5.5:
        priority = "Low"
    elif pred_pic50 >= 7.0 and ro5_violations == 0 and le >= 0.3 and max_sim >= 0.35:
        priority = "High"
    else:
        priority = "Medium"

    return priority, rationale, next_steps, ro5_violations

# =========================
# 8. BATCH SCORING
# =========================
@dataclass
class BatchRow:
    smiles_used: str
    pred_pic50: float
    pred_ic50_nM: float
    priority: str
    ligand_efficiency: float
    binding_efficiency_idx: float
    max_sim: float
    ro5_violations: int
    pains_alert: bool
    molwt: float
    clogp: float
    tpsa: float

def score_smiles_row(smiles: str, model, pains_catalog, df_evidence, strip_salts, compute_ad) -> BatchRow:
    mol = mol_from_smiles(smiles)
    if not mol:
        return BatchRow(smiles, np.nan, np.nan, "Error", np.nan, np.nan, np.nan, 0, False, np.nan, np.nan, np.nan)

    if strip_salts: mol = keep_largest_fragment(mol)
    smiles_used = canonical_smiles(mol)

    p_mean, p_std, fp = rf_predict(model, mol)
    mw, clogp = Descriptors.MolWt(mol), Descriptors.MolLogP(mol)
    hbd, hba = rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol)
    
    le = calculate_ligand_efficiency(p_mean, mol)
    bei = calculate_bei(p_mean, mw)
    pains_hits = check_pains(mol, pains_catalog)

    max_sim = 0.0
    if compute_ad:
        max_sim, _, _, _ = compute_similarity_and_neighbors(fp, df_evidence=df_evidence, topk=1)

    priority, _, _, ro5 = make_decision_summary(p_mean, p_std, max_sim, pains_hits, le, bei, mw, clogp, rdMolDescriptors.CalcTPSA(mol), rdMolDescriptors.CalcNumRotatableBonds(mol), hbd, hba)

    return BatchRow(
        smiles_used=smiles_used, pred_pic50=p_mean, pred_ic50_nM=pic50_to_ic50_nM(p_mean),
        priority=priority, ligand_efficiency=le, binding_efficiency_idx=bei, max_sim=max_sim,
        ro5_violations=ro5, pains_alert=len(pains_hits) > 0,
        molwt=mw, clogp=clogp, tpsa=rdMolDescriptors.CalcTPSA(mol)
    )

# =========================
# 9. APP START
# =========================
st.title("CDK2 Pharmacological Diagnostic Suite")
st.markdown("#### High-Throughput Prediction & ADMET Intelligence for CDK2 Inhibitor Discovery")
st.divider()

try:
    model, df = load_assets()
    pains_catalog = get_pains_filter()
except Exception as e:
    st.error(f"Critical System Failure: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Global Controls")
    st.subheader("Reference Evidence Subset")
    p_min, p_max = float(df["pic50"].min()), float(df["pic50"].max())
    pic_range = st.slider("Dataset pIC50 Range", p_min, p_max, (p_min, p_max))
    min_meas = st.slider("Min measurements", 1, int(df["n_measurements"].max()), 1)

    st.subheader("Computational Engine")
    strip_salts = st.checkbox("Strip Salts (Keep Largest Fragment)", value=True)
    compute_ad = st.checkbox("Compute Applicability Domain (AD)", value=True)
    
df_f = df[(df["pic50"] >= pic_range[0]) & (df["pic50"] <= pic_range[1]) & (df["n_measurements"] >= min_meas)].copy()

tab1, tab2, tab3 = st.tabs(["Lead Diagnostic", "Batch CSV Screening", "Chemical Space"])

# =========================
# Tab 1: Single molecule
# =========================
with tab1:
    col_input, col_results = st.columns([1, 1.4], gap="large")

    with col_input:
        st.subheader("I. Query Definition")
        top_binders = df.sort_values("pic50", ascending=False).head(5)
        refs = {f"Ref: {r.get('molecule_chembl_id','')} (pIC50 {r['pic50']:.1f})": r["smiles"] for _, r in top_binders.iterrows()}
        selected_ref = st.selectbox("Load Validated Reference", ["Custom SMILES"] + list(refs.keys()))
        
        if selected_ref != "Custom SMILES":
            st.session_state["query_smiles"] = refs[selected_ref]

        target_smiles = st.text_input("Target SMILES string", value=st.session_state.get("query_smiles", ""))
        execute = st.button("Execute Diagnostic", type="primary", use_container_width=True)
        st.info(f"Active Evidence Subset: {len(df_f)} / {len(df)} Compounds")

    with col_results:
        if execute and target_smiles:
            mol = mol_from_smiles(target_smiles)
            if mol is None:
                st.error("Structure Error: Invalid SMILES.")
            else:
                if strip_salts: mol = keep_largest_fragment(mol)

                p_mean, p_std, fp = rf_predict(model, mol)
                mw, clogp = Descriptors.MolWt(mol), Descriptors.MolLogP(mol)
                hbd, hba = rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol)
                tpsa, rotb = rdMolDescriptors.CalcTPSA(mol), rdMolDescriptors.CalcNumRotatableBonds(mol)
                
                le_score = calculate_ligand_efficiency(p_mean, mol)
                bei_score = calculate_bei(p_mean, mw)
                pains_hits = check_pains(mol, pains_catalog)

                max_sim, mean_sim, n_ge, neighbors = 0.0, 0.0, 0, pd.DataFrame()
                if compute_ad:
                    max_sim, mean_sim, n_ge, neighbors = compute_similarity_and_neighbors(fp, df_evidence=df_f, topk=5)

                priority, rationale, next_steps, ro5 = make_decision_summary(
                    p_mean, p_std, max_sim, pains_hits, le_score, bei_score, mw, clogp, tpsa, rotb, hbd, hba
                )

                # Output Visuals
                with st.container(border=True):
                    st.markdown(f"### MedChem Decision: <span class='priority-{priority.lower()}'>{priority} Priority Lead</span>", unsafe_allow_html=True)
                    st.markdown("**Scientific Rationale:**")
                    for r in rationale: st.write(f"• {r}")
                    st.markdown("**Actionable Next Steps:**")
                    for s in next_steps: st.write(f"• {s}")

                with st.container(border=True):
                    st.markdown("### A. Target Affinity & Efficiency")
                    cA, cB = st.columns([1, 1.5])
                    cA.image(mol_to_png(mol), use_container_width=True)
                    with cB:
                        m1, m2 = st.columns(2)
                        m1.metric("Pred pIC50", f"{p_mean:.2f}")
                        m2.metric("IC50 (nM)", f"{pic50_to_ic50_nM(p_mean):.1f}")
                        m3, m4 = st.columns(2)
                        m3.metric("Ligand Efficiency (LE)", f"{le_score:.2f}")
                        m4.metric("Binding Effic. (BEI)", f"{bei_score:.1f}")

                with st.container(border=True):
                    st.markdown("### B. Drug-Likeness & ADMET Profiling")
                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("Mol. Weight", f"{mw:.1f}")
                    d2.metric("cLogP", f"{clogp:.2f}")
                    d3.metric("TPSA", f"{tpsa:.1f}")
                    d4.metric("Ro5 Violations", str(ro5))
                    
                    st.markdown(f"**Murcko Scaffold Base:** `{Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))}`")

                if compute_ad and not neighbors.empty:
                    with st.expander("View Nearest Training Neighbors & Applicability Domain", expanded=False):
                        st.metric("Max Tanimoto Similarity", f"{max_sim:.3f}")
                        st.dataframe(neighbors, use_container_width=True, hide_index=True)

# =========================
# Tab 2: Batch scoring
# =========================
with tab2:
    st.subheader("Virtual Screening: High-Throughput Batch Scoring")
    st.write("Upload a CSV library with a column named `smiles`. Output will append predictions, AD, and MedChem Priority.")

    file = st.file_uploader("Upload Compound Library (.csv)", type=["csv"])
    if file is not None:
        batch = pd.read_csv(file)
        if "smiles" not in batch.columns:
            st.error("CSV format rejected: Must include a `smiles` column.")
        else:
            if st.button("Initialize Virtual Screen", type="primary"):
                rows = []
                with st.spinner(f"Scoring {len(batch)} molecules..."):
                    for s in batch["smiles"].astype(str).tolist():
                        r = score_smiles_row(s, model, pains_catalog, df_f, strip_salts, compute_ad)
                        rows.append(asdict(r))

                out = pd.DataFrame(rows)
                st.success("Virtual Screening Complete.")
                st.dataframe(out.head(25), use_container_width=True)

                st.download_button(
                    "Download Triage Results (CSV)",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="cdk2_virtual_screen_results.csv",
                    mime="text/csv",
                )

# =========================
# Tab 3: Chemical space
# =========================
with tab3:
    st.subheader("Structure-Activity Relationship (SAR) Landscape")
    if not df_f.empty:
        fig = px.scatter(
            df_f, x="pic50", y="n_measurements", size="n_measurements", 
            hover_name="molecule_chembl_id" if "molecule_chembl_id" in df_f.columns else None,
            template="plotly_white", labels={"pic50": "Experimental pIC50", "n_measurements": "Assay Measurements"},
            title="Evidence Subset Topology"
        )
        st.plotly_chart(fig, use_container_width=True)
