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
# 1. PAGE CONFIG
# =========================
st.set_page_config(page_title="ChemBLitz Professional | CDK2 Suite", layout="wide")

# =========================
# 2. SAFE PROFESSIONAL CSS (NO OVERLAP)
# =========================
st.markdown(
    """
<style>
html, body {
  font-family: "Times New Roman", Times, serif !important;
  font-size: 18px !important;
  line-height: 1.45 !important;
}
.stMarkdown, .stMarkdown p, .stMarkdown li {
  font-family: "Times New Roman", Times, serif !important;
  font-size: 18px !important;
  line-height: 1.55 !important;
}
section[data-testid="stSidebar"] * {
  font-family: "Times New Roman", Times, serif !important;
  font-size: 16px !important;
  line-height: 1.35 !important;
}
button[data-baseweb="tab"] { font-size: 16px !important; }
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
div[data-testid="stExpander"] summary {
  font-size: 16px !important;
  line-height: 1.25 !important;
}
.stButton > button {
  font-size: 16px !important;
  border-radius: 6px !important;
  padding: 0.55em 1em !important;
}
.block-container { padding-top: 1.2rem !important; }
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

TRIAGE_PIC50 = 7.0
TRIAGE_STD = 0.60
TRIAGE_SIM = 0.30

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

    required = {"smiles", "pic50", "n_measurements", "inchikey", "molecule_chembl_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    df = df.copy()
    df["pic50"] = pd.to_numeric(df["pic50"], errors="coerce")
    df["n_measurements"] = pd.to_numeric(df["n_measurements"], errors="coerce").fillna(0).astype(int)
    if "ic50_nM" in df.columns:
        df["ic50_nM"] = pd.to_numeric(df["ic50_nM"], errors="coerce")

    return model, df


# =========================
# 5. CHEM HELPERS
# =========================
def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True)


def inchikey(mol: Chem.Mol) -> str:
    try:
        return Chem.inchi.MolToInchiKey(mol)
    except Exception:
        return "N/A"


def keep_largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return mol
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


def potency_band(pic50: float) -> str:
    if pic50 >= 8:
        return "Strong"
    if pic50 >= 6:
        return "Moderate"
    return "Weak"


def uncertainty_band(std: float) -> str:
    if std <= 0.35:
        return "Low (good)"
    if std <= 0.60:
        return "Medium"
    return "High (warning)"


def ad_band(max_sim: float) -> str:
    if max_sim >= SIM_HIGH:
        return "In-domain"
    if max_sim >= SIM_MED:
        return "Borderline"
    return "Out-of-domain risk"


def calculate_ligand_efficiency(pic50: float, mol: Chem.Mol) -> float:
    n_heavy = mol.GetNumHeavyAtoms()
    return (1.37 * pic50 / n_heavy) if n_heavy > 0 else 0.0


def check_pains(mol: Chem.Mol, catalog: FilterCatalog) -> list[str]:
    entries = catalog.GetMatches(mol)
    return [e.GetDescription() for e in entries]


def ro5_violations(mol: Chem.Mol) -> int:
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    v = 0
    v += 1 if mw > 500 else 0
    v += 1 if logp > 5 else 0
    v += 1 if hbd > 5 else 0
    v += 1 if hba > 10 else 0
    return v


def veber_pass(mol: Chem.Mol) -> bool:
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    return (tpsa <= 140.0) and (rotb <= 10)


@st.cache_data(show_spinner=False)
def chembl_pref_name_from_chembl_id(chembl_id: str) -> Optional[str]:
    try:
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
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
        if m is None:
            continue
        fps.append(morgan_fp(m))
        idx.append(i)
    return fps, np.array(idx, dtype=int)


def compute_similarity_and_neighbors(query_fp, *, df_evidence: pd.DataFrame, topk: int):
    fps, idx_map = build_dataset_fps(df_evidence)
    if len(fps) == 0:
        return 0.0, 0.0, 0, 0, 0, pd.DataFrame()

    sims = np.array(DataStructs.BulkTanimotoSimilarity(query_fp, fps), dtype=float)

    max_sim = float(sims.max()) if len(sims) else 0.0
    mean_sim = float(sims.mean()) if len(sims) else 0.0
    n_ge_05 = int((sims >= 0.5).sum())
    n_ge_04 = int((sims >= 0.4).sum())
    n_ge_03 = int((sims >= 0.3).sum())

    top = np.argsort(-sims)[:topk]
    rows = []
    for j in top:
        row = df_evidence.iloc[int(idx_map[j])]
        chembl_id = str(row.get("molecule_chembl_id", "")).strip()
        rows.append(
            {
                "similarity": float(sims[j]),
                "molecule_chembl_id": chembl_id,
                "chembl_url": chembl_molecule_url(chembl_id) if chembl_id else "",
                "pic50_exp": float(row.get("pic50", np.nan)),
                "ic50_nM_exp": float(row.get("ic50_nM", np.nan)) if "ic50_nM" in row else np.nan,
                "n_measurements": int(row.get("n_measurements", 0)),
                "smiles": str(row.get("smiles", "")),
            }
        )
    neigh = pd.DataFrame(rows)
    return max_sim, mean_sim, n_ge_05, n_ge_04, n_ge_03, neigh


@st.cache_resource
def add_scaffold_column(df: pd.DataFrame) -> pd.DataFrame:
    """Precompute Murcko scaffolds once (cached)."""
    out = df.copy()
    scaffolds = []
    for s in out["smiles"].astype(str).tolist():
        m = Chem.MolFromSmiles(s)
        if m is None:
            scaffolds.append("")
            continue
        sc = MurckoScaffold.GetScaffoldForMol(m)
        scaffolds.append(Chem.MolToSmiles(sc) if sc is not None else "")
    out["murcko_scaffold"] = scaffolds
    return out


def scaffold_stats(df_scaf: pd.DataFrame, scaffold_smiles: str) -> Tuple[int, float, float, float, pd.DataFrame]:
    sub = df_scaf[df_scaf["murcko_scaffold"] == scaffold_smiles].copy()
    sub = sub.dropna(subset=["pic50"])
    if len(sub) == 0:
        return 0, np.nan, np.nan, np.nan, pd.DataFrame()

    count = int(len(sub))
    p_min = float(sub["pic50"].min())
    p_med = float(sub["pic50"].median())
    p_max = float(sub["pic50"].max())

    top = sub.sort_values("pic50", ascending=False).head(5)
    top_view = top[["molecule_chembl_id", "pic50", "n_measurements", "smiles"]].copy()
    top_view["chembl_url"] = top_view["molecule_chembl_id"].astype(str).apply(
        lambda x: chembl_molecule_url(x) if x else ""
    )
    return count, p_min, p_med, p_max, top_view


def make_decision_summary(
    *,
    pred_pic50: float,
    pred_std: float,
    max_sim: float,
    pains_hits: list[str],
    le: float,
    mol: Chem.Mol,
):
    mw = float(Descriptors.MolWt(mol))
    clogp = float(Descriptors.MolLogP(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    rotb = int(rdMolDescriptors.CalcNumRotatableBonds(mol))

    rationale = []
    next_steps = []

    rationale.append(
        f"Predicted potency: {potency_band(pred_pic50)} "
        f"(pIC50={pred_pic50:.2f}, IC50≈{pic50_to_ic50_nM(pred_pic50):.1f} nM)."
    )
    rationale.append(f"Uncertainty (tree σ): {pred_std:.3f} → {uncertainty_band(pred_std)}.")
    rationale.append(f"Applicability domain: max similarity={max_sim:.3f} → {ad_band(max_sim)}.")
    rationale.append(f"Ligand Efficiency (LE)={le:.2f} (≥0.30 acceptable; ≥0.35 strong).")

    if pains_hits:
        rationale.append(f"PAINS alerts: {', '.join(pains_hits)}.")
        next_steps.append("Run orthogonal counterscreens and confirm signal is not assay interference.")
        next_steps.append("Consider redesign to remove PAINS motifs if feasible.")

    if mw > 500:
        next_steps.append("MolWt > 500: consider scaffold simplification.")
    if clogp > 4.5:
        next_steps.append("High cLogP: increase polarity / reduce hydrophobes (solubility & clearance risk).")
    if tpsa > 140:
        next_steps.append("TPSA > 140: permeability risk; reduce PSA / mask donors.")
    if rotb > 10:
        next_steps.append("RotB > 10: consider rigidification to improve PK and binding entropy.")
    if max_sim < SIM_MED:
        next_steps.append("Low similarity: out-of-domain risk; validate experimentally before optimization.")
        next_steps.append("Search for closer analogs (within-domain) to improve confidence.")

    if not next_steps:
        next_steps.append("Proceed to selectivity profiling (CDK1/4/6) + early ADME screens.")

    # Priority scoring
    score = 0
    score += 2 if pred_pic50 >= 7.0 else (1 if pred_pic50 >= 6.0 else 0)
    score += 1 if pred_std <= TRIAGE_STD else 0
    score += 1 if max_sim >= TRIAGE_SIM else 0
    score -= 2 if pains_hits else 0

    if score >= 3:
        priority = "High"
    elif score >= 2:
        priority = "Medium"
    else:
        priority = "Low"

    return priority, rationale, next_steps


# =========================
# 7. BATCH SCORING
# =========================
@dataclass
class BatchRow:
    ok: bool
    error: str
    id: str
    smiles_in: str
    smiles_used: str
    pred_pic50: float
    pred_ic50_nM: float
    pred_std: float
    potency: str
    ligand_efficiency: float
    max_sim: float
    mean_sim: float
    n_sim_ge_0_5: int
    n_sim_ge_0_4: int
    n_sim_ge_0_3: int
    ad_band: str
    pains_count: int
    pains_hits: str
    scaffold_smiles: str
    scaffold_count_in_dataset: int
    ro5_violations: int
    veber_pass: bool
    molwt: float
    clogp: float
    tpsa: float
    rotb: int
    hbd: int
    hba: int
    rings: int
    qed: float
    chembl_id: str
    chembl_name: str
    chembl_url: str
    priority: str
    pass_triage: bool


def score_smiles_row(
    smiles: str,
    row_id: str,
    *,
    model,
    pains_catalog,
    df_evidence: pd.DataFrame,
    df_full: pd.DataFrame,
    df_scaf: pd.DataFrame,
    strip_salts: bool,
    compute_ad: bool,
    topk_neighbors: int,
) -> BatchRow:
    smiles_in = (smiles or "").strip()
    if not smiles_in:
        return BatchRow(
            ok=False,
            error="Empty SMILES",
            id=row_id,
            smiles_in="",
            smiles_used="",
            pred_pic50=np.nan,
            pred_ic50_nM=np.nan,
            pred_std=np.nan,
            potency="",
            ligand_efficiency=np.nan,
            max_sim=np.nan,
            mean_sim=np.nan,
            n_sim_ge_0_5=0,
            n_sim_ge_0_4=0,
            n_sim_ge_0_3=0,
            ad_band="",
            pains_count=0,
            pains_hits="",
            scaffold_smiles="",
            scaffold_count_in_dataset=0,
            ro5_violations=0,
            veber_pass=False,
            molwt=np.nan,
            clogp=np.nan,
            tpsa=np.nan,
            rotb=0,
            hbd=0,
            hba=0,
            rings=0,
            qed=np.nan,
            chembl_id="",
            chembl_name="",
            chembl_url="",
            priority="",
            pass_triage=False,
        )

    mol = mol_from_smiles(smiles_in)
    if mol is None:
        return BatchRow(
            ok=False,
            error="Invalid SMILES",
            id=row_id,
            smiles_in=smiles_in,
            smiles_used="",
            pred_pic50=np.nan,
            pred_ic50_nM=np.nan,
            pred_std=np.nan,
            potency="",
            ligand_efficiency=np.nan,
            max_sim=np.nan,
            mean_sim=np.nan,
            n_sim_ge_0_5=0,
            n_sim_ge_0_4=0,
            n_sim_ge_0_3=0,
            ad_band="",
            pains_count=0,
            pains_hits="",
            scaffold_smiles="",
            scaffold_count_in_dataset=0,
            ro5_violations=0,
            veber_pass=False,
            molwt=np.nan,
            clogp=np.nan,
            tpsa=np.nan,
            rotb=0,
            hbd=0,
            hba=0,
            rings=0,
            qed=np.nan,
            chembl_id="",
            chembl_name="",
            chembl_url="",
            priority="",
            pass_triage=False,
        )

    if strip_salts:
        mol = keep_largest_fragment(mol)

    smiles_used = canonical_smiles(mol)

    pred_pic50, pred_std, fp = rf_predict(model, mol)
    pred_ic50 = pic50_to_ic50_nM(pred_pic50)
    le = calculate_ligand_efficiency(pred_pic50, mol)
    pains_hits = check_pains(mol, pains_catalog)

    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold_mol) if scaffold_mol is not None else ""

    # scaffold counts in dataset
    sc_count, *_ = scaffold_stats(df_scaf, scaffold_smiles) if scaffold_smiles else (0, np.nan, np.nan, np.nan, pd.DataFrame())

    # dataset match by inchikey
    cid = ""
    cname = ""
    curl = ""
    ik = inchikey(mol)
    m = df_full[df_full["inchikey"].astype(str) == ik]
    if len(m) > 0:
        cid = str(m.iloc[0].get("molecule_chembl_id", "")).strip()
        if cid:
            cname = chembl_pref_name_from_chembl_id(cid) or ""
            curl = chembl_molecule_url(cid)

    max_sim = 0.0
    mean_sim = 0.0
    n05 = n04 = n03 = 0
    if compute_ad:
        max_sim, mean_sim, n05, n04, n03, _ = compute_similarity_and_neighbors(fp, df_evidence=df_evidence, topk=topk_neighbors)

    priority, *_ = make_decision_summary(
        pred_pic50=pred_pic50,
        pred_std=pred_std,
        max_sim=max_sim if compute_ad else 0.0,
        pains_hits=pains_hits,
        le=le,
        mol=mol,
    )

    pass_triage = bool(
        (pred_pic50 >= TRIAGE_PIC50)
        and (pred_std <= TRIAGE_STD)
        and ((max_sim >= TRIAGE_SIM) if compute_ad else True)
        and (len(pains_hits) == 0)
    )

    return BatchRow(
        ok=True,
        error="",
        id=row_id,
        smiles_in=smiles_in,
        smiles_used=smiles_used,
        pred_pic50=float(pred_pic50),
        pred_ic50_nM=float(pred_ic50),
        pred_std=float(pred_std),
        potency=potency_band(pred_pic50),
        ligand_efficiency=float(le),
        max_sim=float(max_sim),
        mean_sim=float(mean_sim),
        n_sim_ge_0_5=int(n05),
        n_sim_ge_0_4=int(n04),
        n_sim_ge_0_3=int(n03),
        ad_band=ad_band(max_sim) if compute_ad else "",
        pains_count=len(pains_hits),
        pains_hits="; ".join(pains_hits),
        scaffold_smiles=scaffold_smiles,
        scaffold_count_in_dataset=int(sc_count),
        ro5_violations=ro5_violations(mol),
        veber_pass=veber_pass(mol),
        molwt=float(Descriptors.MolWt(mol)),
        clogp=float(Descriptors.MolLogP(mol)),
        tpsa=float(rdMolDescriptors.CalcTPSA(mol)),
        rotb=int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        hbd=int(rdMolDescriptors.CalcNumHBD(mol)),
        hba=int(rdMolDescriptors.CalcNumHBA(mol)),
        rings=int(rdMolDescriptors.CalcNumRings(mol)),
        qed=float(QED.qed(mol)),
        chembl_id=cid,
        chembl_name=cname,
        chembl_url=curl,
        priority=priority,
        pass_triage=pass_triage,
    )


# =========================
# 8. APP START
# =========================
st.title("CDK2 Pharmacological Diagnostic Suite")
st.markdown("#### Prediction + Evidence + Interpretation for CDK2 inhibitor discovery")
st.divider()

try:
    model, df = load_assets()
    pains_catalog = get_pains_filter()
except Exception as e:
    st.error(f"Critical System Failure: {e}")
    st.stop()

df_scaf = add_scaffold_column(df)

# Sidebar controls
with st.sidebar:
    st.header("Controls")

    mode = st.radio("Mode", ["Single molecule", "Batch (CSV)"], index=0)

    st.subheader("Evidence subset (applies to AD + neighbors)")
    p_min, p_max = float(df["pic50"].min()), float(df["pic50"].max())
    pic_range = st.slider("pIC50 inclusion range", p_min, p_max, (p_min, p_max))
    min_meas = st.slider("Min measurements", 1, int(df["n_measurements"].max()), 1)

    st.subheader("Analysis options")
    strip_salts = st.checkbox("Keep largest fragment (salt stripping)", value=True)
    compute_ad = st.checkbox("Compute applicability domain + neighbors", value=True)
    topk_neighbors = st.slider("Top-K neighbors", 5, 25, 10)

    st.subheader("Batch triage thresholds")
    st.caption("Used to set pass/fail flags in batch output.")
    triage_pic50 = st.number_input("Min predicted pIC50", value=float(TRIAGE_PIC50), step=0.1)
    triage_std = st.number_input("Max σ (uncertainty)", value=float(TRIAGE_STD), step=0.05)
    triage_sim = st.number_input("Min max similarity", value=float(TRIAGE_SIM), step=0.05)
    TRIAGE_PIC50 = float(triage_pic50)
    TRIAGE_STD = float(triage_std)
    TRIAGE_SIM = float(triage_sim)

df_f = df[(df["pic50"] >= pic_range[0]) & (df["pic50"] <= pic_range[1]) & (df["n_measurements"] >= min_meas)].copy()

tab1, tab2, tab3, tab4 = st.tabs(["Lead Diagnostic", "Library Triage (CSV)", "Chemical Space", "Methodology"])

# =========================
# Tab 1: Single molecule
# =========================
with tab1:
    col_input, col_results = st.columns([1, 1.4], gap="large")

    with col_input:
        st.subheader("I. Query definition")

        top_binders = df.sort_values("pic50", ascending=False).head(5)
        refs = {
            f"Reference: {r.get('molecule_chembl_id','')} (pIC50 {r['pic50']:.1f})": r["smiles"]
            for _, r in top_binders.iterrows()
        }
        selected_ref = st.selectbox("Select validated reference ligand", ["None"] + list(refs.keys()))
        if selected_ref != "None":
            st.session_state["query_smiles"] = refs[selected_ref]

        target_smiles = st.text_input("Input SMILES", value=st.session_state.get("query_smiles", ""))

        execute = st.button("Execute diagnostic analysis", type="primary", use_container_width=True)

        st.markdown("---")
        st.subheader("II. Evidence subset status")
        st.info(f"Evidence subset size: {len(df_f)} compounds (from {len(df)} total)")

        with st.expander("What do these numbers mean?", expanded=False):
            st.markdown(
                """
- **pIC50**: -log10(IC50 in molar). +1 pIC50 ≈ 10× stronger potency.
- **IC50 (nM)**: 10^(9 - pIC50). Reported for intuition.
- **σ (uncertainty)**: std across RF trees (model disagreement). Not a calibrated CI.
- **Applicability Domain (AD)**: similarity to training chemistry. Low similarity = extrapolation risk.
- **Ligand Efficiency (LE)**: potency normalized by size (heavy atom count). Useful for lead-likeness.
- **PAINS**: substructures associated with assay interference. Treat as risk flags, not absolute truth.
- **Scaffold evidence**: how often this Murcko scaffold appears in the dataset + its known activity range.
"""
            )

    with col_results:
        if execute and target_smiles:
            mol = mol_from_smiles(target_smiles)
            if mol is None:
                st.error("Invalid structure: SMILES parsing failed.")
            else:
                if strip_salts:
                    mol = keep_largest_fragment(mol)

                smiles_can = canonical_smiles(mol)
                ik = inchikey(mol)

                p_mean, p_std, fp = rf_predict(model, mol)
                le_score = calculate_ligand_efficiency(p_mean, mol)
                pains_hits = check_pains(mol, pains_catalog)

                scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold_mol) if scaffold_mol is not None else ""

                # exact match (full dataset)
                match = df[df["inchikey"].astype(str) == ik]
                in_dataset = len(match) > 0
                exp_pic50 = float(match["pic50"].median()) if in_dataset else np.nan
                exp_n = int(match["n_measurements"].max()) if in_dataset else 0
                exp_chembl = str(match.iloc[0]["molecule_chembl_id"]).strip() if in_dataset else ""
                exp_name = chembl_pref_name_from_chembl_id(exp_chembl) if exp_chembl else None

                # AD + neighbors evidence
                max_sim = mean_sim = 0.0
                n05 = n04 = n03 = 0
                neighbors = pd.DataFrame()
                if compute_ad:
                    max_sim, mean_sim, n05, n04, n03, neighbors = compute_similarity_and_neighbors(fp, df_evidence=df_f, topk=topk_neighbors)

                # Decision summary
                priority, rationale, next_steps = make_decision_summary(
                    pred_pic50=p_mean,
                    pred_std=p_std,
                    max_sim=max_sim if compute_ad else 0.0,
                    pains_hits=pains_hits,
                    le=le_score,
                    mol=mol,
                )

                with st.container(border=True):
                    st.markdown("## Decision summary")
                    if priority == "High":
                        st.success(f"Priority: **{priority}**")
                    elif priority == "Medium":
                        st.warning(f"Priority: **{priority}**")
                    else:
                        st.info(f"Priority: **{priority}**")

                    st.markdown("**Rationale**")
                    for r in rationale:
                        st.write(f"- {r}")

                    st.markdown("**Recommended next steps**")
                    for s in next_steps:
                        st.write(f"- {s}")

                with st.container(border=True):
                    st.markdown("### Identifiers & dataset match")
                    st.write("Canonical SMILES:")
                    st.code(smiles_can, language="text")
                    st.write("InChIKey:", ik)

                    if in_dataset:
                        st.success(f"Exact match found in dataset: {exp_chembl} (n_measurements={exp_n})")
                        if exp_name:
                            st.write("ChEMBL pref_name:", exp_name)
                        st.write("ChEMBL URL:", chembl_molecule_url(exp_chembl))
                        st.write(f"Experimental pIC50 (median): {exp_pic50:.2f}")
                        st.write(f"Prediction error vs exp (ΔpIC50): {(p_mean - exp_pic50):+.2f}")
                    else:
                        st.warning("No exact match in dataset (by InChIKey).")

                with st.container(border=True):
                    st.markdown("### A. Affinity & efficiency")
                    cA, cB = st.columns([1, 1.5])
                    cA.image(mol_to_png(mol), caption="Query structure (RDKit 2D)")
                    with cB:
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Pred pIC50", f"{p_mean:.2f}")
                        m2.metric("Pred IC50 (nM)", f"{pic50_to_ic50_nM(p_mean):.1f}")
                        m3.metric("σ (trees)", f"{p_std:.3f}")
                        m4.metric("LE", f"{le_score:.2f}")

                        st.caption(f"Potency band: **{potency_band(p_mean)}**")
                        st.caption(f"Uncertainty band: **{uncertainty_band(p_std)}**")

                with st.container(border=True):
                    st.markdown("### B. Applicability domain (evidence)")
                    if compute_ad:
                        a1, a2, a3, a4, a5 = st.columns(5)
                        a1.metric("Max sim", f"{max_sim:.3f}")
                        a2.metric("Mean sim", f"{mean_sim:.3f}")
                        a3.metric("≥0.50", str(n05))
                        a4.metric("≥0.40", str(n04))
                        a5.metric("≥0.30", str(n03))

                        band = ad_band(max_sim)
                        if band == "In-domain":
                            st.success(band)
                        elif band == "Borderline":
                            st.warning(band)
                        else:
                            st.error(band)

                        if len(neighbors):
                            pic = neighbors["pic50_exp"].dropna()
                            if len(pic):
                                st.caption(
                                    f"Neighbor experimental pIC50 spread (Top-{min(len(neighbors), topk_neighbors)}): "
                                    f"min={pic.min():.2f}, median={pic.median():.2f}, max={pic.max():.2f}"
                                )

                            st.markdown("**Top nearest neighbors (experimental evidence)**")
                            st.dataframe(neighbors, use_container_width=True, hide_index=True)
                    else:
                        st.info("AD disabled in sidebar.")

                with st.container(border=True):
                    st.markdown("### C. Scaffold evidence (series context)")
                    if scaffold_smiles:
                        count, pmin, pmed, pmax, top_scaf = scaffold_stats(df_scaf, scaffold_smiles)
                        st.write("Murcko scaffold:")
                        st.code(scaffold_smiles, language="text")
                        st.write(f"Scaffold occurrences in dataset: **{count}**")
                        if count > 0:
                            st.write(f"Scaffold activity (pIC50): min={pmin:.2f}, median={pmed:.2f}, max={pmax:.2f}")
                            st.markdown("**Top scaffold exemplars**")
                            st.dataframe(top_scaf, use_container_width=True, hide_index=True)
                        else:
                            st.info("Scaffold not found in dataset (new chemotype).")
                    else:
                        st.info("No scaffold computed for this structure.")

                with st.container(border=True):
                    st.markdown("### D. Structural integrity & alerts")
                    if pains_hits:
                        st.error(f"PAINS alerts: {', '.join(pains_hits)}")
                    else:
                        st.success("No PAINS alerts detected.")

                with st.container(border=True):
                    st.markdown("### E. Developability / property flags")
                    mw = float(Descriptors.MolWt(mol))
                    clogp = float(Descriptors.MolLogP(mol))
                    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
                    rotb = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
                    hbd = int(rdMolDescriptors.CalcNumHBD(mol))
                    hba = int(rdMolDescriptors.CalcNumHBA(mol))
                    rings = int(rdMolDescriptors.CalcNumRings(mol))
                    qed = float(QED.qed(mol))

                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("MolWt", f"{mw:.1f}")
                    d2.metric("cLogP", f"{clogp:.2f}")
                    d3.metric("TPSA", f"{tpsa:.1f}")
                    d4.metric("QED", f"{qed:.2f}")

                    flags = []
                    flags.append(("Ro5 violations", str(ro5_violations(mol))))
                    flags.append(("Veber", "PASS" if veber_pass(mol) else "FAIL"))
                    flags.append(("Permeability risk", "High" if tpsa > 140 else "OK"))
                    flags.append(("Solubility risk", "High" if clogp > 4.5 else "OK"))
                    flags.append(("Flexibility risk", "High" if rotb > 10 else "OK"))
                    flags.append(("Size risk", "High" if mw > 500 else "OK"))

                    c1, c2, c3 = st.columns(3)
                    for i, (k, v) in enumerate(flags):
                        (c1 if i % 3 == 0 else c2 if i % 3 == 1 else c3).write(f"**{k}:** {v}")

        else:
            st.info("System ready. Define query parameters and execute diagnostic.")

# =========================
# Tab 2: Library triage (CSV)
# =========================
with tab2:
    st.subheader("Library triage (Batch CSV)")
    st.write(
        "Real workflows involve scoring *libraries/series* (dozens to thousands of molecules). "
        "Upload a CSV containing a `smiles` column (optional `id` column)."
    )

    file = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")
    if file is None:
        st.info("Awaiting upload.")
    else:
        try:
            batch = pd.read_csv(file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        if "smiles" not in batch.columns:
            st.error("CSV must include a `smiles` column.")
            st.stop()

        if "id" not in batch.columns:
            batch = batch.copy()
            batch["id"] = [f"row_{i+1}" for i in range(len(batch))]

        st.info(f"Uploaded rows: {len(batch)} | Evidence subset size: {len(df_f)}")

        if st.button("Run library triage", type="primary"):
            rows = []
            with st.spinner("Scoring molecules..."):
                for rid, smi in zip(batch["id"].astype(str).tolist(), batch["smiles"].astype(str).tolist()):
                    r = score_smiles_row(
                        smi,
                        rid,
                        model=model,
                        pains_catalog=pains_catalog,
                        df_evidence=df_f,
                        df_full=df,
                        df_scaf=df_scaf,
                        strip_salts=strip_salts,
                        compute_ad=compute_ad,
                        topk_neighbors=topk_neighbors,
                    )
                    rows.append(asdict(r))

            out = pd.DataFrame(rows)
            st.success("Batch scoring complete.")

            # Show summary
            if "pass_triage" in out.columns:
                st.write(
                    "Pass triage:",
                    f"{int(out['pass_triage'].sum())} / {len(out)}",
                )

            st.dataframe(out, use_container_width=True, hide_index=True)

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download full results CSV",
                data=csv_bytes,
                file_name="cdk2_library_triage_results.csv",
                mime="text/csv",
            )

            if "pass_triage" in out.columns:
                top = out[out["pass_triage"] == True].copy()  # noqa: E712
                if len(top) > 0:
                    top_bytes = top.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download PASS-TRIAGE only CSV",
                        data=top_bytes,
                        file_name="cdk2_library_pass_triage.csv",
                        mime="text/csv",
                    )

# =========================
# Tab 3: Chemical space
# =========================
with tab3:
    st.subheader("Chemical Space (Evidence subset)")
    if "ic50_nM" in df_f.columns:
        fig = px.scatter(
            df_f,
            x="pic50",
            y="ic50_nM",
            size="n_measurements",
            hover_name="molecule_chembl_id",
            template="plotly_white",
            labels={"pic50": "Experimental pIC50", "ic50_nM": "IC50 (nM)"},
        )
    else:
        fig = px.scatter(
            df_f,
            x="pic50",
            y="n_measurements",
            size="n_measurements",
            hover_name="molecule_chembl_id",
            template="plotly_white",
            labels={"pic50": "Experimental pIC50", "n_measurements": "n_measurements"},
        )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# Tab 4: Methodology
# =========================
with tab4:
    st.subheader("Methodology & interpretation")
    st.markdown(
        f"""
### What this tool does
- Predicts **CDK2 pIC50** with a RandomForest model trained on **Morgan fingerprints** (radius={FP_RADIUS}, {FP_NBITS} bits).
- Provides **uncertainty** as tree-to-tree disagreement (σ).
- Provides **evidence** using **nearest-neighbor similarity** and scaffold context.

### Why "Library triage (CSV)" exists
Real discovery work scores **many molecules**: analog series, virtual screening hits, vendor catalogs.
Batch mode supports:
- consistent scoring of libraries
- filtering with **pass_triage** flags
- exporting results for collaboration

### Interpreting output (practical)
- **pIC50**: +1 ≈ 10× potency shift.
- **σ**: not a calibrated CI—use it as a warning flag for model disagreement.
- **AD (similarity)**: low similarity means extrapolation risk.
- **Scaffold evidence**: tells if the chemotype exists in the dataset and its known activity range.
- **PAINS**: risk flags; confirm with orthogonal assays.

### Suggested real workflow
1. Score a library.
2. Filter: **no PAINS**, **pIC50 high**, **σ low**, **in-domain**.
3. Prioritize top candidates for confirmatory assay + selectivity + early ADME.
"""
    )
