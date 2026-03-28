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
# 2. SAFE PROFESSIONAL CSS (NO OVERLAP)
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
  font-size: 16px !important;
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

/* Expander header: fix chevron/text collision */
div[data-testid="stExpander"] summary {
  font-size: 16px !important;
  line-height: 1.25 !important;
}

/* Buttons */
.stButton > button {
  font-size: 16px !important;
  border-radius: 6px !important;
  padding: 0.55em 1em !important;
}

/* Reduce excessive top padding */
.block-container {
  padding-top: 1.2rem !important;
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

    # Validate required columns
    required = {"smiles", "pic50", "n_measurements"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    # Normalize dtypes
    df = df.copy()
    df["pic50"] = pd.to_numeric(df["pic50"], errors="coerce")
    df["n_measurements"] = pd.to_numeric(df["n_measurements"], errors="coerce").fillna(0).astype(int)

    # Optional columns: inchikey, molecule_chembl_id, ic50_nM
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
        return 0.0, 0.0, 0, pd.DataFrame()

    sims = np.array(DataStructs.BulkTanimotoSimilarity(query_fp, fps), dtype=float)

    max_sim = float(sims.max()) if len(sims) else 0.0
    mean_sim = float(sims.mean()) if len(sims) else 0.0
    n_ge = int((sims >= SIM_NEIGHBOR).sum())

    top = np.argsort(-sims)[:topk]
    rows = []
    for j in top:
        row = df_evidence.iloc[int(idx_map[j])]
        rows.append(
            {
                "similarity": float(sims[j]),
                "molecule_chembl_id": str(row.get("molecule_chembl_id", "")),
                "pic50_exp": float(row.get("pic50", np.nan)),
                "ic50_nM_exp": float(row.get("ic50_nM", np.nan)),
                "n_measurements": int(row.get("n_measurements", 0)),
                "smiles": str(row.get("smiles", "")),
            }
        )
    return max_sim, mean_sim, n_ge, pd.DataFrame(rows)


def make_decision_summary(
    *,
    pred_pic50: float,
    pred_std: float,
    max_sim: float,
    pains_hits: list[str],
    le: float,
    mw: float,
    clogp: float,
    tpsa: float,
    rotb: int,
):
    rationale = []
    next_steps = []

    pot = potency_band(pred_pic50)
    unc = uncertainty_band(pred_std)
    dom = ad_band(max_sim)

    rationale.append(f"Predicted potency: {pot} (pIC50={pred_pic50:.2f}, IC50≈{pic50_to_ic50_nM(pred_pic50):.1f} nM).")
    rationale.append(f"Uncertainty (tree σ): {pred_std:.3f} → {unc}.")
    rationale.append(f"Applicability domain: max similarity={max_sim:.3f} → {dom}.")
    rationale.append(f"Ligand Efficiency (LE)={le:.2f} (rule-of-thumb: ≥0.30 acceptable; ≥0.35 strong).")

    if pains_hits:
        rationale.append(f"PAINS alerts detected: {', '.join(pains_hits)}.")
        next_steps.append("Run orthogonal counterscreens and confirm signal is not assay interference.")
        next_steps.append("Consider redesign to remove flagged PAINS motifs if feasible.")

    # Developability heuristics
    if mw > 500:
        next_steps.append("MolWt > 500: consider scaffold simplification.")
    if clogp > 4.5:
        next_steps.append("High cLogP: increase polarity / reduce hydrophobes (solubility & clearance risk).")
    if tpsa > 140:
        next_steps.append("TPSA > 140: permeability risk; reduce PSA / mask donors.")
    if rotb > 10:
        next_steps.append("RotB > 10: consider rigidification to improve PK and binding entropy.")

    if max_sim < SIM_MED:
        next_steps.append("Low similarity: treat as out-of-domain; validate experimentally before optimization.")
        next_steps.append("Search for closer analogs to increase confidence in predictions.")

    if not next_steps:
        next_steps.append("Proceed to selectivity profiling (CDK1/4/6) + early ADME screens.")

    # Priority scoring
    score = 0
    score += 2 if pred_pic50 >= 7.0 else (1 if pred_pic50 >= 6.0 else 0)
    score += 1 if pred_std <= 0.60 else 0
    score += 1 if max_sim >= SIM_MED else 0
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
    smiles_in: str
    smiles_used: str
    pred_pic50: float
    pred_ic50_nM: float
    pred_std: float
    potency: str
    ligand_efficiency: float
    max_sim: float
    mean_sim: float
    neighbors_ge_0_4: int
    ad_band: str
    pains_count: int
    pains_hits: str
    scaffold_smiles: str
    molwt: float
    clogp: float
    tpsa: float
    rotb: int
    hbd: int
    hba: int
    qed: float
    chembl_id: str
    chembl_name: str
    chembl_url: str


def score_smiles_row(
    smiles: str,
    *,
    model,
    pains_catalog,
    df_evidence: pd.DataFrame,
    df_full: pd.DataFrame,
    strip_salts: bool,
    compute_ad: bool,
    topk_neighbors: int,
) -> BatchRow:
    smiles_in = (smiles or "").strip()
    if not smiles_in:
        return BatchRow(
            ok=False,
            error="Empty SMILES",
            smiles_in="",
            smiles_used="",
            pred_pic50=np.nan,
            pred_ic50_nM=np.nan,
            pred_std=np.nan,
            potency="",
            ligand_efficiency=np.nan,
            max_sim=np.nan,
            mean_sim=np.nan,
            neighbors_ge_0_4=0,
            ad_band="",
            pains_count=0,
            pains_hits="",
            scaffold_smiles="",
            molwt=np.nan,
            clogp=np.nan,
            tpsa=np.nan,
            rotb=0,
            hbd=0,
            hba=0,
            qed=np.nan,
            chembl_id="",
            chembl_name="",
            chembl_url="",
        )

    mol = mol_from_smiles(smiles_in)
    if mol is None:
        return BatchRow(
            ok=False,
            error="Invalid SMILES",
            smiles_in=smiles_in,
            smiles_used="",
            pred_pic50=np.nan,
            pred_ic50_nM=np.nan,
            pred_std=np.nan,
            potency="",
            ligand_efficiency=np.nan,
            max_sim=np.nan,
            mean_sim=np.nan,
            neighbors_ge_0_4=0,
            ad_band="",
            pains_count=0,
            pains_hits="",
            scaffold_smiles="",
            molwt=np.nan,
            clogp=np.nan,
            tpsa=np.nan,
            rotb=0,
            hbd=0,
            hba=0,
            qed=np.nan,
            chembl_id="",
            chembl_name="",
            chembl_url="",
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

    # dataset match by inchikey (if present)
    cid = ""
    cname = ""
    curl = ""
    if "inchikey" in df_full.columns:
        ik = inchikey(mol)
        m = df_full[df_full["inchikey"].astype(str) == ik]
        if len(m) > 0:
            cid = str(m.iloc[0].get("molecule_chembl_id", "")).strip()
            if cid:
                cname = chembl_pref_name_from_chembl_id(cid) or ""
                curl = chembl_molecule_url(cid)

    max_sim = 0.0
    mean_sim = 0.0
    n_ge = 0
    if compute_ad:
        max_sim, mean_sim, n_ge, _neigh = compute_similarity_and_neighbors(fp, df_evidence=df_evidence, topk=topk_neighbors)

    return BatchRow(
        ok=True,
        error="",
        smiles_in=smiles_in,
        smiles_used=smiles_used,
        pred_pic50=float(pred_pic50),
        pred_ic50_nM=float(pred_ic50),
        pred_std=float(pred_std),
        potency=potency_band(pred_pic50),
        ligand_efficiency=float(le),
        max_sim=float(max_sim),
        mean_sim=float(mean_sim),
        neighbors_ge_0_4=int(n_ge),
        ad_band=ad_band(max_sim) if compute_ad else "",
        pains_count=len(pains_hits),
        pains_hits="; ".join(pains_hits),
        scaffold_smiles=scaffold_smiles,
        molwt=float(Descriptors.MolWt(mol)),
        clogp=float(Descriptors.MolLogP(mol)),
        tpsa=float(rdMolDescriptors.CalcTPSA(mol)),
        rotb=int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        hbd=int(rdMolDescriptors.CalcNumHBD(mol)),
        hba=int(rdMolDescriptors.CalcNumHBA(mol)),
        qed=float(QED.qed(mol)),
        chembl_id=cid,
        chembl_name=cname,
        chembl_url=curl,
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

df_f = df[(df["pic50"] >= pic_range[0]) & (df["pic50"] <= pic_range[1]) & (df["n_measurements"] >= min_meas)].copy()

tab1, tab2, tab3 = st.tabs(["Lead Diagnostic", "Chemical Space", "Methodology"])

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

                p_mean, p_std, fp = rf_predict(model, mol)
                le_score = calculate_ligand_efficiency(p_mean, mol)
                pains_hits = check_pains(mol, pains_catalog)

                scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold_mol) if scaffold_mol is not None else ""

                # AD + neighbors evidence
                max_sim = 0.0
                mean_sim = 0.0
                n_ge = 0
                neighbors = pd.DataFrame()
                if compute_ad:
                    max_sim, mean_sim, n_ge, neighbors = compute_similarity_and_neighbors(fp, df_evidence=df_f, topk=topk_neighbors)

                # Descriptors
                mw = float(Descriptors.MolWt(mol))
                clogp = float(Descriptors.MolLogP(mol))
                tpsa = float(rdMolDescriptors.CalcTPSA(mol))
                rotb = int(rdMolDescriptors.CalcNumRotatableBonds(mol))

                # Decision summary
                priority, rationale, next_steps = make_decision_summary(
                    pred_pic50=p_mean,
                    pred_std=p_std,
                    max_sim=max_sim if compute_ad else 0.0,
                    pains_hits=pains_hits,
                    le=le_score,
                    mw=mw,
                    clogp=clogp,
                    tpsa=tpsa,
                    rotb=rotb,
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
                        a1, a2, a3 = st.columns(3)
                        a1.metric("Max similarity", f"{max_sim:.3f}")
                        a2.metric("Mean similarity", f"{mean_sim:.3f}")
                        a3.metric("Neighbors ≥ 0.4", str(n_ge))

                        band = ad_band(max_sim)
                        if band == "In-domain":
                            st.success(band)
                        elif band == "Borderline":
                            st.warning(band)
                        else:
                            st.error(band)

                        st.caption("Interpretation: max similarity < 0.30 implies extrapolation risk.")

                        if len(neighbors):
                            st.markdown("**Top nearest neighbors (experimental evidence)**")
                            st.dataframe(neighbors, use_container_width=True, hide_index=True)
                    else:
                        st.info("AD disabled in sidebar.")

                with st.container(border=True):
                    st.markdown("### C. Structural integrity & alerts")
                    if pains_hits:
                        st.error(f"PAINS alerts: {', '.join(pains_hits)}")
                    else:
                        st.success("No PAINS alerts detected.")

                    st.markdown("**Murcko scaffold**")
                    st.code(scaffold_smiles, language="text")

                with st.container(border=True):
                    st.markdown("### D. PhysChem / developability")
                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("MolWt", f"{mw:.1f}")
                    d2.metric("cLogP", f"{clogp:.2f}")
                    d3.metric("TPSA", f"{tpsa:.1f}")
                    d4.metric("QED", f"{QED.qed(mol):.2f}")

                    d5, d6, d7, d8 = st.columns(4)
                    d5.metric("HBD", str(rdMolDescriptors.CalcNumHBD(mol)))
                    d6.metric("HBA", str(rdMolDescriptors.CalcNumHBA(mol)))
                    d7.metric("RotB", str(rotb))
                    d8.metric("Rings", str(rdMolDescriptors.CalcNumRings(mol)))

        else:
            st.info("System ready. Define query parameters and execute diagnostic.")

# =========================
# Tab 2: Chemical space
# =========================
with tab2:
    st.subheader("Chemical Space (Evidence subset)")
    if "ic50_nM" in df_f.columns:
        fig = px.scatter(
            df_f,
            x="pic50",
            y="ic50_nM",
            size="n_measurements",
            hover_name="molecule_chembl_id" if "molecule_chembl_id" in df_f.columns else None,
            template="plotly_white",
            labels={"pic50": "Experimental pIC50", "ic50_nM": "IC50 (nM)"},
        )
    else:
        fig = px.scatter(
            df_f,
            x="pic50",
            y="n_measurements",
            size="n_measurements",
            hover_name="molecule_chembl_id" if "molecule_chembl_id" in df_f.columns else None,
            template="plotly_white",
            labels={"pic50": "Experimental pIC50", "n_measurements": "n_measurements"},
        )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Batch scoring (CSV)")

    st.write("Upload a CSV with a column named `smiles`.")
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:
        try:
            batch = pd.read_csv(file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        if "smiles" not in batch.columns:
            st.error("CSV must include a `smiles` column.")
            st.stop()

        st.info(f"Uploaded rows: {len(batch)} | Evidence subset size: {len(df_f)}")

        if st.button("Run batch scoring", type="primary"):
            rows = []
            with st.spinner("Scoring molecules..."):
                for s in batch["smiles"].astype(str).tolist():
                    r = score_smiles_row(
                        s,
                        model=model,
                        pains_catalog=pains_catalog,
                        df_evidence=df_f,
                        df_full=df,
                        strip_salts=strip_salts,
                        compute_ad=compute_ad,
                        topk_neighbors=topk_neighbors,
                    )
                    rows.append(asdict(r))

            out = pd.DataFrame(rows)
            st.success("Batch scoring complete.")
            st.dataframe(out.head(50), use_container_width=True)

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results CSV",
                data=csv_bytes,
                file_name="cdk2_batch_results.csv",
                mime="text/csv",
            )

# =========================
# Tab 3: Methodology
# =========================
with tab3:
    st.subheader("Methodology & interpretation")
    st.markdown(
        f"""
### What this tool does
- Predicts **CDK2 pIC50** using a RandomForest model trained on **Morgan fingerprints** (radius={FP_RADIUS}, {FP_NBITS} bits).
- Provides **uncertainty** as tree-to-tree disagreement (σ).
- Provides **evidence** via nearest-neighbor similarity to a filtered evidence subset.

### Interpreting output (practical)
- **pIC50**: +1 ≈ 10× potency shift.
- **IC50(nM)**: 10^(9 - pIC50).
- **σ**: model disagreement; treat high σ as low reliability.
- **AD**: similarity-based domain; low similarity implies extrapolation risk.
- **LE**: size-normalized potency. ≥0.30 acceptable; ≥0.35 strong.
- **PAINS**: risk flags for assay interference.

### Suggested real workflow
1. Use prediction + AD + σ to triage.
2. If good: prioritize synthesis/assay, run selectivity (CDK1/4/6), early ADME.
3. If out-of-domain: find closer analogs or validate experimentally before optimizing.
"""
    )
