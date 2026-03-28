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
st.set_page_config(page_title="ChemBLitz Professional | CDK2 Suite", layout="wide")

# Professional CSS (keep your style, but remove the broken Google font import)
st.markdown(
    """
<style>
html, body, [class*="css"], .stMarkdown, p, li, div, span {
    font-family: "Times New Roman", Times, serif !important;
    font-size: 18px !important;
}
.stMetric label { font-size: 18px !important; font-weight: bold !important; color: #1f77b4 !important; }
.stMetric div { font-size: 28px !important; font-weight: 500 !important; }
.stButton>button { height: 3em; font-size: 18px !important; border-radius: 6px; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# 2. PATHS & ASSETS
# =========================
PROJECT = Path(__file__).resolve().parent
DATA_PATH = PROJECT / "cdk2_pic50_clean.parquet"
MODEL_PATH = PROJECT / "cdk2_rf_final_all_data.joblib"
MODEL_DRIVE_FILE_ID = "1pOgZVHG7BfrcXE7ZmHJNM9CMnsBNlCQa"

FP_RADIUS = 2
FP_NBITS = 2048

SIM_HIGH = 0.50
SIM_MED = 0.30
SIM_NEIGHBOR = 0.40  # neighbor-count threshold

# =========================
# 3. CORE ENGINES
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
    data = pd.read_parquet(DATA_PATH)

    # Basic column validation (professional hygiene)
    needed = {"smiles", "pic50", "n_measurements"}
    missing = needed - set(data.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    # Normalize dtypes
    data = data.copy()
    data["pic50"] = pd.to_numeric(data["pic50"], errors="coerce")
    data["n_measurements"] = pd.to_numeric(data["n_measurements"], errors="coerce").fillna(0).astype(int)

    return model, data


# =========================
# 4. CHEM + MODEL HELPERS
# =========================
def mol_to_png(mol: Chem.Mol):
    img = Draw.MolToImage(mol, size=(600, 400))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def keep_largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    """Strip salts/mixtures by keeping the largest fragment by heavy atoms."""
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return mol
    frags = sorted(frags, key=lambda m: m.GetNumHeavyAtoms(), reverse=True)
    return frags[0]


def morgan_fp(mol: Chem.Mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)


def fp_to_array(fp) -> np.ndarray:
    arr = np.zeros((FP_NBITS,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def rf_predict(model, mol: Chem.Mol) -> tuple[float, float, object]:
    """
    Returns:
      mean_pIC50, std_pIC50, fp_bitvect
    """
    fp = morgan_fp(mol)
    X = fp_to_array(fp).reshape(1, -1)
    preds = np.array([t.predict(X)[0] for t in model.estimators_], dtype=float)
    return float(preds.mean()), float(preds.std(ddof=1)), fp


def calculate_ligand_efficiency(pic50: float, mol: Chem.Mol) -> float:
    # LE ≈ (1.37 * pIC50) / HeavyAtomCount  (common quick heuristic)
    n_heavy = mol.GetNumHeavyAtoms()
    return (1.37 * pic50 / n_heavy) if n_heavy > 0 else 0.0


def check_pains(mol: Chem.Mol, catalog: FilterCatalog) -> list[str]:
    entries = catalog.GetMatches(mol)
    return [e.GetDescription() for e in entries]


def pic50_to_ic50_nM(pic50: float) -> float:
    return float(10 ** (9.0 - pic50))


def potency_band(pic50: float) -> str:
    if pic50 >= 8:
        return "Strong"
    if pic50 >= 6:
        return "Moderate"
    return "Weak"


def uncertainty_band(std: float) -> str:
    # heuristic bands for RF-tree std
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
    """
    Returns: (priority_label, rationale_lines, next_steps_lines)
    """
    rationale = []
    next_steps = []

    pot = potency_band(pred_pic50)
    unc = uncertainty_band(pred_std)
    dom = ad_band(max_sim)

    # Core rationale
    rationale.append(f"Predicted potency: {pot} (pIC50={pred_pic50:.2f}, IC50≈{pic50_to_ic50_nM(pred_pic50):.1f} nM).")
    rationale.append(f"Uncertainty (tree σ): {pred_std:.3f} → {unc}.")
    rationale.append(f"Applicability domain: max similarity={max_sim:.3f} → {dom}.")

    if le >= 0.35:
        rationale.append(f"Ligand Efficiency (LE)={le:.2f} is strong for lead-like optimization.")
    elif le >= 0.30:
        rationale.append(f"Ligand Efficiency (LE)={le:.2f} is acceptable.")
    else:
        rationale.append(f"Ligand Efficiency (LE)={le:.2f} is low; may be size-driven potency.")

    if pains_hits:
        rationale.append(f"PAINS alerts detected: {', '.join(pains_hits)}.")
        next_steps.append("Run orthogonal counterscreens and check assay interference risk (PAINS).")
        next_steps.append("Consider redesign to remove PAINS substructure if possible.")

    # Developability heuristics (simple but useful)
    if mw > 500:
        next_steps.append("MolWt > 500: consider scaffold simplification to improve developability.")
    if clogp > 4.5:
        next_steps.append("High cLogP: consider adding polarity / reducing hydrophobic bulk (solubility & clearance risk).")
    if tpsa > 140:
        next_steps.append("TPSA > 140: permeability risk; consider reducing PSA / masking donors.")
    if rotb > 10:
        next_steps.append("High flexibility: consider rigidification to improve potency/PK.")

    if max_sim < SIM_MED:
        next_steps.append("Low similarity to training chemistry: prioritize experimental validation before heavy optimization.")
        next_steps.append("Search for closer analogs (scaffold hopping within domain) to improve confidence.")

    # Priority label (simple scoring)
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

    if not next_steps:
        next_steps.append("Proceed to selectivity profiling (CDK1/4/6) and early ADME screens.")

    return priority, rationale, next_steps


# =========================
# 5. EVIDENCE: similarity + neighbors
# =========================
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


def compute_similarity_and_neighbors(
    query_fp,
    *,
    df_evidence: pd.DataFrame,
    topk: int = 10,
):
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
    neigh = pd.DataFrame(rows)
    return max_sim, mean_sim, n_ge, neigh


# =========================
# 6. Batch scoring
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


def score_smiles_row(
    smiles: str,
    *,
    model,
    pains_catalog,
    df_evidence: pd.DataFrame,
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
        )

    mol = Chem.MolFromSmiles(smiles_in)
    if not mol:
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
        )

    Chem.SanitizeMol(mol)

    if strip_salts:
        mol = keep_largest_fragment(mol)

    smiles_used = Chem.MolToSmiles(mol, canonical=True)

    pred_pic50, pred_std, fp = rf_predict(model, mol)
    pred_ic50 = pic50_to_ic50_nM(pred_pic50)
    le = calculate_ligand_efficiency(pred_pic50, mol)
    pains_hits = check_pains(mol, pains_catalog)
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold_mol) if scaffold_mol is not None else ""

    # AD / neighbors
    max_sim = 0.0
    mean_sim = 0.0
    n_ge = 0
    if compute_ad:
        max_sim, mean_sim, n_ge, _neigh = compute_similarity_and_neighbors(
            fp, df_evidence=df_evidence, topk=topk_neighbors
        )

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
    )


# =========================
# 7. APP LAYOUT
# =========================
st.title("CDK2 Pharmacological Diagnostic Suite")
st.markdown("#### Operational Intelligence for CDK2 inhibitor discovery (prediction + evidence + interpretation)")
st.divider()

try:
    model, df = load_assets()
    pains_catalog = get_pains_filter()
except Exception as e:
    st.error(f"Critical System Failure: {e}")
    st.stop()

# Sidebar controls (professional: single place)
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

tab1, tab2, tab3 = st.tabs(["Lead Diagnostic", "Batch Scoring", "Methodology & Interpretation"])

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
- **σ (uncertainty)**: standard deviation across RF trees (model disagreement). Not a calibrated CI.
- **Applicability Domain (AD)**: similarity to training chemistry. Low similarity = extrapolation risk.
- **Ligand Efficiency (LE)**: potency normalized by size (heavy atom count). Useful for lead-likeness.
- **PAINS**: substructures associated with assay interference. Treat as risk flags, not absolute truth.
"""
            )

    with col_results:
        if execute and target_smiles:
            mol = Chem.MolFromSmiles(target_smiles)
            if not mol:
                st.error("Invalid structure: SMILES parsing failed.")
            else:
                Chem.SanitizeMol(mol)
                if strip_salts:
                    mol = keep_largest_fragment(mol)

                p_mean, p_std, fp = rf_predict(model, mol)
                le_score = calculate_ligand_efficiency(p_mean, mol)
                pains_hits = check_pains(mol, pains_catalog)

                scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold_mol) if scaffold_mol is not None else ""

                # AD + neighbors
                max_sim = 0.0
                mean_sim = 0.0
                n_ge = 0
                neighbors = pd.DataFrame()
                if compute_ad:
                    max_sim, mean_sim, n_ge, neighbors = compute_similarity_and_neighbors(
                        fp, df_evidence=df_f, topk=topk_neighbors
                    )

                # Descriptors
                mw = float(Descriptors.MolWt(mol))
                clogp = float(Descriptors.MolLogP(mol))
                tpsa = float(rdMolDescriptors.CalcTPSA(mol))
                rotb = int(rdMolDescriptors.CalcNumRotatableBonds(mol))

                # Decision summary
                priority, rationale, next_steps = make_decision_summary(
                    pred_pic50=p_mean,
                    pred_std=p_std,
                    max_sim=max_sim,
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

                        st.caption(
                            "Interpretation: if max similarity < 0.30, prediction is extrapolation and may be unreliable."
                        )

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
                        st.success("No PAINS alerts detected (screening clearance).")

                    st.markdown("**Murcko scaffold**")
                    st.code(scaffold_smiles, language="text")

                with st.container(border=True):
                    st.markdown("### D. PhysChem / developability heuristics")
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
# Tab 2: Batch scoring
# =========================
with tab2:
    st.subheader("Batch scoring (CSV)")

    st.write("Upload a CSV with a column named `smiles`. You can download scored results.")

    file = st.file_uploader("Upload CSV", type=["csv"])
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
# Tab 3: Chemical space + methodology
# =========================
with tab3:
    st.subheader("Chemical space (filtered evidence subset)")

    # Build df_f in this scope as well
    df_f2 = df_f.copy()
    if "ic50_nM" in df_f2.columns:
        ycol = "ic50_nM"
        ylab = "IC50 (nM)"
    else:
        ycol = "pic50"
        ylab = "pIC50"

    fig = px.scatter(
        df_f2,
        x="pic50",
        y=ycol,
        size="n_measurements",
        hover_name="molecule_chembl_id" if "molecule_chembl_id" in df_f2.columns else None,
        template="plotly_white",
        labels={"pic50": "Experimental pIC50", ycol: ylab},
        title="Evidence subset landscape",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("How to interpret outputs (practical)")
    st.markdown(
        """
### Potency (pIC50 / IC50)
- **pIC50** is log-scaled potency. A change of **+1.0 pIC50 ≈ 10×** change in IC50.
- **IC50(nM) = 10^(9 - pIC50)**.

### Uncertainty (σ across RF trees)
- This is a **model disagreement** signal, not a calibrated statistical confidence interval.
- Rule of thumb:
  - **σ ≤ 0.35**: low
  - **0.35 < σ ≤ 0.60**: medium
  - **σ > 0.60**: high (treat cautiously)

### Applicability Domain (AD)
- Uses **Tanimoto similarity** between the query fingerprint and training compounds.
- Low similarity implies extrapolation:
  - **max sim ≥ 0.50**: in-domain
  - **0.30–0.50**: borderline
  - **< 0.30**: out-of-domain risk

### Ligand Efficiency (LE)
- Normalizes potency by size:
  - **LE ≥ 0.30**: often acceptable for leads
  - **LE ≥ 0.35**: strong

### PAINS
- PAINS hits increase risk of false positives / assay artifacts.
- Don’t automatically discard—but **validate with orthogonal assays** and consider redesign.

### What to do next (real workflow)
- If **High potency + in-domain + low σ + no PAINS** → prioritize for synthesis/assay and selectivity profiling.
- If **out-of-domain** → find closer analogs or validate experimentally before optimization.
"""
    )
