from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from huggingface_hub import hf_hub_download
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Draw, QED, rdMolDescriptors

# =========================
# App config
# =========================
st.set_page_config(
    page_title="ChemBLitz — CDK2 pIC50 Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT = Path(__file__).resolve().parent
DATA_PATH = PROJECT / "cdk2_pic50_clean.parquet"

# Fingerprints / similarity
FP_RADIUS = 2
FP_NBITS = 2048
SIM_HIGH = 0.50
SIM_MED = 0.30

# Model hub (your repo)
HF_MODEL_REPO = "Utkarsh2405/cdk2-rf-model"
HF_MODEL_FILENAME = "cdk2_rf_final_all_data.joblib"


# =========================
# Utilities
# =========================
def pic50_to_ic50_nM(pic50: float) -> float:
    # IC50(nM) = 10^(9 - pIC50)
    return float(10 ** (9.0 - pic50))


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


def morgan_fp(mol: Chem.Mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)


def fp_to_numpy(fp) -> np.ndarray:
    arr = np.zeros((FP_NBITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def draw_mol_png(mol: Chem.Mol, size=(520, 340)) -> bytes:
    img = Draw.MolToImage(mol, size=size)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def compute_descriptors(mol: Chem.Mol) -> dict:
    return {
        "MolWt": float(Descriptors.MolWt(mol)),
        "cLogP": float(Descriptors.MolLogP(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "HBD": int(rdMolDescriptors.CalcNumHBD(mol)),
        "HBA": int(rdMolDescriptors.CalcNumHBA(mol)),
        "RotB": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "Rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "QED": float(QED.qed(mol)),
    }


def lipinski_violations(d: dict) -> int:
    v = 0
    if d["MolWt"] > 500:
        v += 1
    if d["cLogP"] > 5:
        v += 1
    if d["HBD"] > 5:
        v += 1
    if d["HBA"] > 10:
        v += 1
    return v


def veber_pass(d: dict) -> bool:
    return (d["TPSA"] <= 140.0) and (d["RotB"] <= 10)


def potency_bucket(pic50: float) -> str:
    if pic50 >= 8.0:
        return "Strong"
    if pic50 >= 6.0:
        return "Moderate"
    return "Weak"


def confidence_label(pred_std: float, max_sim: float) -> str:
    # simple, readable rule (can evolve later)
    if max_sim >= SIM_HIGH and pred_std <= 0.35:
        return "High"
    if max_sim >= SIM_MED and pred_std <= 0.60:
        return "Medium"
    return "Low"


def improvement_suggestions(d: dict, max_sim: float) -> list[str]:
    s: list[str] = []
    if d["cLogP"] > 4.5:
        s.append("High cLogP → consider adding polarity or removing hydrophobic substituents.")
    if d["TPSA"] > 140:
        s.append("High TPSA → may reduce permeability; consider reducing polar surface area.")
    if d["RotB"] > 10:
        s.append("High flexibility → consider rigidification (ring closure, reduce rotatable bonds).")
    if d["MolWt"] > 500:
        s.append("High MolWt → consider scaffold simplification.")
    if max_sim < SIM_MED:
        s.append("Low similarity to training chemistry → treat potency prediction cautiously (out-of-domain risk).")
    if not s:
        s.append("No major physchem red flags. Next: optimize selectivity, solubility, and metabolic stability.")
    return s


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
# Data + model
# =========================
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)

    required = {"smiles", "pic50", "n_measurements"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    # Helpful in case types are messy
    df = df.copy()
    df["pic50"] = pd.to_numeric(df["pic50"], errors="coerce")
    df["n_measurements"] = pd.to_numeric(df["n_measurements"], errors="coerce").fillna(0).astype(int)

    return df


@st.cache_resource
def load_model():
    model_file = hf_hub_download(repo_id=HF_MODEL_REPO, filename=HF_MODEL_FILENAME)
    return joblib.load(model_file)


@st.cache_resource
def build_fps_for_df(df: pd.DataFrame):
    fps = []
    idx = []
    for i, s in enumerate(df["smiles"].astype(str).tolist()):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fps.append(morgan_fp(m))
        idx.append(i)
    return fps, np.array(idx, dtype=int)


def rf_predict_with_uncertainty(model, X: np.ndarray) -> tuple[float, float]:
    # Works for RandomForestRegressor
    preds = np.array([t.predict(X)[0] for t in model.estimators_], dtype=float)
    return float(preds.mean()), float(preds.std(ddof=1))


def build_examples(df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    dff = df.dropna(subset=["pic50", "smiles"]).copy()

    top = dff.sort_values("pic50", ascending=False).head(1)
    if len(top):
        r = top.iloc[0]
        out[f"Top binder (pIC50 {r['pic50']:.2f})"] = str(r["smiles"])

    med_pic = float(dff["pic50"].median())
    med_row = dff.iloc[(dff["pic50"] - med_pic).abs().argsort()[:1]]
    if len(med_row):
        r = med_row.iloc[0]
        out[f"Median binder (pIC50 ~{med_pic:.2f})"] = str(r["smiles"])

    low = dff.sort_values("pic50", ascending=True).head(1)
    if len(low):
        r = low.iloc[0]
        out[f"Weak binder (pIC50 {r['pic50']:.2f})"] = str(r["smiles"])

    return out or {"Dataset example": str(df.iloc[0]["smiles"])}


# =========================
# Professional scoring core
# =========================
@dataclass
class ScoreResult:
    ok: bool
    error: str
    smiles_in: str
    smiles_canon: str
    inchikey: str
    pred_pic50: float
    pred_ic50_nM: float
    pred_std: float
    max_sim: float
    mean_sim: float
    neighbors_ge_0_4: int
    confidence: str
    potency_class: str
    molwt: float
    clogp: float
    tpsa: float
    hbd: int
    hba: int
    rotb: int
    rings: int
    qed: float
    in_dataset: bool
    in_filtered_dataset: bool
    chembl_id: str


def score_one(
    smiles: str,
    *,
    model,
    df_full: pd.DataFrame,
    df_evidence: pd.DataFrame,
    compute_similarity: bool,
    topk_neighbors: int = 10,
) -> tuple[ScoreResult, Optional[pd.DataFrame]]:
    smiles = (smiles or "").strip()
    if not smiles:
        return (
            ScoreResult(
                ok=False,
                error="Empty SMILES",
                smiles_in="",
                smiles_canon="",
                inchikey="",
                pred_pic50=np.nan,
                pred_ic50_nM=np.nan,
                pred_std=np.nan,
                max_sim=np.nan,
                mean_sim=np.nan,
                neighbors_ge_0_4=0,
                confidence="",
                potency_class="",
                molwt=np.nan,
                clogp=np.nan,
                tpsa=np.nan,
                hbd=0,
                hba=0,
                rotb=0,
                rings=0,
                qed=np.nan,
                in_dataset=False,
                in_filtered_dataset=False,
                chembl_id="",
            ),
            None,
        )

    mol = mol_from_smiles(smiles)
    if mol is None:
        return (
            ScoreResult(
                ok=False,
                error="Invalid SMILES",
                smiles_in=smiles,
                smiles_canon="",
                inchikey="",
                pred_pic50=np.nan,
                pred_ic50_nM=np.nan,
                pred_std=np.nan,
                max_sim=np.nan,
                mean_sim=np.nan,
                neighbors_ge_0_4=0,
                confidence="",
                potency_class="",
                molwt=np.nan,
                clogp=np.nan,
                tpsa=np.nan,
                hbd=0,
                hba=0,
                rotb=0,
                rings=0,
                qed=np.nan,
                in_dataset=False,
                in_filtered_dataset=False,
                chembl_id="",
            ),
            None,
        )

    can = canonical_smiles(mol)
    ik = inchikey(mol)

    # Prediction
    fp = morgan_fp(mol)
    X = fp_to_numpy(fp).reshape(1, -1)
    pred_pic50, pred_std = rf_predict_with_uncertainty(model, X)
    pred_ic50 = pic50_to_ic50_nM(pred_pic50)

    # Descriptors
    d = compute_descriptors(mol)

    # Dataset match (full + filtered)
    in_full = False
    in_filt = False
    chembl_id = ""

    if "inchikey" in df_full.columns:
        m_full = df_full[df_full["inchikey"].astype(str) == ik]
        in_full = len(m_full) > 0
        if in_full:
            chembl_id = str(m_full.iloc[0].get("molecule_chembl_id", "")).strip()

    if "inchikey" in df_evidence.columns:
        m_f = df_evidence[df_evidence["inchikey"].astype(str) == ik]
        in_filt = len(m_f) > 0

    # Similarity / neighbors evidence
    max_sim = 0.0
    mean_sim = 0.0
    neighbors_ge_0_4 = 0
    conf = "Low"
    neigh_df = None

    if compute_similarity:
        fps, idx_map = build_fps_for_df(df_evidence)
        if len(fps) > 0:
            sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, fps), dtype=float)
            max_sim = float(sims.max()) if len(sims) else 0.0
            mean_sim = float(sims.mean()) if len(sims) else 0.0
            neighbors_ge_0_4 = int((sims >= 0.4).sum())

            top = np.argsort(-sims)[:topk_neighbors]
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
            neigh_df = pd.DataFrame(rows)

        conf = confidence_label(pred_std, max_sim)

    return (
        ScoreResult(
            ok=True,
            error="",
            smiles_in=smiles,
            smiles_canon=can,
            inchikey=ik,
            pred_pic50=float(pred_pic50),
            pred_ic50_nM=float(pred_ic50),
            pred_std=float(pred_std),
            max_sim=float(max_sim),
            mean_sim=float(mean_sim),
            neighbors_ge_0_4=int(neighbors_ge_0_4),
            confidence=conf,
            potency_class=potency_bucket(pred_pic50),
            molwt=float(d["MolWt"]),
            clogp=float(d["cLogP"]),
            tpsa=float(d["TPSA"]),
            hbd=int(d["HBD"]),
            hba=int(d["HBA"]),
            rotb=int(d["RotB"]),
            rings=int(d["Rings"]),
            qed=float(d["QED"]),
            in_dataset=bool(in_full),
            in_filtered_dataset=bool(in_filt),
            chembl_id=chembl_id,
        ),
        neigh_df,
    )


# =========================
# UI
# =========================
st.title("ChemBLitz — CDK2 pIC50 Predictor (Professional)")

if not DATA_PATH.exists():
    st.error(f"Missing dataset file: {DATA_PATH}")
    st.stop()

df = load_data()
examples = build_examples(df)

with st.sidebar:
    st.header("Workflow")

    mode = st.radio("Mode", ["Single molecule", "Batch (CSV)"], horizontal=False)

    st.subheader("Evidence dataset filters")
    pic50_min, pic50_max = float(df["pic50"].min()), float(df["pic50"].max())
    pic50_range = st.slider("pIC50 range", pic50_min, pic50_max, (pic50_min, pic50_max))
    min_n = st.slider("Min measurements per compound", 1, int(df["n_measurements"].max()), 1)

    st.subheader("Evidence options")
    use_filters_for_evidence = st.checkbox(
        "Use filtered dataset for evidence (lookup/similarity/neighbors)",
        value=True,
    )
    compute_similarity = st.checkbox(
        "Compute similarity + neighbors (slower)",
        value=True,
        help="If OFF, predictions still work but no applicability domain / neighbors evidence.",
    )
    topk_neighbors = st.slider("Top-K neighbors", 5, 25, 10)

    st.subheader("Export")
    st.caption("Batch results can be downloaded as CSV after scoring.")

df_filtered = df[
    (df["pic50"] >= pic50_range[0])
    & (df["pic50"] <= pic50_range[1])
    & (df["n_measurements"] >= min_n)
].copy()

df_evidence = df_filtered if use_filters_for_evidence else df

# Load model lazily but cached
model = load_model()

if mode == "Single molecule":
    st.subheader("Single-molecule report")

    c1, c2 = st.columns([1.1, 0.9], gap="large")

    with c1:
        ex_label = st.selectbox("Load dataset example", list(examples.keys()), index=0)
        if st.button("Use example"):
            st.session_state["smiles_single"] = examples[ex_label]

        smiles_in = st.text_area(
            "SMILES",
            value=st.session_state.get("smiles_single", examples[list(examples.keys())[0]]),
            height=90,
        )
        run = st.button("Generate report", type="primary")

        st.info(f"Evidence dataset size: {len(df_evidence)} / {len(df)}")

    with c2:
        if not run:
            st.info("Click **Generate report**.")
        else:
            res, neigh = score_one(
                smiles_in,
                model=model,
                df_full=df,
                df_evidence=df_evidence,
                compute_similarity=compute_similarity,
                topk_neighbors=topk_neighbors,
            )

            if not res.ok:
                st.error(res.error)
            else:
                mol = mol_from_smiles(res.smiles_in)
                if mol is not None:
                    st.image(draw_mol_png(mol), caption="RDKit 2D depiction")

                # Header KPIs
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Predicted pIC50", f"{res.pred_pic50:.3f}")
                k2.metric("Predicted IC50 (nM)", f"{res.pred_ic50_nM:,.2f}")
                k3.metric("Uncertainty (σ)", f"{res.pred_std:.3f}")
                k4.metric("Confidence", res.confidence)

                st.markdown("### Identifiers")
                st.write("Canonical SMILES:")
                st.code(res.smiles_canon, language="text")
                st.write("InChIKey:", res.inchikey)

                st.markdown("### Dataset match")
                st.write("In full dataset:", "✅" if res.in_dataset else "❌")
                st.write("In evidence subset:", "✅" if res.in_filtered_dataset else "❌")

                if res.chembl_id:
                    st.write("ChEMBL ID:", res.chembl_id)
                    name = chembl_pref_name_from_chembl_id(res.chembl_id)
                    if name:
                        st.write("Name:", name)
                    st.write("ChEMBL link:", chembl_molecule_url(res.chembl_id))

                st.markdown("### Applicability domain (similarity)")
                if compute_similarity:
                    a1, a2, a3 = st.columns(3)
                    a1.metric("Max similarity", f"{res.max_sim:.3f}")
                    a2.metric("Mean similarity", f"{res.mean_sim:.3f}")
                    a3.metric("Neighbors (sim ≥ 0.4)", str(res.neighbors_ge_0_4))

                    if res.max_sim >= SIM_HIGH:
                        st.success("In-domain")
                    elif res.max_sim >= SIM_MED:
                        st.warning("Borderline")
                    else:
                        st.error("Out-of-domain risk")
                else:
                    st.info("Similarity disabled in sidebar.")

                st.markdown("### PhysChem & developability heuristics")
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("MolWt", f"{res.molwt:.1f}")
                d2.metric("cLogP", f"{res.clogp:.2f}")
                d3.metric("TPSA", f"{res.tpsa:.1f}")
                d4.metric("QED", f"{res.qed:.2f}")

                d5, d6, d7, d8 = st.columns(4)
                d5.metric("HBD", str(res.hbd))
                d6.metric("HBA", str(res.hba))
                d7.metric("RotB", str(res.rotb))
                d8.metric("Rings", str(res.rings))

                st.write("Lipinski Ro5 violations:", lipinski_violations(
                    {"MolWt": res.molwt, "cLogP": res.clogp, "HBD": res.hbd, "HBA": res.hba}
                ))
                st.write("Veber rule:", "PASS" if veber_pass({"TPSA": res.tpsa, "RotB": res.rotb}) else "FAIL")

                st.markdown("### Recommendations")
                for s in improvement_suggestions(
                    {
                        "MolWt": res.molwt,
                        "cLogP": res.clogp,
                        "TPSA": res.tpsa,
                        "HBD": res.hbd,
                        "HBA": res.hba,
                        "RotB": res.rotb,
                    },
                    res.max_sim if compute_similarity else 0.0,
                ):
                    st.write(f"- {s}")

                if compute_similarity and neigh is not None and len(neigh):
                    st.markdown("### Nearest neighbors (experimental evidence)")
                    st.dataframe(neigh, use_container_width=True, hide_index=True)

else:
    st.subheader("Batch scoring (CSV)")

    st.write(
        "Upload a CSV with a `smiles` column. You'll get predictions + uncertainty + evidence metrics "
        "(if enabled) and can download the results."
    )

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is None:
        st.info("Waiting for CSV upload.")
        st.stop()

    try:
        batch = pd.read_csv(file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    if "smiles" not in batch.columns:
        st.error("CSV must contain a `smiles` column.")
        st.stop()

    st.info(f"Rows uploaded: {len(batch)} | Evidence dataset size: {len(df_evidence)} / {len(df)}")

    if st.button("Run batch scoring", type="primary"):
        with st.spinner("Scoring batch…"):
            rows = []
            for i, s in enumerate(batch["smiles"].astype(str).tolist()):
                res, _neigh = score_one(
                    s,
                    model=model,
                    df_full=df,
                    df_evidence=df_evidence,
                    compute_similarity=compute_similarity,
                    topk_neighbors=topk_neighbors,
                )
                rows.append(res.__dict__)

            out = pd.DataFrame(rows)

        st.success("Batch scoring complete.")
        st.dataframe(out.head(50), use_container_width=True)

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results CSV",
            data=csv_bytes,
            file_name="cdk2_predictions.csv",
            mime="text/csv",
        )

st.markdown("---")
with st.expander("Model card / intended use", expanded=False):
    st.markdown(
        f"""
**Target:** CDK2  
**Endpoint:** pIC50 derived from IC50 measurements  
**Model:** RandomForest regression on Morgan fingerprints (radius={FP_RADIUS}, nBits={FP_NBITS})  
**Model file:** `{HF_MODEL_REPO}/{HF_MODEL_FILENAME}`

**Intended use:** hit prioritization, analog ranking, and hypothesis generation.  
**Not intended for:** clinical decisions or safety-critical use.

**Reliability:** Use applicability domain (similarity) + uncertainty together. Low similarity indicates out-of-domain chemistry.
"""
    )
