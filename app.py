from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from huggingface_hub import hf_hub_download
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Draw, QED, rdMolDescriptors

# =========================
# CONFIG
# =========================
PROJECT = Path(__file__).resolve().parent

DATA_PATH = PROJECT / "cdk2_pic50_clean.parquet"

FP_RADIUS = 2
FP_NBITS = 2048

SIM_HIGH = 0.50
SIM_MED = 0.30

HF_MODEL_REPO = "Utkarsh2405/cdk2-rf-model"
HF_MODEL_FILENAME = "cdk2_rf_final_all_data.joblib"

st.set_page_config(page_title="ChemBLitz — CDK2 Scientific Predictor", layout="wide")


# =========================
# Chemistry helpers
# =========================
def pic50_to_ic50_nM(pic50: float) -> float:
    return float(10 ** (9.0 - pic50))


def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.SanitizeMol(mol)
    return mol


def canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True)


def inchikey(mol: Chem.Mol) -> str:
    try:
        return Chem.inchi.MolToInchiKey(mol)
    except Exception:
        return "N/A"


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
        "MolWt": float(Descriptors.MolWt(mol)),
        "LogP": float(Descriptors.MolLogP(mol)),
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
    if d["LogP"] > 5:
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
        return "Strong (≥ 8)"
    if pic50 >= 6.0:
        return "Moderate (6–8)"
    return "Weak (< 6)"


def confidence_label(pred_std: float, max_sim: float) -> str:
    if (max_sim >= SIM_HIGH) and (pred_std <= 0.35):
        return "High"
    if (max_sim >= SIM_MED) and (pred_std <= 0.60):
        return "Medium"
    return "Low"


def improvement_suggestions(d: dict, max_sim: float) -> list[str]:
    s: list[str] = []
    if d["LogP"] > 4.5:
        s.append("LogP high → consider adding polarity (HBA/HBD) or removing hydrophobic groups.")
    if d["TPSA"] > 140:
        s.append("TPSA high → may reduce permeability; consider masking polar groups.")
    if d["RotB"] > 10:
        s.append("Too flexible → reduce rotatable bonds via rigidification.")
    if d["MolWt"] > 500:
        s.append("MolWt high → consider scaffold simplification.")
    if max_sim < SIM_MED:
        s.append("Out-of-domain risk → low similarity to known CDK2 data; treat cautiously.")
    if not s:
        s.append("No major red flags → next focus on selectivity and metabolic stability.")
    return s


# =========================
# External info
# =========================
@st.cache_data(show_spinner=False)
def chembl_pref_name_from_chembl_id(chembl_id: str) -> Optional[str]:
    try:
        r = requests.get(
            f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json",
            timeout=10,
        )
        if r.status_code != 200:
            return None
        name = r.json().get("pref_name")
        return str(name) if name else None
    except Exception:
        return None


# =========================
# Data + model loading
# =========================
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)


@st.cache_resource
def load_model():
    model_file = hf_hub_download(repo_id=HF_MODEL_REPO, filename=HF_MODEL_FILENAME)
    return joblib.load(model_file)


@st.cache_resource
def build_dataset_fps(df: pd.DataFrame):
    fps: list = []
    idx: list[int] = []
    for i, s in enumerate(df["smiles"].astype(str).tolist()):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fps.append(morgan_fp(m))
        idx.append(i)
    return fps, np.array(idx, dtype=int)


def rf_predict_with_uncertainty(model, X: np.ndarray) -> Tuple[float, float]:
    preds = np.array([t.predict(X)[0] for t in model.estimators_], dtype=float)
    return float(preds.mean()), float(preds.std(ddof=1))


def build_examples(df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    dff = df.dropna(subset=["pic50", "smiles"]).copy()
    dff["pic50"] = pd.to_numeric(dff["pic50"], errors="coerce")
    dff = dff.dropna(subset=["pic50"])

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

    if "n_measurements" in dff.columns:
        mq = dff.sort_values("n_measurements", ascending=False).head(1)
        if len(mq):
            r = mq.iloc[0]
            out[f"Most measured (n={int(r['n_measurements'])})"] = str(r["smiles"])

    if not out:
        out["Dataset example"] = str(df.iloc[0]["smiles"])
    return out


@st.cache_resource
def get_df_and_examples():
    d = load_data()
    return d, build_examples(d)


# =========================
# Session state defaults
# =========================
def init_state():
    if "smiles" not in st.session_state:
        st.session_state["smiles"] = ""
    if "apply_filters" not in st.session_state:
        st.session_state["apply_filters"] = True
    if "pic50_range" not in st.session_state:
        st.session_state["pic50_range"] = None
    if "min_n" not in st.session_state:
        st.session_state["min_n"] = 1


init_state()


# =========================
# UI
# =========================
st.title("ChemBLitz — CDK2 pIC50 Scientific Predictor")
st.caption("CDK2-only. RandomForest on Morgan fingerprints. Model loaded from Hugging Face Model Hub.")

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

df, examples = get_df_and_examples()

# Initialize default SMILES from dataset (once)
if not st.session_state["smiles"]:
    st.session_state["smiles"] = str(df.iloc[0]["smiles"]) if "smiles" in df.columns else "CCO"

tab_pred, tab_val, tab_data, tab_about = st.tabs(
    ["Predict", "Validate (Similarity)", "Dataset Dashboard", "Methodology"]
)

# -------------------------
# Predict tab
# -------------------------
with tab_pred:
    c_left, c_right = st.columns([1.05, 0.95], gap="large")

    with c_left:
        st.subheader("Input")

        ex_label = st.selectbox("Load CDK2 dataset example", list(examples.keys()), index=0)
        if st.button("Use selected example"):
            st.session_state["smiles"] = examples[ex_label]

        st.session_state["smiles"] = st.text_input("SMILES", value=st.session_state["smiles"])

        run = st.button("Predict", type="primary")

        st.markdown("---")
        st.subheader("Global dataset filters (used across tabs)")

        pic50_min, pic50_max = float(df["pic50"].min()), float(df["pic50"].max())
        current_range = st.session_state["pic50_range"] or (pic50_min, pic50_max)

        pic50_range = st.slider("pIC50 range", pic50_min, pic50_max, current_range)
        min_n = st.slider("Min measurements", 1, int(df["n_measurements"].max()), int(st.session_state["min_n"]))

        apply_filters = st.checkbox(
            "Apply filters to lookup, similarity & neighbors",
            value=bool(st.session_state["apply_filters"]),
            help="If ON, dataset match and similarity are computed on the filtered subset instead of the full dataset.",
        )

        # Persist globally for other tabs
        st.session_state["pic50_range"] = pic50_range
        st.session_state["min_n"] = min_n
        st.session_state["apply_filters"] = apply_filters

        df_filtered = df[
            (df["pic50"] >= pic50_range[0])
            & (df["pic50"] <= pic50_range[1])
            & (df["n_measurements"] >= min_n)
        ].copy()

        st.info(f"Filtered dataset size: {len(df_filtered)} / {len(df)}")

    with c_right:
        st.subheader("Output")

        if not run:
            st.info("Click **Predict** to generate outputs.")
        else:
            smiles = st.session_state["smiles"]
            mol = mol_from_smiles(smiles)
            if mol is None:
                st.error("Invalid SMILES.")
                st.stop()

            with st.spinner("Loading model…"):
                model = load_model()

            st.image(draw_mol_png(mol), caption="Input structure (RDKit 2D)")

            can = canonical_smiles(mol)
            ik = inchikey(mol)

            st.write("Canonical SMILES:")
            st.code(can, language="text")
            st.write("InChIKey:", ik)

            # Use filtered or full dataset for lookup depending on toggle
            df_lookup = df_filtered if st.session_state["apply_filters"] else df
            local_match = (
                df_lookup[df_lookup["inchikey"].astype(str) == ik] if "inchikey" in df_lookup.columns else pd.DataFrame()
            )

            chembl_id = None
            if len(local_match) > 0:
                chembl_id = str(local_match.iloc[0].get("molecule_chembl_id", "")).strip() or None
                st.success(f"Found in {'FILTERED' if st.session_state['apply_filters'] else 'FULL'} dataset: {chembl_id or 'N/A'}")
            else:
                st.warning(f"Not found in {'FILTERED' if st.session_state['apply_filters'] else 'FULL'} dataset.")

            pref = chembl_pref_name_from_chembl_id(chembl_id) if chembl_id else None
            if pref:
                st.write("Compound name (ChEMBL):", pref)

            fp = morgan_fp(mol)
            X = fp_to_numpy(fp).reshape(1, -1)
            pred_pic50, pred_std = rf_predict_with_uncertainty(model, X)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted pIC50", f"{pred_pic50:.3f}")
            m2.metric("Predicted IC50 (nM)", f"{pic50_to_ic50_nM(pred_pic50):,.2f}")
            m3.metric("Uncertainty (σ)", f"{pred_std:.3f}")
            m4.metric("Potency class", potency_bucket(pred_pic50))

            st.markdown("### Applicability domain (uses same filter toggle)")
            st.caption("Similarity is expensive the first time (RDKit fingerprints).")

            if st.button("Compute similarity", key="sim_button"):
                df_sim = df_filtered if st.session_state["apply_filters"] else df
                with st.spinner("Computing dataset similarity…"):
                    fps_all, _ = build_dataset_fps(df_sim)
                    sims_all = (
                        np.array(DataStructs.BulkTanimotoSimilarity(fp, fps_all), dtype=float)
                        if len(fps_all)
                        else np.array([])
                    )
                    max_sim = float(sims_all.max()) if len(sims_all) else 0.0
                    mean_sim = float(sims_all.mean()) if len(sims_all) else 0.0

                if max_sim >= SIM_HIGH:
                    st.success(f"In-domain (max similarity: {max_sim:.3f})")
                elif max_sim >= SIM_MED:
                    st.warning(f"Borderline domain (max similarity: {max_sim:.3f})")
                else:
                    st.error(f"Out-of-domain risk (max similarity: {max_sim:.3f})")

                st.write("Mean similarity:", f"{mean_sim:.3f}")
                st.write("Confidence:", confidence_label(pred_std, max_sim))

            st.markdown("### PhysChem & ADMET-like heuristics")
            d = compute_descriptors(mol)
            ro5 = lipinski_violations(d)

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("MolWt", f"{d['MolWt']:.1f}")
            d2.metric("cLogP", f"{d['LogP']:.2f}")
            d3.metric("TPSA", f"{d['TPSA']:.1f}")
            d4.metric("QED", f"{d['QED']:.2f}")

            st.write(f"Lipinski Ro5 violations: **{ro5}**")
            st.write(f"Veber rule: **{'PASS' if veber_pass(d) else 'FAIL'}**")

            # Use max_sim=0.0 here unless similarity computed; still OK
            for s in improvement_suggestions(d, 0.0):
                st.write(f"- {s}")

# -------------------------
# Validate tab
# -------------------------
with tab_val:
    st.subheader("Nearest neighbors (respects global filter toggle)")

    # Recompute filtered view from session state so this tab uses the same settings
    pic50_range = st.session_state["pic50_range"]
    min_n = st.session_state["min_n"]
    apply_filters = st.session_state["apply_filters"]

    df_filtered = df[
        (df["pic50"] >= pic50_range[0])
        & (df["pic50"] <= pic50_range[1])
        & (df["n_measurements"] >= min_n)
    ].copy()

    st.write(
        "Neighbor dataset:",
        f"{'FILTERED' if apply_filters else 'FULL'} "
        f"({len(df_filtered) if apply_filters else len(df)} rows)",
    )

    smiles = st.session_state["smiles"]
    mol_q = mol_from_smiles(smiles)

    topk = st.slider("Top-K neighbors", 5, 25, 10)

    if mol_q is None:
        st.error("Invalid SMILES.")
    else:
        if st.button("Compute nearest neighbors", type="primary"):
            df_nn = df_filtered if apply_filters else df
            with st.spinner("Computing similarity…"):
                fp_q = morgan_fp(mol_q)
                fps_nn, idx_map = build_dataset_fps(df_nn)
                sims = np.array(DataStructs.BulkTanimotoSimilarity(fp_q, fps_nn), dtype=float)
                top = np.argsort(-sims)[:topk]

            rows = []
            for j in top:
                row = df_nn.iloc[int(idx_map[j])]
                rows.append(
                    {
                        "Tanimoto": float(sims[j]),
                        "molecule_chembl_id": str(row.get("molecule_chembl_id", "")),
                        "pic50": float(row.get("pic50", np.nan)),
                        "ic50_nM": float(row.get("ic50_nM", np.nan)),
                        "n_measurements": int(row.get("n_measurements", 0)),
                        "smiles": str(row.get("smiles", "")),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# -------------------------
# Dataset Dashboard tab
# -------------------------
with tab_data:
    st.subheader("Dataset dashboard")

    pic50_range = st.session_state["pic50_range"]
    min_n = st.session_state["min_n"]

    df_filtered = df[
        (df["pic50"] >= pic50_range[0])
        & (df["pic50"] <= pic50_range[1])
        & (df["n_measurements"] >= min_n)
    ].copy()

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(df, x="pic50", title="pIC50 Distribution (FULL)"), use_container_width=True)
    with c2:
        st.plotly_chart(
            px.histogram(df_filtered, x="pic50", title="pIC50 Distribution (FILTERED)"),
            use_container_width=True,
        )

    st.plotly_chart(px.histogram(df, x="n_measurements", title="Measurement Counts (FULL)"), use_container_width=True)

    st.markdown("### Preview (filtered)")
    st.dataframe(df_filtered.head(50), use_container_width=True)

# -------------------------
# About tab
# -------------------------
with tab_about:
    st.subheader("Methodology (scientific context)")
    st.markdown(
        """
**Target:** Cyclin-dependent kinase 2 (CDK2)

**Endpoint:** IC50 → pIC50  
- pIC50 = -log10(IC50 [M])  
- IC50 (nM) = 10^(9 - pIC50)

**Model:** RandomForest regression on Morgan fingerprints  
- radius=2, 2048 bits  
- Uncertainty proxy: std across RF trees (σ)

**Applicability domain:**  
- Tanimoto similarity vs dataset compounds  
- Filter toggle controls whether similarity/neighbors use the filtered subset or the full dataset.

**Model hosting:**  
- Model
