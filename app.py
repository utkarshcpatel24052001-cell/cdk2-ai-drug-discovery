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
# Helpers (chem + plotting)
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
    s = []
    if d["LogP"] > 4.5:
        s.append("LogP high → consider adding polarity (HBA/HBD) or removing hydrophobic ring/alkyl groups.")
    if d["TPSA"] > 140:
        s.append("TPSA high → may reduce permeability; consider reducing polar groups or masking (prodrug strategy).")
    if d["RotB"] > 10:
        s.append("Too flexible → reduce rotatable bonds via ring closure / rigidification.")
    if d["MolWt"] > 500:
        s.append("MolWt high → consider scaffold simplification.")
    if max_sim < SIM_MED:
        s.append("Out-of-domain risk → low similarity to known CDK2 data; treat prediction cautiously.")
    if not s:
        s.append("No major red flags → next focus on selectivity, solubility, and metabolic stability.")
    return s


# =========================
# External info
# =========================
@st.cache_data(show_spinner=False)
def chembl_pref_name_from_chembl_id(chembl_id: str) -> Optional[str]:
    try:
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        js = r.json()
        name = js.get("pref_name")
        return str(name) if name else None
    except Exception:
        return None


# =========================
# Data + model (cached, lazy)
# =========================
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    # Expect: inchikey, smiles, molecule_chembl_id, pic50, ic50_nM, n_measurements
    return df


@st.cache_resource
def load_model():
    # Download from HF cache (fast + stable)
    model_file = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=HF_MODEL_FILENAME,
    )
    return joblib.load(model_file)


@st.cache_resource
def build_dataset_fps(df: pd.DataFrame):
    """
    Build RDKit fingerprints for every row.
    This is expensive on first run, so we call it ONLY when user asks for similarity.
    """
    fps = []
    idx = []
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


def build_real_examples_from_dataset(df: pd.DataFrame) -> dict[str, str]:
    """
    Build “real examples” directly from dataset so examples are always CDK2-related.
    """
    out: dict[str, str] = {}

    dff = df.dropna(subset=["pic50", "smiles"]).copy()
    dff["pic50"] = pd.to_numeric(dff["pic50"], errors="coerce")
    dff = dff.dropna(subset=["pic50"])

    top = dff.sort_values("pic50", ascending=False).head(1)
    if len(top):
        r = top.iloc[0]
        out[f"Top binder (pIC50 {r['pic50']:.2f}) — {r.get('molecule_chembl_id','')}".strip(" —")] = str(
            r["smiles"]
        )

    med_pic = float(dff["pic50"].median())
    med_row = dff.iloc[(dff["pic50"] - med_pic).abs().argsort()[:1]]
    if len(med_row):
        r = med_row.iloc[0]
        out[f"Median binder (pIC50 ~{med_pic:.2f}) — {r.get('molecule_chembl_id','')}".strip(" —")] = str(
            r["smiles"]
        )

    low = dff.sort_values("pic50", ascending=True).head(1)
    if len(low):
        r = low.iloc[0]
        out[f"Weak binder (pIC50 {r['pic50']:.2f}) — {r.get('molecule_chembl_id','')}".strip(" —")] = str(
            r["smiles"]
        )

    if "n_measurements" in dff.columns:
        mq = dff.sort_values("n_measurements", ascending=False).head(1)
        if len(mq):
            r = mq.iloc[0]
            out[f"Most measured (n={int(r['n_measurements'])}) — {r.get('molecule_chembl_id','')}".strip(" —")] = str(
                r["smiles"]
            )

    if not out:
        out["Dataset example"] = str(df.iloc[0]["smiles"])

    return out


@st.cache_resource
def get_df_and_examples():
    df_local = load_data()
    ex_local = build_real_examples_from_dataset(df_local)
    return df_local, ex_local


# =========================
# UI
# =========================
st.title("ChemBLitz — CDK2 pIC50 Scientific Predictor")
st.caption(
    "CDK2-only. RandomForest on Morgan fingerprints. "
    "Model is stored on Hugging Face Model Hub and downloaded on first use."
)

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH} (must be in Space repo root)")
    st.stop()

df, examples = get_df_and_examples()

if "smiles" not in st.session_state:
    st.session_state["smiles"] = str(df.iloc[0]["smiles"]) if "smiles" in df.columns else "CCO"

tab_pred, tab_val, tab_data, tab_about = st.tabs(
    ["Predict", "Validate (Similarity)", "Dataset Dashboard", "Methodology"]
)

with tab_pred:
    c_left, c_right = st.columns([1.05, 0.95], gap="large")

    with c_left:
        st.subheader("Input")

        ex = st.selectbox("Load CDK2 dataset example", list(examples.keys()), index=0)
        if st.button("Use selected example"):
            st.session_state["smiles"] = examples[ex]

        st.session_state["smiles"] = st.text_input(
            "SMILES",
            value=st.session_state["smiles"],
            help="Paste any valid SMILES here.",
        )

        run = st.button("Predict", type="primary")

        st.markdown("---")
        st.subheader("Dataset filters")
        pic50_min, pic50_max = float(df["pic50"].min()), float(df["pic50"].max())
        pic50_range = st.slider("pIC50 range", pic50_min, pic50_max, (pic50_min, pic50_max))
        min_n = st.slider("Min measurements per compound", 1, int(df["n_measurements"].max()), 1)

        df_f = df[
            (df["pic50"] >= pic50_range[0])
            & (df["pic50"] <= pic50_range[1])
            & (df["n_measurements"] >= min_n)
        ].copy()

        st.info(f"Filtered dataset size: {len(df_f)} / {len(df)}")

    with c_right:
        st.subheader("Output (fast)")

        if not run:
            st.info("Click **Predict** to generate outputs.")
        else:
            smiles = st.session_state["smiles"]
            mol = mol_from_smiles(smiles)
            if mol is None:
                st.error("Invalid SMILES.")
                st.stop()

            # Load model only after user clicks Predict
            with st.spinner("Loading model…"):
                model = load_model()

            st.image(draw_mol_png(mol), caption="Input structure (RDKit 2D)")

            can = canonical_smiles(mol)
            ik = inchikey(mol)

            st.write("Canonical SMILES:")
            st.code(can, language="text")
            st.write("InChIKey:", ik)

            local_match = df[df["inchikey"].astype(str) == ik] if "inchikey" in df.columns else pd.DataFrame()
            chembl_id = None
            input_is_in_dataset = len(local_match) > 0

            if input_is_in_dataset:
                row0 = local_match.iloc[0]
                chembl_id = str(row0.get("molecule_chembl_id", "")).strip() or None
                st.success(f"Found in dataset: {chembl_id or 'N/A'}")
            else:
                st.warning("Not found in dataset (by InChIKey).")

            pref = chembl_pref_name_from_chembl_id(chembl_id) if chembl_id else None
            if pref:
                st.write("Compound name (ChEMBL pref_name):", pref)
            elif chembl_id:
                st.write("Compound name:", f"ChEMBL compound {chembl_id} (no pref_name available)")
            else:
                st.write("Compound name:", "Unknown (showing canonical identifiers instead)")

            fp = morgan_fp(mol)
            X = fp_to_numpy(fp).reshape(1, -1)
            pred_pic50, pred_std = rf_predict_with_uncertainty(model, X)
            pred_ic50 = pic50_to_ic50_nM(pred_pic50)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted pIC50", f"{pred_pic50:.3f}")
            m2.metric("Predicted IC50 (nM)", f"{pred_ic50:,.2f}")
            m3.metric("Uncertainty (σ)", f"{pred_std:.3f}")
            m4.metric("Potency class", potency_bucket(pred_pic50))

            st.markdown("### Applicability domain (optional)")
            st.caption("Click the button below to compute similarity vs the dataset (slower on first run).")

            compute_sim = st.button("Compute similarity / applicability domain", key="sim_button")

            if compute_sim:
                with st.spinner("Computing dataset fingerprints + similarity (first time can be slow)…"):
                    fps_all, _idx_map_all = build_dataset_fps(df)
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

                conf = confidence_label(pred_std, max_sim)
                st.write(f"Mean similarity (dataset): {mean_sim:.3f}")
                st.write("Confidence:", conf)
            else:
                st.info("Similarity not computed yet.")

            st.markdown("### PhysChem + ADMET-like heuristics")
            d = compute_descriptors(mol)
            ro5 = lipinski_violations(d)

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("MolWt", f"{d['MolWt']:.1f}")
            d2.metric("cLogP", f"{d['LogP']:.2f}")
            d3.metric("TPSA", f"{d['TPSA']:.1f}")
            d4.metric("QED", f"{d['QED']:.2f}")

            d5, d6, d7, d8 = st.columns(4)
            d5.metric("HBD", str(d["HBD"]))
            d6.metric("HBA", str(d["HBA"]))
            d7.metric("RotB", str(d["RotB"]))
            d8.metric("Rings", str(d["Rings"]))

            st.write(f"Lipinski Ro5 violations: **{ro5}**")
            st.write(f"Veber rule (TPSA ≤ 140 and RotB ≤ 10): **{'PASS' if veber_pass(d) else 'FAIL'}**")

            st.markdown("### Improvement suggestions")
            # If similarity wasn't computed, pass 0.0 for suggestions
            # (still produces useful medchem guidance)
            for s in improvement_suggestions(d, 0.0):
                st.write(f"- {s}")

with tab_val:
    st.subheader("Nearest-neighbor evidence (on demand)")

    st.caption(
        "This tab can be slow the first time because it builds RDKit fingerprints for the selected dataset subset."
    )

    smiles = st.session_state["smiles"]
    mol_q = mol_from_smiles(smiles)

    use_filtered = st.toggle("Use filters for neighbors", value=False)
    topk = st.slider("Top-K neighbors", 5, 25, 10)

    pic50_min, pic50_max = float(df["pic50"].min()), float(df["pic50"].max())
    pic50_range = st.slider(
        "pIC50 range (neighbors)",
        pic50_min,
        pic50_max,
        (pic50_min, pic50_max),
        key="nn_pic50",
    )
    min_n = st.slider(
        "Min measurements (neighbors)",
        1,
        int(df["n_measurements"].max()),
        1,
        key="nn_min_n",
    )

    df_filtered = df[
        (df["pic50"] >= pic50_range[0])
        & (df["pic50"] <= pic50_range[1])
        & (df["n_measurements"] >= min_n)
    ].copy()

    df_nn = df_filtered if use_filtered else df

    if mol_q is None:
        st.error("Invalid SMILES.")
    else:
        if st.button("Compute nearest neighbors", type="primary"):
            with st.spinner("Building fingerprints + computing similarity…"):
                fp_q = morgan_fp(mol_q)
                fps_nn, idx_map_nn = build_dataset_fps(df_nn)

                if len(fps_nn) == 0:
                    st.warning("No valid fingerprints in the selected neighbor set.")
                else:
                    sims = np.array(DataStructs.BulkTanimotoSimilarity(fp_q, fps_nn), dtype=float)
                    top = np.argsort(-sims)[:topk]

                    rows = []
                    for j in top:
                        row = df_nn.iloc[int(idx_map_nn[j])]
                        rows.append(
                            {
                                "Tanimoto": float(sims[j]),
                                "molecule_chembl_id": str(row.get("molecule_chembl_id", "")),
                                "pic50(exp)": float(row.get("pic50", np.nan)),
                                "ic50_nM(exp)": float(row.get("ic50_nM", np.nan)),
                                "n_measurements": int(row.get("n_measurements", 0)),
                                "smiles": str(row.get("smiles", "")),
                            }
                        )
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Click **Compute nearest neighbors** to run similarity search.")

with tab_data:
    st.subheader("Dataset dashboard")

    pic50_min, pic50_max = float(df["pic50"].min()), float(df["pic50"].max())
    pic50_range = st.slider(
        "pIC50 range (dashboard)",
        pic50_min,
        pic50_max,
        (pic50_min, pic50_max),
        key="dash_pic50",
    )
    min_n = st.slider(
        "Min measurements (dashboard)",
        1,
        int(df["n_measurements"].max()),
        1,
        key="dash_min_n",
    )

    df_f = df[
        (df["pic50"] >= pic50_range[0])
        & (df["pic50"] <= pic50_range[1])
        & (df["n_measurements"] >= min_n)
    ].copy()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="pic50", nbins=40, title="pIC50 distribution (FULL)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        figf = px.histogram(df_f, x="pic50", nbins=40, title="pIC50 distribution (FILTERED)")
        st.plotly_chart(figf, use_container_width=True)

    fig3 = px.histogram(df, x="n_measurements", nbins=30, title="n_measurements (FULL)")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Preview (filtered)")
    st.dataframe(df_f.head(50), use_container_width=True)

with tab_about:
    st.subheader("Methodology (scientific context)")
    st.markdown(
        """
**Target:** Cyclin-dependent kinase 2 (CDK2)

**Endpoint:** IC50 values converted to pIC50  
- pIC50 = -log10(IC50 [M])  
- IC50 (nM) = 10^(9 - pIC50)

**Model:** RandomForest regression on Morgan fingerprints  
- Fingerprints: radius=2, 2048 bits  
- Uncertainty proxy: std across RF trees (σ)

**Applicability domain:**  
- Similarity uses Tanimoto similarity vs dataset fingerprints.  
- Low similarity indicates limited training support for that chemistry.

**Model hosting:**  
- Model file is stored in `Utkarsh2405/cdk2-rf-model` and downloaded from Hugging Face on first use.
        """
    )
