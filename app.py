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

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Draw, QED, rdMolDescriptors

# =========================
# CONFIG (CDK2 only)
# =========================
PROJECT = Path(__file__).resolve().parent

# Dataset file is currently in your repo root (based on your repo contents)
DATA_PATH = PROJECT / "cdk2_pic50_clean.parquet"

# Model will be downloaded to this local path (do NOT commit the 139MB file to GitHub)
MODEL_PATH = PROJECT / "models" / "cdk2_rf_final_all_data.joblib"

# Google Drive direct-download URL (uc?id=FILE_ID)
MODEL_DRIVE_URL = "https://drive.google.com/uc?id=1pOgZVHG7BfrcXE7ZmHJNM9CMnsBNlCQa"

FP_RADIUS = 2
FP_NBITS = 2048

# Similarity thresholds (Applicability Domain)
SIM_HIGH = 0.50
SIM_MED = 0.30

st.set_page_config(page_title="ChemBLitz — CDK2 Scientific Predictor", layout="wide")


# =========================
# Download helper (Google Drive)
# =========================
def ensure_model_present(model_path: Path) -> None:
    """
    Ensure the model file exists locally.
    If missing, download from Google Drive using gdown (first run only).
    """
    if model_path.exists():
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import gdown  # type: ignore
    except Exception as e:
        raise RuntimeError("gdown is not installed. Add `gdown` to requirements.txt") from e

    with st.spinner("Downloading model from Google Drive (first run only)…"):
        out = gdown.download(MODEL_DRIVE_URL, str(model_path), quiet=False, fuzzy=True)
        if (not out) or (not model_path.exists()):
            raise RuntimeError(
                "Model download failed. "
                "Check that Google Drive sharing is set to: Anyone with the link = Viewer."
            )


# =========================
# Helpers
# =========================
def pic50_to_ic50_nM(pic50: float) -> float:
    # IC50 (nM) = 10^(9 - pIC50)
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


@st.cache_data(show_spinner=False)
def chembl_pref_name_from_chembl_id(chembl_id: str) -> Optional[str]:
    # ChEMBL REST API: molecule endpoint
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
# Model + data
# =========================
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    # Expected columns: inchikey, smiles, molecule_chembl_id, pic50, ic50_nM, n_measurements
    return df


@st.cache_resource
def load_model():
    ensure_model_present(MODEL_PATH)
    return joblib.load(MODEL_PATH)


@st.cache_resource
def build_dataset_fps(df: pd.DataFrame):
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


def build_real_examples_from_dataset(df: pd.DataFrame) -> dict[str, str]:
    """
    Build “real CDK2-related examples” directly from the dataset:
    strongest, median, weakest, most measured.
    """
    out: dict[str, str] = {}

    dff = df.dropna(subset=["pic50", "smiles"]).copy()
    dff["pic50"] = pd.to_numeric(dff["pic50"], errors="coerce")
    dff = dff.dropna(subset=["pic50"])

    # Top potency
    top = dff.sort_values("pic50", ascending=False).head(1)
    if len(top):
        r = top.iloc[0]
        label = f"Top binder (pIC50 {r['pic50']:.2f}) — {r.get('molecule_chembl_id','')}".strip(" —")
        out[label] = str(r["smiles"])

    # Median potency
    med_pic = float(dff["pic50"].median())
    med_row = dff.iloc[(dff["pic50"] - med_pic).abs().argsort()[:1]]
    if len(med_row):
        r = med_row.iloc[0]
        label = f"Median binder (pIC50 ~{med_pic:.2f}) — {r.get('molecule_chembl_id','')}".strip(" —")
        out[label] = str(r["smiles"])

    # Low potency
    low = dff.sort_values("pic50", ascending=True).head(1)
    if len(low):
        r = low.iloc[0]
        label = f"Weak binder (pIC50 {r['pic50']:.2f}) — {r.get('molecule_chembl_id','')}".strip(" —")
        out[label] = str(r["smiles"])

    # Most measured
    if "n_measurements" in dff.columns:
        mq = dff.sort_values("n_measurements", ascending=False).head(1)
        if len(mq):
            r = mq.iloc[0]
            label = f"Most measured (n={int(r['n_measurements'])}) — {r.get('molecule_chembl_id','')}".strip(" —")
            out[label] = str(r["smiles"])

    if not out:
        out["Dataset example"] = str(df.iloc[0]["smiles"])

    return out


# =========================
# UI
# =========================
st.title("ChemBLitz — CDK2 pIC50 Scientific Predictor")
st.caption(
    "CDK2-only. Prediction + uncertainty + applicability-domain checks + nearest-neighbor evidence + ADMET-like heuristics. "
    "Model is downloaded from Google Drive on first run."
)

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH} (expected in repo root)")
    st.stop()

df = load_data()

# Initialize SMILES in session
if "smiles" not in st.session_state:
    st.session_state["smiles"] = str(df.iloc[0]["smiles"]) if "smiles" in df.columns else "CCO"

examples = build_real_examples_from_dataset(df)

tab_pred, tab_val, tab_data, tab_about = st.tabs(["Predict", "Validate (Similarity)", "Dataset Dashboard", "Methodology"])

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
            help="SMILES is a text representation of a molecule. Paste any valid SMILES here.",
        )

        run = st.button("Predict", type="primary")

        st.markdown("---")
        st.subheader("Dataset filters (affect filtered dashboard + optional filtered neighbors)")
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
        st.subheader("Output (scientific)")

        if not run:
            st.info("Click **Predict** to generate outputs.")
        else:
            # Load model only when needed (prevents first-run download when user hasn't clicked Predict)
            model = load_model()

            smiles = st.session_state["smiles"]
            mol = mol_from_smiles(smiles)
            if mol is None:
                st.error("Invalid SMILES.")
                st.stop()

            can = canonical_smiles(mol)
            ik = inchikey(mol)

            st.image(draw_mol_png(mol), caption="Input structure (RDKit 2D)")
            st.write("Canonical SMILES:")
            st.code(can, language="text")
            st.write("InChIKey:", ik)

            # Local dataset match + filter inclusion
            local_match = df[df["inchikey"].astype(str) == ik] if "inchikey" in df.columns else pd.DataFrame()
            chembl_id = None
            input_is_in_dataset = len(local_match) > 0

            if input_is_in_dataset:
                row0 = local_match.iloc[0]
                chembl_id = str(row0.get("molecule_chembl_id", "")).strip() or None
                st.success(f"Found in dataset: {chembl_id or 'N/A'}")
            else:
                st.warning("Not found in dataset (by InChIKey).")

            input_in_filtered = False
            if input_is_in_dataset and "inchikey" in df_f.columns:
                input_in_filtered = len(df_f[df_f["inchikey"].astype(str) == ik]) > 0
                st.write("Included after filters:", "✅ Yes" if input_in_filtered else "❌ No (excluded by current filters)")

            # Name (preferred) + nomenclature identifier fallback
            pref = chembl_pref_name_from_chembl_id(chembl_id) if chembl_id else None
            if pref:
                st.write("Compound name (ChEMBL pref_name):", pref)
            elif chembl_id:
                st.write("Compound name:", f"ChEMBL compound {chembl_id} (no pref_name available)")
            else:
                st.write("Compound name:", "Unknown (showing canonical identifiers instead)")

            st.caption("Canonical SMILES + InChIKey are stable chemical identifiers (nomenclature-style).")

            # Predict
            fp = morgan_fp(mol)
            X = fp_to_numpy(fp).reshape(1, -1)
            pred_pic50, pred_std = rf_predict_with_uncertainty(model, X)
            pred_ic50 = pic50_to_ic50_nM(pred_pic50)

            # Similarity for applicability domain (FULL dataset)
            fps_all, _idx_map_all = build_dataset_fps(df)
            sims_all = np.array(DataStructs.BulkTanimotoSimilarity(fp, fps_all), dtype=float) if len(fps_all) else np.array([])
            max_sim = float(sims_all.max()) if len(sims_all) else 0.0
            mean_sim = float(sims_all.mean()) if len(sims_all) else 0.0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted pIC50", f"{pred_pic50:.3f}", help="Higher pIC50 means stronger potency.")
            m2.metric("Predicted IC50 (nM)", f"{pred_ic50:,.2f}", help="Converted from pIC50. Lower IC50 is better.")
            m3.metric("Uncertainty (σ)", f"{pred_std:.3f}", help="Std dev across RF trees. Lower = more consistent.")
            m4.metric("Potency class", potency_bucket(pred_pic50))

            st.markdown("### Applicability domain (validity)")
            if max_sim >= SIM_HIGH:
                st.success(f"In-domain (max similarity: {max_sim:.3f})")
            elif max_sim >= SIM_MED:
                st.warning(f"Borderline domain (max similarity: {max_sim:.3f})")
            else:
                st.error(f"Out-of-domain risk (max similarity: {max_sim:.3f})")

            conf = confidence_label(pred_std, max_sim)
            st.write(f"Mean similarity (dataset): {mean_sim:.3f}")
            st.write("Confidence:", conf)

            st.markdown("### PhysChem + ADMET-like heuristics")
            d = compute_descriptors(mol)
            ro5 = lipinski_violations(d)

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("MolWt", f"{d['MolWt']:.1f}", help="Molecular weight. >500 often reduces oral developability.")
            d2.metric("cLogP", f"{d['LogP']:.2f}", help="Hydrophobicity. Too high can reduce solubility.")
            d3.metric("TPSA", f"{d['TPSA']:.1f}", help="Polar surface area. Very high TPSA can reduce permeability.")
            d4.metric("QED", f"{d['QED']:.2f}", help="Drug-likeness score (0–1). Higher is better.")

            d5, d6, d7, d8 = st.columns(4)
            d5.metric("HBD", str(d["HBD"]), help="H-bond donors. Too many can reduce permeability.")
            d6.metric("HBA", str(d["HBA"]), help="H-bond acceptors.")
            d7.metric("RotB", str(d["RotB"]), help="Rotatable bonds. Too many can hurt bioavailability.")
            d8.metric("Rings", str(d["Rings"]), help="Ring count. Impacts rigidity and lipophilicity.")

            st.write(f"Lipinski Ro5 violations: **{ro5}**")
            st.write(f"Veber rule (TPSA ≤ 140 and RotB ≤ 10): **{'PASS' if veber_pass(d) else 'FAIL'}**")

            st.markdown("### Final decision summary")
            flags = []
            if conf == "Low":
                flags.append("Low confidence (uncertainty/similarity).")
            if ro5 >= 2:
                flags.append("Multiple Lipinski violations.")
            if not veber_pass(d):
                flags.append("Veber rule failed (permeability/performance risk).")

            if not flags:
                st.success("Overall: Suitable for screening. Next: selectivity + solubility + metabolic stability checks.")
            else:
                st.warning("Overall: Use caution. " + " ".join(flags))

            st.markdown("### Improvement suggestions")
            for s in improvement_suggestions(d, max_sim):
                st.write(f"- {s}")

            with st.expander("How to interpret these metrics (quick guide)"):
                st.markdown(
                    """
- **pIC50 / IC50:** potency prediction for CDK2. Higher pIC50 (lower IC50) is stronger inhibition.
- **Uncertainty (σ):** consistency across RF trees (proxy for uncertainty). Lower is better.
- **Similarity (Tanimoto):** how close your molecule is to training data. Low similarity ⇒ out-of-domain.
- **MolWt, cLogP, TPSA, RotB, HBD/HBA:** medicinal chemistry properties related to absorption/permeability.
- **QED:** drug-likeness (higher is better).
                    """
                )

with tab_val:
    st.subheader("Nearest-neighbor evidence (updates with current SMILES)")

    smiles = st.session_state["smiles"]
    mol_q = mol_from_smiles(smiles)

    use_filtered = st.toggle(
        "Use filtered dataset for neighbors",
        value=False,
        help="Turn ON to restrict neighbor evidence to the filtered subset.",
    )
    # Use the same filters computed in Predict tab if available; otherwise default to full df
    # (When user hasn't pressed Predict yet, df_f isn't created; we handle gracefully.)
    df_nn = df

    # Recompute filters here as well (so the tab is independent)
    pic50_min, pic50_max = float(df["pic50"].min()), float(df["pic50"].max())
    # Use same defaults; user-adjusted values persist because Streamlit stores widget state
    pic50_range = st.slider("pIC50 range (neighbors)", pic50_min, pic50_max, (pic50_min, pic50_max), key="nn_pic50")
    min_n = st.slider("Min measurements (neighbors)", 1, int(df["n_measurements"].max()), 1, key="nn_min_n")

    df_f2 = df[
        (df["pic50"] >= pic50_range[0])
        & (df["pic50"] <= pic50_range[1])
        & (df["n_measurements"] >= min_n)
    ].copy()

    if use_filtered:
        df_nn = df_f2

    if mol_q is None:
        st.error("Invalid SMILES.")
    else:
        fp_q = morgan_fp(mol_q)
        fps_nn, idx_map_nn = build_dataset_fps(df_nn)

        if len(fps_nn) == 0:
            st.warning("No valid fingerprints in the selected neighbor set.")
        else:
            sims = np.array(DataStructs.BulkTanimotoSimilarity(fp_q, fps_nn), dtype=float)
            topk = 10
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

with tab_data:
    st.subheader("Dataset dashboard (full vs filtered)")

    st.write("These plots show how your filters change which compounds are included.")

    pic50_min, pic50_max = float(df["pic50"].min()), float(df["pic50"].max())
    pic50_range = st.slider("pIC50 range (dashboard)", pic50_min, pic50_max, (pic50_min, pic50_max), key="dash_pic50")
    min_n = st.slider("Min measurements (dashboard)", 1, int(df["n_measurements"].max()), 1, key="dash_min_n")

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
- Uncertainty proxy: std across RF trees

**Applicability domain:**  
- Uses Tanimoto similarity to dataset molecules.  
- Low similarity indicates limited training support for that chemistry.

**Model hosting:**  
- The model file is downloaded from Google Drive on first run (so it doesn't need to be committed to GitHub).
        """
    )
