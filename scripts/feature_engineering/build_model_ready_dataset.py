# -*- coding: utf-8 -*-
"""
Build a model-ready dataset from EMR-like fact and dimension tables.

Inputs (CSV):
  - dataset/cleaned_dataset/fact_txn_cleaned.csv
  - dataset/cleaned_dataset/dim_patient_cleaned.csv
  - dataset/cleaned_dataset/dim_physician_cleaned.csv

Field dictionary (XLSX):
  - dataset/model_ready_dataset/model_feature_dictionary.xlsx
    Used to (a) enforce column order and (b) coerce dtypes where feasible.

Output (CSV):
  - dataset/model_ready_dataset/model_ready_dataset.csv

Business rules implemented (high-level):
  - Cohort: one row per patient, based on each patient's earliest DISEASE_X diagnosis.
  - Age filter: keep patients with age >= 12 at diagnosis year.
  - Symptom onset: most recent symptom date on/before the diagnosis date (tracked but not used for TARGET filtering).
  - Target: TARGET = 1 if patient was NOT prescribed Drug A; TARGET = 0 if patient received Drug A.
  - Target is computed to align with alert logic: 1 = NOT prescribed Drug A (possibly due to an oversight or being considered unsuitable for treatment), 0 = prescribed Drug A.
  - Risk conditions: flags for key comorbidities present on/before diagnosis, plus risk counts and age-based risk.
  - Contraindication level: highest level on/before diagnosis (0/1/2/3).
  - Physician experience: lifetime counts and treatment rate of Drug A among diagnosed patients.
  - Time features: season of diagnosis and symptom→diagnosis day difference.
  - Individual symptom flags: binary indicators for each symptom type present before diagnosis.
  - Column names SYMPTOM_TO_DIAGNOSIS_DAYS and DIAGNOSIS_WITHIN_5DAYS_FLAG are uppercase as requested.

Notes:
  - PHYSICIAN_ID of -1 is treated as missing (NaN) for physician-level aggregations.
  - When a dtype coercion fails (e.g., due to mixed content), the script leaves the original dtype to avoid breaking.
  - All features from the feature dictionary are included regardless of inclusion status.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# Configuration 
# ============================================================
# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "model_ready"
FACT_PATH = DATA_DIR / "fact_txn_cleaned.csv"
PAT_PATH  = DATA_DIR / "dim_patient_cleaned.csv"
DOC_PATH  = DATA_DIR / "dim_physician_cleaned.csv"
DICT_PATH = OUTPUT_DIR / "model_feature_dictionary.xlsx"
OUT_PATH  = OUTPUT_DIR / "model_ready_dataset.csv"

# Domain constants
DX_CODE = "DISEASE_X"
DRUG_A = "DRUG A"
MIN_AGE_YRS = 12               # indication-based minimum age
PHYS_EXP_THRESHOLDS = {        # physician experience banding thresholds
    "High": 200,               # >=200 diagnoses
    "Medium": 20               # >=20 diagnoses; else Low
}

# Optional cohort window, if needed in the future:
# COHORT_START = pd.Timestamp("YYYY-MM-DD")
# COHORT_END   = pd.Timestamp("YYYY-MM-DD")


# ============================================================
# I/O and standardization helpers
# ============================================================
def load_and_standardize() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load fact and dimension tables, and normalize string fields in the fact table.
    Returns:
        fact: transactional events with TXN_* fields
        pat:  patient dimension
        doc:  physician dimension
    """
    fact = pd.read_csv(FACT_PATH, parse_dates=["TXN_DT"])
    pat  = pd.read_csv(PAT_PATH)
    doc  = pd.read_csv(DOC_PATH)

    # Normalize key string fields to uppercase and stripped whitespace to reduce join/noise errors.
    for c in ["TXN_TYPE", "TXN_DESC", "TXN_LOCATION_TYPE", "INSURANCE_TYPE"]:
        fact[c] = fact[c].astype(str).str.strip().str.upper()

    return fact, pat, doc


# ============================================================
# Cohort & core joins
# ============================================================
def build_index_dx(fact: pd.DataFrame) -> pd.DataFrame:
    """
    Build the per-patient index diagnosis row.
    - One row per patient.
    - Uses the patient's earliest DISEASE_X record.
    - Preserves relevant columns for downstream feature logic.
    """
    dx_all = fact[fact["TXN_DESC"] == DX_CODE].copy()

    # If a cohort window is desired, uncomment the following:
    # if "COHORT_START" in globals() and "COHORT_END" in globals():
    #     dx_all = dx_all[(dx_all["TXN_DT"] >= COHORT_START) & (dx_all["TXN_DT"] <= COHORT_END)]

    dx_all = dx_all.sort_values(["PATIENT_ID", "TXN_DT"])
    idx = dx_all.groupby("PATIENT_ID", as_index=False).first()

    # Rename selected fields to diagnosis-centric names
    idx = idx.rename(columns={
        "TXN_DT"           : "DISEASEX_DT",
        "TXN_LOCATION_TYPE": "LOCATION_TYPE",
        "PHYSICIAN_ID"     : "PHYSICIAN_ID",
        "INSURANCE_TYPE"   : "INSURANCE_TYPE_AT_DX",
    })
    return idx


def enrich_patient_physician(idx: pd.DataFrame, pat: pd.DataFrame, doc: pd.DataFrame) -> pd.DataFrame:
    """
    Join patient and physician dimensions; derive patient age; standardize basic demographics.
    - Age is computed as diagnosis year minus birth year (if available > 0).
    - Filters to patients with age >= MIN_AGE_YRS.
    - Fills unknown physician fields with 'UNKNOWN' for robustness.
    """
    # Patient join & age
    pat2 = pat.rename(columns={"GENDER": "PATIENT_GENDER", "BIRTH_YEAR": "BIRTH_YEAR_PAT"})
    base = idx.merge(pat2, on="PATIENT_ID", how="left")

    base["PATIENT_AGE"] = np.where(
        base["BIRTH_YEAR_PAT"].fillna(-1) > 0,
        base["DISEASEX_DT"].dt.year - base["BIRTH_YEAR_PAT"],
        np.nan
    )
    base["PATIENT_GENDER"] = base["PATIENT_GENDER"].fillna("UNKNOWN")

    # Indication-based age filter
    base = base[base["PATIENT_AGE"] >= MIN_AGE_YRS].copy()

    # Physician join
    doc2 = doc.rename(columns={"GENDER": "PHYSICIAN_GENDER", "BIRTH_YEAR": "PHYSICIAN_BIRTH_YEAR"})
    base = base.merge(doc2, on="PHYSICIAN_ID", how="left")
    base["PHYSICIAN_STATE"] = base["STATE"].fillna("UNKNOWN")
    base["PHYSICIAN_TYPE"]  = base["PHYSICIAN_TYPE"].fillna("UNKNOWN")

    return base


# ============================================================
# Symptom-based features (with onset tracking)
# ============================================================
def compute_symptoms(base: pd.DataFrame, fact: pd.DataFrame) -> pd.DataFrame:
    """
    Derive symptom features:
    - SYMPTOM_ONSET_DT: most recent symptom date on/before DISEASEX_DT.
    - SYM_COUNT_5D: number of symptom records within 5 days prior to DISEASEX_DT (inclusive).
    - Individual symptom binary flags (0/1) for each symptom type present in the dataset.
    - SYMPTOM_TO_DIAGNOSIS_DAYS: day difference (diagnosis minus onset)
    - DIAGNOSIS_WITHIN_5DAYS_FLAG: 1 if difference <= 5, else 0
    """
    sym = fact[fact["TXN_TYPE"] == "SYMPTOMS"][["PATIENT_ID", "TXN_DT", "TXN_DESC"]].copy()
    sym = sym.merge(base[["PATIENT_ID", "DISEASEX_DT"]], on="PATIENT_ID", how="inner")

    # Keep only symptoms on or before the diagnosis date
    sym = sym[sym["TXN_DT"] <= sym["DISEASEX_DT"]]

    # Most recent (latest) symptom on/before diagnosis
    sym_onset = (
        sym.sort_values(["PATIENT_ID", "TXN_DT"])
           .groupby("PATIENT_ID", as_index=False)
           .last()
           .rename(columns={"TXN_DT": "SYMPTOM_ONSET_DT"})
    )[["PATIENT_ID", "SYMPTOM_ONSET_DT"]]
    base = base.merge(sym_onset, on="PATIENT_ID", how="left")

    # Count symptoms within 5 days prior to diagnosis, inclusive
    sym5 = sym[sym["TXN_DT"].between(
        sym["DISEASEX_DT"] - pd.Timedelta(days=5),
        sym["DISEASEX_DT"], inclusive="both"
    )]
    sym_count = sym5.groupby("PATIENT_ID").size().rename("SYM_COUNT_5D").reset_index()
    base = base.merge(sym_count, on="PATIENT_ID", how="left")
    base["SYM_COUNT_5D"] = base["SYM_COUNT_5D"].fillna(0).astype(int)

    # Create binary features for each individual symptom type
    base = create_symptom_binary_features(base, sym)

    return base


def create_symptom_binary_features(base: pd.DataFrame, sym: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary (0/1) features for each individual symptom type.
    
    This function creates a separate binary feature for each symptom type found in the dataset.
    Each feature indicates whether the patient had that specific symptom before diagnosis.
    
    Parameters:
    - base: Base patient dataframe
    - sym: Symptom transaction dataframe (already filtered to diagnosis date)
    
    Returns:
    - Updated dataframe with individual symptom binary features
    """
    # Define all symptom types found in the dataset
    symptom_types = [
        "ACUTE_PHARYNGITIS",    # Acute pharyngitis
        "ACUTE_URI",            # Acute upper respiratory infection
        "CHILLS",               # Chills
        "CONGESTION",           # Congestion
        "COUGH",                # Cough
        "DIARRHEA",             # Diarrhea
        "DIFFICULTY_BREATHING", # Difficulty breathing
        "FATIGUE",              # Fatigue
        "FEVER",                # Fever
        "HEADACHE",             # Headache
        "LOSS_OF_TASTE_OR_SMELL", # Loss of taste or smell
        "MUSCLE_ACHE",          # Muscle ache
        "NAUSEA_AND_VOMITING",  # Nausea and vomiting
        "SORE_THROAT"           # Sore throat
    ]
    
    # Create binary feature for each symptom type
    for symptom in symptom_types:
        # Create symptom binary flag
        symptom_flag = (
            sym[sym["TXN_DESC"] == symptom]
            .groupby("PATIENT_ID")
            .size()
            .rename(f"SYMPTOM_{symptom}")
            .reset_index()
        )
        
        # Convert count to binary flag (1 if symptom present, 0 if not)
        symptom_flag[f"SYMPTOM_{symptom}"] = 1
        
        # Merge with base dataframe
        base = base.merge(symptom_flag, on="PATIENT_ID", how="left")
        
        # Fill missing values with 0 (symptom not present)
        base[f"SYMPTOM_{symptom}"] = base[f"SYMPTOM_{symptom}"].fillna(0).astype(int)
    
    return base


# ============================================================
# Treatment label (simplified)
# ============================================================
def label_target(base: pd.DataFrame, fact: pd.DataFrame) -> pd.DataFrame:
    """
    Compute TARGET to align with alert logic:
    - TARGET = 1 if patient was NOT prescribed Drug A (potential missed prescription; alert candidates).
    - TARGET = 0 if patient received Drug A (or prescribing is not clinically necessary).
    - No treatment window restrictions.
    """
    treat = fact[(fact["TXN_TYPE"] == "TREATMENTS") & (fact["TXN_DESC"] == DRUG_A)][["PATIENT_ID", "TXN_DT"]].copy()

    # Simple flag: 1 if patient ever received Drug A, 0 otherwise
    treated = treat.groupby("PATIENT_ID").size().rename("HAS_TREAT").reset_index()
    treated["HAS_TREAT"] = 1
    base = base.merge(treated, on="PATIENT_ID", how="left")
    base["HAS_TREAT"] = base["HAS_TREAT"].fillna(0).astype(int)

    # Final target: 1 = NOT prescribed Drug A (alert), 0 = prescribed/not necessary
    base["TARGET"] = (1 - base["HAS_TREAT"]).astype(int)
    return base


# ============================================================
# Comorbidity risk features
# ============================================================
def risk_features(base: pd.DataFrame, fact: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comorbidity/risk flags present on/before diagnosis.
    Also compute:
      - RISK_NUM: number of risk flags present
      - RISK_AGE_FLAG: 1 if age >= 65 else 0
    """
    cond = fact[fact["TXN_TYPE"] == "CONDITIONS"][["PATIENT_ID", "TXN_DT", "TXN_DESC"]].copy()
    cond = cond.merge(base[["PATIENT_ID", "DISEASEX_DT"]], on="PATIENT_ID", how="inner")
    cond = cond[cond["TXN_DT"] <= cond["DISEASEX_DT"]]

    def to_flag(df: pd.DataFrame, names: list[str], colname: str) -> pd.DataFrame:
        out = df[df["TXN_DESC"].isin(names)].groupby("PATIENT_ID").size().rename(colname).reset_index()
        out[colname] = 1
        return out

    risk_immuno   = to_flag(cond, ["IMMUNOCOMPROMISED"], "RISK_IMMUNO")
    risk_cvd      = to_flag(cond, ["HEART_DISEASE", "HYPERTENSION", "STROKE"], "RISK_CVD")
    risk_diabetes = to_flag(cond, ["DIABETES"], "RISK_DIABETES")
    risk_obesity  = to_flag(cond, ["OBESITY"], "RISK_OBESITY")

    for df_flag in [risk_immuno, risk_cvd, risk_diabetes, risk_obesity]:
        base = base.merge(df_flag, on="PATIENT_ID", how="left")

    for c in ["RISK_IMMUNO", "RISK_CVD", "RISK_DIABETES", "RISK_OBESITY"]:
        base[c] = base[c].fillna(0).astype(int)

    base["RISK_NUM"] = base[["RISK_IMMUNO", "RISK_CVD", "RISK_DIABETES", "RISK_OBESITY"]].sum(axis=1)
    base["RISK_AGE_FLAG"] = (base["PATIENT_AGE"] >= 65).astype(int)

    return base


# ============================================================
# Contraindication level
# ============================================================
def contraindication_level(base: pd.DataFrame, fact: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PRIOR_CONTRA_LVL as the highest contraindication level on/before diagnosis.
    Mapping:
      - LOW_CONTRAINDICATION    -> 1
      - MEDIUM_CONTRAINDICATION -> 2
      - HIGH_CONTRAINDICATION   -> 3
      - None/missing            -> 0
    """
    contra = fact[fact["TXN_TYPE"] == "CONTRAINDICATIONS"][["PATIENT_ID", "TXN_DT", "TXN_DESC"]].copy()
    contra = contra.merge(base[["PATIENT_ID", "DISEASEX_DT"]], on="PATIENT_ID", how="inner")
    contra = contra[contra["TXN_DT"] <= contra["DISEASEX_DT"]]

    lvl_map = {
        "LOW_CONTRAINDICATION": 1,
        "MEDIUM_CONTRAINDICATION": 2,
        "HIGH_CONTRAINDICATION": 3
    }
    contra["LVL"] = contra["TXN_DESC"].map(lvl_map).fillna(0).astype(int)

    prior_contra = contra.groupby("PATIENT_ID")["LVL"].max().rename("PRIOR_CONTRA_LVL").reset_index()
    base = base.merge(prior_contra, on="PATIENT_ID", how="left")
    base["PRIOR_CONTRA_LVL"] = base["PRIOR_CONTRA_LVL"].fillna(0).astype(int)

    return base


# ============================================================
# Time features & season
# ============================================================
def add_time_and_season(base: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features:
      - DX_SEASON: season of the diagnosis date
      - SYMPTOM_TO_DIAGNOSIS_DAYS: day difference (diagnosis minus onset)
      - DIAGNOSIS_WITHIN_5DAYS_FLAG: 1 if difference <= 5, else 0
    """
    def season(d: pd.Timestamp) -> str:
        m = d.month
        if m in (12, 1, 2):  return "Winter"
        if m in (3, 4, 5):   return "Spring"
        if m in (6, 7, 8):   return "Summer"
        return "Fall"

    base["DX_SEASON"] = base["DISEASEX_DT"].apply(season)

    # Compute onset→diagnosis day difference; fix negative values defensively if any occur.
    base["SYMPTOM_ONSET_DT"] = pd.to_datetime(base["SYMPTOM_ONSET_DT"])
    base["SYMPTOM_TO_DIAGNOSIS_DAYS"] = (base["DISEASEX_DT"] - base["SYMPTOM_ONSET_DT"]).dt.days

    neg_mask = base["SYMPTOM_TO_DIAGNOSIS_DAYS"] < 0
    if neg_mask.any():
        # Align onset to the diagnosis date for these rows and recompute
        base.loc[neg_mask, "SYMPTOM_ONSET_DT"] = base.loc[neg_mask, "DISEASEX_DT"]
        base["SYMPTOM_TO_DIAGNOSIS_DAYS"] = (base["DISEASEX_DT"] - base["SYMPTOM_ONSET_DT"]).dt.days

    base["DIAGNOSIS_WITHIN_5DAYS_FLAG"] = (base["SYMPTOM_TO_DIAGNOSIS_DAYS"].fillna(0) <= 5).astype(int)
    return base


# ============================================================
# Physician-level features
# ============================================================
def physician_experience(base: pd.DataFrame, fact: pd.DataFrame) -> pd.DataFrame:
    """
    Compute physician-level metrics over the full historical range:
      - PHYS_TOTAL_DX: number of DISEASE_X diagnoses done by physician
      - PHYS_TREAT_RATE_ALL: % of their diagnosed patients who ever received Drug A
      - PHYS_EXPERIENCE_LEVEL: High/Medium/Low based on PHYS_TOTAL_DX
    """
    dx_hist = fact[fact["TXN_DESC"] == DX_CODE][["PHYSICIAN_ID", "PATIENT_ID"]].copy()
    treat_hist = fact[(fact["TXN_TYPE"] == "TREATMENTS") & (fact["TXN_DESC"] == DRUG_A)][["PATIENT_ID"]].drop_duplicates()

    # Treat PHYSICIAN_ID == -1 as missing
    dx_hist["PHYSICIAN_ID"] = dx_hist["PHYSICIAN_ID"].replace(-1, np.nan)
    base["PHYSICIAN_ID"]    = base["PHYSICIAN_ID"].replace(-1, np.nan)

    # Total diagnoses by physician
    phys_total = dx_hist.dropna(subset=["PHYSICIAN_ID"]).groupby("PHYSICIAN_ID").size().rename("PHYS_TOTAL_DX").reset_index()

    # Physician treatment rate: among their diagnosed patients, proportion who ever received Drug A
    den_hist = dx_hist.dropna(subset=["PHYSICIAN_ID"]).drop_duplicates().merge(
        treat_hist.assign(TREATED=1), on="PATIENT_ID", how="left"
    )
    den_hist["TREATED"] = den_hist["TREATED"].fillna(0)
    phys_rate = den_hist.groupby("PHYSICIAN_ID")["TREATED"].mean().rename("PHYS_TREAT_RATE_ALL").reset_index()

    # Merge both aggregates back to patient rows
    base = base.merge(phys_total, on="PHYSICIAN_ID", how="left").merge(phys_rate, on="PHYSICIAN_ID", how="left")

    # Experience level banding based on total diagnoses
    def exp_level(n: float) -> str:
        if pd.isna(n): return "Low"
        if n >= PHYS_EXP_THRESHOLDS["High"]:   return "High"
        if n >= PHYS_EXP_THRESHOLDS["Medium"]: return "Medium"
        return "Low"

    base["PHYS_EXPERIENCE_LEVEL"] = base["PHYS_TOTAL_DX"].map(exp_level)
    return base


# ============================================================
# Feature dictionary alignment (order + dtype where feasible)
# ============================================================
def load_feature_dictionary(dict_path: Path) -> pd.DataFrame:
    """
    Load the feature dictionary with all features included.
    Expected columns:
      - Variable: target column name
      - Type:    intended data type hint; used for gentle coercion
      - Included in Model: indicates whether feature should be included (for reference only)
    """
    feat_dict = pd.read_excel(dict_path)
    required = {"Variable", "Type"}
    missing = required - set(feat_dict.columns)
    if missing:
        raise ValueError(f"Feature dictionary is missing required column(s): {missing}")

    # Drop empty rows and normalize casing/whitespace
    feat_dict = feat_dict[feat_dict["Variable"].notna()].copy()
    feat_dict["Variable"] = feat_dict["Variable"].astype(str).str.strip()

    # Include all features regardless of inclusion status
    # The "Included in Model" column is kept for reference but not used for filtering

    return feat_dict


def coerce_types(df: pd.DataFrame, feat_dict: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to coerce each column to the type declared in the dictionary, where feasible.
    This uses a conservative approach: if coercion fails, the original dtype is kept.
    """
    def to_dtype(tstr: str) -> str | None:
        t = str(tstr).lower()
        if "int" in t:
            return "Int64"            # nullable integer
        if "float" in t:
            return "float64"
        if "object" in t or "string" in t or "category" in t:
            return "object"           # keep as generic object (safe default)
        if "date" in t or "datetime" in t:
            return "datetime64[ns]"
        return None                   # unknown or "keep as is"

    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    for _, row in feat_dict.iterrows():
        col = row["Variable"]
        if col in df.columns:
            target = to_dtype(row["Type"])
            if target == "datetime64[ns]":
                df.loc[:, col] = pd.to_datetime(df[col], errors="coerce")
            elif target:
                try:
                    df.loc[:, col] = df[col].astype(target)
                except Exception:
                    # If coercion fails (mixed content, etc.), keep as-is.
                    pass
    return df


def reorder_and_fill(df: pd.DataFrame, feat_dict: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder columns according to the dictionary.
    - Columns missing from df are created and filled with <NA>.
    - Extra columns not listed in the dictionary are removed to keep only specified variables.
    - DX_SEASON is always included even if not in the dictionary.
    """
    ordered_cols = list(feat_dict["Variable"])
    
    # Always include DX_SEASON if it exists in df
    if "DX_SEASON" in df.columns and "DX_SEASON" not in ordered_cols:
        ordered_cols.append("DX_SEASON")

    # Ensure all required columns exist
    for c in ordered_cols:
        if c not in df.columns:
            df[c] = pd.Series([pd.NA] * len(df))

    # Keep only columns that are in the ordered list
    df = df[ordered_cols]
    return df


# ============================================================
# Main orchestration
# ============================================================
def main():
    # 1) Build the base patient-level dataset
    fact, pat, doc = load_and_standardize()
    idx = build_index_dx(fact)
    base = enrich_patient_physician(idx, pat, doc)
    base = compute_symptoms(base, fact)
    base = label_target(base, fact)
    base = risk_features(base, fact)
    base = contraindication_level(base, fact)
    base = add_time_and_season(base)
    base = physician_experience(base, fact)

    # 2) Load the feature dictionary and align output to its order and types
    feat_dict = load_feature_dictionary(DICT_PATH)

    # Normalize to uppercase to avoid casing mismatches.
    base.columns = [c.upper() for c in base.columns]
    feat_dict["Variable"] = feat_dict["Variable"].str.upper()

    # Ensure requested uppercase names for these two features
    base = base.rename(columns={
        "SYMPTOM_TO_DIAGNOSIS_DAYS": "SYMPTOM_TO_DIAGNOSIS_DAYS",
        "DIAGNOSIS_WITHIN_5DAYS_FLAG": "DIAGNOSIS_WITHIN_5DAYS_FLAG",
    })

    # Reorder + fill + coerce types (best-effort)
    final_df = reorder_and_fill(base, feat_dict)
    final_df = coerce_types(final_df, feat_dict)

    # 3) Save the final, dictionary-aligned dataset
    final_df.to_csv(OUT_PATH, index=False)

    # 4) Lightweight reporting to stdout
    dx_min = final_df["DISEASEX_DT"].min() if "DISEASEX_DT" in final_df.columns else pd.NaT
    dx_max = final_df["DISEASEX_DT"].max() if "DISEASEX_DT" in final_df.columns else pd.NaT
    print(f"[OK] Wrote file: {OUT_PATH}")
    print(f"Rows: {final_df.shape[0]}, Cols: {final_df.shape[1]}")
    print("DISEASEX_DT range:", dx_min.date() if pd.notna(dx_min) else None, "->", dx_max.date() if pd.notna(dx_max) else None)
    if "DIAGNOSIS_WITHIN_5DAYS_FLAG" in final_df.columns:
        print("DIAGNOSIS_WITHIN_5DAYS_FLAG mean:", float(final_df["DIAGNOSIS_WITHIN_5DAYS_FLAG"].mean()))


if __name__ == "__main__":
    main()