"""
load_data.py  -  load and clean the raw CICIDS2017 CSVs.

Other scripts (train.py) should do:   from load_data import load_raw
Run directly to inspect the data:     python src\\load_data.py
"""
import numpy as np
import pandas as pd

from config import RAW_DIR, LABEL_COL

HASH_COL = "_row_hash"

EXPECTED_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]


def _check_files_exist():
    """Fail loudly if any expected file is missing."""
    missing = [f for f in EXPECTED_FILES if not (RAW_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} file(s) in {RAW_DIR}:\n  " + "\n  ".join(missing)
        )


def _clean_labels(labels: pd.Series) -> pd.Series:
    """Strip spaces and replace garbled characters in label text.
    'Web Attack <garbage> Brute Force' -> 'Web Attack - Brute Force'
    """
    labels = labels.astype(str).str.strip()
    # Any run of non-ASCII / odd characters becomes ONE hyphen. (The garbage in
    # these files is several characters long, so we must match the whole run.)
    labels = labels.str.replace(r"[^A-Za-z0-9\s\-\.]+", "-", regex=True)
    labels = labels.str.replace(r"\s*-\s*", " - ", regex=True)   # uniform spacing
    labels = labels.str.replace(r"\s+", " ", regex=True).str.strip()
    # FTP-Patator / SSH-Patator keep their real hyphen
    labels = labels.str.replace(r"^(FTP|SSH) - Patator$", r"\1-Patator", regex=True)
    return labels


def _load_one(path) -> pd.DataFrame:
    # latin-1 never fails to decode, so a bad byte can't crash the read
    df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    df.columns = df.columns.str.strip()
    df[LABEL_COL] = _clean_labels(df[LABEL_COL])

    feature_cols = [c for c in df.columns if c != LABEL_COL]
    numeric = df[feature_cols].apply(pd.to_numeric, errors="coerce")  # still float64

    # Hash each row from the FULL-precision values, before shrinking to float32.
    # float32 rounds large numbers, so comparing rows after the cast could call
    # two different flows "duplicates". Duplicate checks use this hash instead.
    df[HASH_COL] = pd.util.hash_pandas_object(numeric, index=False).values

    df[feature_cols] = numeric.astype("float32")   # halves memory
    return df


def _report_conflicts(df: pd.DataFrame) -> pd.Series:
    """Rows with identical features but different labels. Returns a boolean mask."""
    n_labels = df.groupby(HASH_COL)[LABEL_COL].transform("nunique")
    mask = n_labels > 1
    print(f"Rows with identical features but different labels: {int(mask.sum()):,}")
    if mask.any():
        print("  Labels involved:\n" + df.loc[mask, LABEL_COL].value_counts().to_string())
    return mask


def load_raw(drop_conflicts: bool = True) -> pd.DataFrame:
    _check_files_exist()

    data = pd.concat([_load_one(RAW_DIR / f) for f in EXPECTED_FILES], ignore_index=True)
    print(f"Raw shape: {data.shape}")
    assert LABEL_COL in data.columns, f"'{LABEL_COL}' column not found"
    raw_counts = data[LABEL_COL].value_counts()

    # Drop the duplicated column if (and only if) it really is identical
    if "Fwd Header Length.1" in data.columns:
        if (data["Fwd Header Length"] == data["Fwd Header Length.1"]).all():
            data = data.drop(columns=["Fwd Header Length.1"])
            print("Dropped 'Fwd Header Length.1' (identical copy)")
        else:
            print("WARNING: 'Fwd Header Length.1' differs from 'Fwd Header Length'; kept")

    feature_cols = [c for c in data.columns if c not in (LABEL_COL, HASH_COL)]
    print(f"Feature columns: {len(feature_cols)}")

    # Cleaning steps, with row counts after each
    n0 = len(data)
    data = data.replace([np.inf, -np.inf], np.nan)
    print(f"NaN cells after inf->NaN: {int(data[feature_cols].isna().sum().sum()):,}")
    data = data.dropna()
    n1 = len(data)
    print(f"Removed by dropna:           {n0 - n1:>9,}  (now {n1:,})")

    data = data.drop_duplicates(subset=[HASH_COL, LABEL_COL])
    n2 = len(data)
    print(f"Removed by drop_duplicates:  {n1 - n2:>9,}  (now {n2:,})")

    mask = _report_conflicts(data)
    if drop_conflicts:
        data = data[~mask]
        print(f"Removed conflicting rows:    {int(mask.sum()):>9,}  (now {len(data):,})")

    data = data.drop(columns=[HASH_COL]).reset_index(drop=True)
    print(f"\nFinal shape: {data.shape}")

    # How much did cleaning cost each class? (matters most for the rare ones)
    final_counts = data[LABEL_COL].value_counts()
    summary = pd.DataFrame({"raw": raw_counts, "final": final_counts}).fillna(0).astype(int)
    summary["removed"] = summary["raw"] - summary["final"]
    summary["kept_%"] = (100 * summary["final"] / summary["raw"]).round(1)
    print(f"\nNumber of classes: {data[LABEL_COL].nunique()}")
    print(summary.sort_values("raw", ascending=False).to_string())
    return data


if __name__ == "__main__":
    df = load_raw()
    print(f"\nMemory used: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")