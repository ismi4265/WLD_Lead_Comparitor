import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any

import pandas as pd


# ----------------------------
# Helpers: read & normalize
# ----------------------------

def safe_read_csv(path: Path) -> pd.DataFrame:
    """
    Try several encodings and return a clean DataFrame (all columns as str).
    """
    last_err = None
    for enc in (None, "utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc).replace({"": pd.NA})
        except Exception as e:
            last_err = e
    raise last_err


def normalize_phone_series(s: pd.Series) -> pd.Series:
    """
    Normalize phone numbers by stripping all non-digits and keeping last 10 digits.
    Example:
        "(612) 418-3038"  -> "6124183038"
        "+1 952 297 2579" -> "9522972579"
    """
    if s is None:
        return pd.Series(dtype=object)
    return s.map(
        lambda x: (lambda d: d[-10:] if d else None)(
            re.sub(r"\D", "", str(x))
        ) if pd.notna(x) else None
    )


def last_first_to_first_last(value: str) -> str:
    """
    Convert 'Last, First[ Middle...]' -> 'First[ Middle...] Last'.
    Non-matching strings are returned trimmed as-is.
    Used for Kickstart PatNum ('Dexter, Ann' -> 'Ann Dexter').
    """
    if pd.isna(value):
        return None
    s = str(value).strip()
    m = re.match(r"^([^,]+),\s*(.+)$", s)
    if m:
        last = m.group(1).strip()
        first = m.group(2).strip()
        return f"{first} {last}"
    return s


def normalize_name_series(s: pd.Series) -> pd.Series:
    """
    Normalize names:
    - For 'Last, First' -> 'First Last'
    - Collapse extra whitespace
    - Lowercase
    Works for both Kickstart PatNum and WLD Patient.
    """
    if s is None:
        return pd.Series(dtype=object)
    fixed = s.map(lambda x: last_first_to_first_last(x) if pd.notna(x) else None)
    return fixed.map(lambda x: re.sub(r"\s+", " ", x).strip().lower() if isinstance(x, str) else None)


def to_dt_series(s: pd.Series, length: int) -> pd.Series:
    """
    Safe datetime conversion: if s is None or missing, return all NaT.
    """
    if s is None:
        return pd.Series([pd.NaT] * length)
    return pd.to_datetime(s, errors="coerce")


# ----------------------------
# Matching and QA
# ----------------------------

def safe_merge(left: pd.DataFrame, right: pd.DataFrame, key: str, label: str) -> pd.DataFrame:
    """
    Inner join on non-null key; tag with _match_type.
    Keeps ks_id and wld_id to identify unique pairs.
    """
    if key not in left.columns or key not in right.columns:
        return pd.DataFrame()
    l_ok = left[left[key].notna()]
    r_ok = right[right[key].notna()]
    if l_ok.empty or r_ok.empty:
        return pd.DataFrame()
    m = pd.merge(l_ok, r_ok, on=key, how="inner", suffixes=("_ks", "_wld"))
    if not m.empty:
        m["_match_type"] = label
    return m


def dataset_quality(df: pd.DataFrame) -> Dict[str, int]:
    """
    Basic data-quality stats for a dataframe that has _phone_key and _name_key.
    """
    n = len(df)
    if n == 0:
        return {"invalid_phones": 0, "blank_names": 0, "duplicate_phones": 0}

    invalid_phones = int(df.get("_phone_key", pd.Series([None] * n)).isna().sum())
    blank_names = int(df.get("_name_key", pd.Series([None] * n)).isna().sum())

    duplicate_phones = 0
    if "_phone_key" in df.columns and df["_phone_key"].notna().any():
        dup_mask = df["_phone_key"].duplicated(keep=False) & df["_phone_key"].notna()
        duplicate_phones = int(dup_mask.sum())

    return {
        "invalid_phones": invalid_phones,
        "blank_names": blank_names,
        "duplicate_phones": duplicate_phones,
    }


# ----------------------------
# Main CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Match Kickstart leads to WLD leads by phone (primary) and name (fallback), with summary metrics."
    )
    ap.add_argument("--kickstart", required=True, help="Path to Kickstart CSV (expects PatNum, WirelessPhone, FirstVisitPay)")
    ap.add_argument("--wld-calls", required=True, help="Path to WLD Phone Call Leads CSV (e.g., Phone Call Leads_October.csv)")
    ap.add_argument("--wld-forms", required=True, help="Path to WLD Form Submissions CSV (e.g., Form Submissions_October.csv)")

    # Column names (overrideable if exports change)
    ap.add_argument("--kickstart-name-col", default="PatNum", help="Kickstart name column (default: PatNum)")
    ap.add_argument("--kickstart-phone-col", default="WirelessPhone", help="Kickstart phone column (default: WirelessPhone)")

    ap.add_argument("--wld-calls-name-col", default="Patient", help="WLD calls name column (default: Patient)")
    ap.add_argument("--wld-calls-phone-col", default="Phone Number", help="WLD calls phone column (default: 'Phone Number')")

    ap.add_argument("--wld-forms-name-col", default="Patient", help="WLD forms name column (default: Patient)")
    ap.add_argument("--wld-forms-phone-col", default="Phone", help="WLD forms phone column (default: Phone)")

    ap.add_argument("--value-column", default="FirstVisitPay", help="Kickstart value column to sum (default: FirstVisitPay)")
    ap.add_argument("--outdir", default="outputs", help="Output directory (default: outputs)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load data
    # -------------------------
    ks = safe_read_csv(Path(args.kickstart))
    calls = safe_read_csv(Path(args.wld_calls))
    forms = safe_read_csv(Path(args.wld_forms))

    # Sanity checks for required columns
    for col in [args.kickstart_phone_col, args.kickstart_name_col]:
        if col not in ks.columns:
            raise KeyError(f"Kickstart column '{col}' not found. Available: {list(ks.columns)}")

    for col in [args.wld_calls_phone_col, args.wld_calls_name_col]:
        if col not in calls.columns:
            raise KeyError(f"WLD Calls column '{col}' not found. Available: {list(calls.columns)}")

    for col in [args.wld_forms_phone_col, args.wld_forms_name_col]:
        if col not in forms.columns:
            raise KeyError(f"WLD Forms column '{col}' not found. Available: {list(forms.columns)}")

    # -------------------------
    # Build normalized keys
    # -------------------------
    ks["_phone_key"] = normalize_phone_series(ks.get(args.kickstart_phone_col))
    ks["_name_key"] = normalize_name_series(ks.get(args.kickstart_name_col))

    calls["_phone_key"] = normalize_phone_series(calls.get(args.wld_calls_phone_col))
    calls["_name_key"] = normalize_name_series(calls.get(args.wld_calls_name_col))

    forms["_phone_key"] = normalize_phone_series(forms.get(args.wld_forms_phone_col))
    forms["_name_key"] = normalize_name_series(forms.get(args.wld_forms_name_col))

    calls["_source"] = "Phone Call"
    forms["_source"] = "Form"
    wld = pd.concat([forms, calls], ignore_index=True)

    # -------------------------
    # Assign stable IDs for pairs
    # -------------------------
    ks = ks.reset_index().rename(columns={"index": "ks_id"})
    wld = wld.reset_index().rename(columns={"index": "wld_id"})

    # -------------------------
    # Matching (pre-dedup) – phone then name
    # -------------------------
    m_phone = safe_merge(ks, wld, "_phone_key", "phone")
    m_name = safe_merge(ks, wld, "_name_key", "name")

    matches_list = [m for m in (m_phone, m_name) if not m.empty]
    if matches_list:
        all_matches = pd.concat(matches_list, ignore_index=True)
    else:
        all_matches = pd.DataFrame()

    if not all_matches.empty:
        # phone matches are considered stronger than name matches
        prio = {"phone": 0, "name": 1}
        all_matches["_prio"] = all_matches["_match_type"].map(prio)
        # sort so we can drop duplicates per (ks_id, wld_id) and keep phone if both phone+name match
        all_matches.sort_values(by=["ks_id", "wld_id", "_prio"], inplace=True)
        matched = all_matches.drop_duplicates(subset=["ks_id", "wld_id"], keep="first").copy()
    else:
        matched = all_matches

    # -------------------------
    # Unmatched sets based on IDs
    # -------------------------
    if matched.empty:
        unmatched_ks = ks.copy()
        unmatched_wld = wld.copy()
    else:
        matched_ks_ids = set(matched["ks_id"].tolist())
        matched_wld_ids = set(matched["wld_id"].tolist())
        unmatched_ks = ks[~ks["ks_id"].isin(matched_ks_ids)].copy()
        unmatched_wld = wld[~wld["wld_id"].isin(matched_wld_ids)].copy()

    unmatched_ks.to_csv(outdir / "unmatched_kickstart.csv", index=False)
    unmatched_wld.to_csv(outdir / "unmatched_wld.csv", index=False)

    # -------------------------
    # QA flags + timing (non-fatal if dates missing)
    # -------------------------
    if not matched.empty:
        matched = matched.copy()
        matched["invalid_phone_flag"] = matched["_phone_key"].isna()
        matched["dup_phone_flag"] = matched["_phone_key"].notna() & (
            matched["_phone_key"].map(matched["_phone_key"].value_counts()) > 1
        )
        matched["weak_match_flag"] = (matched["_match_type"] == "name") & matched["_phone_key"].isna()

        # WLD lead date (first column on WLD side containing 'date')
        wld_date_cols = [c for c in matched.columns if c.endswith("_wld") and "date" in c.lower()]
        if wld_date_cols:
            lead_dt = to_dt_series(matched[wld_date_cols[0]], len(matched))
        else:
            lead_dt = pd.Series([pd.NaT] * len(matched))

        first_appt_dt = to_dt_series(matched.get("FirstApptCreated_ks"), len(matched))
        first_visit_dt = to_dt_series(matched.get("FirstVisit_ks"), len(matched))

        matched["lead_date"] = lead_dt
        matched["days_to_first_appt"] = (first_appt_dt - lead_dt).dt.days
        matched["days_to_first_visit"] = (first_visit_dt - lead_dt).dt.days

    # Save matched rows (with ks_id, wld_id)
    matched.to_csv(outdir / "matched_rows.csv", index=False)

    # -------------------------
    # Patient value sum from Kickstart
    # -------------------------
    patient_value_sum = 0.0
    if not matched.empty:
        vcol_ks = f"{args.value_column}_ks"
        if vcol_ks in matched.columns:
            vuse = vcol_ks
        elif args.value_column in matched.columns:
            vuse = args.value_column
        else:
            vuse = None

        if vuse is not None:
            patient_value_sum = pd.to_numeric(matched.get(vuse), errors="coerce").fillna(0).sum()

    # -------------------------
    # Summary metrics
    # -------------------------
    pairs_count = int(len(matched))
    unique_kick = int(matched["ks_id"].nunique()) if pairs_count else 0
    unique_wld = int(matched["wld_id"].nunique()) if pairs_count else 0

    total_kick = int(len(ks))
    total_wld = int(len(wld))

    conversion_from_kick = round((unique_kick / total_kick * 100.0), 2) if total_kick else 0.0
    conversion_from_wld = round((unique_wld / total_wld * 100.0), 2) if total_wld else 0.0

    # Simple source breakdown on WLD side (if Source exists)
    source_breakdown: Dict[str, Dict[str, Any]] = {}
    if not matched.empty:
        wld_source_cols = [c for c in matched.columns if c.endswith("_wld") and "source" in c.lower()]
        if wld_source_cols:
            src_col = wld_source_cols[0]
            # totals for all WLD
            wld_src_cols_all = [c for c in wld.columns if "source" in c.lower()]
            if wld_src_cols_all:
                wld_src = wld[wld_src_cols_all[0]]
                total_by_src = wld_src.value_counts(dropna=False).to_dict()
            else:
                total_by_src = {}
            matched_by_src = matched[src_col].value_counts(dropna=False).to_dict()
            for k, v in matched_by_src.items():
                label = "" if (k is None or (isinstance(k, float) and pd.isna(k))) else str(k)
                total = total_by_src.get(k, 0)
                rate = (v / total * 100.0) if total else None
                source_breakdown[label] = {
                    "matches": int(v),
                    "total": int(total),
                    "rate_percent": round(rate, 2) if rate is not None else None,
                }

    # Time-to-conversion stats (optional, may be all NaN)
    avg_days_to_first_appt = None
    median_days_to_first_appt = None
    avg_days_to_first_visit = None
    if not matched.empty:
        if "days_to_first_appt" in matched.columns:
            s = matched["days_to_first_appt"].dropna()
            if not s.empty:
                avg_days_to_first_appt = float(round(s.mean(), 2))
                median_days_to_first_appt = float(s.median())
        if "days_to_first_visit" in matched.columns:
            s2 = matched["days_to_first_visit"].dropna()
            if not s2.empty:
                avg_days_to_first_visit = float(round(s2.mean(), 2))

    ks_quality = dataset_quality(ks)
    wld_quality = dataset_quality(wld)

    summary = {
        # main numbers you care about:
        "number_of_matches": unique_kick,          # unique Kickstart leads matched
        "number_of_match_pairs": pairs_count,      # Kickstart–WLD pairs
        "number_of_kick_leads_matched": unique_kick,
        "number_of_wld_leads_matched": unique_wld,
        "total_kick_leads": total_kick,
        "total_wld_leads": total_wld,
        "conversion_rate_from_kick": conversion_from_kick,
        "conversion_rate_from_wld": conversion_from_wld,
        "patient_value_sum": float(round(patient_value_sum, 2)),

        # extra detail:
        "match_rate_by_key": matched["_match_type"].value_counts().to_dict() if not matched.empty else {},
        "source_breakdown": source_breakdown,
        "avg_days_to_first_appt": avg_days_to_first_appt,
        "median_days_to_first_appt": median_days_to_first_appt,
        "avg_days_to_first_visit": avg_days_to_first_visit,
        "data_quality": {
            "kickstart": ks_quality,
            "wld_combined": wld_quality,
        },
        "value_column_used_from_kickstart": args.value_column,
        "primary_key_order": ["phone", "name"],
    }

    # Write summary
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    (outdir / "summary.txt").write_text(
        "number_of_matches (Kick leads): {nm}\n"
        "number_of_match_pairs (Kick-WLD): {pairs}\n"
        "total_kick_leads: {tk}\n"
        "total_wld_leads: {tw}\n"
        "conversion_rate_from_kick: {crk:.2f}%\n"
        "conversion_rate_from_wld: {crw:.2f}%\n"
        "patient_value_sum: {pvs:.2f}\n".format(
            nm=summary["number_of_matches"],
            pairs=summary["number_of_match_pairs"],
            tk=summary["total_kick_leads"],
            tw=summary["total_wld_leads"],
            crk=summary["conversion_rate_from_kick"],
            crw=summary["conversion_rate_from_wld"],
            pvs=summary["patient_value_sum"],
        )
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
