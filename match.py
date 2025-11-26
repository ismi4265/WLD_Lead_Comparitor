import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Any
from math import inf

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


def coalesce_columns(df: pd.DataFrame, candidates) -> (pd.Series, list):
    """
    Return a single Series made by combining the first available non-null values
    across the candidate columns, in order. Also returns the list of columns used.
    """
    seen = set()
    cols = []
    for c in candidates:
        if c and c not in seen and c in df.columns:
            cols.append(c)
            seen.add(c)

    if not cols:
        return None, []

    series = df[cols[0]].copy()
    for c in cols[1:]:
        series = series.combine_first(df[c])
    return series, cols


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


NICKNAME_CANON = {
    # common short → canonical
    "andy": "andrew",
    "drew": "andrew",
    "tony": "anthony",
    "beth": "elizabeth",
    "betsy": "elizabeth",
    "liz": "elizabeth",
    "lizzy": "elizabeth",
    "lisa": "elizabeth",
    "kate": "katherine",
    "katie": "katherine",
    "katy": "katherine",
    "cathy": "catherine",
    "cath": "catherine",
    "chris": "christopher",
    "dan": "daniel",
    "danny": "daniel",
    "dave": "david",
    "jim": "james",
    "jimmy": "james",
    "joe": "joseph",
    "joey": "joseph",
    "tom": "thomas",
    "tommy": "thomas",
    "bill": "william",
    "billy": "william",
    "will": "william",
    "bobby": "robert",
    "bob": "robert",
    "rob": "robert",
    "robbie": "robert",
    "mike": "michael",
    "mikey": "michael",
    "steve": "steven",
    "steven": "steven",
    "stephen": "steven",
    "jen": "jennifer",
    "jenny": "jennifer",
    "ben": "benjamin",
    "benji": "benjamin",
    "ali": "allison",
    "ally": "allison",
    "allie": "allison",
}


def normalize_name_series(s: pd.Series) -> pd.Series:
    """
    Normalize names:
    - For 'Last, First' -> 'First Last'
    - Strip punctuation (quotes, commas, periods, hyphens, etc.)
    - Collapse extra whitespace and lowercase
    Works for both Kickstart PatNum and WLD Patient.
    """
    if s is None:
        return pd.Series(dtype=object)
    fixed = s.map(lambda x: last_first_to_first_last(x) if pd.notna(x) else None)
    return fixed.map(
        lambda x: (
            lambda cleaned: cleaned if cleaned else None
        )(
            re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9\s]", " ", str(x)))
            .strip()
            .lower()
        )
        if isinstance(x, str)
        else None
    )


def normalize_name_variants(s: pd.Series) -> (pd.Series, pd.Series):
    """
    Returns two aligned Series:
    - strict normalized name
    - normalized name with common nicknames expanded to canonical forms
    """
    base = normalize_name_series(s)

    def apply_nickname(name: str):
        if not isinstance(name, str):
            return None
        parts = name.split()
        if not parts:
            return None
        first = parts[0]
        canon = NICKNAME_CANON.get(first, first)
        parts[0] = canon
        renamed = " ".join(parts).strip()
        return renamed if renamed else None

    nick = base.map(apply_nickname)
    return base, nick


def to_dt_series(s: pd.Series, length: int) -> pd.Series:
    """
    Safe datetime conversion: if s is None or missing, return all NaT.
    """
    if s is None:
        return pd.Series([pd.NaT] * length)
    return pd.to_datetime(s, errors="coerce")

def _name_similarity(a: str, b: str) -> float:
    """
    Aggressive similarity:
    - plain ratio
    - token-sorted ratio (handles swapped order)
    - boost if last names match and first initials/prefixes align
    """
    if not a or not b:
        return 0.0
    a = a.strip()
    b = b.strip()
    ratio_full = SequenceMatcher(None, a, b).ratio()

    def token_sort(s: str) -> str:
        return " ".join(sorted(s.split()))

    ratio_token = SequenceMatcher(None, token_sort(a), token_sort(b)).ratio()
    score = max(ratio_full, ratio_token)

    a_parts = a.split()
    b_parts = b.split()
    if a_parts and b_parts:
        a_last = a_parts[-1]
        b_last = b_parts[-1]
        if a_last == b_last:
            a_first = a_parts[0]
            b_first = b_parts[0]
            if a_first and b_first and a_first[0] == b_first[0]:
                score = max(score, 0.94)
            if a_first and b_first and (a_first.startswith(b_first) or b_first.startswith(a_first)):
                score = max(score, 0.92)
    return score


def name_parts_from_normalized(name: str):
    """
    Given a normalized name ("first middle last"), return (first, last).
    """
    if not isinstance(name, str):
        return None, None
    parts = name.strip().split()
    if not parts:
        return None, None
    first = parts[0]
    last = parts[-1] if len(parts) > 1 else None
    return first or None, last or None


def soundex_code(s: str) -> str:
    """
    Basic Soundex implementation for last-name similarity.
    """
    if not s:
        return ""
    s = re.sub(r"[^A-Za-z]", "", s).upper()
    if not s:
        return ""
    first = s[0]
    mapping = {
        **dict.fromkeys(list("BFPV"), "1"),
        **dict.fromkeys(list("CGJKQSXZ"), "2"),
        **dict.fromkeys(list("DT"), "3"),
        "L": "4",
        **dict.fromkeys(list("MN"), "5"),
        "R": "6",
    }
    digits = [mapping.get(ch, "") for ch in s[1:]]
    # remove consecutive duplicates and zeros/blanks
    cleaned = []
    prev = ""
    for d in digits:
        if d and d != prev:
            cleaned.append(d)
        prev = d
    code = (first + "".join(cleaned) + "000")[:4]
    return code


def build_initial_last_key(series: pd.Series) -> pd.Series:
    """
    First initial + last name (from normalized name).
    """
    if series is None:
        return pd.Series(dtype=object)
    def mk(name):
        first, last = name_parts_from_normalized(name)
        if not first or not last:
            return None
        return f"{first[0]} {last}"
    return series.map(lambda x: mk(x) if pd.notna(x) else None)


def build_soundex_key(series: pd.Series) -> pd.Series:
    """
    Soundex(last) + first initial for loose last-name phonetic matches.
    """
    if series is None:
        return pd.Series(dtype=object)
    def mk(name):
        first, last = name_parts_from_normalized(name)
        if not last or not first:
            return None
        return f"{soundex_code(last)}|{first[0]}"
    return series.map(lambda x: mk(x) if pd.notna(x) else None)


def build_last_phonefrag_key(name_series: pd.Series, phone_series: pd.Series, frag_len: int = 5) -> pd.Series:
    """
    Combine last name with last N digits of phone (defaults to 5, falls back to 4 if shorter).
    """
    if name_series is None or phone_series is None:
        return pd.Series(dtype=object)
    def mk(name, phone):
        if pd.isna(name) or pd.isna(phone):
            return None
        _, last = name_parts_from_normalized(name)
        if not last:
            return None
        frag = str(phone)[-frag_len:] if len(str(phone)) >= frag_len else str(phone)[-4:]
        if not frag:
            return None
        return f"{last}|{frag}"
    return pd.Series([mk(n, p) for n, p in zip(name_series, phone_series)])

# needs to be more vague, maybe 50% match between names?
def fuzzy_name_match(unmatched_ks: pd.DataFrame, unmatched_wld: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Fuzzy match names between remaining unmatched KS and WLD rows using SequenceMatcher.
    Returns a dataframe shaped like safe_merge output with _match_type 'name_fuzzy' and _match_ratio.
    """
    if threshold <= 0 or unmatched_ks.empty or unmatched_wld.empty:
        return pd.DataFrame()

    def pick_key(df: pd.DataFrame):
        if "_name_nick_key" in df.columns:
            return df["_name_nick_key"]
        if "_name_key" in df.columns:
            return df["_name_key"]
        return None

    ks_key = pick_key(unmatched_ks)
    wld_key = pick_key(unmatched_wld)
    if ks_key is None or wld_key is None:
        return pd.DataFrame()

    ks_pool = unmatched_ks[ks_key.notna()]
    wld_pool = unmatched_wld[wld_key.notna()]
    if ks_pool.empty or wld_pool.empty:
        return pd.DataFrame()

    wld_used = set()
    records = []

    for _, ks_row in ks_pool.iterrows():
        ks_name = ks_row["_name_nick_key"] if "_name_nick_key" in ks_row else ks_row.get("_name_key")
        best_ratio = 0.0
        best_row = None
        for _, wld_row in wld_pool.iterrows():
            if wld_row["wld_id"] in wld_used:
                continue
            wld_name = wld_row["_name_nick_key"] if "_name_nick_key" in wld_row else wld_row.get("_name_key")
            ratio = _name_similarity(ks_name, wld_name)
            if ratio > best_ratio:
                best_ratio = ratio
                best_row = wld_row
        if best_row is not None and best_ratio >= threshold:
            wld_used.add(best_row["wld_id"])
            rec = {}
            for c in ks_row.index:
                rec[f"{c}_ks"] = ks_row[c]
            for c in best_row.index:
                rec[f"{c}_wld"] = best_row[c]
            rec["_match_type"] = "name_fuzzy"
            rec["_match_ratio"] = best_ratio
            records.append(rec)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


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

    ap.add_argument(
        "--kickstart-clinic-filter",
        nargs="+",
        help="Filter Kickstart rows to these Clinic values (case-insensitive). Provide one or more values; otherwise all clinics are used.",
    )
    ap.add_argument(
        "--fuzzy-name-threshold",
        type=float,
        default=0.0,
        help="Optional fuzzy name matching threshold (0-1). >0 enables SequenceMatcher fallback on remaining unmatched by name/phone. Example: 0.88. Use 0 to disable.",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load data
    # -------------------------
    ks = safe_read_csv(Path(args.kickstart))
    calls = safe_read_csv(Path(args.wld_calls))
    forms = safe_read_csv(Path(args.wld_forms))

    # Optional clinic filter on Kickstart
    if args.kickstart_clinic_filter:
        if "Clinic" not in ks.columns:
            raise KeyError("Kickstart clinic filter requested but 'Clinic' column not found.")
        clinics = {c.lower() for c in args.kickstart_clinic_filter}
        ks = ks[ks["Clinic"].str.lower().isin(clinics)]
        print(f"Info: filtered Kickstart to clinics {sorted(clinics)}; remaining rows: {len(ks)}")

    # Sanity checks for required columns
    for col in [args.kickstart_phone_col, args.kickstart_name_col]:
        if col not in ks.columns:
            raise KeyError(f"Kickstart column '{col}' not found. Available: {list(ks.columns)}")

    if args.wld_calls_name_col not in calls.columns:
        raise KeyError(f"WLD Calls column '{args.wld_calls_name_col}' not found. Available: {list(calls.columns)}")

    if args.wld_forms_name_col not in forms.columns:
        raise KeyError(f"WLD Forms column '{args.wld_forms_name_col}' not found. Available: {list(forms.columns)}")

    # -------------------------
    # Build normalized keys
    # -------------------------
    ks["_phone_key"] = normalize_phone_series(ks.get(args.kickstart_phone_col))
    ks["_name_key"], ks["_name_nick_key"] = normalize_name_variants(ks.get(args.kickstart_name_col))
    ks["_initial_last_key"] = build_initial_last_key(ks["_name_nick_key"])
    ks["_soundex_key"] = build_soundex_key(ks["_name_nick_key"])
    ks["_last_phonefrag_key"] = build_last_phonefrag_key(ks["_name_nick_key"], ks["_phone_key"])

    calls_phone_raw, calls_cols_used = coalesce_columns(
        calls,
        [args.wld_calls_phone_col, "Number", "Phone Number"]
    )
    if not calls_cols_used:
        raise KeyError(
            f"No phone column found for WLD Calls. Tried: {[args.wld_calls_phone_col, 'Number', 'Phone Number']}. "
            f"Available: {list(calls.columns)}"
        )
    if calls_cols_used[0] != args.wld_calls_phone_col:
        print(f"Info: using WLD Calls phone column '{calls_cols_used[0]}' (fallback from '{args.wld_calls_phone_col}')")
    elif len(calls_cols_used) > 1:
        print(f"Info: combining WLD Calls phone columns {calls_cols_used}")

    calls["_phone_key"] = normalize_phone_series(calls_phone_raw)
    calls["_name_key"], calls["_name_nick_key"] = normalize_name_variants(calls.get(args.wld_calls_name_col))
    calls["_initial_last_key"] = build_initial_last_key(calls["_name_nick_key"])
    calls["_soundex_key"] = build_soundex_key(calls["_name_nick_key"])
    calls["_last_phonefrag_key"] = build_last_phonefrag_key(calls["_name_nick_key"], calls["_phone_key"])

    forms_phone_raw, forms_cols_used = coalesce_columns(
        forms,
        [args.wld_forms_phone_col, "Phone", "Phone Number"]
    )
    if not forms_cols_used:
        raise KeyError(
            f"No phone column found for WLD Forms. Tried: {[args.wld_forms_phone_col, 'Phone', 'Phone Number']}. "
            f"Available: {list(forms.columns)}"
        )
    if forms_cols_used[0] != args.wld_forms_phone_col:
        print(f"Info: using WLD Forms phone column '{forms_cols_used[0]}' (fallback from '{args.wld_forms_phone_col}')")
    elif len(forms_cols_used) > 1:
        print(f"Info: combining WLD Forms phone columns {forms_cols_used}")

    forms["_phone_key"] = normalize_phone_series(forms_phone_raw)
    forms["_name_key"], forms["_name_nick_key"] = normalize_name_variants(forms.get(args.wld_forms_name_col))
    forms["_initial_last_key"] = build_initial_last_key(forms["_name_nick_key"])
    forms["_soundex_key"] = build_soundex_key(forms["_name_nick_key"])
    forms["_last_phonefrag_key"] = build_last_phonefrag_key(forms["_name_nick_key"], forms["_phone_key"])

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
    m_name_nick = safe_merge(ks, wld, "_name_nick_key", "name_nick")
    m_initial_last = safe_merge(ks, wld, "_initial_last_key", "initial_last")
    m_soundex = safe_merge(ks, wld, "_soundex_key", "soundex_initial_last")
    m_phonefrag = safe_merge(ks, wld, "_last_phonefrag_key", "last_phonefrag")

    matches_list = [m for m in (m_phone, m_name, m_name_nick, m_initial_last, m_soundex, m_phonefrag) if not m.empty]
    if matches_list:
        all_matches = pd.concat(matches_list, ignore_index=True)
    else:
        all_matches = pd.DataFrame()

    if not all_matches.empty:
        # priority and score (for date-aware tie-breaks)
        prio = {
            "phone": 0,
            "name": 1,
            "name_nick": 2,
            "initial_last": 3,
            "soundex_initial_last": 4,
            "last_phonefrag": 5,
            "name_fuzzy": 6,
        }
        base_score = {
            "phone": 1.0,
            "name": 0.95,
            "name_nick": 0.92,
            "initial_last": 0.9,
            "soundex_initial_last": 0.85,
            "last_phonefrag": 0.83,
            "name_fuzzy": 0.75,
        }
        all_matches["_prio"] = all_matches["_match_type"].map(prio).fillna(99)
        all_matches["_score"] = all_matches["_match_type"].map(base_score).fillna(0.7)

        # Date proximity bonus for weak/fuzzy ties
        wld_date_cols_all = [c for c in all_matches.columns if c.endswith("_wld") and "date" in c.lower()]
        lead_dt_all = to_dt_series(all_matches[wld_date_cols_all[0]], len(all_matches)) if wld_date_cols_all else pd.Series([pd.NaT] * len(all_matches))
        ks_dt = pd.Series([pd.NaT] * len(all_matches))
        for c in ("FirstApptCreated_ks", "FirstVisit_ks"):
            if c in all_matches.columns:
                ks_dt = ks_dt.combine_first(to_dt_series(all_matches[c], len(all_matches)))
        delta = (ks_dt - lead_dt_all).dt.days
        delta_abs = delta.abs()
        bonus = 0.2 * (1 / (1 + delta_abs))
        bonus = bonus.mask(delta_abs.isna(), 0)
        all_matches["_score"] = all_matches["_score"] + bonus.fillna(0)

        # sort so we can drop duplicates per (ks_id, wld_id) with prio then score
        all_matches.sort_values(by=["ks_id", "wld_id", "_prio", "_score"], ascending=[True, True, True, False], inplace=True)
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

    # -------------------------
    # Fuzzy name fallback on remaining unmatched
    # -------------------------
    fuzzy_df = fuzzy_name_match(unmatched_ks, unmatched_wld, args.fuzzy_name_threshold)
    if not fuzzy_df.empty:
        matched = pd.concat([matched, fuzzy_df], ignore_index=True) if not matched.empty else fuzzy_df
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
