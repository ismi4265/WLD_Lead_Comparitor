
import argparse, json, re
from pathlib import Path
import pandas as pd

def normalize_phone_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype=object)
    return s.map(lambda x: (lambda d: d[-10:] if d else None)(re.sub(r"\D","",str(x))) if pd.notna(x) else None)

def last_first_to_first_last(value: str) -> str:
    if pd.isna(value):
        return None
    s = str(value).strip()
    m = re.match(r'^([^,]+),\s*(.+)$', s)
    if m:
        last = m.group(1).strip()
        first = m.group(2).strip()
        return f"{first} {last}"
    return s

def normalize_name_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype=object)
    fixed = s.map(lambda x: last_first_to_first_last(x) if pd.notna(x) else None)
    return fixed.map(lambda x: re.sub(r"\s+"," ", x).strip().lower() if isinstance(x,str) else None)

def safe_read_csv(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in (None, "utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc).replace({"": pd.NA})
        except Exception as e:
            last_err = e
    raise last_err

def main():
    ap = argparse.ArgumentParser(description="Match Kickstart leads to WLD leads by phone then name.")
    ap.add_argument("--kickstart", required=True)
    ap.add_argument("--wld-calls", required=True)
    ap.add_argument("--wld-forms", required=True)
    ap.add_argument("--kickstart-name-col", default="PatNum")
    ap.add_argument("--kickstart-phone-col", default="WirelessPhone")
    ap.add_argument("--wld-calls-name-col", default="Patient")
    ap.add_argument("--wld-calls-phone-col", default="Phone Number")
    ap.add_argument("--wld-forms-first-col", default="First Name")
    ap.add_argument("--wld-forms-last-col", default="Last Name")
    ap.add_argument("--wld-forms-phone-col", default="Phone Number")
    ap.add_argument("--value-column", default="FirstVisitPay")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    ks = safe_read_csv(Path(args.kickstart))
    calls = safe_read_csv(Path(args.wld_calls))
    forms = safe_read_csv(Path(args.wld_forms))

    ks["_phone_key"] = normalize_phone_series(ks.get(args.kickstart_phone_col))
    ks["_name_key"] = normalize_name_series(ks.get(args.kickstart_name_col))

    calls["_phone_key"] = normalize_phone_series(calls.get(args.wld_calls_phone_col))
    calls["_name_key"] = normalize_name_series(calls.get(args.wld_calls_name_col))

    first = forms.get(args.wld_forms_first_col)
    last = forms.get(args.wld_forms_last_col)
    if first is not None and last is not None:
        forms["_name_raw"] = (first.fillna("").astype(str).str.strip() + " " + last.fillna("").astype(str).str.strip()).str.strip()
    else:
        forms["_name_raw"] = forms.get("Name", pd.Series([None]*len(forms)))
    forms["_name_key"] = normalize_name_series(forms["_name_raw"])
    forms["_phone_key"] = normalize_phone_series(forms.get(args.wld_forms_phone_col))

    calls["_source"] = "Phone Call"
    forms["_source"] = "Form"
    wld = pd.concat([calls, forms], ignore_index=True)

    def safe_merge(left, right, key, label):
        if key not in left.columns or key not in right.columns:
            return pd.DataFrame()
        left_ok = left[left[key].notna()]
        right_ok = right[right[key].notna()]
        if left_ok.empty or right_ok.empty:
            return pd.DataFrame()
        m = pd.merge(left_ok, right_ok, on=key, how="inner", suffixes=("_ks", "_wld"))
        if not m.empty:
            m["_match_type"] = label
        return m

    m_phone = safe_merge(ks, wld, "_phone_key", "phone")
    m_name = safe_merge(ks, wld, "_name_key", "name")

    matches = [m for m in (m_phone, m_name) if not m.empty]
    matched = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame()

    if not matched.empty:
        prio = {"phone":0, "name":1}
        matched["_prio"] = matched["_match_type"].map(prio)
        matched.sort_values(by=["_prio"], inplace=True)
        subset = [c for c in ["_phone_key","_name_key"] if c in matched.columns]
        matched = matched.drop_duplicates(subset=subset, keep="first")

    matched.to_csv(outdir/"matched_rows.csv", index=False)

    matches_count = len(matched)
    rows_in_wld = len(wld)
    percent_converted = (matches_count / rows_in_wld * 100.0) if rows_in_wld else 0.0

    value_sum = 0.0
    vcol = f"{args.value_column}_ks"
    if matches_count and (vcol in matched.columns or args.value_column in matched.columns):
        vuse = vcol if vcol in matched.columns else args.value_column
        value_sum = pd.to_numeric(matched.get(vuse), errors="coerce").fillna(0).sum()

    summary = {
        "number_of_matches": int(matches_count),
        "rows_in_wld_combined": int(rows_in_wld),
        "percent_converted": round(percent_converted, 2),
        "patient_value_sum": round(float(value_sum), 2),
        "value_column_used_from_kickstart": args.value_column,
        "primary_key_order": ["phone","name"]
    }
    (outdir/"summary.json").write_text(json.dumps(summary, indent=2))
    (outdir/"summary.txt").write_text(
        f"number_of_matches: {summary['number_of_matches']}\n"
        f"rows_in_wld_combined: {summary['rows_in_wld_combined']}\n"
        f"percent_converted: {summary['percent_converted']:.2f}%\n"
        f"patient_value_sum: {summary['patient_value_sum']:.2f}\n"
        f"value_column_used_from_kickstart: {summary['value_column_used_from_kickstart']}\n"
    )

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
