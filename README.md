# WLD Lead Matcher

This application compares **Kickstart leads** with **West Lakes Dentistry (WLD)** leads and identifies where the same person appears in both systems. It produces clean reports showing:

- Which Kickstart leads converted into WLD leads
- Total number of matches
- Total patient value from matched leads
- Unmatched leads on both sides
- “Near matches” (possible matches that need manual review)

You **do not** need coding experience to use this tool.

## What You Need

- **Python 3.10 or newer**

Check with:
```
python --version
```

## Setup Instructions

### 1. Download the project

Download this folder to your computer and place it somewhere easy, such as:
- Windows: `C:\Users\YOURNAME\Documents\WLD_Lead_Matcher`
- Mac: `/Users/YOURNAME/Documents/WLD_Lead_Matcher`

### 2. Place your CSV files inside the project folder

Put these three files in the same folder as `match.py`:
- Kickstart CSV
- WLD Phone Call Leads CSV
- WLD Form Submissions CSV

### 3. Open a terminal in the project folder

#### Windows:
```
cd "C:\Users\YOURNAME\Documents\WLD_Lead_Matcher"
```

#### Mac:
```
cd "/Users/YOURNAME/Documents/WLD_Lead_Matcher"
```

### 4. Create a Python environment (only once per computer)
```
python -m venv .venv
```

Activate it:

#### Windows:
```
.venv\Scripts\activate
```

#### Mac:
```
source .venv/bin/activate
```

### 5. Install required packages
```
pip install -r requirements.txt
```

---

## Running the Matcher

### Windows (one line):
```
python match.py --kickstart "KICKSTART LEADS.csv" --wld-calls "WLD_Phone_Calls.csv" --wld-forms "WLD_Form_Submissions.csv" --value-column FirstVisitPay --outdir output_results
```

### Mac (multi-line allowed):
```
python match.py   --kickstart "KICKSTART LEADS.csv"   --wld-calls "WLD_Phone_Calls.csv"   --wld-forms "WLD_Form_Submissions.csv"   --value-column FirstVisitPay   --outdir output_results
```

**Tips / common flags**
- WLD call exports usually name the phone column `Number` (not `Phone Number`). The script auto-uses `Number`, or pass `--wld-calls-phone-col Number`.
- Emails are matched automatically if present; override column names with `--kickstart-email-col`, `--wld-calls-email-col`, `--wld-forms-email-col`.
- Filter Kickstart by clinic(s): `--kickstart-clinic-filter Chaska "WLD Implants"`.
- Enable/loosen fuzzy names: `--fuzzy-name-threshold 0.85` (0 disables).

---

## Output Files

Inside the output folder you will find:

- `matched_rows.csv` — Final matched Kickstart–WLD pairs
- `unmatched_kickstart.csv`
- `unmatched_wld.csv`
- `near_matches.csv` — Weaker matches (nickname/initial/phonetic/phone-fragment/fuzzy)
- `summary.json`
- `summary.txt`

---

## How Matching Works

Order (strongest → weakest):
1. Phone (last 10 digits)
2. Email
3. Exact normalized name (handles “Last, First” and strips punctuation)
4. Nickname-expanded name (e.g., Drew → Andrew)
5. First initial + last name
6. Phonetic last name (Soundex) + first initial
7. Last name + last 4–5 digits of phone
8. Fuzzy name (optional, via `--fuzzy-name-threshold`)
Date proximity is used only to break ties between weak matches.

---

## Patient Data Safety

The `.gitignore` ensures:
- CSV files **are NOT uploaded** to GitHub
- Output files **are NOT uploaded** to GitHub

---

## Troubleshooting

- “python not found” → Install Python
- “No module named pandas” → run `pip install -r requirements.txt`
- File not found → ensure filenames match exactly

---

You're ready to go!
