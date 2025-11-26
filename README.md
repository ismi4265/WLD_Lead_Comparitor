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

**Tip about WLD call exports:** the phone column is usually named `Number` (not `Phone Number`). The script now auto-falls back to `Number`, but you can also pass it explicitly with `--wld-calls-phone-col Number`.

---

## Output Files

Inside the output folder you will find:

- `matched_rows.csv` — Final matched Kickstart–WLD pairs
- `unmatched_kickstart.csv`
- `unmatched_wld.csv`
- `near_matches.csv`
- `summary.json`
- `summary.txt`

---

## How Matching Works

1. **Phone number** (most accurate)
2. **Email address**
3. **Normalized name**
4. **Near matches** (same last name + first initial + area code OR same last name + email domain)

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
