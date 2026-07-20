# k-AnonymityDataExperiments

A working sample using **[ANJANA](https://github.com/IFCA-Advanced-Computing/anjana)**
(anonymization) + **[pyCANON](https://github.com/IFCA-Advanced-Computing/pycanon)**
(independent verification) — the Python alternative to ARX for structured/tabular
data. Trying another approach and Python based ibraries compared to [ARX](https://arx.deidentifier.org/)

## Getting started in Codespaces
In your Codespace terminal, run:

```bash
pip install uv
uv add anjana pycanon pandas
```
Then run anonymize demo file:

   ```bash
   uv run python anonymize_demo.py 
   ```
Output

```

   Original dataset:
 age gender  zipcode disease
  34   male    81667     flu
  45 female    81675     flu
  66   male    81925  cancer
  70 female    81931     flu
  34 female    81931  cancer
  70   male    81931  cancer
  45   male    81928     flu
  34 female    81931  cancer

--- k-anonymity only (k=2) ---
 age gender zipcode disease
 <50      *   816**     flu
 <50      *   816**     flu
>=50      *   819**  cancer
>=50      *   819**     flu
 <50      *   819**  cancer
>=50      *   819**  cancer
 <50      *   819**     flu
 <50      *   819**  cancer
Verified k: 2
Verified l: 1   <- notice this can be 1, i.e. no real protection on 'disease'

--- k-anonymity + l-diversity (k=2, l=2) ---
 age gender zipcode disease
 <50      *   81***     flu
 <50      *   81***     flu
>=50      *   81***  cancer
>=50      *   81***     flu
 <50      *   81***  cancer
>=50      *   81***  cancer
 <50      *   81***     flu
 <50      *   81***  cancer
Verified k: 3
Verified l: 2
The data verifies t-closeness with t=0.5

--- k-anonymity + t-closeness (k=2, t=0.5) ---
 age gender zipcode disease
 <50      *   816**     flu
 <50      *   816**     flu
>=50      *   819**  cancer
>=50      *   819**     flu
 <50      *   819**  cancer
>=50      *   819**  cancer
 <50      *   819**     flu
 <50      *   819**  cancer
Verified k: 2
Verified t: 0.500

Full privacy report (k-anonymity + l-diversity result):
The dataset verifies:
                 - k-anonymity with k = 3
                 - (alpha,k)-anonymity with alpha = 0.6666666666666666 and k = 3
                 - l-diversity with l = 2
                 - entropy l-diversity with l = 1
                 - (c,l)-diversity with c = 1 and l = 2
                 - basic beta-likeness with beta = 0.33333333333333326
                 - enhanced beta-likeness with beta = 0.33333333333333326
                 - t-closeness with t = 0.16666666666666666
                 - delta-disclosure privacy with delta = 0.40546510810816444
```

## What it does

- Anonymizes a tiny dataset (`age`, `gender`, `zipcode` as quasi-identifiers,
  `disease` as the sensitive attribute) to satisfy 2-anonymity, using
  generalization hierarchies defined in `hierarchies/*.csv`.
- Then **independently re-checks** the result with `pycanon` — a separate
  library computing k-anonymity, l-diversity, t-closeness, and several other
  metrics from scratch — instead of just trusting anjana's own claim.

## Adapting to your own data

1. Swap the `data = pd.DataFrame(...)` block for `pd.read_csv("your_file.csv")`.
2. Write one hierarchy CSV per quasi-identifier: column 0 = the exact original
   values appearing in your data, each following column a more generalized
   version. (Same format ARX uses, so if you ever migrated from ARX hierarchy
   files, they carry over almost as-is.)
3. Adjust `K` and `SUPPRESSION_LIMIT` to your risk tolerance, then run
   `report.print_report(...)` to see the full picture — k-anonymity alone is
   rarely sufficient for a real release; check l-diversity/t-closeness on your
   sensitive columns too.

## Alternative Approach # 1 using ```diffprivlib```
Please note : This library is primarily intended for research.
```bash
uv add diffprivlib "scikit-learn<1.9"
```
Then run anonymize demo file:

   ```bash
 uv run python diffprivlib_demo.py
 ```
 Output

```
Original dataset:
 age gender  zipcode disease
  34   male    81667     flu
  45 female    81675     flu
  66   male    81925  cancer
  70 female    81931     flu
  34 female    81931  cancer
  70   male    81931  cancer
  45   male    81928     flu
  34 female    81931  cancer

--- Differentially private aggregate statistics (total epsilon=1.0) ---
Cancer count   -> true:   4   DP:  11.81
Flu count      -> true:   4   DP:   2.60
Mean age       -> true:  49.75   DP:  27.48

Epsilon spent: 1.000 (budget was 1.0)

Note: no row-level data was ever generalized, suppressed, or published here - only noisy aggregate answers. That's the core difference from the anjana/k-anonymity approach: DP protects *queries*, not a released dataset. If you need to hand someone a CSV they can explore freely, DP alone doesn't give you that - you'd pair it with synthetic data generation, or fall back to k-anonymity/l-diversity.
```
## Alternative Approach # 2 using ```opendp```
```bash
uv add opendp "polars==1.36.1" pyarrow pandas
```
Then run anonymize demo file:

   ```bash
 uv run python opendp_demo.py
 ```
 Output

```
Original dataset:
 age gender  zipcode disease
  34   male    81667     flu
  45 female    81675     flu
  66   male    81925  cancer
  70 female    81931     flu
  34 female    81931  cancer
  70   male    81931  cancer
  45   male    81928     flu
  34 female    81931  cancer

--- Query 1: private counts by disease ---
shape: (1, 4)
┌────────┬──────────────┬─────────────────┬───────┐
│ column ┆ aggregate    ┆ distribution    ┆ scale │
│ ---    ┆ ---          ┆ ---             ┆ ---   │
│ str    ┆ str          ┆ str             ┆ f64   │
╞════════╪══════════════╪═════════════════╪═══════╡
│ len    ┆ Frame Length ┆ Integer Laplace ┆ 2.0   │
└────────┴──────────────┴─────────────────┴───────┘
shape: (2, 2)
┌─────────┬─────┐
│ disease ┆ len │
│ ---     ┆ --- │
│ str     ┆ u32 │
╞═════════╪═════╡
│ flu     ┆ 11  │
│ cancer  ┆ 0   │
└─────────┴─────┘

(true counts, for comparison only - wouldn't normally be shown alongside a real release)
disease
flu       4
cancer    4
Name: count, dtype: int64

--- Query 2: private mean age ---
shape: (2, 4)
┌────────┬───────────┬─────────────────┬───────┐
│ column ┆ aggregate ┆ distribution    ┆ scale │
│ ---    ┆ ---       ┆ ---             ┆ ---   │
│ str    ┆ str       ┆ str             ┆ f64   │
╞════════╪═══════════╪═════════════════╪═══════╡
│ age    ┆ Sum       ┆ Integer Laplace ┆ 400.0 │
│ age    ┆ Length    ┆ Integer Laplace ┆ 4.0   │
└────────┴───────────┴─────────────────┴───────┘
raw DP mean age:     226.20
clamped to (0,100):  100.00   <- noise can push the raw value outside physically plausible bounds on small datasets; clamping after release is a cosmetic fix only, it doesn't add privacy or remove it
(true mean age: 49.75, for comparison only)

--- Attempting a 3rd, unplanned query (should be blocked) ---
[expected] blocked: Privacy allowance has been exhausted
This is the actual point of pre-declaring your budget: you can't just decide to ask 'one more thing' after seeing the results without it being a deliberate, tracked decision.

```
## Alternative Approach # 3 using ```snsql```

***  smartnoise-sql is stuck on an old, pinned opendp version and hasn't been updated to track OpenDP's current releases.  Neeed a new isolated wnvironment to run. ***

```bash
uv add smartnoise-sql

```
Then run anonymize demo file:

   ```bash
 uv run python smartnoisesql_demo.py
 ```
 Output

```
python smartnoisesql_demo.py
[['disease', 'n'], ['cancer', 3], ['flu', 3]]
[['mean_age'], [34.49513909991469]]
cumulative privacy spent (epsilon, delta): (np.float64(6.0), np.float64(0.001499499999999987))
```

## Further reading

- anjana paper (open access): *An Open Source Python Library for Anonymizing
  Sensitive Data*, Scientific Data (2024/2025) — https://www.nature.com/articles/s41597-024-04019-z
- pycanon paper: *A Python library to check the level of anonymity of a
  dataset*, Scientific Data (2022) — https://www.nature.com/articles/s41597-022-01894-2
- Exposing Privacy Risks in Anonymizing Clinical Data:
Combinatorial Refinement Attacks on k-Anonymity Without
Auxiliary Information https://arxiv.org/pdf/2509.03350, https://www.youtube.com/watch?v=pHeXFVl9NgE