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

## Further reading

- anjana paper (open access): *An Open Source Python Library for Anonymizing
  Sensitive Data*, Scientific Data (2024/2025) — https://www.nature.com/articles/s41597-024-04019-z
- pycanon paper: *A Python library to check the level of anonymity of a
  dataset*, Scientific Data (2022) — https://www.nature.com/articles/s41597-022-01894-2
  - Exposing Privacy Risks in Anonymizing Clinical Data:
Combinatorial Refinement Attacks on k-Anonymity Without
Auxiliary Information https://arxiv.org/pdf/2509.03350, https://www.youtube.com/watch?v=pHeXFVl9NgE