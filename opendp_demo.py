"""
OpenDP for a REAL dataset - using the Context API (Polars-based), not the
low-level scalar mechanisms from the toy comparison demo.

This is the interface OpenDP actually recommends for real deployments. Two
things make it fundamentally different from `opendp_demo.py`:

1. You declare your ENTIRE privacy budget and analysis plan up front, before
   touching the data. You don't get a Laplace mechanism per query and hope
   the math adds up - the Context object tracks spend automatically and will
   hard-block any query once the budget is exhausted. Verified: attempting a
   3rd query after a 2-query budget raises `ValueError: Privacy allowance has
   been exhausted` - not a warning, a refusal.

2. You work with Polars expressions (`.dp.noise()`, `.dp.mean()`, group_by,
   filter, etc.) instead of hand-building sensitivity/scale math. This is
   what makes it usable on a real dataset with many columns and group-by
   breakdowns, rather than one scalar count at a time.

--------------------------------------------------------------------------
BEFORE YOU RUN THIS ON REAL PATIENT DATA - decisions you must make first:
--------------------------------------------------------------------------

1. PRIVACY UNIT: what does "one person" mean in your data?
   If each row is already one unique patient, `contributions=1` (below) is
   correct. If a patient can appear in MULTIPLE rows (e.g. one row per visit
   or per lab result), you must set `contributions` to the maximum number of
   rows any single patient could contribute - otherwise the privacy
   guarantee silently doesn't hold for patients with many visits.

2. PRIVACY BUDGET (epsilon): how much total noise are you willing to trade
   for accuracy, across ALL queries you'll ever run against this budget?
   There's no universal "correct" epsilon - it depends on data sensitivity
   and how many times the data will be queried. Commonly cited ranges in
   research/statistical-agency practice run from ~0.1 (very strict) to ~10
   (loose) depending on context; healthcare data releases tend toward the
   stricter end. Decide this BEFORE looking at results, not after.

3. PRE-REGISTER YOUR QUERIES: decide what you want to ask before you start.
   Peeking at a noisy result and then deciding to ask a follow-up query
   based on what you saw undermines the formal guarantee (this is the same
   discipline as pre-registering a clinical trial analysis plan). The
   `split_evenly_over` / budget-exhaustion enforcement exists specifically
   to make this discipline hard to skip.

4. MARGINS ARE A PRIVACY DECISION, NOT JUST PLUMBING: declaring
   `max_length` or `by=[...]` as a `Margin` tells OpenDP "treat this as
   PUBLIC information, not protected by the budget." Setting these too
   loosely (e.g. exposing exact group membership) can leak information
   outside the formal guarantee. Set margins conservatively - a generous
   upper bound on row count is fine; the exact row count usually is not.

Install:
    uv add opendp polars pyarrow
    (OpenDP pins a specific Polars version internally - if you see a
    "DSL_SCHEMA_HASH... not compatible" error, check the OpenDP release
    notes for the Polars version it currently expects and pin to that
    exact version, e.g. `uv add "polars==1.36.1"`.)
"""

import polars as pl
import opendp.prelude as dp
import pandas as pd

dp.enable_features("contrib")

# ---------------------------------------------------------------------------
# 1. Load data as a Polars LazyFrame (Context API requires lazy frames)
# ---------------------------------------------------------------------------
inputdata = pd.read_csv("input_data/patient_data.csv")# pandas, from the shared loader

data = pl.from_pandas(inputdata).lazy()

print("Original dataset:")
print(inputdata.to_string(index=False))

# ---------------------------------------------------------------------------
# 2. Declare the privacy unit, budget, and public margins - ALL UP FRONT
# ---------------------------------------------------------------------------
TOTAL_EPSILON = 1.0
N_QUERIES = 2  # we're about to ask exactly 2 questions - decide this now

context = dp.Context.compositor(
    data=data,
    # one row = one patient in this dataset; bump `contributions` if a
    # patient can appear in more than one row in your real data
    privacy_unit=dp.unit_of(contributions=1),
    privacy_loss=dp.loss_of(epsilon=TOTAL_EPSILON),
    split_evenly_over=N_QUERIES,
    margins=[
        # "there are at most 1000 rows total" - a loose, safe public bound,
        # not the real (protected) row count
        dp.polars.Margin(max_length=1000),
        # "the set of disease values (flu, cancer, ...) is public" - true
        # here since we're not trying to hide which diagnoses EXIST, only
        # how many patients have each one
        dp.polars.Margin(by=["disease"], max_length=1000, invariant="keys"),
    ],
)

# ---------------------------------------------------------------------------
# 3. Query 1: private counts per disease (a private histogram)
# ---------------------------------------------------------------------------
# This is the realistic healthcare question: "how many patients have each
# diagnosis?" without revealing exact counts that could re-identify a small
# clinic's patient roster.
query_counts = context.query().group_by("disease").agg(pl.len().dp.noise())

print("\n--- Query 1: private counts by disease ---")
print(query_counts.summarize())  # shows the noise mechanism & scale BEFORE spending budget
counts_result = query_counts.release().collect()
print(counts_result)

true_counts = inputdata["disease"].value_counts()
print("\n(true counts, for comparison only - wouldn't normally be shown alongside a real release)")
print(true_counts)

# ---------------------------------------------------------------------------
# 4. Query 2: private mean age
# ---------------------------------------------------------------------------
query_mean_age = context.query().select(pl.col("age").dp.mean(bounds=(0, 100)))

print("\n--- Query 2: private mean age ---")
print(query_mean_age.summarize())
mean_result = query_mean_age.release().collect()
raw_mean_age = mean_result["age"][0]
clamped_mean_age = max(0.0, min(100.0, raw_mean_age))
print(f"raw DP mean age:     {raw_mean_age:.2f}")
print(f"clamped to (0,100):  {clamped_mean_age:.2f}   <- noise can push the raw value outside physically plausible bounds on small datasets; clamping after release is a cosmetic fix only, it doesn't add privacy or remove it")
print(f"(true mean age: {inputdata['age'].mean():.2f}, for comparison only)")

# ---------------------------------------------------------------------------
# 5. Budget is now exhausted - this is enforced, not just documented
# ---------------------------------------------------------------------------
print("\n--- Attempting a 3rd, unplanned query (should be blocked) ---")
try:
    extra_query = context.query().select(pl.len().dp.noise())
    extra_query.release().collect()
    print("[unexpected] extra query succeeded")
except ValueError as e:
    print(f"[expected] blocked: {e}")
    print(
        "This is the actual point of pre-declaring your budget: you can't "
        "just decide to ask 'one more thing' after seeing the results "
        "without it being a deliberate, tracked decision."
    )
