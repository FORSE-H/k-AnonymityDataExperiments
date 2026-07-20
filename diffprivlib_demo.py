"""
Differential privacy demo using diffprivlib (IBM).

Unlike anjana/pycanon (generalization + suppression on row-level data), this
approach never generalizes or publishes individual rows at all. Instead, it
answers specific aggregate questions ("how many patients have cancer?",
"what's the average age?") and adds calibrated random noise to each answer,
with a mathematically provable privacy guarantee (epsilon) rather than a
threshold you tune by trial and error like k/l/t.

Install: pip install diffprivlib pandas
Run:     python diffprivlib_demo.py
"""

from diffprivlib.accountant import BudgetAccountant
from diffprivlib.mechanisms import Laplace
import diffprivlib.tools as dpt
import pandas as pd

data = pd.read_csv("input_data/patient_data.csv")

# ---------------------------------------------------------------------------
# Privacy budget
# ---------------------------------------------------------------------------
# Total epsilon available to spend across ALL queries below. Smaller epsilon
# = stronger privacy but noisier answers. There's no universal "good" value;
# it depends on how many times you'll query this data and how sensitive the
# answers are. 1.0 total, split across 3 queries, is intentionally tight for
# a demo - real deployments often reason carefully about this per use case.
TOTAL_EPSILON = 1.0
N_QUERIES = 3
EPS_PER_QUERY = TOTAL_EPSILON / N_QUERIES

#data = load_data()
print("Original dataset:")
print(data.to_string(index=False))

accountant = BudgetAccountant()

# ---------------------------------------------------------------------------
# 1 & 2. Private counts: how many patients have each disease?
# ---------------------------------------------------------------------------
# Sensitivity = 1: adding or removing a single patient changes any of these
# counts by at most 1, which is what calibrates the noise.
count_mechanism = Laplace(epsilon=EPS_PER_QUERY, sensitivity=1)

true_cancer = int((data["disease"] == "cancer").sum())
true_flu = int((data["disease"] == "flu").sum())

dp_cancer = count_mechanism.randomise(true_cancer)
dp_flu = count_mechanism.randomise(true_flu)

# ---------------------------------------------------------------------------
# 3. Private mean age
# ---------------------------------------------------------------------------
# diffprivlib's `tools` module mirrors numpy/pandas but with DP baked in.
# `bounds` clips ages to a public range first (required so a single extreme
# outlier can't skew the sensitivity/noise calculation).
dp_mean_age = dpt.mean(
    data["age"].values,
    epsilon=EPS_PER_QUERY,
    bounds=(0, 100),
    accountant=accountant,
)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print(f"\n--- Differentially private aggregate statistics (total epsilon={TOTAL_EPSILON}) ---")
print(f"Cancer count   -> true: {true_cancer:>3}   DP: {dp_cancer:6.2f}")
print(f"Flu count      -> true: {true_flu:>3}   DP: {dp_flu:6.2f}")
print(f"Mean age       -> true: {data['age'].mean():6.2f}   DP: {dp_mean_age:6.2f}")

# The mean-age query recorded itself via `accountant`; add the two manual
# count queries so the total reflects everything spent.
manual_spend = EPS_PER_QUERY * 2
total_spent = accountant.total()[0] + manual_spend
print(f"\nEpsilon spent: {total_spent:.3f} (budget was {TOTAL_EPSILON})")

print(
    "\nNote: no row-level data was ever generalized, suppressed, or "
    "published here - only noisy aggregate answers. That's the core "
    "difference from the anjana/k-anonymity approach: DP protects *queries*, "
    "not a released dataset. If you need to hand someone a CSV they can "
    "explore freely, DP alone doesn't give you that - you'd pair it with "
    "synthetic data generation, or fall back to k-anonymity/l-diversity."
)
