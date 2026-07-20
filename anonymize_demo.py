"""
Sample project: anonymizing structured tabular healthcare-style data using
ANJANA (anonymization) + pyCANON (verification).

Both libraries are open source, Apache-2.0 licensed, built and maintained by
IFCA (Instituto de Fisica de Cantabria), a Spanish public research institute
(CSIC / University of Cantabria), funded through the EU Horizon Europe SIESTA
project. Published and peer-reviewed in Nature Scientific Data.

- anjana:  https://github.com/IFCA-Advanced-Computing/anjana
- pycanon: https://github.com/IFCA-Advanced-Computing/pycanon

Unlike ARX, both install with a plain `pip install`, no manual jar wrangling.
"""

import pandas as pd
from anjana.anonymity import k_anonymity, l_diversity, t_closeness
from pycanon import anonymity, report

# 1. Input dataset --------------------------------------------------------
# age/gender/zipcode = quasi-identifiers, disease = sensitive attribute
data = pd.DataFrame(
    {
        "age": [34, 45, 66, 70, 34, 70, 45, 34],
        "gender": ["male", "female", "male", "female", "female", "male", "male", "female"],
        "zipcode": [81667, 81675, 81925, 81931, 81931, 81931, 81928, 81931],
        "disease": ["flu", "flu", "cancer", "flu", "cancer", "cancer", "flu", "cancer"],
    }
)

# 2. Generalization hierarchies --------------------------------------------
# Loaded from CSV. Column 0 = original value, each subsequent column is a
# more general level (same convention ARX uses for hierarchy files).
hierarchies = {
    "age": dict(pd.read_csv("hierarchies/age.csv", header=None)),
    "gender": dict(pd.read_csv("hierarchies/gender.csv", header=None)),
    "zipcode": dict(pd.read_csv("hierarchies/zipcode.csv", header=None)),
}

identifiers = []  # columns to suppress outright (e.g. name, SSN) - none here
quasi_identifiers = ["age", "gender", "zipcode"]
sensitive_attribute = "disease"

K = 2                 # desired k-anonymity level
L = 2                 # desired l-diversity level (distinct sensitive values per group)
T = 0.5               # desired t-closeness threshold (lower = closer to global distribution)
SUPPRESSION_LIMIT = 20  # max % of rows allowed to be fully suppressed
print("Original dataset:")
print(data.to_string(index=False))


# 3a. k-anonymity only -------------------------------------------------------
# Hides identity, but says nothing about the sensitive attribute: a group can
# still be 100% "cancer", which leaks the diagnosis to anyone who narrows a
# person down to that group.
k_only = k_anonymity(data, identifiers, quasi_identifiers, K, SUPPRESSION_LIMIT, hierarchies)
print(f"\n--- k-anonymity only (k={K}) ---")
print(k_only.to_string(index=False))
print(f"Verified k: {anonymity.k_anonymity(k_only, quasi_identifiers)}")
print(f"Verified l: {anonymity.l_diversity(k_only, quasi_identifiers, [sensitive_attribute])}"
      "   <- notice this can be 1, i.e. no real protection on 'disease'")

# 3b. k-anonymity + l-diversity ----------------------------------------------
# Forces every group to contain at least L distinct sensitive values, so no
# group can be all-"cancer" or all-"flu".
l_div = l_diversity(data, identifiers, quasi_identifiers, sensitive_attribute, K, L, SUPPRESSION_LIMIT, hierarchies)
print(f"\n--- k-anonymity + l-diversity (k={K}, l={L}) ---")
print(l_div.to_string(index=False))
print(f"Verified k: {anonymity.k_anonymity(l_div, quasi_identifiers)}")
print(f"Verified l: {anonymity.l_diversity(l_div, quasi_identifiers, [sensitive_attribute])}")

# 3c. k-anonymity + t-closeness ----------------------------------------------
# A stricter guarantee than l-diversity: each group's distribution of the
# sensitive attribute must stay close (within t) to the overall dataset's
# distribution, not just "contain 2+ values".
t_close = t_closeness(data, identifiers, quasi_identifiers, sensitive_attribute, K, T, SUPPRESSION_LIMIT, hierarchies)
print(f"\n--- k-anonymity + t-closeness (k={K}, t={T}) ---")
print(t_close.to_string(index=False))
print(f"Verified k: {anonymity.k_anonymity(t_close, quasi_identifiers)}")
print(f"Verified t: {anonymity.t_closeness(t_close, quasi_identifiers, [sensitive_attribute]):.3f}")

print("\nFull privacy report (k-anonymity + l-diversity result):")
report.print_report(l_div, quasi_identifiers, [sensitive_attribute])
