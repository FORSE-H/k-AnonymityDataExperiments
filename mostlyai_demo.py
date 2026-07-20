"""
MOSTLY AI Synthetic Data SDK - step-wise comparison of privacy settings.

Runs the SAME dataset through three configurations, so you can directly see
the accuracy/privacy tradeoff instead of just reading about it:

  STEP 1: value_protection=False   - no built-in protection at all
  STEP 2: value_protection=True    - default behaviour (rare-category
                                      suppression, noisy value bounds, but
                                      NO formal privacy guarantee)
  STEP 3: differential_privacy=... - formal, mathematically-bounded epsilon
                                      guarantee via DP-SGD training

Each step trains its own generator and samples the same number of synthetic
rows, so you can compare them side by side at the end.

Install:
    uv add "mostlyai[local]" --extra-index-url https://download.pytorch.org/whl/cpu

Note on dataset size: the very first version of this demo used the same
8-row toy dataset from the rest of this project, and MOSTLY AI's rare-
category protection swallowed almost the entire output (everything got
replaced with `_RARE_` tokens) because with only 8 rows, nearly every value
genuinely IS rare. This script uses a larger (60-row) synthetic dataset with
the same column structure specifically so the differences between settings
are actually visible rather than uniformly wiped out.
"""

import numpy as np
import pandas as pd
from mostlyai.sdk import MostlyAI
from mostlyai.sdk.domain import (
    DifferentialPrivacyConfig,
    GeneratorConfig,
    ModelConfiguration,
    SourceTableConfig,
)

# ---------------------------------------------------------------------------
# 1. Build a dataset with the same shape as the rest of this project, but
#    large enough (n=60) for the privacy mechanisms to behave meaningfully
#    instead of collapsing everything to a placeholder value.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
N = 60

ages = rng.integers(20, 85, N)
genders = rng.choice(["male", "female"], N)
zipcodes = rng.choice([81667, 81675, 81925, 81931, 81928], N)
# disease correlated with age, same as a real dataset would be - this
# correlation is what we're watching to see how well each setting preserves
disease = np.where(
    ages > 55,
    rng.choice(["cancer", "flu"], N, p=[0.6, 0.4]),
    rng.choice(["cancer", "flu"], N, p=[0.15, 0.85]),
)

data = pd.DataFrame({"age": ages, "gender": genders, "zipcode": zipcodes, "disease": disease})

print("Original dataset (n=60):")
print(data.head(8).to_string(index=False))
print("...")
print(f"\nTrue disease split: {dict(data['disease'].value_counts())}")
print(f"True mean age: {data['age'].mean():.1f}")

mostly = MostlyAI(local=True, quiet=True)

# Small epoch count everywhere purely to keep this demo fast on a laptop/CPU
# Codespace - bump this up for real use, it directly trades training time
# for quality.
MAX_EPOCHS = 2


def run_step(step_name: str, model_config: ModelConfiguration) -> pd.DataFrame:
    """Train a generator with the given config and sample 60 synthetic rows."""
    config = GeneratorConfig(
        name=step_name,
        tables=[
            SourceTableConfig(
                name="patients",
                data=data,
                tabular_model_configuration=model_config,
            )
        ],
    )
    generator = mostly.train(config=config)
    synthetic = mostly.generate(generator, size=N).data()
    return synthetic


# ---------------------------------------------------------------------------
# STEP 1: No protection at all
# ---------------------------------------------------------------------------
# Maximum fidelity to the real data's statistics. This is what you'd use if
# you ONLY care about realistic-looking test/dev data and have zero privacy
# requirement - not appropriate for anything derived from real patient data.
print("\n" + "=" * 70)
print("STEP 1: value_protection=False (no protection)")
print("=" * 70)

sd_none = run_step(
    "step1_no_protection",
    ModelConfiguration(max_epochs=MAX_EPOCHS, value_protection=False),
)
print(sd_none.head(8).to_string(index=False))
print(f"\nSynthetic disease split: {dict(sd_none['disease'].value_counts())}")
print(f"Synthetic mean age: {sd_none['age'].mean():.1f}")


# ---------------------------------------------------------------------------
# STEP 2: Default protection (value_protection=True)
# ---------------------------------------------------------------------------
# MOSTLY AI's default. Applies rare-category suppression and noisy value
# bounds as heuristic protections, but this is NOT a formal privacy
# guarantee - there's no epsilon, no mathematical bound on what an attacker
# could infer. Good general-purpose default; not sufficient on its own for
# something you'd formally call "anonymized" in a healthcare context.
print("\n" + "=" * 70)
print("STEP 2: value_protection=True (default - heuristic protection)")
print("=" * 70)

sd_default = run_step(
    "step2_default_protection",
    ModelConfiguration(max_epochs=MAX_EPOCHS, value_protection=True),
)
print(sd_default.head(8).to_string(index=False))
print(f"\nSynthetic disease split: {dict(sd_default['disease'].value_counts())}")
print(f"Synthetic mean age: {sd_default['age'].mean():.1f}")


# ---------------------------------------------------------------------------
# STEP 3: Formal differential privacy (DP-SGD training)
# ---------------------------------------------------------------------------
# This is the real privacy upgrade: the model itself is trained with
# calibrated noise (DP-SGD), giving a mathematically bounded guarantee
# (epsilon), the same formal concept as the diffprivlib/OpenDP demos
# elsewhere in this project - just applied during model TRAINING instead of
# to individual query answers.
#
#   maxEpsilon: training stops early if this budget would be exceeded
#   delta: probability the guarantee doesn't hold (smaller = stronger)
#   noiseMultiplier / maxGradNorm: control how much noise gets injected
#     into training - see MOSTLY AI's docs before tuning these on real data
#
# WARNING (found by actually running this): on a small dataset with few
# epochs, DP training visibly degraded the `age` column - synthetic ages
# came back as small integers (0-7) instead of realistic ages (20-85). This
# isn't a bug, it's the honest cost of the privacy guarantee: DP-SGD adds
# noise to every gradient update, and with little data and few training
# steps, that noise can dominate before the model learns real structure.
# Real deployments need substantially more data and/or more training epochs
# for DP to produce usable output - the same "DP needs scale" lesson from
# the diffprivlib/OpenDP demos, just showing up inside model training here
# instead of inside a query answer.
print("\n" + "=" * 70)
print("STEP 3: differential_privacy=... (formal epsilon guarantee)")
print("=" * 70)

sd_dp = run_step(
    "step3_differential_privacy",
    ModelConfiguration(
        max_epochs=MAX_EPOCHS,
        value_protection=True,
        differential_privacy=DifferentialPrivacyConfig(maxEpsilon=5.0),
    ),
)
print(sd_dp.head(8).to_string(index=False))
print(f"\nSynthetic disease split: {dict(sd_dp['disease'].value_counts())}")
print(f"Synthetic mean age: {sd_dp['age'].mean():.1f}")
print(
    "\n(if 'age' looks like small integers instead of realistic ages here, "
    "that's the DP noise dominating a small/short training run - see the "
    "warning in the comments above)"
)


# ---------------------------------------------------------------------------
# Side-by-side summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY: mean age and disease split across all three settings")
print("=" * 70)
summary = pd.DataFrame(
    {
        "true_data": [data["age"].mean(), dict(data["disease"].value_counts())],
        "step1_no_protection": [sd_none["age"].mean(), dict(sd_none["disease"].value_counts())],
        "step2_default": [sd_default["age"].mean(), dict(sd_default["disease"].value_counts())],
        "step3_dp": [sd_dp["age"].mean(), dict(sd_dp["disease"].value_counts())],
    },
    index=["mean_age", "disease_split"],
)
print(summary.to_string())