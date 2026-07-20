import pandas as pd
import snsql
from snsql import Privacy

data = pd.read_csv("input_data/patient_data.csv")

# Metadata: declare column bounds/types once, instead of per-query
metadata = {
    "": {
        "patients": {
            "patients": {
                "row_privacy": True,   # each row = one patient (no separate patient-ID column)
                "max_ids": 1,
                "censor_dims": False,  # see note below - this defaults to True
                "age": {"type": "int", "lower": 0, "upper": 100},
                "gender": {"type": "string"},
                "zipcode": {"type": "int", "lower": 0, "upper": 99999},
                "disease": {"type": "string"},
            }
        }
    }
}

privacy = Privacy(epsilon=2.0, delta=1/1000)
reader = snsql.from_df(data, privacy=privacy, metadata=metadata)

# Just... SQL. This is the whole appeal.
result = reader.execute("SELECT disease, COUNT(*) AS n FROM patients.patients GROUP BY disease")
print(result)

result2 = reader.execute("SELECT AVG(age) AS mean_age FROM patients.patients")
print(result2)

print("cumulative privacy spent (epsilon, delta):", reader.odometer.spent)