"""Make a deliberately corrupted copy of the data, to demonstrate the data gate."""
import numpy as np
import pandas as pd
import yaml

cfg = yaml.safe_load(open("config.yaml"))
df = pd.read_csv(cfg["data"]["path"])
rng = np.random.default_rng(0)
df.loc[rng.choice(len(df), 40, replace=False), "Age"] = -5                 # impossible ages
df.loc[rng.choice(len(df), int(0.3 * len(df)), replace=False), "Credit_Limit"] = np.nan
df.loc[0, "Income_Bracket"] = "Platinum"                                  # unknown category
df.to_csv("data/corrupted_demo.csv", index=False)
print("wrote data/corrupted_demo.csv")
