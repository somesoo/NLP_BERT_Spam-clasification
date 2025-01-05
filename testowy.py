import pandas as pd

df = pd.read_csv("7000_training_dataset.csv")
df2 = pd.read_csv("500_test_dataset.csv")

print(df["label"].sum())
print(df2["label"].sum())
