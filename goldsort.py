import pandas as pd

df = pd.read_csv(".\OutputDir\goldsorted.csv", sep=",", header=0)
print(df.sort_values(axis=0, by=0))
df.to_csv(".\OutputDir\goldsorted.csv")