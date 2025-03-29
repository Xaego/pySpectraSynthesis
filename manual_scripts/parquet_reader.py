import pandas as pd

df = pd.read_parquet(r"/data/light_sources.parquet")
print(df.head())