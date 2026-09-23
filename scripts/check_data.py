from incidentsense.config import DATA_PATH
from incidentsense.data import load_incidents

df = load_incidents(DATA_PATH)
print(f"Rows: {len(df)}")
print(f"Classes: {df['root_cause'].nunique()} -> {sorted(df['root_cause'].unique())}")
print(df[['severity','service','root_cause']].head(5).to_string(index=False))
