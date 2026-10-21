import os, pandas as pd

h_dir = r"D:\Github\deep-learning-for-computer-vision\data\raw\hybrid"
for f in sorted(os.listdir(h_dir)):
    if f.endswith('.csv'):
        df = pd.read_csv(os.path.join(h_dir, f))
        dk = df['dataset_key'].unique().tolist() if 'dataset_key' in df.columns else []
        print(f"{f:30s}: {len(df):6d} rows | datasets: {dk}")
