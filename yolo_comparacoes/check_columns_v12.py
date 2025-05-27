import pandas as pd

v12_results = 'C:/Users/Utilizador/runs/detect/train7/results.csv'
v12_df = pd.read_csv(v12_results)
v12_df.columns = v12_df.columns.str.strip()  # Remove espaços em branco
print("Colunas YOLOv12:", v12_df.columns.tolist())