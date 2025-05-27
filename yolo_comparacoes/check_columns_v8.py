import pandas as pd

file_path = 'C:/Users/Utilizador/runs/detect/combined_yolov8m3/results.csv'
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()
print("Colunas do CSV YOLOv8:")
print(df.columns.tolist())
print("\nPrimeiras linhas do CSV:")
print(df.head())