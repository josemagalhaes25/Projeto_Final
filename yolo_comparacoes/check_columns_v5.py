import pandas as pd

# Substitui este caminho pelo caminho do teu ficheiro CSV
csv_path = 'C:/Users/Utilizador/Desktop/yolov5/yolov5/runs/train/olive_trees_model5/results.csv'

# Carrega o CSV
df = pd.read_csv(csv_path)

# Mostra as colunas
print("Colunas do CSV:")
print(df.columns)

# Mostra as primeiras linhas para veres os dados
print("\nPrimeiras linhas do CSV:")
print(df.head())