import os
import shutil
from pathlib import Path
import yaml

# Caminhos de entrada
dataset1 = Path("datasets/dataset1")
dataset2 = Path("datasets/dataset2")

# Caminho de saída ajustado para ficar dentro de 'datasets'
output_dir = Path("datasets/combined_dataset")

# Substituir 'val' por 'valid'
subsets = ['train', 'valid', 'test']
for subset in subsets:
    (output_dir / subset / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / subset / "labels").mkdir(parents=True, exist_ok=True)

def copiar_conteudo(subset, source_dir):
    for tipo in ['images', 'labels']:
        src_path = source_dir / subset / tipo
        dst_path = output_dir / subset / tipo
        if src_path.exists():
            for file in src_path.glob("*.*"):
                nome_novo = f"{source_dir.name}_{file.name}"
                shutil.copy(file, dst_path / nome_novo)

# Copiar imagens e labels dos dois datasets
for subset in subsets:
    copiar_conteudo(subset, dataset1)
    copiar_conteudo(subset, dataset2)

# Criar novo data.yaml
with open(dataset1 / "data.yaml", "r") as f:
    data_yaml = yaml.safe_load(f)

data_yaml["train"] = "datasets/combined_dataset/train/images"
data_yaml["val"] = "datasets/combined_dataset/valid/images"  # Notar que aqui também usamos 'valid'
if (dataset1 / "test/images").exists() or (dataset2 / "test/images").exists():
    data_yaml["test"] = "datasets/combined_dataset/test/images"

with open(output_dir / "data.yaml", "w") as f:
    yaml.dump(data_yaml, f)

print("Fusão completa com sucesso em 'datasets/combined_dataset'!")