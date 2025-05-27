import os
import shutil
from pathlib import Path

# Caminhos
root_dir = Path('datasets')
output_dir = root_dir / 'combined_dataset_v12'
splits = ['train', 'valid', 'test']
subdirs = ['images', 'labels']
datasets = ['dataset1', 'dataset2']  # substitui pelos nomes exatos se forem diferentes

# Criação das pastas de destino
for split in splits:
    for sub in subdirs:
        dest = output_dir / split / sub
        dest.mkdir(parents=True, exist_ok=True)

# Copiar os ficheiros de cada dataset
for dataset in datasets:
    for split in splits:
        for sub in subdirs:
            src_dir = root_dir / dataset / split / sub
            dst_dir = output_dir / split / sub
            if src_dir.exists():
                for file in src_dir.iterdir():
                    shutil.copy(file, dst_dir)

print("Fusão completa em:", output_dir)