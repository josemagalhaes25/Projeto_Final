import shutil
import os

def copy_dataset(src, dst):
    for split in ['train', 'valid', 'test']:
        for subfolder in ['images', 'labels']:
            src_path = os.path.join(src, split, subfolder)
            dst_path = os.path.join(dst, split, subfolder)
            os.makedirs(dst_path, exist_ok=True)
            for file in os.listdir(src_path):
                src_file = os.path.join(src_path, file)
                dst_file = os.path.join(dst_path, file)
                if not os.path.exists(dst_file):  # evita duplicados
                    shutil.copy(src_file, dst_path)

# Caminhos relativos à raiz do projeto
dataset1 = 'datasets/dataset1'
dataset2 = 'datasets/dataset2'
combined = 'datasets/combined_dataset'

copy_dataset(dataset1, combined)
copy_dataset(dataset2, combined)

print("Datasets fundidos com sucesso em:", combined)