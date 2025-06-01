import sys
from pathlib import Path
import torch
import cv2
import numpy as np

# 1. Configuração de caminhos ESPECÍFICA para sua estrutura
ROOT = Path(__file__).resolve().parents[1]  # Volta para a pasta yolov5
SOURCE = ROOT / 'datasets' / 'combined_dataset' / 'valid'/ 'images' # Caminho real
WEIGHTS = ROOT / 'modelo_yolov5' / 'runs' / 'train' / 'olive_trees_model' / 'weights' / 'best.pt'
SAVE_DIR = ROOT / 'modelo_yolov5' / 'runs' / 'val_images'

# 2. Verificação dos caminhos
print(f" Procurando imagens em: {SOURCE}")
if not SOURCE.exists():
    print(f" Pasta não encontrada! Verifique se existe:")
    print(f" - {SOURCE}")
    sys.exit(1)

# 3. Cria pasta de saída
SAVE_DIR.mkdir(parents=True, exist_ok=True)
print(f" Pasta de saída: {SAVE_DIR}")

# 4. Carrega modelo (ajuste os imports conforme sua versão)
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

device = select_device('0')
model = DetectMultiBackend(WEIGHTS, device=device)
stride = model.stride
imgsz = check_img_size(640, s=stride)

# 5. Processamento
img_files = list(SOURCE.glob('*.jpg')) + list(SOURCE.glob('*.png'))
print(f"\n {len(img_files)} imagens encontradas")

for img_path in img_files:
    try:
        # Carrega imagem
        img0 = cv2.imread(str(img_path))
        if img0 is None:
            print(f" Imagem corrompida: {img_path.name}")
            continue

        # Pré-processamento
        img = cv2.resize(img0, (imgsz, imgsz))
        img = img.transpose((2, 0, 1))[::-1]  # BGR→RGB, HWC→CHW
        img = np.ascontiguousarray(img)
        
        # Inferência
        img_tensor = torch.from_numpy(img).to(device).float() / 255.0
        pred = model(img_tensor.unsqueeze(0))[0]
        pred = non_max_suppression(pred, 0.25, 0.45)

        # Desenha bounding boxes
        annotator = Annotator(img0.copy(), line_width=2, example=str(model.names))
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img_tensor.shape[1:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

        # Salva resultado
        output_path = SAVE_DIR / img_path.name
        cv2.imwrite(str(output_path), annotator.result())
        print(f" Salvo: {output_path}")

    except Exception as e:
        print(f" Erro em {img_path.name}: {str(e)}")

print(f"\n Todas imagens processadas em: {SAVE_DIR}")