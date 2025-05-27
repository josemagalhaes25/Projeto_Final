import pandas as pd
import matplotlib.pyplot as plt
import os

def load_results(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Remove espaços em branco nas colunas
    return df

# Criar a pasta 'plots' se não existir
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# Caminhos para os CSVs
v5_results = 'C:/Users/Utilizador/Desktop/yolov5/yolov5/runs/train/olive_trees_model5/results.csv'
v8_results = 'C:/Users/Utilizador/runs/detect/combined_yolov8m3/results.csv'
v12_results = 'C:/Users/Utilizador/runs/detect/train7/results.csv'

# Carregar dados
v5_df = load_results(v5_results)
v8_df = load_results(v8_results)
v12_df = load_results(v12_results)

# Gráfico mAP@0.5
plt.figure(figsize=(10,6))
plt.plot(v5_df['epoch'], v5_df['metrics/mAP_0.5'], label='YOLOv5', color='blue')
plt.plot(v8_df['epoch'], v8_df['metrics/mAP50(B)'], label='YOLOv8', color='green')
plt.plot(v12_df['epoch'], v12_df['metrics/mAP50(B)'], label='YOLOv12', color='red')
plt.title('Comparação de mAP@0.5 entre YOLOv5, YOLOv8 e YOLOv12')
plt.xlabel('Epochs')
plt.ylabel('mAP@0.5')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'grafico_map_0.5.png'), dpi=300, bbox_inches='tight')
plt.show()

# Gráfico mAP@0.5:0.95
plt.figure(figsize=(10,6))
plt.plot(v5_df['epoch'], v5_df['metrics/mAP_0.5:0.95'], label='YOLOv5', color='blue')
plt.plot(v8_df['epoch'], v8_df['metrics/mAP50-95(B)'], label='YOLOv8', color='green')
plt.plot(v12_df['epoch'], v12_df['metrics/mAP50-95(B)'], label='YOLOv12', color='red')
plt.title('Comparação de mAP@0.5:0.95 entre YOLOv5, YOLOv8 e YOLOv12')
plt.xlabel('Epochs')
plt.ylabel('mAP@0.5:0.95')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'grafico_map_0.5_0.95.png'), dpi=300, bbox_inches='tight')
plt.show()

# Gráfico Box Loss
plt.figure(figsize=(10,6))
plt.plot(v5_df['epoch'], v5_df['train/box_loss'], label='YOLOv5 Box Loss', color='blue')
plt.plot(v8_df['epoch'], v8_df['train/box_loss'], label='YOLOv8 Box Loss', color='green')
plt.plot(v12_df['epoch'], v12_df['train/box_loss'], label='YOLOv12 Box Loss', color='red')
plt.title('Comparação de Box Loss entre YOLOv5, YOLOv8 e YOLOv12')
plt.xlabel('Epochs')
plt.ylabel('Box Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'grafico_box_loss.png'), dpi=300, bbox_inches='tight')
plt.show()

# Gráfico F1 Score (se existir)
if 'f1_score' in v5_df.columns and 'f1_score' in v8_df.columns and 'f1_score' in v12_df.columns:
    plt.figure(figsize=(10,6))
    plt.plot(v5_df['epoch'], v5_df['f1_score'], label='YOLOv5 F1 Score', color='blue')
    plt.plot(v8_df['epoch'], v8_df['f1_score'], label='YOLOv8 F1 Score', color='green')
    plt.plot(v12_df['epoch'], v12_df['f1_score'], label='YOLOv12 F1 Score', color='red')
    plt.title('Comparação de F1 Score entre YOLOv5, YOLOv8 e YOLOv12')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'grafico_f1_score.png'), dpi=300, bbox_inches='tight')
    plt.show()

# Gráfico de tempo de treino (se disponível)
if 'time' in v8_df.columns and 'time' in v12_df.columns:
    plt.figure(figsize=(10,6))
    plt.plot(v8_df['epoch'], v8_df['time'], label='YOLOv8 Train Time', color='green')
    plt.plot(v12_df['epoch'], v12_df['time'], label='YOLOv12 Train Time', color='red')
    plt.title('Comparação de Tempo de Treinamento entre YOLOv8 e YOLOv12')
    plt.xlabel('Epochs')
    plt.ylabel('Tempo de Treinamento (s)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'grafico_tempo_treinamento.png'), dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("Dados de tempo de treino não disponíveis para YOLOv5 ou YOLOv8/YOLOv12")