from ultralytics import YOLO
import torch

torch.cuda.empty_cache()  # Libera cache da GPU
torch.cuda.ipc_collect()  # Libera mem�ria compartilhada entre processos

def main():
    modelo = YOLO('yolo11s-cls.pt')
    results = modelo.train(
        data='dataset',
        epochs=50,
        patience=10,
        imgsz=640
    )

if __name__ == '__main__':
    main()


