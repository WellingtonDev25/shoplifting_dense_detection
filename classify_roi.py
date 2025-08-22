import cv2
from ultralytics import YOLO
import time

# Carrega o modelo de classifica��o
modeloTipo = YOLO('cls-small-suspeitos-novas-imagens.pt')

def classificar_imagem(crop):
    tipoFinal = 'normal'
    resultados = modeloTipo.predict(crop,verbose=False)
    # end_time = time.perf_counter()  # marca o final do tempo

    # elapsed_time_ms = (end_time - start_time) * 1000  # converte para milissegundos
    # print(f"Tempo de processamento: {elapsed_time_ms:.2f} ms")

    for resultado in resultados:
        if resultado.probs is not None:
            top1 = resultado.probs.top1
            score = resultado.probs.top1conf  # confian�a da predi��o
            nomes = resultado.names
            tipoFinal = nomes[top1]
            # if tipoFinal in ['suspeito'] and score>=0.99:
            if tipoFinal in ['suspeitos'] and score>=0.90:
                return tipoFinal

    return "normal"