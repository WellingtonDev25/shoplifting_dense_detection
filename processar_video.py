import cv2
import numpy as np
from ultralytics import YOLO
import torch
from tqdm import tqdm
import time
from processar_roi import processar_roi_image
from classify_roi import classificar_imagem
import uuid

# ================== CONFIG ==================
# VIDEO_PATH          = "rtsp://admin:Ii3432841296@192.168.1.21:554/cam/realmonitor?channel=1&subtype=0" 
VIDEO_PATH          = "videos/The Lemonade Looting Bearded Beggar.mp4"    # caminho do v�deo ou RTSP
MODEL_PATH          = "yolo11n.pt"    # modelo YOLO
CONF_THRESH         = 0.35            # confian�a m�nima
FRAME_STRIDE_PER_ID = 4               # chama processar_roi a cada N frames POR TRACK ID
ROI_OUT_W, ROI_OUT_H = 640, 480       # tamanho do ROI ap�s letterbox
SHOW_FPS            = True
TRACKER_CFG         = "bytetrack.yaml"  # ou "botsort.yaml"
# ============================================

def letterbox_resize(img, out_w, out_h):
    """Redimensiona preservando propor��o e preenche com bordas (preto) at� (out_w,out_h)."""
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return None
    scale = min(out_w / w, out_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    top = (out_h - nh) // 2
    left = (out_w - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas

def clamp_box(xyxy, w, h):
    x1, y1, x2, y2 = map(int, xyxy)
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def draw_label(img, x1, y1, text, color=(0,0,255)):
    """Ret�ngulo de fundo + texto leg�vel."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.6, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 4
    cv2.rectangle(img, (x1, max(0, y1 - th - 2*pad)), (x1 + tw + 2*pad, y1), color, -1)
    cv2.putText(img, text, (x1 + pad, y1 - pad), font, scale, (255,255,255), thickness, cv2.LINE_AA)

def main():
    # Metadados do v�deo (para barra de progresso)
    cap_meta = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_cap = cap_meta.get(cv2.CAP_PROP_FPS) or 30.0
    cap_meta.release()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(MODEL_PATH)
    try:
        model.to(device)
    except Exception:
        pass

    print(f"[INFO] Abrindo {VIDEO_PATH} | FPS�{fps_cap:.2f} | tracker={TRACKER_CFG}")

    # Estados por track id
    next_due_frame = {}   # id -> pr�ximo frame_idx em que deve processar
    label_by_id    = {}   # id -> �ltimo r�tulo conhecido

    frame_idx = 0
    t_last = time.time()
    smoothed_fps = None

    pbar = tqdm(total=total_frames if total_frames > 0 else None,
                unit="frame", dynamic_ncols=True, desc="Rastreando")

    # Usando stream=True para receber 1 Results por frame
    # classes=[0] => s� 'person'
    stream = model.track(
        source=VIDEO_PATH,
        stream=True,
        conf=CONF_THRESH,
        classes=[0],
        tracker=TRACKER_CFG,
        verbose=False,
        persist=True  # mant�m estado do modelo/track
    )

    try:
        for res in stream:
            # Imagem original deste frame
            frame = res.orig_img  # BGR
            if frame is None:
                break
            # frame = cv2.resize(frame,(1280,720))
            h, w = frame.shape[:2]

            # Pega caixas e ids do rastreador
            boxes = res.boxes
            if boxes is not None and boxes.xyxy is not None:
                xyxys = boxes.xyxy.cpu().numpy()
                ids = boxes.id
                ids = ids.int().cpu().numpy() if ids is not None else None

                # Processar cada detecção (com ID válido)
                if ids is not None:
                    for xyxy, tid in zip(xyxys, ids):
                        box = clamp_box(xyxy, w, h)
                        if box is None:
                            continue
                        x1, y1, x2, y2 = box

                        # ROI recortada + letterbox 640x480
                        roi = frame[y1:y2, x1:x2]
                        if roi.size == 0:
                            continue
                        roi_lb = letterbox_resize(roi, ROI_OUT_W, ROI_OUT_H)
                        if roi_lb is None:
                            continue

                        # agenda por ID: s� chama a cada FRAME_STRIDE_PER_ID
                        if tid not in next_due_frame:
                            next_due_frame[tid] = 0  # processa j� no primeiro encontro
                        if frame_idx >= next_due_frame[tid]:
                            img_dense = processar_roi_image(roi_lb)  # <<< SUA INFER�NCIA AQUI
                            cv2.imshow('dense',img_dense)
                            label = classificar_imagem(img_dense)  # <<< SUA INFER�NCIA AQUI
                            # print(label)
                            # if label == "suspeitos":
                            #     idname = uuid.uuid4()
                            #     cv2.imwrite(f'suspeitas/{idname}.jpg',roi_lb)
                            #     print('imagem salva ',idname)
                            
                            label_by_id[tid] = label
                            next_due_frame[tid] = frame_idx + FRAME_STRIDE_PER_ID

                        # desenha bbox + label atual (se existir)
                        label_txt = label_by_id.get(tid, "...")
                        color = (0, 0, 255)  # vermelho padr�o
                        # op��o: cores por classe
                        if label_txt == "normal": color = (0, 200, 0)
                        elif label_txt == "roupa": color = (0, 128, 255)
                        elif label_txt == "saia": color = (255, 0, 200)
                        elif label_txt == "bolsa": color = (255, 128, 0)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        draw_label(frame, x1, y1, f"ID {tid}: {label_txt}", color=color)

            # FPS na tela
            if SHOW_FPS:
                t_now = time.time()
                dt = t_now - t_last
                t_last = t_now
                inst_fps = 1.0 / dt if dt > 0 else 0.0
                smoothed_fps = inst_fps if smoothed_fps is None else 0.9*smoothed_fps + 0.1*inst_fps
                cv2.putText(frame, f"FPS: {smoothed_fps:4.1f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

            cv2.imshow("YOLO + Rastreamento + ROI", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC ou q
                break

            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
