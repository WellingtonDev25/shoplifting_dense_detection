import cv2
import time
import numpy as np
from ultralytics import YOLO
from processar_roi import processar_roi_image
from classify_roi import classificar_imagem

# ==================== CONFIG ====================
MODEL_PATH      = "yolo11n.pt"
VIDEO_PATH      = "videos/The Cashword Copping Caffeine Concealer.mp4"

skip_frames     = True   # False para detectar todo frame
detect_interval = 4      # detectar a cada N frames quando skip_frames=True

ROI_MIN_SIZE    = 30000  # área mínima da ROI (pixels) para processar/classificar
TRACKER_CFG     = "bytetrack.yaml"
PERSON_CLASS_ID = 0      # manter apenas pessoas

SHOW_FPS        = True
DRAW_THICKNESS  = 3
FONT            = cv2.FONT_HERSHEY_SIMPLEX
# =================================================

def letterbox_resize(img, out_w=640, out_h=480):
    """Redimensiona mantendo proporção e preenche até (out_w,out_h)."""
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return None
    scale = min(out_w / w, out_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    top  = (out_h - nh) // 2
    left = (out_w - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas

# dicionário: track_id -> (x1, y1, x2, y2, cor_BGR, classificacao)
bboxes_id = {}

# Modelo e vídeo
modelo = YOLO(MODEL_PATH)
video  = cv2.VideoCapture(VIDEO_PATH)

prev_time = time.time()
fps = 0.0
frame_count = 1

try:
    while True:
        ok, img = video.read()
        if not ok or img is None:
            break

        # opcional: padronizar tamanho de exibição
        img = cv2.resize(img, (1280, 720))
        imgShow = img.copy()

        # ================= FPS =================
        if SHOW_FPS:
            now = time.time()
            dt  = now - prev_time
            fps = 1.0 / dt if dt > 0 else 0.0
            prev_time = now

        # ======== DETECÇÃO (conforme intervalo) ========
        run_detection = (not skip_frames) or (frame_count % detect_interval == 0)
        if run_detection:
            # limpa o snapshot e repovoa neste frame de detecção
            bboxes_id.clear()

            resultados = modelo.track(
                img,
                persist=True,
                tracker=TRACKER_CFG,
                verbose=False
            )

            # results do ultralytics para 1 frame: lista com 1 Results
            for res in resultados:
                boxes = res.boxes
                if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
                    continue

                xyxys = boxes.xyxy.cpu().numpy()              # (N, 4)
                clss  = boxes.cls.int().cpu().numpy()         # (N,)
                ids_t = boxes.id                               # pode ser None
                ids   = ids_t.int().cpu().numpy() if ids_t is not None else None

                if ids is None:
                    # sem IDs => nada a rastrear neste frame
                    continue

                # percorre detecções lado a lado
                for (x1, y1, x2, y2), cls_id, track_id in zip(xyxys, clss, ids):
                    # ignora sem ID válido (só por segurança)
                    if track_id is None:
                        continue

                    # mantém apenas pessoas (ajuste se quiser outras classes)
                    if int(cls_id) != PERSON_CLASS_ID:
                        continue

                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    # ROI
                    roi = img[y1:y2, x1:x2]
                    h, w = roi.shape[:2]
                    area_px = h * w  # área real em pixels
                    if area_px <= 30000:
                        continue

                    roi_lb   = letterbox_resize(roi)
                    if roi_lb is None:
                        continue

                    # seu pipeline de ROI
                    roi_proc      = processar_roi_image(roi_lb)
                    classificacao = classificar_imagem(roi_proc)  # ex.: "normal" | "suspeitos" | etc.

                    # cor por classificação (exemplo simples)
                    cor = (0, 255, 0) if classificacao == "normal" else (0, 0, 255)

                    # ✅ usa track_id como chave (e não 'id' do Python)
                    bboxes_id[int(track_id)] = (x1, y1, x2, y2, cor, classificacao)

        # ============== DESENHO (todo frame) ==============
        for track_id, (x1, y1, x2, y2, cor, classificacao) in bboxes_id.items():
            cv2.rectangle(imgShow, (x1, y1), (x2, y2), cor, DRAW_THICKNESS)
            cv2.putText(imgShow, f"{track_id} - {classificacao}",
                        (x1, max(0, y1 - 8)), FONT, 0.6, cor, 2, cv2.LINE_AA)

        if SHOW_FPS:
            cv2.putText(imgShow, f"FPS: {fps:.2f}", (10, 30), FONT, 1, (255, 255, 0), 2, cv2.LINE_AA)

        # ============== EXIBIÇÃO ==============
        cv2.imshow("YOLOv8 + Rastreamento", imgShow)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC ou 'q'
            break

        frame_count += 1

finally:
    video.release()
    cv2.destroyAllWindows()
