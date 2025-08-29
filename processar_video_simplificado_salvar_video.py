import os
import cv2
import time
import uuid
import numpy as np
from collections import deque, Counter
from ultralytics import YOLO

from processar_roi import processar_roi_image
from classify_roi import classificar_imagem

# ==================== CONFIG ====================
MODEL_PATH       = "yolo11n.pt"
VIDEO_PATH       = "videos/The Cashword Copping Caffeine Concealer.mp4"

skip_frames      = True    # False => processa todo frame | True => processa a cada N
detect_interval  = 4       # quando skip_frames=True, processa a cada N frames

ROI_MIN_AREA_PX  = 30000   # área mínima da ROI (w*h) para classificar
TRACKER_CFG      = "bytetrack.yaml"
PERSON_CLASS_ID  = 0       # apenas 'person' (COCO id=0)

SHOW_FPS         = True
DRAW_THICKNESS   = 3
FONT             = cv2.FONT_HERSHEY_SIMPLEX

# Evento: salvar 6s (3 antes + 3 depois)
PRE_SECONDS      = 3
POST_SECONDS     = 3
OUTPUT_DIR       = "eventos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Smoother: confirma suspeito com 3 consecutivas
SMOOTH_WIN_LABEL = 5
SUSPECT_CONSEC_N = 3
# =================================================

def letterbox_resize(img, out_w=640, out_h=480):
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

def draw_label(img, x1, y1, text, color=(0, 0, 255)):
    scale, thickness = 0.6, 2
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    pad = 4
    y1t = max(0, y1 - th - 2*pad)
    cv2.rectangle(img, (x1, y1t), (x1 + tw + 2*pad, y1), color, -1)
    cv2.putText(img, text, (x1 + pad, y1 - pad), FONT, scale, (255, 255, 255), thickness, cv2.LINE_AA)

class Smoother:
    """Suaviza rótulo e confirma 'suspeito' quando últimos N são 'suspeito'."""
    def __init__(self, label_win=5, suspect_win=3):
        self.label_hist   = {}  # tid -> deque(labels)
        self.suspect_hist = {}  # tid -> deque(bool)
        self.prev_confirm = {}  # tid -> bool (borda de subida)
        self.label_win    = label_win
        self.suspect_win  = suspect_win

    def update(self, tid, label):
        dq = self.label_hist.setdefault(tid, deque(maxlen=self.label_win))
        dq.append(label)
        s = self.suspect_hist.setdefault(tid, deque(maxlen=self.suspect_win))
        s.append(label == "suspeito")

    def smoothed_label(self, tid, default="..."):
        dq = self.label_hist.get(tid)
        if not dq:
            return default
        return Counter(dq).most_common(1)[0][0]

    def rising_confirm(self, tid):
        """True só no frame de transição para 'confirmado' (3/3)."""
        s = self.suspect_hist.get(tid)
        now = (s is not None and len(s) == s.maxlen and all(s))
        prev = self.prev_confirm.get(tid, False)
        self.prev_confirm[tid] = now
        return (now and not prev)

# dicionário de caixas desenhadas entre detecções:
# track_id -> (x1, y1, x2, y2, cor_BGR, classificacao)
bboxes_id = {}

# Modelo e vídeo
modelo = YOLO(MODEL_PATH)
video  = cv2.VideoCapture(VIDEO_PATH)

# Metadados para writer e buffers
fps_cap = video.get(cv2.CAP_PROP_FPS) or 30.0
W = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)  or 1280)
H = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
pre_maxlen = int(round(PRE_SECONDS * fps_cap)) + 2
pre_buffer = deque(maxlen=pre_maxlen)  # guarda frames já desenhados (pré-roll)

# Estado de gravação
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = None
post_left = 0
wrote_current_via_flush = False

smoother = Smoother(label_win=SMOOTH_WIN_LABEL, suspect_win=SUSPECT_CONSEC_N)

prev_time = time.time()
fps = 0.0
frame_count = 1

try:
    while True:
        ok, img = video.read()
        if not ok or img is None:
            break

        # padroniza tamanho de exibição/gravação
        img = cv2.resize(img, (W, H))
        draw = img.copy()

        # ================= FPS =================
        if SHOW_FPS:
            now = time.time()
            dt  = now - prev_time
            fps = 1.0 / dt if dt > 0 else 0.0
            prev_time = now

        # ======== DETECÇÃO/CLASSIFICAÇÃO (com skip) ========
        run_detection = (not skip_frames) or (frame_count % detect_interval == 0)
        event_triggered = False

        if run_detection:
            # limpa instantâneo e detecta neste frame
            bboxes_id.clear()

            resultados = modelo.track(
                draw,                # frame atual
                persist=True,
                tracker=TRACKER_CFG,
                verbose=False
            )

            for res in resultados:
                boxes = res.boxes
                if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
                    continue

                xyxys = boxes.xyxy.cpu().numpy()
                clss  = boxes.cls.int().cpu().numpy() if boxes.cls is not None else []
                ids_t = boxes.id
                ids   = ids_t.int().cpu().numpy() if ids_t is not None else None
                if ids is None:
                    continue

                for (x1, y1, x2, y2), cls_id, tid in zip(xyxys, clss, ids):
                    if tid is None or int(cls_id) != PERSON_CLASS_ID:
                        continue
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                    w_box = max(0, x2 - x1)
                    h_box = max(0, y2 - y1)
                    if w_box == 0 or h_box == 0 or (w_box * h_box) < ROI_MIN_AREA_PX:
                        continue

                    roi = img[y1:y2, x1:x2]
                    roi_lb = letterbox_resize(roi)
                    if roi_lb is None:
                        continue

                    # pipeline ROI -> classificação
                    roi_proc = processar_roi_image(roi_lb)
                    label = str(classificar_imagem(roi_proc)).strip().lower()
                    if label not in ("normal", "suspeito"):
                        label = "normal"

                    tid = int(tid)
                    smoother.update(tid, label)
                    show_label = smoother.smoothed_label(tid, default="...")
                    color = (0, 200, 0) if show_label == "normal" else (0, 0, 255) if show_label == "suspeito" else (180, 180, 180)

                    # guarda para desenhar em todos os frames até a próxima detecção
                    bboxes_id[tid] = (x1, y1, x2, y2, color, show_label)

                    # evento dispara quando confirmamos 3 consecutivas
                    if smoother.rising_confirm(tid):
                        event_triggered = True

        # ============== DESENHO (todo frame) ==============
        for track_id, (x1, y1, x2, y2, cor, classificacao) in bboxes_id.items():
            cv2.rectangle(draw, (x1, y1), (x2, y2), cor, DRAW_THICKNESS)
            cv2.putText(draw, f"{track_id} - {classificacao}",
                        (x1, max(0, y1 - 8)), FONT, 0.6, cor, 2, cv2.LINE_AA)

        if SHOW_FPS:
            cv2.putText(draw, f"FPS: {fps:.2f}", (10, 30), FONT, 1, (255, 255, 0), 2, cv2.LINE_AA)

        # ============== BUFFER/GRAVAÇÃO DE EVENTO ==============
        pre_buffer.append(draw.copy())

        if event_triggered:
            if writer is None:
                fname = f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}.mp4"
                out_path = os.path.join(OUTPUT_DIR, fname)
                writer = cv2.VideoWriter(out_path, fourcc, fps_cap, (W, H))
                # despeja pré-roll (inclui o frame atual já desenhado)
                for f in pre_buffer:
                    writer.write(f)
                post_left = int(round(POST_SECONDS * fps_cap))
                wrote_current_via_flush = True
                print("[EVENTO] Iniciado:", out_path)
            else:
                # já gravando: estende pós-roll
                post_left = max(post_left, int(round(POST_SECONDS * fps_cap)))

        if writer is not None:
            if not wrote_current_via_flush:
                writer.write(draw)
            else:
                wrote_current_via_flush = False

            post_left -= 1
            if post_left <= 0:
                writer.release()
                writer = None
                print("[EVENTO] Finalizado.")

        # ============== EXIBIÇÃO ==============
        cv2.imshow("YOLO + Rastreamento (skip opcional) + Eventos 3s/3s", draw)
        if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
            break

        frame_count += 1

finally:
    if writer is not None:
        writer.release()
    video.release()
    cv2.destroyAllWindows()
