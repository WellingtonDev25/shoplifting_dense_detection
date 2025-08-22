import cv2
import numpy as np
import torch
import time
import queue
from collections import deque, defaultdict
from threading import Lock
from multiprocessing import Process, Queue, Event, set_start_method, Manager

from ultralytics import YOLO
from processar_roi import processar_roi_image
from classify_roi import classificar_imagem

# ================== CONFIG ==================
VIDEO_PATH            = "videos/Apenas suspeitos.mp4"   # ou RTSP
MODEL_PATH            = "yolo11n.pt"
CONF_THRESH           = 0.35
TRACKER_CFG           = "bytetrack.yaml"
TRACK_CLASSES         = [0]                     # ex.: [0] só 'person'; None = todas
FRAME_STRIDE_PER_ID   = 3                        # enfileira a cada N frames por ID
ROI_OUT_W, ROI_OUT_H  = 640, 480
SHOW_FPS              = True

# GPU worker (micro-batch)
BATCH_SIZE            = 8                        # até N ROIs por rodada
BATCH_MAX_DELAY_MS    = 40                       # ou processa ao atingir esse tempo
QUEUE_MAX_IN          = 256                      # capacidade da fila de entrada
QUEUE_MAX_OUT         = 256                      # capacidade da fila de saída

# Smoothing
SMOOTH_K              = 5                        # últimos K resultados por ID

# Limpeza
IDLE_CLEANUP_EVERY    = 30                       # frames
# ============================================


# ----------------- util -----------------
def letterbox_resize(img, out_w, out_h):
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

def draw_label(img, x1, y1, text, color=(0, 0, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.6, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad = 4
    cv2.rectangle(img, (x1, max(0, y1 - th - 2 * pad)), (x1 + tw + 2 * pad, y1), color, -1)
    cv2.putText(img, text, (x1 + pad, y1 - pad), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

class SmootherK:
    def __init__(self, k=5):
        self.buf = deque(maxlen=k)
    def update(self, label):
        self.buf.append(label)
    def result(self, default="..."):
        if not self.buf: return default
        counts = defaultdict(int)
        for x in self.buf: counts[x] += 1
        max_cnt = max(counts.values())
        candidatos = [x for x, c in counts.items() if c == max_cnt]
        return candidatos[0] if len(candidatos) == 1 else self.buf[-1]


# --------------- WORKER (processo) ---------------
def gpu_worker(in_q: Queue, out_q: Queue, stop_evt: Event):
    """
    Recebe (tid:int, ts:float, roi:np.ndarray BGR 640x480).
    Mantém só o mais recente por ID e processa em micro-batches.
    Envia (tid:int, ts_proc:float, label:str).
    """
    # Se suas funções usam torch internamente, ative benchmarks e warmup:
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    last_by_id = {}  # tid -> (ts, roi)

    # Warmup opcional (crie um dummy ROI para “acordar” kernels/cuDNN)
    dummy = np.zeros((ROI_OUT_H, ROI_OUT_W, 3), np.uint8)
    try:
        with torch.no_grad(), torch.cuda.amp.autocast():
            _ = classificar_imagem(processar_roi_image(dummy))
    except Exception:
        pass

    def drain_until_batch(deadline_ms):
        """Agrupa o mais recente de cada ID até BATCH_SIZE ou estourar deadline."""
        nonlocal last_by_id
        t0 = time.time() * 1000.0
        # Pegar pelo menos 1
        while True:
            remaining = max(1.0, deadline_ms - (time.time() * 1000.0 - t0))
            try:
                item = in_q.get(timeout=remaining / 1000.0)
                if item is None:  # sentinela
                    return None
                tid, ts, roi = item
                last_by_id[tid] = (ts, roi)  # mantém só o mais recente por ID
                # Tente preencher até BATCH_SIZE enquanto há tempo
                if len(last_by_id) >= BATCH_SIZE:
                    break
            except queue.Empty:
                break  # estourou tempo
        # Seleciona até BATCH_SIZE IDs mais recentes
        if not last_by_id:
            return []
        items = sorted(last_by_id.items(), key=lambda kv: kv[1][0], reverse=True)[:BATCH_SIZE]
        # Remove selecionados do buffer
        for tid, _ in items:
            last_by_id.pop(tid, None)
        # Retorna lista [(tid, ts, roi), ...]
        return [(tid, ts_roi[0], ts_roi[1]) for tid, ts_roi in items]

    while not stop_evt.is_set():
        batch = drain_until_batch(BATCH_MAX_DELAY_MS)
        if batch is None:
            break  # encerrando
        if not batch:
            continue

        # Processa lote (sequencial ou batched, dependendo do seu código)
        # Aqui assumo que suas funções aceitam 1 ROI por vez.
        results = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for (tid, ts, roi_lb) in batch:
                try:
                    img_dense = processar_roi_image(roi_lb)        # GPU
                    label = classificar_imagem(img_dense)          # GPU
                except Exception:
                    label = "erro"
                results.append((tid, time.time(), label))

        # Publica resultados
        for tid, ts_proc, label in results:
            try:
                out_q.put_nowait((tid, ts_proc, label))
            except queue.Full:
                # Se a saída estiver cheia, preferimos descartar o mais antigo:
                try:
                    out_q.get_nowait()
                    out_q.put_nowait((tid, ts_proc, label))
                except queue.Empty:
                    pass


# --------------- PRODUTOR (rastreamento + render) ---------------
def main():
    set_start_method("spawn", force=True)

    # Modelo de rastreamento
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(MODEL_PATH)
    try:
        model.to(device)
    except Exception:
        pass

    print(f"[INFO] Abrindo {VIDEO_PATH} | tracker={TRACKER_CFG} | device={device}")

    # Filas p/ comunicação c/ worker
    in_q  = Queue(maxsize=QUEUE_MAX_IN)
    out_q = Queue(maxsize=QUEUE_MAX_OUT)
    stop_evt = Event()

    # Lança processo worker
    worker = Process(target=gpu_worker, args=(in_q, out_q, stop_evt), daemon=True)
    worker.start()

    # Estados no produtor
    next_due_frame = {}                    # id -> próximo frame_idx para enfileirar
    smoothers = defaultdict(lambda: SmootherK(SMOOTH_K))
    last_label = {}                        # id -> string
    last_seen  = {}                        # id -> ts (para limpeza)
    frame_idx = 0
    smoothed_fps = None
    t_last = time.time()

    # Stream de rastreamento (1 Results por frame)
    stream = model.track(
        source=VIDEO_PATH,
        stream=True,
        conf=CONF_THRESH,
        classes=TRACK_CLASSES,
        tracker=TRACKER_CFG,
        verbose=False,
        persist=True,
    )

    try:
        for res in stream:
            frame = res.orig_img
            if frame is None:
                break
            h, w = frame.shape[:2]

            # 1) Drena resultados do worker sem bloquear (atualiza smoothing)
            while True:
                try:
                    tid, ts_proc, label = out_q.get_nowait()
                except queue.Empty:
                    break
                last_label[tid] = label
                smoothers[tid].update(label)
                last_seen[tid] = time.time()

            # 2) Coleta detecções e enfileira ROIs com stride por ID
            boxes = res.boxes
            if boxes is not None and boxes.xyxy is not None:
                xyxys = boxes.xyxy.int().cpu().numpy()
                ids = boxes.id
                ids = ids.int().cpu().numpy() if ids is not None else None

                if ids is not None:
                    for xyxy, tid in zip(xyxys, ids):
                        x1, y1, x2, y2 = xyxy
                        x1 = max(0, min(x1, w - 1)); y1 = max(0, min(y1, h - 1))
                        x2 = max(0, min(x2, w - 1)); y2 = max(0, min(y2, h - 1))
                        if x2 <= x1 or y2 <= y1: 
                            continue

                        roi = frame[y1:y2, x1:x2]
                        if roi.size == 0:
                            continue
                        roi_lb = letterbox_resize(roi, ROI_OUT_W, ROI_OUT_H)
                        if roi_lb is None:
                            continue

                        # Stride por ID
                        if tid not in next_due_frame:
                            next_due_frame[tid] = 0
                        if frame_idx >= next_due_frame[tid]:
                            # Envia para worker, sem bloquear o render
                            try:
                                # (opção) comprimir: ok, mas adiciona latência e CPU
                                # ok, buf = cv2.imencode('.jpg', roi_lb, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1]
                                in_q.put_nowait((int(tid), time.time(), roi_lb))
                                # in_q.put_nowait((int(tid), time.time(), buf))  # se for mandar JPEG
                            except queue.Full:
                                pass  # se cheio, dropa — worker já mantém só o mais recente por ID
                            next_due_frame[tid] = frame_idx + FRAME_STRIDE_PER_ID

                        # Desenha bbox + label suavizado atual
                        label_txt = smoothers[tid].result(default=last_label.get(tid, "..."))

                        color = (0, 0, 255)
                        if label_txt == "normal": color = (0, 200, 0)
                        elif label_txt == "roupa": color = (0, 128, 255)
                        elif label_txt == "saia": color = (255, 0, 200)
                        elif label_txt == "bolsa": color = (255, 128, 0)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        draw_label(frame, x1, y1, f"ID {int(tid)}: {label_txt}", color=color)
                        last_seen[tid] = time.time()

            # 3) FPS
            if SHOW_FPS:
                t_now = time.time()
                dt = t_now - t_last
                t_last = t_now
                inst_fps = 1.0 / dt if dt > 0 else 0.0
                smoothed_fps = inst_fps if smoothed_fps is None else 0.9 * smoothed_fps + 0.1 * inst_fps
                cv2.putText(frame, f"FPS: {smoothed_fps:4.1f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

            # 4) Render
            cv2.imshow("YOLO + Rastreamento (produtor) + Micro-batch GPU (worker)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

            frame_idx += 1
            if frame_idx % IDLE_CLEANUP_EVERY == 0:
                # opcional: remover smoothers antigos (IDs que sumiram)
                now = time.time()
                ids_to_del = [tid for tid, ts in last_seen.items() if (now - ts) > 10.0]
                for tid in ids_to_del:
                    smoothers.pop(tid, None)
                    last_label.pop(tid, None)
                    last_seen.pop(tid, None)

    finally:
        stop_evt.set()
        try:
            in_q.put_nowait(None)  # sentinela
        except queue.Full:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
