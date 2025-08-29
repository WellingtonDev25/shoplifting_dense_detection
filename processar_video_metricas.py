import cv2
import time
import numpy as np
from ultralytics import YOLO
from processar_roi import processar_roi_image
from classify_roi import classificar_imagem

# ==================== CONFIG ====================
MODEL_PATH       = "yolo11n.pt"
VIDEO_PATH       = "videos/The Lemonade Looting Bearded Beggar.mp4"

skip_frames      = True    # False para detectar todo frame
detect_interval  = 4       # detectar a cada N frames quando skip_frames=True

ROI_MIN_SIZE     = 30000   # área mínima da ROI (pixels) para processar/classificar
TRACKER_CFG      = "bytetrack.yaml"
PERSON_CLASS_ID  = 0       # manter apenas pessoas

# ---- Throttle por ID (NOVO) ----
FRAME_STRIDE_PER_ID = 8    # roda Dense+Classif a cada N frames por track_id
# -------------------------------

SHOW_FPS         = True
SHOW_TIMES       = True
DRAW_THICKNESS   = 3
FONT             = cv2.FONT_HERSHEY_SIMPLEX
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

# dicionários de estado
# track_id -> (x1, y1, x2, y2, cor_BGR, classificacao)
bboxes_id = {}
# track_id -> último rótulo conhecido ("normal"/"suspeito")
last_label_by_id = {}
# track_id -> próximo frame_idx em que deve rodar Dense+Classif
next_dense_due = {}

# Modelo e vídeo
modelo = YOLO(MODEL_PATH)
video  = cv2.VideoCapture(VIDEO_PATH)

# ===== Acumuladores de tempo =====
perf = time.perf_counter  # relógio de alta resolução
# loop
loop_total = 0.0
loop_count = 0
# yolo
yolo_total = 0.0
yolo_calls = 0
# letterbox
lb_total = 0.0
lb_count = 0
# dense
dense_total = 0.0
dense_count = 0
# classificação
cls_total = 0.0
cls_count = 0

# ===== para overlay (último frame) =====
last_loop_ms  = 0.0
last_yolo_ms  = 0.0
last_lb_ms    = 0.0  # soma no frame
last_dense_ms = 0.0  # soma no frame
last_cls_ms   = 0.0  # soma no frame
last_rois     = 0    # qtd de ROIs processadas no frame

# FPS
prev_time = time.time()
fps = 0.0
frame_count = 1

try:
    while True:
        loop_t0 = perf()

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

        # zera métricas do frame para ROI
        last_lb_ms = 0.0
        last_dense_ms = 0.0
        last_cls_ms = 0.0
        last_rois = 0
        last_yolo_ms = 0.0

        if run_detection:
            # limpa o snapshot e repovoa neste frame de detecção
            bboxes_id.clear()

            yolo_t0 = perf()
            resultados = modelo.track(
                img,
                persist=True,
                tracker=TRACKER_CFG,
                verbose=False
            )
            yolo_dt = perf() - yolo_t0
            yolo_total += yolo_dt
            yolo_calls += 1
            last_yolo_ms = yolo_dt * 1000.0

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
                    if area_px <= ROI_MIN_SIZE:
                        continue

                    tid = int(track_id)

                    # --------- THROTTLE POR ID (NOVO) ---------
                    # processa Dense+Classif somente se "venceu o prazo" deste ID
                    if frame_count >= next_dense_due.get(tid, 0):
                        # letterbox timing
                        lb_t0 = perf()
                        roi_lb = letterbox_resize(roi)
                        lb_dt = perf() - lb_t0
                        if roi_lb is None:
                            continue
                        lb_total += lb_dt
                        lb_count += 1
                        last_lb_ms += lb_dt * 1000.0

                        # "dense" timing (processar_roi_image)
                        dense_t0 = perf()
                        roi_proc = processar_roi_image(roi_lb)
                        dense_dt = perf() - dense_t0
                        dense_total += dense_dt
                        dense_count += 1
                        last_dense_ms += dense_dt * 1000.0

                        # classificação timing
                        cls_t0 = perf()
                        classificacao = classificar_imagem(roi_proc)  # "normal" | "suspeito"
                        cls_dt = perf() - cls_t0
                        cls_total += cls_dt
                        cls_count += 1
                        last_cls_ms += cls_dt * 1000.0
                        last_rois += 1

                        # salva rótulo e agenda próximo processamento para este ID
                        last_label_by_id[tid] = classificacao
                        next_dense_due[tid] = frame_count + FRAME_STRIDE_PER_ID
                    else:
                        # reutiliza último rótulo conhecido (sem rodar Dense/Classif)
                        classificacao = last_label_by_id.get(tid, "normal")
                    # ------------------------------------------

                    # cor por classificação (exemplo simples)
                    cor = (0, 255, 0) if classificacao == "normal" else (0, 0, 255)

                    # armazena para desenhar em todos os frames até a próxima detecção
                    bboxes_id[tid] = (x1, y1, x2, y2, cor, classificacao)

        # ============== DESENHO (todo frame) ==============
        for track_id, (x1, y1, x2, y2, cor, classificacao) in bboxes_id.items():
            cv2.rectangle(imgShow, (x1, y1), (x2, y2), cor, DRAW_THICKNESS)
            cv2.putText(imgShow, f"{track_id} - {classificacao}",
                        (x1, max(0, y1 - 8)), FONT, 0.6, cor, 2, cv2.LINE_AA)

        # ===== Tempo total do loop =====
        last_loop_ms = (perf() - loop_t0) * 1000.0
        loop_total  += last_loop_ms / 1000.0
        loop_count  += 1

        # ===== Overlay de métricas =====
        if SHOW_FPS or SHOW_TIMES:
            y = 30
            dy = 22
            if SHOW_FPS:
                cv2.putText(imgShow, f"FPS: {fps:.2f}", (10, y), FONT, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                y += dy
            if SHOW_TIMES:
                cv2.putText(imgShow, f"Loop: {last_loop_ms:.1f} ms", (10, y), FONT, 0.6, (0, 255, 0), 2, cv2.LINE_AA); y += dy
                cv2.putText(imgShow, f"YOLO: {last_yolo_ms:.1f} ms", (10, y), FONT, 0.6, (0, 255, 255), 2, cv2.LINE_AA); y += dy
                if last_rois > 0:
                    cv2.putText(imgShow, f"Letterbox: {last_lb_ms:.1f} ms (avg {last_lb_ms/last_rois:.1f})", (10, y),
                                FONT, 0.6, (0, 200, 255), 2, cv2.LINE_AA); y += dy
                    cv2.putText(imgShow, f"Dense: {last_dense_ms:.1f} ms (avg {last_dense_ms/last_rois:.1f})", (10, y),
                                FONT, 0.6, (255, 200, 0), 2, cv2.LINE_AA); y += dy
                    cv2.putText(imgShow, f"Classif: {last_cls_ms:.1f} ms (avg {last_cls_ms/last_rois:.1f})", (10, y),
                                FONT, 0.6, (255, 0, 200), 2, cv2.LINE_AA); y += dy
                else:
                    cv2.putText(imgShow, "ROIs: 0", (10, y), FONT, 0.6, (200, 200, 200), 2, cv2.LINE_AA); y += dy

        # ============== EXIBIÇÃO ==============
        cv2.imshow("YOLOv8 + Rastreamento (timings + throttle por ID)", imgShow)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC ou 'q'
            break

        frame_count += 1

finally:
    video.release()
    cv2.destroyAllWindows()

    # ======= RESUMO FINAL =======
    def safe_avg(total, count): return (total / count) if count > 0 else 0.0
    avg_loop_ms = safe_avg(loop_total * 1000.0, loop_count)
    avg_yolo_ms = safe_avg(yolo_total * 1000.0, yolo_calls)
    avg_lb_ms   = safe_avg(lb_total * 1000.0, lb_count)
    avg_dense_ms= safe_avg(dense_total * 1000.0, dense_count)
    avg_cls_ms  = safe_avg(cls_total * 1000.0, cls_count)

    print("\n========== RESUMO DE TEMPOS ==========")
    print(f"Frames processados: {loop_count}")
    print(f"Chamadas YOLO:      {yolo_calls}")
    print(f"ROIs processadas:   {lb_count} (letterbox) | {dense_count} (dense) | {cls_count} (classif)")
    print("--------------------------------------")
    print(f"Média Loop:         {avg_loop_ms:.2f} ms  (~{1000.0/max(avg_loop_ms,1e-6):.1f} FPS)")
    print(f"Média YOLO:         {avg_yolo_ms:.2f} ms por chamada")
    print(f"Média Letterbox:    {avg_lb_ms:.2f} ms por ROI")
    print(f"Média Dense:        {avg_dense_ms:.2f} ms por ROI")
    print(f"Média Classificação:{avg_cls_ms:.2f} ms por ROI")
    print("======================================\n")
