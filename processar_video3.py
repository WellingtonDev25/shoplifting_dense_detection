import cv2
import numpy as np
from ultralytics import YOLO
import torch
from tqdm import tqdm
import time
from processar_roi import processar_roi_image
from classify_roi import classificar_imagem

# ================== CONFIG ==================
# VIDEO_PATH          = "rtsp://admin:Ii3432841296@192.168.1.21:554/cam/realmonitor?channel=1&subtype=0" 
VIDEO_PATH           = "videos/rob02.mp4"   # caminho do vídeo ou RTSP
MODEL_PATH           = "yolo11n-pose.pt"        # modelo YOLO de pose
CONF_THRESH          = 0.35                     # confiança mínima
FRAME_STRIDE_PER_ID  = 4                        # chama processar_roi a cada N frames POR TRACK ID
ROI_OUT_W, ROI_OUT_H = 640, 480                 # tamanho do ROI após letterbox
SHOW_FPS             = True
TRACKER_CFG          = "bytetrack.yaml"         # ou "botsort.yaml"

# Desenho de keypoints
cor_linha      = (0, 255, 0)
cor_ponto      = (0, 0, 255)
cor_braco      = (255, 0, 0)
cor_mao        = (0, 255, 255)
espessura_linha = 2
raio_ponto      = 3
raio_mao        = 5
padding_bbox    = 20   # padding extra no recorte
margem_expand   = 0    # se quiser expandir além do bbox (0 = desativado)

# Skeleton COCO-17 (pares de índices)
skeleton = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 1), (1, 3), (0, 2), (2, 4)
]
pares_braco = {(5,7), (7,9), (6,8), (8,10)}  # pares considerados "braço/mão"

# ============================================

def letterbox_resize(img, out_w, out_h):
    """Redimensiona preservando proporção e preenche com bordas (preto) até (out_w,out_h)."""
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

def draw_label(img, x1, y1, text, color=(0,0,255)):
    """Retângulo de fundo + texto legível."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.6, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 4
    cv2.rectangle(img, (x1, max(0, y1 - th - 2*pad)), (x1 + tw + 2*pad, y1), color, -1)
    cv2.putText(img, text, (x1 + pad, y1 - pad), font, scale, (255,255,255), thickness, cv2.LINE_AA)

def desenhar_keypoints(frame_disp, pontos_xy):
    """
    pontos_xy: np.ndarray shape (17, 2) em coordenadas da imagem.
    Desenha linhas do esqueleto e pontos (mãos com cor/raio diferentes).
    """
    # linhas
    for i, j in skeleton:
        pi = pontos_xy[i]; pj = pontos_xy[j]
        if pi[0] > 0 and pi[1] > 0 and pj[0] > 0 and pj[1] > 0:
            pt1 = (int(pi[0]), int(pi[1]))
            pt2 = (int(pj[0]), int(pj[1]))
            cor = cor_braco if ((i, j) in pares_braco or (j, i) in pares_braco) else cor_linha
            cv2.line(frame_disp, pt1, pt2, cor, espessura_linha, cv2.LINE_AA)

    # pontos
    for idx, (x, y) in enumerate(pontos_xy):
        if x > 0 and y > 0:
            centro = (int(x), int(y))
            if idx in [9, 10]:  # mãos
                cv2.circle(frame_disp, centro, raio_mao, cor_mao, -1, cv2.LINE_AA)
            else:
                cv2.circle(frame_disp, centro, raio_ponto, cor_ponto, -1, cv2.LINE_AA)

def main():
    # Metadados do vídeo (para barra de progresso)
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

    print(f"[INFO] Abrindo {VIDEO_PATH} | FPS≈{fps_cap:.2f} | tracker={TRACKER_CFG} | modelo={MODEL_PATH}")

    # Estados por track id
    next_due_frame = {}   # id -> próximo frame_idx em que deve processar
    label_by_id    = {}   # id -> último rótulo conhecido

    frame_idx = 0
    t_last = time.time()
    smoothed_fps = None

    pbar = tqdm(total=total_frames if total_frames > 0 else None,
                unit="frame", dynamic_ncols=True, desc="Rastreando")

    # Stream com rastreamento + pose; classes=[0] => pessoa
    stream = model.track(
        source=VIDEO_PATH,
        stream=True,
        conf=CONF_THRESH,
        classes=[0],
        tracker=TRACKER_CFG,
        verbose=False,
        persist=True
    )

    try:
        for res in stream:
            frame_orig = res.orig_img  # BGR (sem overlays)
            if frame_orig is None:
                break
            h, w = frame_orig.shape[:2]

            # Vamos desenhar manualmente sobre uma cópia para exibição
            frame_disp = frame_orig.copy()

            boxes = res.boxes
            kps   = res.keypoints  # Keypoints object (pode ser None)
            if boxes is not None and boxes.xyxy is not None and kps is not None and kps.xy is not None:
                xyxys = boxes.xyxy.cpu().numpy()
                ids = boxes.id
                ids = ids.int().cpu().numpy() if ids is not None else None
                kps_xy = kps.xy.cpu().numpy()  # shape (N, 17, 2)

                if ids is not None:
                    for idx_det, (xyxy, tid) in enumerate(zip(xyxys, ids)):
                        # keypoints desse det
                        pontos = kps_xy[idx_det]  # (17,2), float
                        # desenhar keypoints na imagem de exibição
                        desenhar_keypoints(frame_disp, pontos)

                        # ROI com base no bbox da pessoa (pode usar keypoints min/max + padding)
                        x1, y1, x2, y2 = map(int, xyxy)
                        # opcional: usar keypoints para bbox mais justo
                        pontos_validos = pontos[(pontos[:,0] > 0) & (pontos[:,1] > 0)]
                        if len(pontos_validos) > 0:
                            kx_min, ky_min = pontos_validos.min(axis=0).astype(int)
                            kx_max, ky_max = pontos_validos.max(axis=0).astype(int)
                            x1 = min(x1, kx_min); y1 = min(y1, ky_min)
                            x2 = max(x2, kx_max); y2 = max(y2, ky_max)

                        # padding e margem
                        x1 = max(x1 - padding_bbox - margem_expand, 0)
                        y1 = max(y1 - padding_bbox - margem_expand, 0)
                        x2 = min(x2 + padding_bbox + margem_expand, w)
                        y2 = min(y2 + padding_bbox + margem_expand, h)
                        if x2 <= x1 or y2 <= y1:
                            continue

                        # Recorte NA IMAGEM ORIGINAL
                        roi = frame_orig[y1:y2, x1:x2]
                        if roi.size == 0:
                            continue
                        roi_lb = letterbox_resize(roi, ROI_OUT_W, ROI_OUT_H)
                        if roi_lb is None:
                            continue

                        # Chamadas ao pipeline somente a cada N frames por ID
                        if tid not in next_due_frame:
                            next_due_frame[tid] = 0
                        if frame_idx >= next_due_frame[tid]:
                            img_dense = processar_roi_image(roi_lb)
                            label = classificar_imagem(img_dense)
                            label_by_id[tid] = label
                            next_due_frame[tid] = frame_idx + FRAME_STRIDE_PER_ID

                        # Desenhar caixa e rótulo
                        label_txt = label_by_id.get(tid, "...")
                        color = (0, 0, 255)
                        if label_txt == "normal": color = (0, 200, 0)
                        elif label_txt == "roupa": color = (0, 128, 255)
                        elif label_txt == "saia": color = (255, 0, 200)
                        elif label_txt == "bolsa": color = (255, 128, 0)

                        cv2.rectangle(frame_disp, (x1, y1), (x2, y2), color, 2)
                        draw_label(frame_disp, x1, y1, f"ID {tid}: {label_txt}", color=color)

            # FPS
            if SHOW_FPS:
                t_now = time.time()
                dt = t_now - t_last
                t_last = t_now
                inst_fps = 1.0 / dt if dt > 0 else 0.0
                smoothed_fps = inst_fps if smoothed_fps is None else 0.9*smoothed_fps + 0.1*inst_fps
                cv2.putText(frame_disp, f"FPS: {smoothed_fps:4.1f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

            cv2.imshow("YOLO Pose + Rastreamento (keypoints manuais) + ROI original", frame_disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
