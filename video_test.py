import cv2
import time
from utils.helper import GetLogger, Predictor

# =========================
# CONFIGURAÇÃO
# =========================
INPUT_SOURCE = "video.mp4"   # pode ser "video.mp4", 0 (webcam) ou "rtsp://..."
SHOW_SEGMENT = True          # True = mostra out_frame_seg; False = mostra out_frame
RESIZE_TO    = None          # (largura, altura), ex: (1280, 720) ou None
WINDOW_NAME  = "Preview em tempo real"

# =========================
# SETUP
# =========================
logger = GetLogger.logger(__name__)
predictor = Predictor()

cap = cv2.VideoCapture(INPUT_SOURCE)
if not cap.isOpened():
    raise RuntimeError(f"Não foi possível abrir: {INPUT_SOURCE}")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

paused = False
frame_idx = 0
t0 = time.time()
t_last = t0

def maybe_resize(img, size_wh):
    if size_wh is None: return img
    w, h = size_wh
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

# =========================
# LOOP
# =========================
while True:
    if not paused:
        ok, frame = cap.read()
        if not ok:
            logger.info("Fim do vídeo ou falha na leitura.")
            break

        out_frame, out_frame_seg = predictor.predict(frame)
        vis = out_frame_seg if SHOW_SEGMENT else out_frame
        vis = maybe_resize(vis, RESIZE_TO)

        # FPS instantâneo
        now = time.time()
        dt = max(now - t_last, 1e-6)
        t_last = now
        inst_fps = 1.0 / dt
        cv2.putText(vis, f"FPS: {inst_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, vis)
        frame_idx += 1

    key = cv2.waitKey(1) & 0xFF
    if key == 27:       # ESC
        break
    elif key == ord(' '):
        paused = not paused

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()

elapsed = max(time.time() - t0, 1e-6)
logger.info(f"Frames: {frame_idx} | Tempo: {elapsed:.2f}s | FPS médio: {frame_idx/elapsed:.2f}")
