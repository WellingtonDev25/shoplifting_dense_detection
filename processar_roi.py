# processar_roi.py
import os
import cv2
import numpy as np
from threading import Lock
from utils.helper import GetLogger, Predictor

_logger = GetLogger.logger(__name__)

# =======================
# PARÂMETROS DE DESEMPENHO
# =======================
# Tamanho usado APENAS na inferência (downscale); a saída continua 640x480.
# Teste (384, 288) e, se precisar de mais FPS, tente (320, 240).
_INFER_W, _INFER_H = (384, 288)

# Pular frames: roda DensePose a cada N frames e reaproveita a última saída nos intermediários.
_FRAME_STRIDE = 1  # 2 = infere a cada 2 frames. Ajuste para 3 se precisar de mais FPS.

# Permitir override via variáveis de ambiente (opcional)
try:
    _env_size = os.getenv("DENSEPOSE_INFER_SIZE", "").strip().lower()
    if "x" in _env_size:
        _INFER_W, _INFER_H = tuple(int(x) for x in _env_size.split("x"))
except Exception:
    pass

try:
    _env_stride = os.getenv("DENSEPOSE_FRAME_STRIDE")
    if _env_stride:
        _FRAME_STRIDE = max(1, int(_env_stride))
except Exception:
    pass

# =======================
# LAZY INIT DO PREDICTOR
# =======================
_predictor = None
_predictor_init_lock = Lock()

# Serializa a inferência para evitar contenção em GPU com múltiplas threads chamando ao mesmo tempo
_infer_lock = Lock()

# Cache do último resultado (mantemos as duas variantes para respeitar show_segment)
_last_out_frame = None      # denso com anotações
_last_out_frame_seg = None  # denso segmentado
_frame_counter = 0


def _get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        with _predictor_init_lock:
            if _predictor is None:
                _predictor = Predictor()
                _logger.info("Predictor DensePose inicializado.")
    return _predictor


def _ensure_640x480(bgr_img: np.ndarray) -> np.ndarray:
    """Garante e retorna imagem BGR 640x480 (contígua em memória)."""
    if bgr_img is None:
        raise ValueError("Imagem None recebida.")
    if not isinstance(bgr_img, np.ndarray):
        raise TypeError("A imagem deve ser um numpy.ndarray (BGR).")

    h, w = bgr_img.shape[:2]
    if (w, h) != (640, 480):
        img = cv2.resize(bgr_img, (640, 480), interpolation=cv2.INTER_AREA)
        _logger.debug(f"Redimensionado para 640x480 (recebido {w}x{h}).")
    else:
        img = bgr_img
    # Evita cópias internas no framework garantindo contiguidade
    return np.ascontiguousarray(img)


def _downscale_for_infer(bgr_640x480: np.ndarray) -> np.ndarray:
    """Downscale para resolução de inferência (contígua)."""
    small = cv2.resize(bgr_640x480, (_INFER_W, _INFER_H), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(small)


def _to_uint8(bgr_img: np.ndarray) -> np.ndarray:
    if bgr_img.dtype != np.uint8:
        bgr_img = np.clip(bgr_img, 0, 255).astype(np.uint8)
    return bgr_img


def processar_roi_image(img_bgr_640x480: np.ndarray, show_segment: bool = True) -> np.ndarray:
    """
    Aplica DensePose em uma imagem BGR 640x480 e retorna a imagem com o efeito.

    - Roda a inferência em resolução menor (_INFER_W x _INFER_H) para ganhar FPS e faz upscale.
    - Pula frames (FRAME_STRIDE) reaproveitando a última saída.
    - Mantém cache de duas variantes: segmentada e anotada.

    Parâmetros
    ----------
    img_bgr_640x480 : np.ndarray
        Imagem BGR (OpenCV) de tamanho 640x480 (ou será redimensionada).
    show_segment : bool, opcional (default=True)
        Se True, retorna o frame segmentado (out_frame_seg).
        Se False, retorna o frame com as anotações padrão (out_frame).

    Retorna
    -------
    np.ndarray
        Imagem BGR com o efeito DensePose aplicado (uint8, 640x480).
    """
    global _last_out_frame, _last_out_frame_seg, _frame_counter

    # 1) Normaliza entrada e decide se infere ou usa cache
    img_in_640x480 = _ensure_640x480(img_bgr_640x480)

    _frame_counter += 1
    do_infer = (_frame_counter % _FRAME_STRIDE) == 0

    # Se não for inferir neste frame, devolve o cache se existir
    if not do_infer:
        if show_segment and _last_out_frame_seg is not None:
            return _last_out_frame_seg
        if (not show_segment) and _last_out_frame is not None:
            return _last_out_frame
        # Se ainda não existe cache (ex.: primeiros frames), cai para inferência

    # 2) Inferência (serializada para evitar contenção)
    with _infer_lock:
        predictor = _get_predictor()
        small = _downscale_for_infer(img_in_640x480)

        # Chamada ao modelo (espera-se que aceite qualquer tamanho razoável)
        out_frame_small, out_frame_seg_small = predictor.predict(small)

        # 3) Upscale de volta para 640x480
        out_frame = cv2.resize(out_frame_small, (640, 480), interpolation=cv2.INTER_LINEAR)
        out_frame_seg = cv2.resize(out_frame_seg_small, (640, 480), interpolation=cv2.INTER_LINEAR)

        # 4) Normaliza tipo
        out_frame = _to_uint8(out_frame)
        out_frame_seg = _to_uint8(out_frame_seg)

        # 5) Atualiza cache
        _last_out_frame = out_frame
        _last_out_frame_seg = out_frame_seg

        # 6) Retorna conforme solicitado
        return out_frame_seg if show_segment else out_frame
