import os
import cv2
from pathlib import Path
import numpy as np
from utils.helper import GetLogger, Predictor

# =========================
# CONFIGURAÇÃO
# =========================
INPUT_DIR   = Path("datasets_recortes/suspeito")         # pasta de entrada
OUTPUT_DIR  = Path("datasets_recortes/suspeito_dense")   # pasta de saída
RECURSIVE   = True                                   # True = varrer subpastas
SAVE_BOTH   = False                                  # True = salvar também o out_frame "normal"
SUFFIX_SEG  = "_densepose"                           # sufixo do arquivo processado
SUFFIX_RAW  = "_proc"                                # sufixo opcional do out_frame (se SAVE_BOTH=True)
EXTS        = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# =========================
# DESEMPENHO (inferência em baixa resolução + upscale)
# =========================
# Tamanho utilizado APENAS para a inferência (downscale). A saída é redimensionada
# de volta ao tamanho ORIGINAL da imagem de entrada.
_INFER_W, _INFER_H = (384, 288)  # ajuste para (320, 240) se precisar de mais FPS

# Permitir override por variável de ambiente: ex. DENSEPOSE_INFER_SIZE=320x240
_env_size = os.getenv("DENSEPOSE_INFER_SIZE", "").strip().lower()
if "x" in _env_size:
    try:
        _INFER_W, _INFER_H = tuple(int(x) for x in _env_size.split("x"))
    except Exception:
        pass

# =========================
# SETUP
# =========================
logger = GetLogger.logger(__name__)
predictor = Predictor()

if not INPUT_DIR.exists():
    raise FileNotFoundError(f"Pasta de entrada não encontrada: {INPUT_DIR}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Lista de arquivos
paths = list(INPUT_DIR.rglob("*")) if RECURSIVE else list(INPUT_DIR.glob("*"))
img_paths = [p for p in paths if p.is_file() and p.suffix.lower() in EXTS]

if not img_paths:
    logger.warning("Nenhuma imagem encontrada para processar.")
else:
    logger.info(f"Encontradas {len(img_paths)} imagens para processar.")
    logger.info(f"Inferência em {_INFER_W}x{_INFER_H} com upscale para o tamanho original.")

# =========================
# HELPERS
# =========================
def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def _downscale_for_infer(bgr_img: np.ndarray) -> np.ndarray:
    """Reduz a imagem para o tamanho de inferência, garantindo contiguidade."""
    small = cv2.resize(bgr_img, (_INFER_W, _INFER_H), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(small)

def _upscale_to_original(img_small: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Redimensiona de volta para o tamanho original."""
    return cv2.resize(img_small, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

# =========================
# PROCESSAMENTO
# =========================
processed = 0
failed = 0

for idx, in_path in enumerate(img_paths, 1):
    try:
        # Lê imagem (BGR)
        img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
        if img is None:
            failed += 1
            logger.warning(f"[{idx}/{len(img_paths)}] Falha ao ler: {in_path}")
            continue

        h, w = img.shape[:2]

        # Downscale apenas para inferência
        small = _downscale_for_infer(img)

        # Predição em low-res (esperado: (out_frame, out_frame_seg))
        out_frame_small, out_frame_seg_small = predictor.predict(small)

        # Upscale dos resultados para o tamanho original
        out_frame     = _upscale_to_original(out_frame_small,     w, h)
        out_frame_seg = _upscale_to_original(out_frame_seg_small, w, h)

        # Normaliza tipo/intervalo
        out_frame     = _to_uint8(out_frame)
        out_frame_seg = _to_uint8(out_frame_seg)

        # Caminhos de saída (preserva estrutura)
        rel = in_path.relative_to(INPUT_DIR)
        rel_dir = rel.parent
        out_dir = OUTPUT_DIR / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = in_path.stem
        ext  = in_path.suffix  # mantém extensão original

        # Salva densepose (out_frame_seg)
        out_seg_path = out_dir / f"{stem}{SUFFIX_SEG}{ext}"
        ok = cv2.imwrite(str(out_seg_path), out_frame_seg)
        if not ok:
            raise RuntimeError("cv2.imwrite retornou False para out_frame_seg")

        # (Opcional) Salva também o out_frame "normal"
        if SAVE_BOTH:
            out_raw_path = out_dir / f"{stem}{SUFFIX_RAW}{ext}"
            ok2 = cv2.imwrite(str(out_raw_path), out_frame)
            if not ok2:
                logger.warning(f"Falhou ao salvar out_frame: {out_raw_path}")

        processed += 1
        if idx % 25 == 0 or idx == len(img_paths):
            logger.info(f"Progresso: {idx}/{len(img_paths)} | Sucesso: {processed} | Falhas: {failed}")

    except Exception as e:
        failed += 1
        logger.exception(f"[{idx}/{len(img_paths)}] Erro ao processar {in_path}: {e}")

logger.info(f"Concluído. Total: {len(img_paths)} | Sucesso: {processed} | Falhas: {failed}")
