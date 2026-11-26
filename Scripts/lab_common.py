"""Utilitários partilhados por todos os scripts (Lab3).

Estrutura esperada:
    Lab3/
      Scripts/                 <- estes ficheiros
      retinaface-pytorch/      <- o repositório do RetinaFace
      data/                    <- datasets
      results/                 <- métricas, tabelas e figuras
      checkpoints/             <- modelos treinados
"""
import os
import re
from pathlib import Path

import cv2
import numpy as np

# Os caminhos são calculados a partir da localização deste ficheiro,
# por isso os scripts funcionam seja qual for a pasta onde os corres.
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def read_image(path):
    """Lê uma imagem BGR de 3 canais. Devolve None se falhar.

    O cv2.imread não abre caminhos com acentos no Windows (ex.: '2º Ano',
    'Visão'). Com np.fromfile o Python lê os bytes (suporta Unicode) e o
    cv2.imdecode só descodifica a imagem.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # força 3 canais (BGR)
    except OSError:
        return None
    return img


def write_image(path, img, quality=95):
    """Equivalente seguro do cv2.imwrite para caminhos com acentos."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if path.suffix.lower() in (".jpg", ".jpeg") else []
    ok, buffer = cv2.imencode(path.suffix, img, params)
    if not ok:
        raise IOError(f"Não consegui codificar {path}")
    buffer.tofile(str(path))


def list_images(folder):
    """Lista todas as imagens de uma pasta e subpastas (os.walk, ordenado)."""
    found = []
    for dirpath, _, filenames in os.walk(folder):
        for name in filenames:
            if name.lower().endswith(IMAGE_EXTENSIONS):
                found.append(Path(dirpath) / name)
    return sorted(found)


def parse_utkface_name(path):
    """[idade]_[género]_[raça]_[data].jpg  ->  (idade, género, raça).

    Género: 0 = masculino, 1 = feminino (convenção do UTKFace).
    Raça: 0 White, 1 Black, 2 Asian, 3 Indian, 4 Others (-1 se faltar no nome).
    Levanta ValueError se o nome não tiver idade e género válidos.
    """
    parts = Path(path).name.split("_")
    age = int(parts[0])
    gender = int(parts[1])
    if gender not in (0, 1):
        raise ValueError("género inválido")
    try:
        race = int(parts[2])
        if race not in (0, 1, 2, 3, 4):
            race = -1
    except (ValueError, IndexError):
        race = -1  # alguns ficheiros do UTKFace não têm a raça no nome
    return age, gender, race


def parse_fgnet_name(path):
    """FG-NET: '001A02.JPG' -> (sujeito 1, idade 2). Devolve None se não encaixar."""
    match = re.match(r"^(\d{3})A(\d{2})", Path(path).stem, flags=re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def onnx_providers():
    """Escolhe o CUDA se o onnxruntime o suportar, senão o CPU."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
    return ["CPUExecutionProvider"], -1  # (providers, ctx_id do InsightFace)


# ----------------------------------------------------------------------------
# Manifesto do UTKFace (split fixo + deteções) e pasta dos recortes
# ----------------------------------------------------------------------------
MANIFEST = DATA / "utkface_manifest.csv"


def crops_dir(mode="aligned", size=128, zoom=0.7, margin=0.25):
    """Pasta dos recortes, cujo nome guarda a configuração (evita misturar versões)."""
    name = f"aligned_s{size}_z{zoom}" if mode == "aligned" else f"bbox_s{size}_m{margin}"
    return DATA / "UTKFace_crops" / name


def load_manifest():
    import pandas as pd

    if not MANIFEST.exists():
        raise SystemExit(f"Falta {MANIFEST}. Corre primeiro o prepare_utkface.py.")
    return pd.read_csv(MANIFEST)
