
import cv2
import numpy as np

ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],  # olho esquerdo
        [73.5318, 51.5014],  # olho direito
        [56.0252, 71.7366],  # nariz
        [41.5493, 92.3655],  # canto esquerdo da boca
        [70.7299, 92.2041],  # canto direito da boca
    ],
    dtype=np.float32,
)


def get_template(size=112, zoom=1.0):
    """Template à escala `size`. zoom < 1 encolhe a cara dentro do recorte
    (mais contexto: testa, cabelo), útil para a idade."""
    template = ARCFACE_TEMPLATE * (size / 112.0)
    if zoom != 1.0:
        center = np.array([size / 2.0, size / 2.0], dtype=np.float32)
        template = (template - center) * zoom + center
    return template


def align_face(frame, landmarks, size=112, zoom=1.0, return_matrix=False):
    """Devolve o recorte alinhado (size x size, BGR) ou None se falhar."""
    points = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    template = get_template(size, zoom)

    # 4 graus de liberdade (rotação, escala, tx, ty): similaridade, sem perspetiva
    matrix, _ = cv2.estimateAffinePartial2D(points, template, method=cv2.LMEDS)
    if matrix is None:
        return (None, None) if return_matrix else None

    aligned = cv2.warpAffine(frame, matrix, (size, size), borderValue=0)
    return (aligned, matrix) if return_matrix else aligned


def crop_bbox(frame, box, size=112, margin=0.25):
    """Recorte simples SEM alinhamento (baseline para a comparação):
    quadrado centrado na caixa, com lado = max(w, h) * (1 + margin)."""
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + margin)
    scale = size / side
    # mapeia o quadrado [cx-side/2, cx+side/2] para [0, size]
    matrix = np.array(
        [[scale, 0, -scale * (cx - side / 2)], [0, scale, -scale * (cy - side / 2)]],
        dtype=np.float32,
    )
    return cv2.warpAffine(frame, matrix, (size, size), borderValue=0)


def make_crop(frame, box, landmarks, mode="aligned", size=128, zoom=0.7, margin=0.25):
    """Ponto único de entrada: devolve o recorte segundo o modo escolhido.

    O MESMO modo, tamanho e zoom têm de ser usados no treino e na aplicação.
    """
    if mode == "aligned":
        return align_face(frame, landmarks, size=size, zoom=zoom)
    if mode == "bbox":
        return crop_bbox(frame, box, size=size, margin=margin)
    raise ValueError(f"modo desconhecido: {mode}")
