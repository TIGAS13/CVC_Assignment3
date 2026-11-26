
import sys
from pathlib import Path

import cv2
import numpy as np

from lab_common import ROOT, onnx_providers


def _empty():
    return (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0, 5, 2), dtype=np.float32),
    )


class RetinaFaceDetector:
    """Envolve o código do repositório retinaface-pytorch: letterbox 640x640, forward, decode, limiar e NMS."""

    name = "retinaface"

    def __init__(self, weights, network="mobilenetv2", repo_path=None,
                 conf_threshold=0.02, nms_threshold=0.4,
                 pre_nms_topk=5000, post_nms_topk=750, input_size=640):
        import torch

        # Põe o repositório do RetinaFace no sys.path para importar layers, config, ...
        repo_path = Path(repo_path or ROOT / "retinaface-pytorch").resolve()
        sys.path.insert(0, str(repo_path))
        from layers import PriorBox
        from config import get_config
        from models import RetinaFace
        from utils.box_utils import decode, decode_landmarks, nms

        self._decode, self._decode_landmarks, self._nms = decode, decode_landmarks, nms
        self.torch = torch
        self.cfg = get_config(network)
        if self.cfg is None:
            raise KeyError(f"Config para '{network}' não encontrada")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.size = input_size
        self.rgb_mean = (104, 117, 123)

        self.model = RetinaFace(cfg=self.cfg).to(self.device)
        state_dict = torch.load(str(weights), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # A imagem é sempre 640x640, por isso as âncoras calculam-se uma só vez.
        priorbox = PriorBox(self.cfg, image_size=(input_size, input_size))
        self.priors = priorbox.generate_anchors().to(self.device)

    def _letterbox(self, frame):
        """Redimensiona mantendo a proporção e preenche o resto a zeros."""
        size = self.size
        ratio = frame.shape[0] / frame.shape[1]
        if ratio > 1.0:
            new_h, new_w = size, int(size / ratio)
        else:
            new_w, new_h = size, int(size * ratio)
        factor = new_h / frame.shape[0]
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = cv2.resize(frame, (new_w, new_h))
        return canvas, factor

    def detect(self, frame):
        torch = self.torch
        image, factor = self._letterbox(frame)
        image = np.float32(image) - self.rgb_mean
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            loc, conf, landms = self.model(tensor)
        loc, conf, landms = loc.squeeze(0), conf.squeeze(0), landms.squeeze(0)

        boxes = self._decode(loc, self.priors, self.cfg["variance"])
        landms = self._decode_landmarks(landms, self.priors, self.cfg["variance"])

        s = self.size
        box_scale = torch.tensor([s, s] * 2, device=self.device)
        boxes = (boxes * box_scale / factor).cpu().numpy()
        lm_scale = torch.tensor([s, s] * 5, device=self.device)
        landms = (landms * lm_scale / factor).cpu().numpy()
        scores = conf.cpu().numpy()[:, 1]

        keep = scores > self.conf_threshold
        boxes, landms, scores = boxes[keep], landms[keep], scores[keep]
        if len(scores) == 0:
            return _empty()

        order = scores.argsort()[::-1][: self.pre_nms_topk]
        boxes, landms, scores = boxes[order], landms[order], scores[order]

        dets = np.hstack((boxes, scores[:, None])).astype(np.float32, copy=False)
        keep = self._nms(dets, self.nms_threshold)[: self.post_nms_topk]
        return boxes[keep], scores[keep], landms[keep].reshape(-1, 5, 2)


class ScrfdDetector:
    """SCRFD através do pacote InsightFace (modelo det_10g do pack buffalo_l).
    Corre em onnxruntime, sem dependências do mmcv."""

    name = "scrfd"

    def __init__(self, det_size=640, conf_threshold=0.02, model_pack="buffalo_l"):
        from insightface.app import FaceAnalysis

        providers, ctx_id = onnx_providers()
        self.app = FaceAnalysis(name=model_pack, allowed_modules=["detection"], providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_thresh=conf_threshold, det_size=(det_size, det_size))

    def detect(self, frame):
        bboxes, kpss = self.app.det_model.detect(frame, max_num=0, metric="default")
        if bboxes.shape[0] == 0:
            return _empty()
        return bboxes[:, :4], bboxes[:, 4], kpss.reshape(-1, 5, 2)


def build_detector(name, weights=None, network="mobilenetv2", repo_path=None,
                   conf_threshold=0.02, det_size=640):
    if name == "retinaface":
        weights = weights or (Path(repo_path or ROOT / "retinaface-pytorch") / "weights" / "retinaface_mv2.pth")
        return RetinaFaceDetector(weights, network, repo_path, conf_threshold=conf_threshold, input_size=det_size)
    if name == "scrfd":
        return ScrfdDetector(det_size=det_size, conf_threshold=conf_threshold)
    raise ValueError(f"Detetor desconhecido: {name}")
