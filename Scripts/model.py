import torch
import torch.nn as nn
from torchvision import models

AGE_GROUPS = ["0-12", "13-19", "20-39", "40-59", "60+"]


def age_to_group_idx(age_value: float) -> int:
    """Classifica o valor contínuo da regressão numa das 5 faixas etárias."""
    age_rounded = int(max(0, round(float(age_value))))
    if age_rounded <= 12:
        return 0  # "0-12"
    elif age_rounded <= 19:
        return 1  # "13-19"
    elif age_rounded <= 39:
        return 2  # "20-39"
    elif age_rounded <= 59:
        return 3  # "40-59"
    else:
        return 4  # "60+"


def age_to_group_label(age_value: float) -> str:
    return AGE_GROUPS[age_to_group_idx(age_value)]


class MultiTaskFaceAttributes(nn.Module):
    """
    Mesma backbone para todas as tarefas:
      - Cabeça 1 (age_head): Regressão contínua da idade (saída escalar).
      - Cabeça 2 (gender_head): Classificação binária de género (1 logit).
    """

    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.backbone_name = backbone_name.lower()

        if self.backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            net = models.resnet18(weights=weights)
            in_features = net.fc.in_features  # 512
            net.fc = nn.Identity()
            self.backbone = net

        elif self.backbone_name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            net = models.mobilenet_v3_small(weights=weights)
            in_features = net.classifier[-1].in_features  # 1024
            net.classifier[-1] = nn.Identity()
            self.backbone = net

        else:
            raise ValueError(f"Backbone não suportada: {backbone_name}. Usa 'resnet18' ou 'mobilenet_v3_small'.")

        # Cabeça de Regressão de Idade (saída contínua >= 0)
        self.age_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        # Cabeça de Classificação de Género (logit binário)
        self.gender_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        age_pred = self.age_head(features).squeeze(-1)        # (B,) contínuo
        gender_logit = self.gender_head(features).squeeze(-1) # (B,) logit
        return age_pred, gender_logit