import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from model import MultiTaskFaceAttributes, AGE_GROUPS, age_to_group_idx
from train_attributes import FaceAttributesDataset, load_samples, evaluate


def main():
    parser = argparse.ArgumentParser(description="Avaliação dos modelos de Regressão de Idade + Género")
    parser.add_argument("--checkpoint", type=str, required=True, help="Caminho para o .pt treinado")
    parser.add_argument("--data-dir", type=str, required=True, help="Pasta das imagens (aligned ou bbox)")
    parser.add_argument("--val-csv", type=str, default=None, help="CSV de validação/teste (opcional)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-dir", type=str, default="../results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    backbone_name = ckpt.get("backbone", "resnet18")
    crop_mode = ckpt.get("crop_mode", "aligned")
    img_size = ckpt.get("img_size", 224)

    model = MultiTaskFaceAttributes(backbone_name=backbone_name, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.val_csv:
        val_samples = load_samples(args.data_dir, args.val_csv)
    else:
        all_samples = load_samples(args.data_dir)
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(all_samples))
        split = int(0.8 * len(all_samples))
        val_samples = [all_samples[i] for i in indices[split:]]

    val_loader = DataLoader(FaceAttributesDataset(val_samples, val_tf), batch_size=args.batch_size, shuffle=False)
    metrics = evaluate(model, val_loader, nn.SmoothL1Loss(beta=2.0), nn.BCEWithLogitsLoss(), device, 0.1, 1.0)

    # Medição de latência (ms/imagem) e número de parâmetros
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = ((time.perf_counter() - t0) / 100.0) * 1000.0
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"\n=== Resultados: {backbone_name} ({crop_mode}) ===")
    print(f"Parâmetros:          {params_m:.2f} M")
    print(f"Latência (1 face):   {latency_ms:.2f} ms")
    print(f"MAE (Idade):         {metrics['mae']:.2f} anos")
    print(f"RMSE (Idade):        {metrics['rmse']:.2f} anos")
    print(f"Erro <= 5 anos:      {metrics['within_5']*100:.1f}%")
    print(f"Acc Faixa Etária:    {metrics['age_group_acc']*100:.2f}%")
    print(f"Acc Género:          {metrics['gender_acc']*100:.2f}%")

    # Guardar gráficos
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{backbone_name}_{crop_mode}"

    y_true, y_pred = metrics["y_true"], metrics["y_pred"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Dispersão Idade Real vs Prevista
    axes[0].scatter(y_true, y_pred, alpha=0.25, s=10)
    axes[0].plot([0, 100], [0, 100], "r--", linewidth=1.5, label="Ideal (y = x)")
    axes[0].set_title(f"Regressão Idade - {tag}\nMAE={metrics['mae']:.2f}a | RMSE={metrics['rmse']:.2f}a")
    axes[0].set_xlabel("Idade Real (anos)")
    axes[0].set_ylabel("Idade Prevista (anos)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Matriz de Confusão das 5 Faixas Etárias (derivadas da regressão)
    true_g = np.array([age_to_group_idx(a) for a in y_true])
    pred_g = np.array([age_to_group_idx(a) for a in y_pred])
    cm = np.zeros((len(AGE_GROUPS), len(AGE_GROUPS)), dtype=int)
    for t, p in zip(true_g, pred_g):
        cm[t, p] += 1

    im = axes[1].imshow(cm, cmap="Blues")
    axes[1].set_xticks(range(len(AGE_GROUPS)))
    axes[1].set_yticks(range(len(AGE_GROUPS)))
    axes[1].set_xticklabels(AGE_GROUPS)
    axes[1].set_yticklabels(AGE_GROUPS)
    axes[1].set_xlabel("Faixa Prevista (via Regressão)")
    axes[1].set_ylabel("Faixa Real")
    axes[1].set_title(f"Faixas Etárias ({metrics['age_group_acc']*100:.1f}% Acc)")
    for i in range(len(AGE_GROUPS)):
        for j in range(len(AGE_GROUPS)):
            axes[1].text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    fig_path = out_dir / f"eval_{tag}.png"
    plt.savefig(fig_path, dpi=200)
    print(f"Gráfico guardado em: {fig_path}")


if __name__ == "__main__":
    main()