import os
import csv
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import MultiTaskFaceAttributes, AGE_GROUPS, age_to_group_idx


class FaceAttributesDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        samples: lista de tuplos (img_path, age_float, gender_int)
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, age, gender = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return (
            img,
            torch.tensor(age, dtype=torch.float32),
            torch.tensor(gender, dtype=torch.float32),
        )


def load_samples(data_dir: str, csv_path: str = None):
    samples = []
    data_dir = Path(data_dir)

    if csv_path and os.path.isfile(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_path = row.get("image_path") or row.get("filename") or row.get("file")
                full_path = data_dir / rel_path if not os.path.isabs(rel_path) else Path(rel_path)
                if full_path.exists():
                    age = float(row["age"])
                    gender = int(float(row["gender"]))
                    samples.append((str(full_path), age, gender))
    else:
        # Fallback: formato UTKFace (ex.: 25_0_1_20170116.jpg -> age=25, gender=0)
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in data_dir.rglob(ext):
                parts = p.stem.split("_")
                if len(parts) >= 2 and parts[0].isdigit() and parts[1] in ("0", "1"):
                    samples.append((str(p), float(parts[0]), int(parts[1])))

    if not samples:
        raise RuntimeError(f"Nenhuma imagem encontrada em {data_dir} (csv={csv_path}).")
    return samples


def evaluate(model, loader, criterion_age, criterion_gender, device, w_age, w_gender):
    model.eval()
    total_loss = 0.0
    all_age_true, all_age_pred = [], []
    all_gender_true, all_gender_pred = [], []

    with torch.no_grad():
        for imgs, ages, genders in loader:
            imgs = imgs.to(device)
            ages = ages.to(device)
            genders = genders.to(device)

            age_pred, gender_logit = model(imgs)
            loss_age = criterion_age(age_pred, ages)
            loss_gender = criterion_gender(gender_logit, genders)
            loss = w_age * loss_age + w_gender * loss_gender

            total_loss += loss.item() * imgs.size(0)
            all_age_true.extend(ages.cpu().numpy().tolist())
            all_age_pred.extend(age_pred.cpu().numpy().tolist())

            gender_preds = (torch.sigmoid(gender_logit) >= 0.5).long()
            all_gender_true.extend(genders.long().cpu().numpy().tolist())
            all_gender_pred.extend(gender_preds.cpu().numpy().tolist())

    n = len(all_age_true)
    y_true = np.array(all_age_true)
    y_pred = np.array(all_age_pred)

    # Métricas de Regressão
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # Classificação derivada da Regressão (discretização nas 5 faixas etárias)
    true_groups = np.array([age_to_group_idx(a) for a in y_true])
    pred_groups = np.array([age_to_group_idx(a) for a in y_pred])
    age_group_acc = float(np.mean(true_groups == pred_groups))

    # Métrica de Classificação de Género (mesma backbone)
    gender_acc = float(np.mean(np.array(all_gender_true) == np.array(all_gender_pred)))

    return {
        "loss": total_loss / n,
        "mae": mae,
        "rmse": rmse,
        "age_group_acc": age_group_acc,
        "gender_acc": gender_acc,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main():
    parser = argparse.ArgumentParser(description="Treino Multi-Task (Regressão Idade + Género)")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "mobilenet_v3_small"])
    parser.add_argument("--data-dir", type=str, required=True, help="Pasta com imagens alinhadas (aligned)")
    parser.add_argument("--train-csv", type=str, default=None, help="CSV de treino (opcional)")
    parser.add_argument("--val-csv", type=str, default=None, help="CSV de validação (opcional)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--w-age", type=float, default=0.1, help="Peso da loss de regressão de idade")
    parser.add_argument("--w-gender", type=float, default=1.0, help="Peso da loss de género")
    parser.add_argument("--checkpoint-out", type=str, default=None)
    parser.add_argument("--log-out", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    train_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.train_csv and args.val_csv:
        train_samples = load_samples(args.data_dir, args.train_csv)
        val_samples = load_samples(args.data_dir, args.val_csv)
    else:
        all_samples = load_samples(args.data_dir, args.train_csv)
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(all_samples))
        split = int(0.8 * len(all_samples))
        train_samples = [all_samples[i] for i in indices[:split]]
        val_samples = [all_samples[i] for i in indices[split:]]

    train_loader = DataLoader(FaceAttributesDataset(train_samples, train_tf), batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(FaceAttributesDataset(val_samples, val_tf), batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = MultiTaskFaceAttributes(backbone_name=args.backbone, pretrained=True).to(device)

    # SmoothL1Loss (Huber) é robusta a outliers na regressão de idade
    criterion_age = nn.SmoothL1Loss(beta=2.0)
    criterion_gender = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_path = Path(args.checkpoint_out or f"../checkpoints/{args.backbone}_aligned.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_out or f"{args.backbone}_aligned_log.csv")

    best_val_mae = float("inf")

    with open(log_path, "w", newline="", encoding="utf-8") as log_f:
        writer = csv.writer(log_f)
        writer.writerow([
            "epoch", "train_loss", "train_mae", "train_gender_acc",
            "val_loss", "val_mae", "val_rmse", "val_age_group_acc", "val_gender_acc"
        ])

        for epoch in range(1, args.epochs + 1):
            model.train()
            tr_loss = 0.0
            tr_age_true, tr_age_pred = [], []
            tr_g_true, tr_g_pred = [], []

            for imgs, ages, genders in train_loader:
                imgs, ages, genders = imgs.to(device), ages.to(device), genders.to(device)

                optimizer.zero_grad()
                age_pred, gender_logit = model(imgs)
                loss = args.w_age * criterion_age(age_pred, ages) + args.w_gender * criterion_gender(gender_logit, genders)
                loss.backward()
                optimizer.step()

                tr_loss += loss.item() * imgs.size(0)
                tr_age_true.extend(ages.cpu().numpy().tolist())
                tr_age_pred.extend(age_pred.detach().cpu().numpy().tolist())
                tr_g_true.extend(genders.long().cpu().numpy().tolist())
                tr_g_pred.extend((torch.sigmoid(gender_logit) >= 0.5).long().cpu().numpy().tolist())

            scheduler.step()

            n_tr = len(tr_age_true)
            train_loss = tr_loss / n_tr
            train_mae = float(np.mean(np.abs(np.array(tr_age_true) - np.array(tr_age_pred))))
            train_gender_acc = float(np.mean(np.array(tr_g_true) == np.array(tr_g_pred)))

            val_metrics = evaluate(model, val_loader, criterion_age, criterion_gender, device, args.w_age, args.w_gender)

            writer.writerow([
                epoch, train_loss, train_mae, train_gender_acc,
                val_metrics["loss"], val_metrics["mae"], val_metrics["rmse"],
                val_metrics["age_group_acc"], val_metrics["gender_acc"]
            ])

            print(
                f"Epoch {epoch:02d}/{args.epochs:02d} | "
                f"Val MAE: {val_metrics['mae']:.2f} anos | "
                f"Val Group Acc: {val_metrics['age_group_acc']:.3f} | "
                f"Val Gender Acc: {val_metrics['gender_acc']:.3f}"
            )

            if val_metrics["mae"] < best_val_mae:
                best_val_mae = val_metrics["mae"]
                torch.save({
                    "backbone": args.backbone,
                    "img_size": args.img_size,
                    "age_groups": AGE_GROUPS,
                    "state_dict": model.state_dict(),
                    "val_mae": val_metrics["mae"],
                    "val_age_group_acc": val_metrics["age_group_acc"],
                    "val_gender_acc": val_metrics["gender_acc"],
                }, ckpt_path)

    print(f"\nMelhor modelo guardado em: {ckpt_path} (Val MAE = {best_val_mae:.2f} anos)")


if __name__ == "__main__":
    main()