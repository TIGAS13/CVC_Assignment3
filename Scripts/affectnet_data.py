"""Carregamento do AffectNet (versão AffectNetCustom) para treino de emoções.

Estrutura esperada (uma pasta por classe, o nome da pasta é o índice da emoção):
    AffectNetCustom/
      train/0 ... train/7
      val/0   ... val/7
      test/0  ... test/7

Uso como módulo:
    from affectnet_data import build_dataloaders, EMOTIONS
    loaders = build_dataloaders(img_size=224, batch_size=64, balance=True)
    for imgs, labels in loaders["train"]: ...

Uso como script (estatísticas + grelha de exemplos):
    python affectnet_data.py
    python affectnet_data.py --root ../AffectNetCustom --grid
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from lab_common import IMAGE_EXTENSIONS, RESULTS, ROOT

# Mesma ordem que labels_sentimento.txt (índice = nome da pasta)
EMOTIONS = ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
SPLITS = ("train", "val", "test")
DEFAULT_ROOT = ROOT / "AffectNetCustom"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_samples(root=DEFAULT_ROOT, split="train"):
    """Devolve uma lista ordenada de (caminho, label) para um split."""
    split_dir = Path(root) / split
    if not split_dir.is_dir():
        raise SystemExit(f"Não encontrei {split_dir}")

    samples = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        if not class_dir.name.isdigit() or int(class_dir.name) >= len(EMOTIONS):
            print(f"AVISO: pasta ignorada (não é uma classe válida): {class_dir}")
            continue
        label = int(class_dir.name)
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((str(path), label))

    if not samples:
        raise SystemExit(f"Nenhuma imagem encontrada em {split_dir}")
    return samples


def get_transforms(img_size=224, train=False):
    """Transformações compatíveis com as backbones pré-treinadas no ImageNet.

    Sem flip vertical nem rotações grandes: alterariam a expressão facial.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class AffectNetDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

    @property
    def labels(self):
        return [label for _, label in self.samples]


def class_counts(samples):
    counts = Counter(label for _, label in samples)
    return np.array([counts.get(i, 0) for i in range(len(EMOTIONS))])


def class_weights(samples):
    """Pesos inversamente proporcionais à frequência (para a CrossEntropyLoss).

    Normalizados para média 1; classes sem amostras ficam com peso 0.
    """
    counts = class_counts(samples).astype(np.float64)
    weights = np.where(counts > 0, counts.sum() / (len(counts) * np.maximum(counts, 1)), 0.0)
    return torch.tensor(weights / weights[weights > 0].mean(), dtype=torch.float32)


def make_balanced_sampler(samples):
    """Sampler que tira cada classe com a mesma probabilidade (com reposição)."""
    counts = class_counts(samples)
    per_sample = [1.0 / counts[label] for _, label in samples]
    return WeightedRandomSampler(per_sample, num_samples=len(samples), replacement=True)


def build_datasets(root=DEFAULT_ROOT, img_size=224):
    return {
        split: AffectNetDataset(load_samples(root, split), get_transforms(img_size, train=(split == "train")))
        for split in SPLITS
        if (Path(root) / split).is_dir()
    }


def build_dataloaders(root=DEFAULT_ROOT, img_size=224, batch_size=64, num_workers=2, balance=False):
    """DataLoaders para train/val/test.

    balance=True usa um WeightedRandomSampler no treino (alternativa a passar
    class_weights() à loss; não usar as duas coisas ao mesmo tempo).
    """
    datasets = build_datasets(root, img_size)
    loaders = {}
    for split, ds in datasets.items():
        is_train = split == "train"
        sampler = make_balanced_sampler(ds.samples) if (is_train and balance) else None
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train and sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=is_train,
        )
    return loaders


def print_stats(root):
    print(f"AffectNet em: {root}\n")
    header = f"{'Emoção':<10}" + "".join(f"{s:>9}" for s in SPLITS)
    print(header)
    print("-" * len(header))
    per_split = {s: class_counts(load_samples(root, s)) for s in SPLITS if (Path(root) / s).is_dir()}
    for i, name in enumerate(EMOTIONS):
        print(f"{i} {name:<8}" + "".join(f"{per_split[s][i]:>9}" for s in per_split))
    print("-" * len(header))
    print(f"{'Total':<10}" + "".join(f"{per_split[s].sum():>9}" for s in per_split))

    if "train" in per_split:
        c = per_split["train"]
        print(f"\nDesequilíbrio no treino: classe maior / menor = {c.max() / max(c.min(), 1):.1f}x")
        w = class_weights(load_samples(root, "train"))
        print("Pesos para a loss: " + ", ".join(f"{EMOTIONS[i]}={w[i]:.2f}" for i in range(len(EMOTIONS))))


def save_grid(root, out_path, per_class=6, seed=0):
    import matplotlib.pyplot as plt

    samples = load_samples(root, "train")
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(EMOTIONS), per_class, figsize=(per_class * 1.6, len(EMOTIONS) * 1.7))
    for label, name in enumerate(EMOTIONS):
        paths = [p for p, l in samples if l == label]
        chosen = rng.choice(paths, size=min(per_class, len(paths)), replace=False) if paths else []
        for j in range(per_class):
            ax = axes[label, j]
            ax.axis("off")
            if j < len(chosen):
                ax.imshow(Image.open(chosen[j]).convert("RGB"))
        axes[label, 0].set_title(name, loc="left", fontsize=9)
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    print(f"\nGrelha de exemplos guardada em: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Carregamento e estatísticas do AffectNet")
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--grid", action="store_true", help="guardar uma grelha de exemplos por classe")
    p.add_argument("--check-batch", action="store_true", help="carregar um batch para testar o pipeline")
    args = p.parse_args()

    print_stats(args.root)
    if args.grid:
        save_grid(args.root, RESULTS / "affectnet" / "samples_grid.png")
    if args.check_batch:
        loaders = build_dataloaders(args.root, batch_size=16, num_workers=0, balance=True)
        imgs, labels = next(iter(loaders["train"]))
        print(f"\nBatch de treino: imgs {tuple(imgs.shape)}, labels {labels.tolist()}")


if __name__ == "__main__":
    main()
