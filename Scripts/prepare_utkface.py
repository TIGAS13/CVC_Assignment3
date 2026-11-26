
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from alignment import make_crop
from lab_common import DATA, MANIFEST, crops_dir, list_images, parse_utkface_name, read_image, write_image

KPS_COLUMNS = [f"k{i}" for i in range(10)]


def build_manifest(utk_dir, seed, val_frac, test_frac):
    rows, invalid, seen = [], 0, set()
    for path in list_images(utk_dir):
        if path.name in seen:  # o mesmo ficheiro repetido noutra pasta/parte
            continue
        try:
            age, gender, race = parse_utkface_name(path)
        except (ValueError, IndexError):
            invalid += 1
            continue
        seen.add(path.name)
        rows.append({"name": path.name, "path": path.relative_to(utk_dir).as_posix(),
                     "age": age, "gender": gender, "race": race})
    if not rows:
        raise SystemExit(f"Não encontrei imagens UTKFace válidas em {utk_dir}")
    df = pd.DataFrame(rows)

    # split aleatório com seed fixa (o UTKFace não traz identidades, por isso
    # não é possível garantir que a mesma pessoa não aparece em dois splits)
    order = np.random.RandomState(seed).permutation(len(df))
    n_test, n_val = int(len(df) * test_frac), int(len(df) * val_frac)
    split = np.array(["train"] * len(df), dtype=object)
    split[order[:n_test]] = "test"
    split[order[n_test:n_test + n_val]] = "val"
    df["split"] = split

    df["det_done"], df["det_ok"], df["det_score"] = 0, 0, 0.0
    for col in ["x1", "y1", "x2", "y2"] + KPS_COLUMNS:
        df[col] = 0.0
    print(f"Manifesto: {len(df)} imagens válidas | {invalid} nomes inválidos ignorados")
    return df


def detect_faces(df, utk_dir, todo, det_size):
    from detectors import ScrfdDetector

    detector = ScrfdDetector(det_size=det_size, conf_threshold=0.3)
    for n, idx in enumerate(tqdm(todo, desc="Deteção SCRFD")):
        img = read_image(utk_dir / df.at[idx, "path"])
        df.at[idx, "det_done"] = 1
        if img is not None:
            boxes, scores, lms = detector.detect(img)
            if len(scores):
                # fica com a maior cara (cada imagem do UTKFace tem uma cara principal)
                best = int(((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).argmax())
                df.loc[idx, ["x1", "y1", "x2", "y2"]] = boxes[best]
                df.loc[idx, KPS_COLUMNS] = lms[best].reshape(-1)
                df.at[idx, "det_score"] = float(scores[best])
                df.at[idx, "det_ok"] = 1
        if (n + 1) % 1000 == 0:  # guarda de vez em quando para poder retomar
            df.to_csv(MANIFEST, index=False)
    df.to_csv(MANIFEST, index=False)


def main():
    p = argparse.ArgumentParser(description="Prepara o UTKFace")
    p.add_argument("--utk-dir", default=str(DATA / "UTKFace"))
    p.add_argument("--mode", choices=["aligned", "bbox"], default="aligned")
    p.add_argument("--size", type=int, default=128, help="lado do recorte (px)")
    p.add_argument("--zoom", type=float, default=0.7, help="(aligned) <1 = mais contexto à volta da cara")
    p.add_argument("--margin", type=float, default=0.25, help="(bbox) margem à volta da caixa")
    p.add_argument("--det-size", type=int, default=320)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--limit", type=int, default=0, help="máx. de imagens a detetar nesta execução (0 = todas)")
    args = p.parse_args()

    from pathlib import Path
    utk_dir = Path(args.utk_dir)
    df = pd.read_csv(MANIFEST) if MANIFEST.exists() else build_manifest(utk_dir, args.seed, args.val_frac, args.test_frac)

    todo = list(df.index[df.det_done == 0])
    if args.limit:
        todo = todo[: args.limit]
    if todo:
        detect_faces(df, utk_dir, todo, args.det_size)

    # --- recortes -----------------------------------------------------------
    out = crops_dir(args.mode, args.size, args.zoom, args.margin)
    ok_rows = df[df.det_ok == 1]
    made = failed = 0
    for _, row in tqdm(ok_rows.iterrows(), total=len(ok_rows), desc=f"Recortes ({args.mode})"):
        target = out / row["name"]
        if target.exists():
            continue
        img = read_image(utk_dir / row["path"])
        box = [row.x1, row.y1, row.x2, row.y2]
        crop = None if img is None else make_crop(img, box, row[KPS_COLUMNS].values.astype(float),
                                                  mode=args.mode, size=args.size, zoom=args.zoom, margin=args.margin)
        if crop is None:
            failed += 1
            continue
        write_image(target, crop)
        made += 1

    done = df[df.det_done == 1]
    print(f"\nDeteção: {int(done.det_ok.sum())}/{len(done)} imagens com cara ({done.det_ok.mean():.1%})")
    print(f"Recortes novos: {made} | falhados: {failed} | pasta: {out}")
    print("Imagens por split (com cara detetada):")
    print(df[df.det_ok == 1].groupby("split").size().to_string())
    if len(done) and done.det_ok.mean() < 0.95:
        print("AVISO: >5% sem cara detetada. Tenta --det-size 640 (apaga data/utkface_manifest.csv antes).")


if __name__ == "__main__":
    main()
