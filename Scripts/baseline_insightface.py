
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from lab_common import DATA, RESULTS, load_manifest, onnx_providers, read_image
from Legacy.metrics_age_gender import full_report


def main():
    p = argparse.ArgumentParser(description="Baseline InsightFace genderage no UTKFace")
    p.add_argument("--utk-dir", default=str(DATA / "UTKFace"))
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    from insightface.app import FaceAnalysis
    from insightface.app.common import Face

    providers, ctx_id = onnx_providers()
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "genderage"], providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(320, 320))
    genderage = app.models["genderage"]

    df = load_manifest()
    df = df[(df.det_ok == 1) & (df.split == args.split)]
    if args.limit:
        df = df.head(args.limit)

    rows = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"genderage ({args.split})"):
        img = read_image(Path(args.utk_dir) / row.path)
        if img is None:
            continue
        face = Face(bbox=np.array([row.x1, row.y1, row.x2, row.y2], dtype=np.float32))
        gender, age = genderage.get(img, face)
        rows.append({
            "name": row.name, "age": row.age, "gender": row.gender, "race": row.race,
            "age_pred": float(age),
            # InsightFace: 1 = masculino; UTKFace: 0 = masculino -> converter
            "gender_pred": 0 if int(gender) == 1 else 1,
        })

    result = pd.DataFrame(rows)
    out_dir = RESULTS / "age_gender" / f"insightface_genderage_{args.split}"
    full_report(result, out_dir, f"InsightFace genderage ({args.split})")
    result.to_csv(out_dir / "predictions.csv", index=False)


if __name__ == "__main__":
    main()
