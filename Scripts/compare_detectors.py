"""Junta os resultados de results/wider/*.json numa tabela Markdown
(AP por subconjunto + ponto de operação + tempos) para colar no relatório.

    python compare_detectors.py
"""
import json

from lab_common import RESULTS


def fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def main():
    files = sorted((RESULTS / "wider").glob("*.json"))
    if not files:
        raise SystemExit("Sem resultados em results/wider. Corre primeiro o wider_eval.py.")
    runs = [json.load(open(f, encoding="utf-8")) for f in files]
    names = [f"{r['detector']}@{r['det_size']}" for r in runs]

    print("### AP no WIDER FACE val (IoU >= 0.5)\n")
    print("| Subconjunto | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(runs))
    for subset in runs[0]["ap"]:
        n = runs[0]["n_faces"].get(subset, "")
        cells = [fmt(r["ap"].get(subset)) for r in runs]
        print(f"| {subset} (n={n}) | " + " | ".join(cells) + " |")

    print("\n### Ponto de operação e tempos\n")
    print("| Detetor | Precisão | Recall | Limiar | Latência média (ms) | p95 (ms) | FPS |")
    print("|---|---|---|---|---|---|---|")
    for name, r in zip(names, runs):
        op, t = r["operating_point"], r.get("timing") or {}
        print(f"| {name} | {fmt(op['precision'])} | {fmt(op['recall'])} | {op['threshold']} | "
              f"{fmt(t.get('mean_ms'), 1)} | {fmt(t.get('p95_ms'), 1)} | {fmt(t.get('fps_mean'), 1)} |")
    print("\n(Tempos medidos só na chamada detect(), sem leitura de imagem; descartadas as 5 primeiras.)")


if __name__ == "__main__":
    main()
