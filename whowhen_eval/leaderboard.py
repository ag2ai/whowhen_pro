"""Leaderboard scorer with the per-MAS-averaged metric recipe.

For each (model, modality) cell, computes 4 axes:

  Who   = mean over MAS of agent-attribution accuracy   (multi-agent MAS only)
  When  = mean over MAS of step-localization accuracy
  What  = macro-F1 over the observed mode classes (global within the cell)
  All   = mean over MAS of joint accuracy (Who AND When AND What all correct)

Each axis is computed *per-MAS first, then averaged across MASes* (i.e.
each MAS gets equal weight regardless of size). The What axis is the
exception: mode classification is a global problem (the taxonomy is
shared across MASes), so we compute macro-F1 across all observed classes
within the cell, not per-MAS.

The composite leaderboard score per axis is the arithmetic mean across
the modalities (text/image/image_gui/video). Models without all
modalities get the composite computed only on the modalities they have,
marked with a dagger in the printed table.

Usage
-----
::

    # Print the leaderboard for every model found under ./results
    python -m whowhen_eval.leaderboard --results ./results

    # Restrict to a few models
    python -m whowhen_eval.leaderboard --models gpt-5.4 claude-sonnet-4-6
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Optional

# MASes whose agent attribution is degenerate — architecturally one acting
# agent, so ``ground_truth.agent`` carries no information. They are
# excluded from the Who denominator but still count toward When/What/All.
# Multi-agent frameworks: debate, dylan, macnet, magentic-one, metagpt,
# dvd, and coact (image_gui split: orchestrator/coder/computer).
SINGLE_AGENT_FRAMEWORKS = {
    "smolagents", "alfagent", "mathchat", "pixelcraft", "eva",
    # image_gui split (GUI agents): all single-agent except coact
    "openai_cua", "agentoccam", "gemini",
}

AXES = ("Who", "When", "What", "All")


def load_jsonl(path: Path):
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def collect(model_dir: Path) -> list[dict]:
    """Read one model's result directory into per-trace dicts."""
    rows: list[dict] = []
    if not model_dir.exists():
        return rows
    for jp in sorted(model_dir.glob("*.jsonl")):
        for r in load_jsonl(jp):
            if r.get("error"):
                continue
            sc = r.get("score") or {}
            gt = r.get("ground_truth") or {}
            pr = r.get("prediction") or {}
            rows.append({
                "fw": r.get("framework"),
                "modality": r.get("modality"),
                "a_correct": bool(sc.get("agent")),
                "s_correct": bool(sc.get("step")),
                "m_correct": bool(sc.get("mode")),
                "gt_mode": str(gt.get("mode") or ""),
                "pr_mode": str(pr.get("error_mode") or ""),
            })
    return rows


def macro_f1(preds: list[str], golds: list[str]) -> float:
    """Macro-F1 over the union of predicted+gold labels with non-zero gold support."""
    labels = sorted(set(golds) | set(preds))
    f1s: list[float] = []
    for lab in labels:
        tp = sum(1 for p, g in zip(preds, golds) if p == lab and g == lab)
        fp = sum(1 for p, g in zip(preds, golds) if p == lab and g != lab)
        fn = sum(1 for p, g in zip(preds, golds) if p != lab and g == lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if (tp + fn) > 0:  # only count classes with gold support in this cell
            f1s.append(f1)
    return mean(f1s) if f1s else 0.0


def cell_metrics(rows: list[dict]) -> Optional[dict[str, Optional[float]]]:
    """Compute Who/When/What/All for one (model, modality) cell.

    Returns None if the cell has no rows (model didn't run on this modality).
    Who is None if there are no multi-agent rows in the cell.
    """
    if not rows:
        return None
    by_fw: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_fw[r["fw"]].append(r)

    # WHO — multi-agent only
    multi_agent_fws = [fw for fw in by_fw if fw not in SINGLE_AGENT_FRAMEWORKS]
    if multi_agent_fws:
        per_mas_who = {fw: sum(1 for r in by_fw[fw] if r["a_correct"]) / len(by_fw[fw])
                       for fw in multi_agent_fws}
        who: Optional[float] = mean(per_mas_who.values())
    else:
        who = None

    # WHEN — all MAS
    per_mas_when = {fw: sum(1 for r in v if r["s_correct"]) / len(v) for fw, v in by_fw.items()}
    when = mean(per_mas_when.values())

    # WHAT — global macro-F1 over modes
    what = macro_f1([r["pr_mode"] for r in rows], [r["gt_mode"] for r in rows])

    # ALL — joint accuracy averaged across MAS
    per_mas_all = {fw: sum(1 for r in v if r["a_correct"] and r["s_correct"] and r["m_correct"]) / len(v)
                   for fw, v in by_fw.items()}
    all_ = mean(per_mas_all.values())

    return {"Who": who, "When": when, "What": what, "All": all_}


def composite(cells: dict[str, Optional[dict]]) -> dict[str, Optional[float]]:
    """Macro across modalities (each modality the model has counts equally)."""
    out: dict[str, Optional[float]] = {}
    for ax in AXES:
        vals = [c[ax] for c in cells.values() if c is not None and c.get(ax) is not None]
        out[ax] = mean(vals) if vals else None
    return out


def discover_models(results_dir: Path) -> list[tuple[str, Path]]:
    """List all model dirs under the results directory."""
    if not results_dir.is_dir():
        return []
    return [(d.name, d) for d in sorted(results_dir.iterdir())
            if d.is_dir() and any(d.glob("*.jsonl"))]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m whowhen_eval.leaderboard",
        description="Score the leaderboard with per-MAS-averaged metrics.",
    )
    ap.add_argument("--results", type=Path, default=Path("./results"),
                    help="Results root: one subdirectory per model (default ./results)")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Optional subset of model directory names to include.")
    ap.add_argument("--modalities", nargs="*",
                    default=list(("text", "image", "image_gui", "video")),
                    help="Modalities to report (cells in the table).")
    args = ap.parse_args(argv)

    models = discover_models(args.results)
    if args.models:
        keep = set(args.models)
        models = [(n, p) for n, p in models if n in keep]
    if not models:
        print(f"No model directories found in {args.results}")
        return 1

    # Per-modality cells
    cells_per_model: dict[str, dict[str, Optional[dict]]] = {}
    composites: dict[str, dict[str, Optional[float]]] = {}
    for name, mdir in models:
        rows = collect(mdir)
        cells = {mod: cell_metrics([r for r in rows if r["modality"] == mod])
                 for mod in args.modalities}
        cells_per_model[name] = cells
        composites[name] = composite(cells)

    def fmt_axes(cell: Optional[dict]) -> str:
        if cell is None:
            return f"{'  —':^28}"
        bits = []
        for ax in AXES:
            v = cell.get(ax)
            bits.append(f"{v*100:5.1f}" if v is not None else "  —  ")
        return " " + " ".join(bits) + " "

    width = 28
    header_top = (f"{'Model':<22}|"
                  + "".join(f"{m.upper():^{width}}|" for m in args.modalities)
                  + f"{'COMPOSITE':^{width}}")
    header_sub = (f"{'':<22}|"
                  + (" ".join(f"{ax:>5}" for ax in AXES) + " " + "|")
                  * (len(args.modalities) + 1))
    print(header_top)
    print(header_sub)
    print("-" * len(header_top))

    for name, _ in models:
        line = f"{name:<22}|"
        for mod in args.modalities:
            line += fmt_axes(cells_per_model[name][mod]) + "|"
        comp = composites[name]
        any_missing = any(cells_per_model[name][mod] is None for mod in args.modalities)
        marker = "†" if any_missing else " "
        bits = []
        for ax in AXES:
            v = comp.get(ax)
            bits.append(f"{v*100:5.1f}" if v is not None else "  —  ")
        line += " " + " ".join(bits) + marker + "|"
        print(line)

    print()
    print("Notes:")
    print("  Who   = mean across multi-agent MASes of agent-attribution accuracy.")
    print("  When  = mean across MASes of step-localization accuracy.")
    print("  What  = macro-F1 over observed mode classes (global within each cell).")
    print("  All   = mean across MASes of joint (Who ∧ When ∧ What) accuracy.")
    print(f"  Single-agent MASes excluded from Who: {sorted(SINGLE_AGENT_FRAMEWORKS)}")
    print("  COMPOSITE = arithmetic mean across the listed modalities.")
    print("  †         = composite computed on a subset of modalities.")

    print()
    print("=== Leaderboard (sorted by composite All) ===")
    ranked = sorted(models, key=lambda nm: -(composites[nm[0]].get("All") or 0))
    for rank, (name, _) in enumerate(ranked, 1):
        c = composites[name]
        bits = " ".join(f"{ax}={(c[ax]*100):5.1f}%" if c[ax] is not None else f"{ax}=  —"
                        for ax in AXES)
        print(f"  {rank}. {name:<46} {bits}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
