"""Round-trip test for ``whowhen_eval.score``.

For each (framework, sample-trace) pair: render the trace through the
framework's renderer to extract the actual ``Step <coord>`` string the LLM
would see at the ground-truth injection location, build a synthetic
prediction that matches GT exactly, run ``score()``, and assert all three
axes return True.

A failure here means either:
  (a) the renderer emits a step coord the scorer can't recompose from GT —
      i.e. a GT↔render misalignment; or
  (b) the GT has fields the scorer doesn't read.

A second case checks the ``accepted_predictions`` machinery on a macnet
C.3 trace, where two different (step, mode) pairs are both legitimate
answers.

Samples are discovered by streaming the dataset's JSONL splits
(``data/{text,image,video}.jsonl``) through the same index/row-loading
helpers the runner uses, so this test also exercises that path. Rows are
never all held in memory: the image split alone is several GB.

Usage::

    export WHOWHEN_DATA=/path/to/whowhen-pro     # dataset checkout root
    python -m tests.test_scorer                  # one trace per framework
    python -m tests.test_scorer --framework debate
    python -m tests.test_scorer --id bigcodebench_macnet_0002
    python -m tests.test_scorer --data-root /path/to/whowhen-pro
    pytest tests/test_scorer.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

from whowhen_eval.render import get_renderer
from whowhen_eval.run import (
    MODALITIES,
    TraceRef,
    iter_split_lines,
    iter_trace_refs,
    load_release,
    parse_row_meta,
    split_path,
)
from whowhen_eval.score import score


# The released frameworks. Every one must round-trip.
FRAMEWORKS = (
    "smolagents", "alfagent", "debate", "dylan", "macnet", "magentic-one",
    "mathchat", "metagpt", "pixelcraft", "dvd", "eva",
    # image_gui split (GUI agents)
    "coact", "openai_cua", "agentoccam", "gemini",
)

# Which split a framework's traces live in. Everything not listed here is
# in text.jsonl; the hint exists so that finding, say, a macnet sample
# never reads the multi-GB image split.
FRAMEWORK_SPLIT = {
    "pixelcraft": "image", "dvd": "video", "eva": "video",
    "coact": "image_gui", "openai_cua": "image_gui",
    "agentoccam": "image_gui", "gemini": "image_gui",
}

# Preferred sample: a C.3 (multi-agent over-reliance) trace, because it is
# the mode with the most scorer machinery behind it.
PREFERRED_MODE = "C.3"

# ``ground_truth`` is a JSON string *inside* the row, so its quotes arrive
# escaped. Probing the raw bytes lets the image split be scanned without
# ``json.loads``-ing rows that carry tens of MB of base64. A probe hit is
# re-checked against the parsed mode, so false positives are harmless;
# false negatives only cost the C.3 preference, never correctness.
_MODE_PROBE = b'mode\\":\\"' + PREFERRED_MODE.encode()

# Env var naming the dataset checkout, so no path is baked into the test.
DATA_ENV = "WHOWHEN_DATA"

# Pull the rendered "Step <coord> | Agent: <name>" out of a TranscriptBlock's
# text. We grep on the rendered string rather than poke at coord_str helpers
# directly, so the test verifies what the LLM actually sees.
_STEP_LINE_RE = re.compile(r"^\s*Step\s+(\S+?)\s*\|\s*Agent:\s*(\S+)", flags=re.MULTILINE)


def resolve_data_root(explicit: Optional[Path] = None) -> Path:
    """Dataset root from ``--data-root``, else ``$WHOWHEN_DATA``."""
    root = explicit or (Path(os.environ[DATA_ENV]) if os.environ.get(DATA_ENV) else None)
    if root is None:
        raise SystemExit(
            f"set {DATA_ENV} to your dataset checkout, or pass --data-root"
        )
    if not any(split_path(root, s).is_file() for s in MODALITIES):
        raise SystemExit(
            f"{root} does not look like a dataset checkout "
            "(no data/<split>.jsonl)"
        )
    return root


def _extract_rendered_steps(rr) -> list[tuple[str, str]]:
    """Return list of ``(coord, agent)`` pairs in trajectory order."""
    pairs: list[tuple[str, str]] = []
    for block in rr.blocks:
        for m in _STEP_LINE_RE.finditer(block.text or ""):
            pairs.append((m.group(1), m.group(2)))
    return pairs


def _gt_locator(gt: dict, framework: str) -> tuple[Optional[str], set[str]]:
    """Predict the rendered step coord (and acceptable agent set) for the
    GT's injection location, *based on the framework's documented
    rendering rule*. Used as the synthetic 'perfect prediction'.

    Returns ``(rendered_step_coord, acceptable_agents)``; the coord is
    ``None`` if the framework's coord isn't reconstructible without
    looking at the renderer's output.
    """
    agents: set[str] = set()
    a = gt.get("agent")
    if isinstance(a, str) and a:
        agents.add(a)
    if isinstance(gt.get("agents"), list):
        agents.update(str(x) for x in gt["agents"] if x)

    if framework in ("macnet", "pixelcraft", "debate", "dylan"):
        rd = gt.get("round")
        pos = gt.get("position")
        if rd is not None and pos is not None:
            return f"{rd}.{pos}", agents
        if rd is not None:
            return f"{rd}", agents  # debate/dylan: round-only acceptable
        return None, agents
    if framework in ("magentic-one", "smolagents", "alfagent",
                     "coact", "openai_cua", "agentoccam", "gemini"):
        # GT.step is already the rendered coord.
        s = gt.get("step")
        if framework in ("smolagents", "alfagent") and not agents:
            agents = {"agent"}  # renderer hardcodes the single-agent label
        if framework == "openai_cua" and not agents:
            agents = {"computer_use_agent"}  # render/gui.py per-step label
        if framework in ("agentoccam", "gemini") and not agents:
            agents = {"web_agent"}           # render/gui.py per-step label
        return (str(s) if s is not None else None), agents
    if framework == "mathchat":
        rd = gt.get("round")
        pos = gt.get("position")
        if rd is not None and pos is not None:
            return str(2 * int(rd) + int(pos)), agents
        return None, agents
    if framework == "metagpt":
        s = gt.get("stage")
        return (str(s) if s is not None else None), agents
    if framework == "dvd":
        # GT.step is native trajectory idx; renderer emits Step (idx - 2).
        s = gt.get("step")
        return (str(int(s) - 2) if s is not None else None), agents
    if framework == "eva":
        # GT.step is native trajectory idx; renderer emits dense Step
        # ((idx - 2) // 2) because each step folds (assistant + tool).
        s = gt.get("step")
        coord = str((int(s) - 2) // 2) if s is not None else None
        if not agents:
            agents = {"agent"}  # renderer hardcodes single-agent label
        return coord, agents
    return None, agents


def _coord_matches(rendered_coord: str, expected_coord: Optional[str],
                   framework: str) -> bool:
    """Whether ``rendered_coord`` matches what GT points at.

    For most frameworks: exact equality. For debate/dylan, GT only resolves
    to round-level — the rendered ``R.P`` matches if its round equals the
    expected round.
    """
    if not rendered_coord or not expected_coord:
        return False
    if framework in ("debate", "dylan"):
        return rendered_coord.split(".", 1)[0] == expected_coord
    return rendered_coord == expected_coord


def run_one(data_root: Path, ref: TraceRef) -> dict:
    # ``load_release`` decodes the row's JSON-string columns and, for the
    # video split, points ``__source_dir__`` at video_assets/<framework>/
    # so eva/dvd can resolve their relative frame paths.
    release = load_release(data_root, ref)
    framework = release.get("framework")
    gt = release.get("ground_truth") or {}

    rr = get_renderer(framework)(release)
    rendered = _extract_rendered_steps(rr)

    # GT → canonical (coord, agent_set) per the framework's rendering rule.
    expected_coord, expected_agents = _gt_locator(gt, framework)

    # Walk the actual rendered transcript and find ANY step block where:
    #   (a) coord matches the GT-derived coord (exact, or round-prefix for
    #       debate/dylan), AND
    #   (b) the agent label at that block is one of GT's agents.
    # If such a block exists, the GT genuinely lands on a real rendered
    # step with the right agent label.
    matched_block: Optional[tuple[str, str]] = None
    for r_coord, r_agent in rendered:
        if not _coord_matches(r_coord, expected_coord, framework):
            continue
        if expected_agents and r_agent.lower() not in {a.lower() for a in expected_agents}:
            continue
        matched_block = (r_coord, r_agent)
        break

    # Build the synthetic perfect prediction directly from the matched
    # rendered block (when found) — that's what the LLM would see.
    if matched_block:
        synth_step = f"step {matched_block[0]}"
        synth_agent = matched_block[1]
    else:
        # No literal match — fall back so the scorer still runs and we can
        # see *which* axis fails. Use the canonical (coord, first agent).
        synth_step = f"step {expected_coord}" if expected_coord else None
        synth_agent = next(iter(expected_agents)) if expected_agents else None

    pred = {
        "agent_name": synth_agent,
        "step_coord": synth_step,
        "error_mode": gt.get("mode"),
    }
    sc = score(pred, gt, framework)

    return {
        "trace_id": ref.trace_id,
        "split": ref.split,
        "framework": framework,
        "gt_summary": {k: gt.get(k) for k in
                       ("agent", "agents", "round", "position", "step", "stage", "mode")
                       if k in gt},
        "rendered_n_steps": len(rendered),
        "expected_coord": expected_coord,
        "expected_agents": sorted(expected_agents) if expected_agents else [],
        "matched_in_render": matched_block is not None,
        "matched_block": matched_block,
        "pred": pred,
        "score": sc,
        # Ok = a literal `Step <coord> | Agent: <gt_agent>` line was found
        # in the rendered transcript AND the scorer accepts it 3/3.
        "ok": matched_block is not None and sc == {"agent": True, "step": True, "mode": True},
    }


def _print_result(r: dict) -> None:
    flag = "OK" if r["ok"] else "FAIL"
    print(f"[{flag}] {r['framework']:13s} {r['trace_id']} ({r['split']})")
    print(f"        gt           = {r['gt_summary']}")
    print(f"        rendered     = {r['rendered_n_steps']} steps")
    print(f"        looking for  = coord={r['expected_coord']!r}, agent in {r['expected_agents']}")
    print(f"        found in ren = {r['matched_in_render']}  matched_block={r['matched_block']}")
    print(f"        synth pred   = {r['pred']}")
    print(f"        scorer       = {r['score']}")


def _row_ground_truth(line: bytes) -> dict:
    """``ground_truth`` of one raw row, or ``{}`` if it can't be read."""
    try:
        gt = json.loads(json.loads(line).get("ground_truth") or "{}")
    except Exception:  # noqa: BLE001 — a bad row is just not a candidate
        return {}
    return gt if isinstance(gt, dict) else {}


def _scan_split(
    data_root: Path,
    split: str,
    wanted: set[str],
    first: dict[str, TraceRef],
    preferred: dict[str, TraceRef],
) -> None:
    """Stream one split, filling ``first`` / ``preferred`` for ``wanted``.

    Stops as soon as every wanted framework has a ``PREFERRED_MODE``
    sample. On the image split, rows are parsed only when the byte probe
    says the label could be the preferred mode — the rest are skipped
    after a few hundred bytes of prefix.
    """
    path = split_path(data_root, split)
    if not path.is_file() or not wanted:
        return
    probe_first = split in ("image", "image_gui")  # rows here run to tens of MB
    for offset, line in iter_split_lines(path):
        meta = parse_row_meta(line)
        if meta is None:
            continue
        trace_id, fw, bench = meta
        if fw not in wanted:
            continue
        ref = TraceRef(split=split, offset=offset, trace_id=trace_id,
                       framework=fw, benchmark=bench)
        first.setdefault(fw, ref)
        if fw in preferred:
            continue
        if probe_first and _MODE_PROBE not in line:
            continue
        if _row_ground_truth(line).get("mode") == PREFERRED_MODE:
            preferred[fw] = ref
            if all(f in preferred for f in wanted):
                return


def default_samples(data_root: Path,
                    frameworks: Iterable[str] = FRAMEWORKS) -> list[TraceRef]:
    """One trace per framework. Prefers a C.3 case (multi-agent test)."""
    frameworks = tuple(frameworks)
    first: dict[str, TraceRef] = {}
    preferred: dict[str, TraceRef] = {}

    # Pass 1: each framework in the split it is expected to live in.
    homes: dict[str, set[str]] = {}
    for fw in frameworks:
        homes.setdefault(FRAMEWORK_SPLIT.get(fw, "text"), set()).add(fw)
    for split in MODALITIES:
        _scan_split(data_root, split, homes.get(split, set()), first, preferred)

    # Pass 2: whatever the hint missed — scan the other splits for it, so a
    # relocated framework degrades to "slower", not "not found".
    missing = {fw for fw in frameworks if fw not in first}
    for split in MODALITIES:
        elsewhere = {fw for fw in missing if FRAMEWORK_SPLIT.get(fw, "text") != split}
        if not elsewhere:
            continue
        _scan_split(data_root, split, elsewhere, first, preferred)
        missing = {fw for fw in frameworks if fw not in first}

    out: list[TraceRef] = []
    for fw in frameworks:
        ref = preferred.get(fw) or first.get(fw)
        if ref is None:
            raise SystemExit(f"no traces found for framework {fw!r} under {data_root}")
        out.append(ref)
    return out


def samples_for_framework(data_root: Path, framework: str,
                          limit: int = 3) -> list[TraceRef]:
    """First ``limit`` traces of one framework, in split/file order."""
    return list(itertools.islice(
        iter_trace_refs(data_root, framework=framework), limit))


def samples_for_ids(data_root: Path, ids: Iterable[str]) -> list[TraceRef]:
    """Look up specific traces by ``id``, streaming until all are found."""
    wanted = list(dict.fromkeys(ids))
    found: dict[str, TraceRef] = {}
    for ref in iter_trace_refs(data_root):
        if ref.trace_id in wanted:
            found[ref.trace_id] = ref
            if len(found) == len(wanted):
                break
    missing = [i for i in wanted if i not in found]
    if missing:
        raise SystemExit(f"no trace with id {missing} under {data_root}")
    return [found[i] for i in wanted]


# ---------------------------------------------------------------------------
# accepted_predictions: macnet C.3
# ---------------------------------------------------------------------------


def _find_macnet_c3_with_alternates(data_root: Path) -> Optional[tuple[TraceRef, dict]]:
    """A macnet C.3 trace whose alternates include the ``(R.P+1, C.3)``
    critic-turn reading and an ``(R.P, R.*)`` critic-content reading.

    macnet is a text-split framework, so this streams text.jsonl and
    stops at the first qualifying row.
    """
    split = FRAMEWORK_SPLIT.get("macnet", "text")
    path = split_path(data_root, split)
    if not path.is_file():
        return None
    for offset, line in iter_split_lines(path):
        meta = parse_row_meta(line)
        if meta is None or meta[1] != "macnet":
            continue
        gt = _row_ground_truth(line)
        if gt.get("mode") != "C.3":
            continue
        aps = gt.get("accepted_predictions") or []
        has_c3_alt = any(a.get("mode") == "C.3" and a.get("step_coord") == "1.1" for a in aps)
        has_r_alt = any(str(a.get("mode", "")).startswith("R.")
                        and a.get("step_coord") == "1.0" for a in aps)
        if has_c3_alt and has_r_alt:
            trace_id, fw, bench = meta
            return TraceRef(split=split, offset=offset, trace_id=trace_id,
                            framework=fw, benchmark=bench), gt
    return None


def check_macnet_accepted_alternates(data_root: Path) -> dict:
    """Both accepted alternates must score ``step=True`` and ``mode=True``.

    ``accepted_predictions`` is scored per-axis against the union of GT +
    alternates, so a judge that reads the injected critic turn (step 1.1,
    C.3) and one that reads the critic's fabricated content (step 1.0,
    R.*) are both credited.
    """
    found = _find_macnet_c3_with_alternates(data_root)
    if found is None:
        return {"ok": False, "detail": "no macnet C.3 trace with both alternates"}
    ref, gt = found
    agent = str(gt.get("agent") or "node_1")

    results = {}
    for label, coord, mode in (
        ("critic-turn (1.1, C.3)", "1.1", "C.3"),
        ("critic-content (1.0, R.*)", "1.0",
         next(a["mode"] for a in gt["accepted_predictions"]
              if str(a.get("mode", "")).startswith("R.") and a.get("step_coord") == "1.0")),
    ):
        sc = score({"agent_name": agent, "step_coord": f"step {coord}",
                    "error_mode": mode}, gt, "macnet")
        results[label] = sc

    ok = all(sc["step"] and sc["mode"] for sc in results.values())
    return {"ok": ok, "trace_id": ref.trace_id, "results": results}


# ---------------------------------------------------------------------------
# pytest entrypoints
# ---------------------------------------------------------------------------


def _pytest_data_root() -> Path:
    if not os.environ.get(DATA_ENV):
        import pytest
        pytest.skip(f"{DATA_ENV} not set")
    return resolve_data_root()


def test_round_trip_all_frameworks() -> None:
    data_root = _pytest_data_root()
    failures = [r for r in (run_one(data_root, ref)
                            for ref in default_samples(data_root))
                if not r["ok"]]
    assert not failures, [(r["framework"], r["score"]) for r in failures]


def test_macnet_accepted_alternates() -> None:
    data_root = _pytest_data_root()
    outcome = check_macnet_accepted_alternates(data_root)
    assert outcome["ok"], outcome


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=None,
                    help=f"Dataset checkout root (default: ${DATA_ENV})")
    ap.add_argument("--framework", default=None,
                    help="Only test this framework (first 3 traces)")
    ap.add_argument("--id", dest="ids", nargs="*", default=None,
                    help="Only test these trace ids")
    args = ap.parse_args(argv)

    data_root = resolve_data_root(args.data_root)

    if args.ids:
        refs = samples_for_ids(data_root, args.ids)
    elif args.framework:
        refs = samples_for_framework(data_root, args.framework)
    else:
        refs = default_samples(data_root)

    n_pass = n_fail = 0
    for ref in refs:
        try:
            r = run_one(data_root, ref)
        except Exception as e:  # noqa: BLE001
            print(f"[ERR ] {ref.trace_id}: {type(e).__name__}: {e}")
            n_fail += 1
            continue
        _print_result(r)
        if r["ok"]:
            n_pass += 1
        else:
            n_fail += 1

    # accepted_predictions case (skipped when a subset was requested)
    if not args.ids and not args.framework:
        outcome = check_macnet_accepted_alternates(data_root)
        flag = "OK" if outcome["ok"] else "FAIL"
        print(f"\n[{flag}] macnet C.3 accepted_predictions alternates")
        if "trace_id" in outcome:
            print(f"        trace        = {outcome['trace_id']}")
            for label, sc in outcome["results"].items():
                print(f"        {label:26s} -> {sc}")
        else:
            print(f"        {outcome['detail']}")
        if outcome["ok"]:
            n_pass += 1
        else:
            n_fail += 1

    print("\n=== summary ===")
    print(f"  pass: {n_pass}")
    print(f"  fail: {n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
