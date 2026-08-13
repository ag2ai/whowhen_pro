"""All-at-once attribution eval runner.

Reads the dataset's JSONL splits — ``<data-root>/data/text.jsonl``,
``image.jsonl``, ``image_gui.jsonl`` and ``video.jsonl``, one row per
trace — builds an all-at-once prompt for each trace via
:func:`whowhen_eval.prompts.all_at_once`, sends it to the requested model
through LiteLLM, parses the response, scores it, and appends one JSONL
record per trace to::

    <out>/<model_safe>/<benchmark>.jsonl

Dataset layout
--------------
::

    <data-root>/
    ├── data/text.jsonl          # one row per trace
    ├── data/image.jsonl         # images are embedded base64 in the rows
    ├── data/video.jsonl
    ├── video_assets/<framework>/assets/frames/...
    └── taxonomy.yaml

A row is ``{"id", "framework", "benchmark", "task", "trajectory",
"ground_truth", "extras"}``, where the last four are JSON-encoded
*strings*; :func:`release_from_row` decodes them back into the release
dict the renderers expect (``modality`` is the split name, and video rows
get ``__source_dir__`` pointing at ``video_assets/<framework>/`` so
dvd/eva can resolve their relative frame paths).

Memory
------
``image.jsonl`` is multiple GB, so nothing ever holds a whole split in
memory. Discovery streams each split once in binary mode and keeps only a
``TraceRef`` per row — ``(split, byte offset, id, framework, benchmark)``,
read from the line's prefix without JSON-parsing it. Evaluation re-opens
the split, seeks to the offset and parses that single row.

Concurrency
-----------
``litellm.completion`` is synchronous, so we dispatch via
``asyncio.to_thread`` and bound concurrent calls with an
``asyncio.Semaphore`` — the in-flight count is exactly ``--concurrency``
(default 8).

Resumability
------------
Before dispatching, we read the destination JSONL and collect every
already-stored ``trace_id``. Those traces are skipped. Pass
``--no-resume`` to start clean (the JSONL is left in place; new records
just get appended after the existing ones).

Examples
--------
::

    export OPENAI_API_KEY=sk-...
    python -m whowhen_eval.run --model gpt-5.4 --data-root ./whowhen-pro

    export GEMINI_API_KEY=...
    python -m whowhen_eval.run --model gemini/gemini-3-flash-preview \\
        --data-root ./whowhen-pro --modality image --concurrency 16

    python -m whowhen_eval.run --model gpt-5.4 --data-root ./whowhen-pro \\
        --benchmark charxiv --framework pixelcraft --max-traces 5 --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from .llm import generate
from .parse import parse_all_at_once
from .prompts import Taxonomy, all_at_once, load_taxonomy, user_msg
from .render import get_renderer
from .score import score as score_prediction
from .store import ResultsStore

# The dataset splits. A split name is also the trace's modality.
# ``image_gui`` is the GUI-agent split (desktop/web screenshots); its
# images are embedded base64 in the rows, like ``image``.
MODALITIES = ("text", "image", "image_gui", "video")

# Frameworks with a ``render(release)`` implementation in
# ``whowhen_eval.render``. Only used to sanity-check ``--framework``.
FRAMEWORKS = (
    "smolagents", "alfagent", "debate", "dylan", "macnet", "magentic-one",
    "mathchat", "metagpt", "pixelcraft", "dvd", "eva",
    # image_gui split (GUI agents; one shared renderer)
    "coact", "openai_cua", "agentoccam", "gemini",
)

# The build writes rows with compact separators and ``id``/``framework``/
# ``benchmark`` first, so the three metadata fields can be read off the
# line's first few hundred bytes — no need to JSON-parse a row whose
# embedded base64 images can run to tens of MB. Any line that doesn't
# match (escapes in a value, a differently-ordered writer) falls back to a
# full ``json.loads`` in :func:`parse_row_meta`, so this is an
# optimisation only, never a source of silently wrong metadata.
_ROW_META_RE = re.compile(
    rb'\{"id":"([^"\\]*)","framework":"([^"\\]*)","benchmark":"([^"\\]*)"'
)
_META_PREFIX_BYTES = 512


# ---------------------------------------------------------------------------
# Trace discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceRef:
    """Where one trace lives, plus the metadata the filters need.

    Holding a byte offset rather than the row itself is what keeps the
    runner's footprint flat across the multi-GB image split.
    """
    split: str          # text | image | video — also the trace's modality
    offset: int         # byte offset of the row's first byte in the split
    trace_id: str
    framework: str
    benchmark: str


def split_path(data_root: Path, split: str) -> Path:
    """Path of one split's JSONL under a dataset checkout."""
    return data_root / "data" / f"{split}.jsonl"


def parse_row_meta(line: bytes) -> Optional[tuple[str, str, str]]:
    """``(id, framework, benchmark)`` for one raw JSONL line.

    Tries the cheap prefix match first and falls back to a full parse, so
    a row this function can't shortcut is read correctly rather than
    dropped. Returns ``None`` only for a line that isn't a usable row.
    """
    m = _ROW_META_RE.match(line, 0, _META_PREFIX_BYTES)
    if m is not None:
        try:
            trace_id, framework, benchmark = (g.decode("utf-8") for g in m.groups())
        except UnicodeDecodeError:
            pass
        else:
            if trace_id:
                return trace_id, framework, benchmark
    try:
        row = json.loads(line)
    except Exception:  # noqa: BLE001 — a torn/blank line is not fatal
        return None
    if not isinstance(row, dict) or not row.get("id"):
        return None
    return (str(row["id"]), str(row.get("framework") or ""),
            str(row.get("benchmark") or ""))


def iter_split_lines(path: Path) -> Iterator[tuple[int, bytes]]:
    """Yield ``(byte_offset, raw_line)`` for every non-empty line.

    Binary mode with an explicit ``tell()`` before each ``readline()``:
    the offset recorded is exactly the one ``seek()`` needs later, and at
    most one line is resident at a time.
    """
    with path.open("rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                return
            if line.strip():
                yield offset, line


def iter_trace_refs(
    data_root: Path,
    benchmark: str = "all",
    framework: Optional[str] = None,
    modality: Optional[str] = None,
) -> Iterator[TraceRef]:
    """Stream a :class:`TraceRef` for every trace matching the filters.

    ``benchmark="all"`` covers every benchmark in every split;
    ``modality`` (text/image/video) restricts the walk to one split. The
    error mode is *not* part of the index — it lives in
    ``ground_truth.mode`` and is read when the row itself is parsed.
    """
    if not (data_root / "data").is_dir():
        raise FileNotFoundError(
            f"no data/ directory under {data_root} — --data-root must point at "
            "the root of the dataset checkout (data/<split>.jsonl, "
            "video_assets/, taxonomy.yaml)"
        )
    splits = [modality] if modality else list(MODALITIES)
    present = [s for s in splits if split_path(data_root, s).is_file()]
    if not present:
        raise FileNotFoundError(
            "no split JSONL found — expected "
            + ", ".join(f"data/{s}.jsonl" for s in splits)
            + f" under {data_root}"
        )
    for split in present:
        path = split_path(data_root, split)
        n_bad = 0
        for offset, line in iter_split_lines(path):
            meta = parse_row_meta(line)
            del line  # drop the row bytes as soon as the metadata is out
            if meta is None:
                n_bad += 1
                continue
            trace_id, fw, bench = meta
            if framework and fw != framework:
                continue
            if benchmark != "all" and bench != benchmark:
                continue
            yield TraceRef(split=split, offset=offset, trace_id=trace_id,
                           framework=fw, benchmark=bench)
        if n_bad:
            print(f"[warn] {n_bad} unreadable line(s) in {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Row -> release
# ---------------------------------------------------------------------------


def _decode(value: Any, default: Any) -> Any:
    """Decode one JSON-encoded row field, tolerating already-decoded
    values (e.g. a row handed over by ``datasets`` rather than read raw)."""
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    return json.loads(value)


def release_from_row(row: dict, split: str, data_root: Path) -> dict:
    """Rebuild the release dict the renderers expect from one JSONL row.

    ``extras`` is spliced in at the top level, which is where the
    framework-specific keys (agent roster, topology, ...) lived in the
    per-file layout the renderers were written against.
    """
    release: dict[str, Any] = {
        "id": row.get("id"),
        "framework": row.get("framework"),
        "benchmark": row.get("benchmark"),
        "modality": split,
        "task": _decode(row.get("task"), {}),
        "trajectory": _decode(row.get("trajectory"), []),
        "ground_truth": _decode(row.get("ground_truth"), {}),
    }
    extras = _decode(row.get("extras"), {})
    if isinstance(extras, dict):
        release.update(extras)
    elif extras:
        release["extras"] = extras
    if split == "video":
        # dvd/eva rows carry frame paths relative to their framework's
        # shared asset store (``assets/frames/...``).
        release["__source_dir__"] = str(
            data_root / "video_assets" / str(row.get("framework") or "")
        )
    return release


def load_row(data_root: Path, ref: TraceRef) -> dict:
    """Read and parse exactly the one row ``ref`` points at."""
    with split_path(data_root, ref.split).open("rb") as f:
        f.seek(ref.offset)
        line = f.readline()
    row = json.loads(line)
    if row.get("id") != ref.trace_id:
        raise ValueError(
            f"stale offset: data/{ref.split}.jsonl@{ref.offset} holds "
            f"id={row.get('id')!r}, expected {ref.trace_id!r}"
        )
    return row


def load_release(data_root: Path, ref: TraceRef) -> dict:
    """Load one trace and reconstruct its release dict."""
    return release_from_row(load_row(data_root, ref), ref.split, data_root)


# ---------------------------------------------------------------------------
# Per-trace eval
# ---------------------------------------------------------------------------


def build_prompt(release: dict, framework: str,
                 taxonomy: Taxonomy) -> list[dict[str, Any]]:
    """Render the trace and assemble the all-at-once content parts."""
    rr = get_renderer(framework)(release)
    return all_at_once(
        rr,
        problem=(release.get("task") or {}).get("query") or "",
        taxonomy=taxonomy,
    )


_NO_USAGE = {"input_tokens": None, "output_tokens": None, "total_tokens": None}


async def evaluate_one(
    *,
    sem: asyncio.Semaphore,
    model: str,
    data_root: Path,
    ref: TraceRef,
    taxonomy: Taxonomy,
    temperature: float,
    max_tokens: int,
    reasoning_effort: Optional[str],
    dry_run: bool,
) -> dict[str, Any]:
    """Evaluate one trace and return a JSONL-ready record dict.

    The row is read here, one seek at a time, so the caller only ever
    holds the lightweight :class:`TraceRef` index. Catches every
    exception so a single bad trace can't kill the run — the failure is
    captured in ``record["error"]`` and the row is still written so
    downstream analysis can flag it.
    """
    base: dict[str, Any] = {
        "protocol": "all_at_once",
        "model": model,
        "trace_id": ref.trace_id,
        "framework": ref.framework,
        "benchmark": ref.benchmark,
        "modality": ref.split,
        "error_mode": None,
        "trace_path": f"data/{ref.split}.jsonl",
        "ground_truth": None,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    try:
        release = load_release(data_root, ref)
    except Exception as e:  # noqa: BLE001 — surface as record-level error
        return {**base, "raw_output": None, "prediction": None, "score": None,
                "usage": dict(_NO_USAGE), "duration_s": 0.0,
                "error": f"row_load: {type(e).__name__}: {e}"}

    gt = release.get("ground_truth")
    base["ground_truth"] = gt
    base["error_mode"] = gt.get("mode") if isinstance(gt, dict) else None

    try:
        parts = build_prompt(release, ref.framework, taxonomy)
    except Exception as e:  # noqa: BLE001 — surface as record-level error
        return {**base, "raw_output": None, "prediction": None, "score": None,
                "usage": dict(_NO_USAGE), "duration_s": 0.0,
                "error": f"prompt_build: {type(e).__name__}: {e}"}

    if dry_run:
        # Record the prompt size so smoke tests can sanity-check the cell
        # distribution without burning tokens.
        n_parts = len(parts)
        n_imgs = sum(1 for p in parts if p.get("type") == "image_url")
        n_chars = sum(len(p.get("text") or "") for p in parts if p.get("type") == "text")
        return {**base, "raw_output": None, "prediction": None, "score": None,
                "usage": dict(_NO_USAGE), "duration_s": 0.0,
                "dry_run": {"parts": n_parts, "images": n_imgs, "text_chars": n_chars},
                "error": None}

    t0 = time.monotonic()
    raw: Optional[str] = None
    usage = dict(_NO_USAGE)
    err: Optional[str] = None
    try:
        async with sem:
            raw, usage = await asyncio.to_thread(
                generate,
                model,
                [user_msg(parts)],
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    duration = time.monotonic() - t0

    pred_dict: Optional[dict[str, Any]] = None
    if raw is not None:
        parsed = parse_all_at_once(raw, taxonomy.codes)
        pred_dict = {
            "agent_name": parsed.agent_name,
            "step_coord": parsed.step_coord,
            "error_mode": parsed.error_mode,
            "reason": parsed.reason,
            "parse_warnings": parsed.parse_warnings,
        }

    return {
        **base,
        "raw_output": raw,
        "prediction": pred_dict,
        "score": score_prediction(pred_dict, release.get("ground_truth"),
                                  ref.framework),
        "usage": usage,
        "duration_s": round(duration, 3),
        "error": err,
    }


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> int:
    taxonomy = load_taxonomy(args.data_root)
    print(f"[taxonomy] {len(taxonomy.codes)} modes: {', '.join(taxonomy.codes)}",
          file=sys.stderr)

    sem = asyncio.Semaphore(args.concurrency)

    if args.framework and args.framework not in FRAMEWORKS:
        print(f"[warn] unknown --framework {args.framework!r}; known frameworks: "
              + ", ".join(FRAMEWORKS), file=sys.stderr)

    # Bucket trace refs by benchmark so each benchmark's JSONL is one
    # store. Only the refs are kept — the rows stay on disk until eval.
    by_bench: dict[str, list[TraceRef]] = {}
    for ref in iter_trace_refs(
        args.data_root,
        benchmark=args.benchmark,
        framework=args.framework,
        modality=args.modality,
    ):
        by_bench.setdefault(ref.benchmark, []).append(ref)

    if not by_bench:
        print(f"No traces matched benchmark={args.benchmark} "
              f"framework={args.framework} modality={args.modality}",
              file=sys.stderr)
        return 1

    total_done = 0
    total_skipped = 0
    total_failed = 0

    for bench, items in by_bench.items():
        store = ResultsStore.for_cell(args.out, args.model, bench)
        already = store.done_trace_ids() if args.resume else set()

        # Filter undone, then cap by --max-traces.
        candidates = [it for it in items if it.trace_id not in already]
        n_resumed = len(items) - len(candidates)
        pending = candidates[: args.max_traces] if args.max_traces is not None else candidates
        n_capped = len(candidates) - len(pending)

        total_skipped += n_resumed
        cap_note = f", {n_capped} held back by --max-traces" if n_capped else ""
        print(f"[{bench}] {len(pending)} pending, {n_resumed} skipped "
              f"(already on disk){cap_note} -> {store.path}",
              file=sys.stderr)
        if not pending:
            continue

        async def _one(ref: TraceRef):
            record = await evaluate_one(
                sem=sem, model=args.model, data_root=args.data_root,
                ref=ref, taxonomy=taxonomy,
                temperature=args.temperature, max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
                dry_run=args.dry_run,
            )
            # Dry-run records carry no LLM output; don't pollute the JSONL.
            if not args.dry_run:
                store.append(record)
            return record

        n_done = n_failed = 0
        n_total = len(pending)
        t0 = time.monotonic()
        # Fire all coros concurrently; the semaphore caps in-flight LLM calls.
        coros = [_one(ref) for ref in pending]
        for fut in asyncio.as_completed(coros):
            rec = await fut
            if rec.get("error"):
                n_failed += 1
            n_done += 1
            if n_done % max(1, n_total // 20) == 0 or n_done == n_total:
                elapsed = time.monotonic() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (n_total - n_done) / rate if rate > 0 else 0
                print(f"  [{bench}] {n_done}/{n_total} "
                      f"({100 * n_done / n_total:.0f}%) "
                      f"failed={n_failed} {rate:.2f}/s eta={eta:.0f}s",
                      file=sys.stderr)
        total_done += n_done
        total_failed += n_failed

    print(
        f"\n=== run summary ===\n"
        f"  benchmarks: {len(by_bench)}\n"
        f"  done:       {total_done}\n"
        f"  skipped:    {total_skipped} (already on disk)\n"
        f"  failed:     {total_failed}"
        + ("\n  (dry run — nothing written)" if args.dry_run else ""),
        file=sys.stderr,
    )
    return 0 if total_failed == 0 else 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m whowhen_eval.run",
        description="All-at-once failure-attribution eval runner",
    )
    p.add_argument("--model", required=True,
                   help="LiteLLM model id. Examples: gpt-5.4, "
                        "claude-sonnet-4-6, gemini/gemini-3-flash-preview, "
                        "xai/grok-4, deepseek/deepseek-chat. Set the "
                        "provider's API key in the environment "
                        "(OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, ...)")
    p.add_argument("--data-root", required=True, type=Path,
                   help="Root of the dataset checkout (holds "
                        "data/{text,image,video}.jsonl, video_assets/ and "
                        "taxonomy.yaml)")
    p.add_argument("--benchmark", default="all",
                   help='Benchmark name (e.g. "mmsearch") or "all" (default)')
    p.add_argument("--modality", default=None, choices=list(MODALITIES),
                   help="Optional: limit to one modality")
    p.add_argument("--framework", default=None,
                   help="Optional: limit to one framework (smolagents, pixelcraft, ...)")
    p.add_argument("--concurrency", type=int, default=8,
                   help="Max concurrent LLM calls (default 8)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=2048,
                   help="Output token cap. Default 2048 leaves headroom for "
                        "reasoning models, whose internal thinking counts "
                        "toward this budget.")
    p.add_argument("--reasoning-effort", default=None,
                   help="Reasoning effort, passed through to the provider "
                        "(e.g. minimal/low/medium/high). Dropped "
                        "automatically for models that don't accept it.")
    p.add_argument("--out", type=Path, default=Path("./results"),
                   help="Output root (default: ./results)")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="Don't skip trace_ids already on disk")
    p.set_defaults(resume=True)
    p.add_argument("--dry-run", action="store_true",
                   help="Build prompts but don't call the LLM (writes nothing)")
    p.add_argument("--max-traces", type=int, default=None,
                   help="Cap pending traces per benchmark (smoke-testing knob)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
