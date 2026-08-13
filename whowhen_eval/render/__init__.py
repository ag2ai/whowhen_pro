"""Per-framework trajectory renderers.

Each release JSON declares its ``framework`` (``smolagents``, ``debate``,
``dylan``, ``macnet``, ``mathchat``, ``metagpt``, ``magentic`` /
``magentic-one``, ``pixelcraft``, ``dvd``, ``eva``, plus the GUI four:
``coact``, ``openai_cua``, ``agentoccam``, ``gemini``); ``get_renderer()``
dispatches to the matching module's ``render()`` function.

A renderer's job is to turn the release JSON into a ``RenderResult`` of
``TranscriptBlock``s — each block bundles a step coordinate, the step's
multi-line text rendering, and any image parts attached to the step. The
prompt-building functions in ``whowhen_eval.prompts`` walk these blocks to produce
the final OpenAI-style content-parts list (interleaved text + image_url
parts).

The renderer must NEVER expose the ``is_injected`` / ``is_replayed`` flags
or any per-turn ``input_messages`` in the rendered output — the judge sees
only the trace.
"""
from __future__ import annotations

from .base import (  # noqa: F401  (re-export)
    RenderResult,
    StepCoord,
    TASK_ANCHOR,
    TranscriptBlock,
)


def get_renderer(framework: str):
    """Return the ``render`` callable for ``framework``.

    Lazy-imports each framework module so a missing dependency or syntax
    error in one framework's renderer doesn't break the dispatch table for
    the others.
    """
    fw = (framework or "").strip().lower()
    if fw == "smolagents":
        from . import smolagents as mod
        return mod.render
    if fw == "alfagent":
        from . import alfagent as mod
        return mod.render
    if fw == "debate":
        from . import debate as mod
        return mod.render
    if fw == "dylan":
        from . import dylan as mod
        return mod.render
    if fw == "macnet":
        from . import macnet as mod
        return mod.render
    if fw == "mathchat":
        from . import mathchat as mod
        return mod.render
    if fw == "metagpt":
        from . import metagpt as mod
        return mod.render
    if fw in ("magentic", "magentic-one"):
        from . import magentic as mod
        return mod.render
    if fw == "pixelcraft":
        from . import pixelcraft as mod
        return mod.render
    if fw == "dvd":
        from . import dvd as mod
        return mod.render
    if fw == "eva":
        from . import eva as mod
        return mod.render
    if fw in ("coact", "openai_cua", "agentoccam", "gemini"):
        # The four GUI frameworks (image_gui split) share one release
        # trajectory schema, so a single renderer serves them all.
        from . import gui as mod
        return mod.render
    raise ValueError(f"no renderer registered for framework={framework!r}")
