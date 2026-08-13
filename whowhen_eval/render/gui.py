"""GUI-agent renderer (``image_gui`` split: coact, openai_cua, agentoccam, gemini).

One renderer serves all four GUI frameworks because the ``image_gui``
release writes a single shared trajectory schema for them:

    [
        {kind: "user", content, images?},
        {kind: "action", step_number, agent, reasoning, action,
                         observation, observation_images?, action_raw?,
                         error?, is_final_answer},
        ...
    ]

Step coordinate: flat ``step N`` — ``step_number`` is 1-indexed and
sequential, and ``ground_truth.step`` is written in the *same* coordinate
space (identity mapping; no scorer offset needed).

Topology: ``coact`` is multi-agent (orchestrator / coder / computer —
the hierarchical delegation is flattened into one sequential step list,
with each step labelled by its acting agent). The other three are
single-agent (``computer_use_agent`` for openai_cua on OSWorld,
``web_agent`` for agentoccam / gemini on WebVoyager).

Screenshot handling
-------------------
Every action step carries at most one screenshot in
``observation_images`` — the screen state the agent saw when it acted.
The block layout follows the pixelcraft text→image→body pattern::

    Step 7 | Agent: coder
    [screenshot]
    <inline image>
    [reasoning] ... [/reasoning]
    [action] ... [/action]
    [observation] url_before=... | url_after=... [/observation]

GUI screenshots need a *width*-capped resize rather than the
longest-side ``max_dim`` cap in ``pil_image_part``: WebVoyager
full-page captures run to 1920x6000+, and a longest-side cap of 1024
would shrink them to ~320px wide — unreadable UI text. Capping width at
1024 keeps text legible while tall pages stay tall (1920x6105 →
1024x3256; desktop 1920x1080 → 1024x576). JPEG q80 re-encode matches
the other image-split renderers' bandwidth budget.

The ``[observation]`` body carries the url/title transitions recorded by
the web frameworks (empty for OSWorld desktop traces). ``action_raw``
(the structured form of the action) is intentionally not rendered — the
human-readable ``action`` string is the judged surface, matching what
the agent's own model emitted.
"""
from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from .base import (
    RenderResult,
    StepCoord,
    TASK_ANCHOR,
    TranscriptBlock,
    task_image_parts,
)


# Width cap + JPEG quality for screenshots (see module docstring for why
# this is width-based rather than pil_image_part's longest-side cap).
_MAX_WIDTH = 1024
_JPEG_QUALITY = 80


def _screenshot_part(release_image: dict) -> dict[str, Any]:
    """Convert a release screenshot to an ``image_url`` part, capping
    *width* at ``_MAX_WIDTH`` (aspect preserved) and re-encoding JPEG."""
    data = release_image.get("data")
    if not isinstance(data, str) or not data:
        raise ValueError("release image entry has empty/non-str 'data'")
    try:
        from PIL import Image  # local import, keeps module import cheap
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Pillow required for screenshot resizing") from e
    img = Image.open(BytesIO(base64.b64decode(data)))
    if img.width > _MAX_WIDTH:
        new_h = max(1, round(img.height * _MAX_WIDTH / img.width))
        img = img.resize((_MAX_WIDTH, new_h))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, "JPEG", quality=_JPEG_QUALITY, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}


def render(release: dict) -> RenderResult:
    blocks: list[TranscriptBlock] = []
    step_index: list[tuple[str, StepCoord]] = []

    # Task images — GUI tasks are text-only prompts, so this is normally
    # empty; kept for schema parity with the other image renderers.
    task_imgs = task_image_parts(release)
    if task_imgs:
        blocks.append(TranscriptBlock(coord=TASK_ANCHOR, text="", images=task_imgs))

    for entry in release.get("trajectory") or []:
        kind = entry.get("kind")
        if kind != "action":
            # ``user`` is framing (rendered as the User Question section);
            # this schema has no other step kinds.
            continue

        step_number = entry.get("step_number")
        agent = str(entry.get("agent") or "agent")
        coord = str(step_number)

        reasoning = (entry.get("reasoning") or "").strip()
        action = (entry.get("action") or "").strip()
        observation = (entry.get("observation") or "").strip()
        error = entry.get("error")

        step_imgs: list[dict[str, Any]] = []
        for img in entry.get("observation_images") or []:
            if isinstance(img, dict) and img.get("data"):
                step_imgs.append(_screenshot_part(img))

        header = f"Step {coord} | Agent: {agent}"
        if step_imgs:
            header += "\n[screenshot]"

        body_parts: list[str] = []
        if reasoning:
            body_parts.append(f"[reasoning]\n{reasoning}\n[/reasoning]")
        if action:
            body_parts.append(f"[action]\n{action}\n[/action]")
        if observation:
            body_parts.append(f"[observation]\n{observation}\n[/observation]")
        if error:
            body_parts.append(f"[error]\n{error}\n[/error]")
        body = "\n".join(body_parts) if body_parts else "(empty step)"

        blocks.append(TranscriptBlock(
            coord=coord,
            text=header,
            images=step_imgs,
            body_text=body,
        ))
        native: StepCoord = (
            (step_number,) if isinstance(step_number, int) else (len(step_index) + 1,)
        )
        step_index.append((coord, native))

    # This schema has no ``final_answer`` framing entry — the terminal
    # action (``is_final_answer: true``) is the agent's last real step.
    final_answer = None

    fw = str(release.get("framework") or "")
    return RenderResult(
        blocks=blocks,
        step_format_hint=(
            "Step coords are sequential integers across the trajectory, "
            "1-indexed."
        ),
        step_index=step_index,
        trajectory_length=len(step_index),
        final_answer=final_answer,
        extras={
            "framework": fw,
            "benchmark": release.get("benchmark"),
            "modality": release.get("modality"),
            "topology": "multi" if fw == "coact" else "single",
            "agents": release.get("agents") or [],
        },
    )
