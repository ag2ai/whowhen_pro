"""Per-framework scoring of attribution predictions.

Public entrypoint: ``score(pred, ground_truth, framework) -> dict``.

Returns a flat 3-bool dict::

    {"agent": bool, "step": bool, "mode": bool}

Each axis is judged independently. An axis is True if the predicted value
matches any acceptable value on that axis — drawn from the canonical GT
fields plus any ``ground_truth.accepted_predictions`` alternates. We do
not require the predicted (agent, step, mode) tuple to be coherent: e.g.
the model can match agent + mode from different alternates, and still
get both booleans set. If you want strict tuple-matching, recompose from
the same data — the per-axis representation is just the lossless cell.

Per-framework dispatch lives in ``_REGISTRY``. A framework needs its own
scorer whenever its ``ground_truth`` coordinate is written in the *native*
trajectory's index space while the renderer emits a different one. Those
mappings are the load-bearing part of this file:

===============  ================================================
framework        GT → rendered step coordinate
===============  ================================================
smolagents       ``gt.step`` (identity)
alfagent         ``gt.step`` (identity)
mathchat         ``2 * gt.round + gt.position``
metagpt          ``gt.stage``
dvd              ``gt.step - 2``   (trajectory[0..1] are framing)
eva              ``(gt.step - 2) // 2``  (assistant+tool fold into 1 step)
debate, dylan    round only — any position within ``gt.round`` counts
openai_cua       ``gt.step`` (identity; agent axis = fixed label)
agentoccam       ``gt.step`` (identity; agent axis = fixed label)
gemini           ``gt.step`` (identity; agent axis = fixed label)
default          ``gt.step_coord`` / ``gt.step`` / ``round.position``
===============  ================================================

``macnet``, ``pixelcraft`` and ``magentic-one`` use the default: their GT
already carries the rendered coordinate (``round``/``position``, or a
pre-composed ``step`` string). ``coact`` (image_gui split) also uses the
default: multi-agent, GT carries the named agent plus a 1-indexed
``step`` that IS the rendered coordinate.
"""
from __future__ import annotations

from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Normalizers
# ---------------------------------------------------------------------------


def _norm_str(s: Any) -> str:
    return str(s).strip().lower() if s else ""


def _norm_mode(s: Any) -> str:
    return _norm_str(s).upper()


def _norm_step(s: Any) -> str:
    """Strip the ``step ``/``round ``/``turn `` prefix and lowercase.

    ``"step 1.0"`` and ``"1.0"`` both map to ``"1.0"``; ``"round 1"`` maps
    to ``"1"``. Renderers and parsers may emit either form, so we collapse
    both before comparing.

    Accepts integer input (e.g. ``0``, ``1``) — ``_norm_str`` would treat
    integer 0 as falsy and return ``""``, which is wrong for step coords
    where 0 is a valid value (PL.1 traces point at a planning step at
    coord 0).
    """
    if s is None or s == "":
        return ""
    out = str(s).strip().lower()
    for prefix in ("step ", "round ", "turn "):
        if out.startswith(prefix):
            return out[len(prefix):].strip()
    return out


def _agents_from_field(value: Any) -> set[str]:
    if isinstance(value, str):
        return {_norm_str(value)} if value else set()
    if isinstance(value, list):
        return {_norm_str(x) for x in value if x}
    return set()


def _agents_from_gt(gt: dict) -> set[str]:
    """Multi-agent C.3 in debate/dylan stores the answer in ``agents``
    (list); single-agent traces store it in ``agent`` (string). Accept
    either."""
    out: set[str] = set()
    out |= _agents_from_field(gt.get("agent"))
    out |= _agents_from_field(gt.get("agents"))
    return out


def _step_from_gt(gt: dict) -> str:
    """Compose the canonical step string from ``round``/``position``.

    Frameworks differ in granularity:
    - macnet, pixelcraft: ``round`` + ``position`` (e.g. ``1.0``).
    - debate, dylan: ``round`` only (e.g. ``1``).
    - magentic-one: ``step`` (string, may already be normalized).
    """
    if gt.get("step_coord") not in (None, ""):
        return _norm_step(gt["step_coord"])
    if gt.get("step") not in (None, ""):
        return _norm_step(gt["step"])
    rd = gt.get("round")
    pos = gt.get("position")
    if rd is None:
        return ""
    if pos is None:
        return str(rd)
    return f"{rd}.{pos}"


def _accepted(gt: dict, ok_agents: set[str], ok_steps: set[str],
              ok_modes: set[str]) -> None:
    """Fold ``ground_truth.accepted_predictions`` into the three accept sets.

    Alternates are read per-axis: an alternate contributes its agent to
    ``ok_agents``, its coord to ``ok_steps`` and its mode to ``ok_modes``
    independently, matching the per-axis scoring contract described in the
    module docstring.
    """
    for ap in gt.get("accepted_predictions") or []:
        a = _norm_str(ap.get("agent_name"))
        if a:
            ok_agents.add(a)
        s = _norm_step(ap.get("step_coord"))
        if s:
            ok_steps.add(s)
        m = _norm_mode(ap.get("mode"))
        if m:
            ok_modes.add(m)


def _verdict(pred: dict, ok_agents: set[str], ok_steps: set[str],
             ok_modes: set[str]) -> dict:
    """Compare the parsed prediction against the three accept sets."""
    pa = _norm_str(pred.get("agent_name"))
    ps = _norm_step(pred.get("step_coord"))
    pm = _norm_mode(pred.get("error_mode"))
    return {
        "agent": bool(pa) and pa in ok_agents,
        "step": bool(ps) and ps in ok_steps,
        "mode": bool(pm) and pm in ok_modes,
    }


_MISS = {"agent": False, "step": False, "mode": False}


# ---------------------------------------------------------------------------
# Default scorer (macnet, pixelcraft, magentic-one)
# ---------------------------------------------------------------------------


def _default_score(pred: Optional[dict], gt: dict) -> dict:
    if not pred:
        return dict(_MISS)

    ok_agents = _agents_from_gt(gt)
    ok_steps: set[str] = set()
    canon_step = _step_from_gt(gt)
    if canon_step:
        ok_steps.add(canon_step)
    ok_modes: set[str] = set()
    canon_mode = _norm_mode(gt.get("mode"))
    if canon_mode:
        ok_modes.add(canon_mode)

    _accepted(gt, ok_agents, ok_steps, ok_modes)
    return _verdict(pred, ok_agents, ok_steps, ok_modes)


# ---------------------------------------------------------------------------
# Per-framework registry
# ---------------------------------------------------------------------------


ScoreFn = Callable[[Optional[dict], dict], dict]
_REGISTRY: dict[str, ScoreFn] = {}


def register(framework: str) -> Callable[[ScoreFn], ScoreFn]:
    def decorator(fn: ScoreFn) -> ScoreFn:
        _REGISTRY[framework] = fn
        return fn
    return decorator


def score(pred: Optional[dict], gt: Optional[dict], framework: str) -> dict:
    """Dispatch to the framework's scorer (or the shared default)."""
    fn = _REGISTRY.get(framework, _default_score)
    return fn(pred, gt or {})


# ---------------------------------------------------------------------------
# Smolagents
# ---------------------------------------------------------------------------
#
# Single-agent framework: the renderer hardcodes ``Agent: agent`` for every
# step (see render/smolagents.py: ``framework_agent = "agent"``). GT does
# not record an agent name (``agent: None``) because there's only one. So
# the agent axis: pred matches the literal renderer label ``"agent"``.
# Step axis: GT records ``step`` directly as a flat int in the renderer's
# own index space.


@register("smolagents")
def _score_smolagents(pred: Optional[dict], gt: dict) -> dict:
    if not pred:
        return dict(_MISS)

    canon_mode = _norm_mode(gt.get("mode"))
    canon_step = str(gt.get("step")) if gt.get("step") is not None else ""

    ok_agents = {"agent"}  # what the renderer emits
    ok_steps = {canon_step} if canon_step else set()
    ok_modes = {canon_mode} if canon_mode else set()

    _accepted(gt, ok_agents, ok_steps, ok_modes)
    return _verdict(pred, ok_agents, ok_steps, ok_modes)


# ---------------------------------------------------------------------------
# ALFAgent (ALFWorld text agent)
# ---------------------------------------------------------------------------
#
# Same shape as smolagents: single agent, renderer hardcodes
# ``Agent: agent``, GT.agent is None, and GT.step is the trajectory's
# native ``step_number`` which the renderer emits verbatim as the coord.


@register("alfagent")
def _score_alfagent(pred: Optional[dict], gt: dict) -> dict:
    return _score_smolagents(pred, gt)


# ---------------------------------------------------------------------------
# GUI single-agent frameworks (image_gui split)
# ---------------------------------------------------------------------------
#
# Same shape as smolagents: one acting agent, ``gt.agent`` is null, and
# ``gt.step`` is 1-indexed in the renderer's own coordinate space
# (identity). The difference is the label: render/gui.py labels each step
# with the trace's actual agent name (``computer_use_agent`` for
# openai_cua on OSWorld; ``web_agent`` for agentoccam / gemini on
# WebVoyager), so that's what the judge sees and what we accept.
#
# ``coact`` — the multi-agent GUI framework — is NOT registered here: the
# default scorer handles it (named ``gt.agent``, identity ``gt.step``).


def _score_gui_single(pred: Optional[dict], gt: dict, label: str) -> dict:
    if not pred:
        return dict(_MISS)

    canon_mode = _norm_mode(gt.get("mode"))
    canon_step = str(gt.get("step")) if gt.get("step") is not None else ""

    ok_agents = {label}  # what render/gui.py emits for every step
    ok_steps = {canon_step} if canon_step else set()
    ok_modes = {canon_mode} if canon_mode else set()

    _accepted(gt, ok_agents, ok_steps, ok_modes)
    return _verdict(pred, ok_agents, ok_steps, ok_modes)


@register("openai_cua")
def _score_openai_cua(pred: Optional[dict], gt: dict) -> dict:
    return _score_gui_single(pred, gt, "computer_use_agent")


@register("agentoccam")
def _score_agentoccam(pred: Optional[dict], gt: dict) -> dict:
    return _score_gui_single(pred, gt, "web_agent")


@register("gemini")
def _score_gemini(pred: Optional[dict], gt: dict) -> dict:
    return _score_gui_single(pred, gt, "web_agent")


# ---------------------------------------------------------------------------
# MathChat
# ---------------------------------------------------------------------------
#
# Renderer flattens the (round, position) dialog into a 1-indexed ordinal
# ``step N`` where ``N = 2 * round + position`` (per mathchat.py docstring).
# The bootstrap framing turn (round=0, position=0, user_proxy) is rendered
# as an unnumbered ``User Input`` block — never injected as a target. GT
# stores ``round`` and ``position`` (always position=1 in shipped traces);
# we recompose N to compare with the model's flat coord.


@register("mathchat")
def _score_mathchat(pred: Optional[dict], gt: dict) -> dict:
    if not pred:
        return dict(_MISS)

    ok_agents = _agents_from_gt(gt)
    canon_mode = _norm_mode(gt.get("mode"))
    rd = gt.get("round")
    pos = gt.get("position")
    canon_step = str(2 * int(rd) + int(pos)) if rd is not None and pos is not None else ""

    ok_steps = {canon_step} if canon_step else set()
    ok_modes = {canon_mode} if canon_mode else set()

    _accepted(gt, ok_agents, ok_steps, ok_modes)
    return _verdict(pred, ok_agents, ok_steps, ok_modes)


# ---------------------------------------------------------------------------
# MetaGPT
# ---------------------------------------------------------------------------
#
# Renderer emits ``step S`` where S is the 0-indexed SOP stage (architect=0,
# engineer=1, reviewer=2). GT records ``stage`` rather than round/position;
# the default scorer wouldn't pick it up.


@register("metagpt")
def _score_metagpt(pred: Optional[dict], gt: dict) -> dict:
    if not pred:
        return dict(_MISS)

    ok_agents = _agents_from_gt(gt)
    canon_mode = _norm_mode(gt.get("mode"))
    stage = gt.get("stage")
    canon_step = str(int(stage)) if stage is not None else ""

    ok_steps = {canon_step} if canon_step else set()
    ok_modes = {canon_mode} if canon_mode else set()

    _accepted(gt, ok_agents, ok_steps, ok_modes)
    return _verdict(pred, ok_agents, ok_steps, ok_modes)


# ---------------------------------------------------------------------------
# DVD (DeepVideoDiscovery)
# ---------------------------------------------------------------------------
#
# Renderer at ``render/dvd.py`` emits ``Step N`` where N = native
# ``trajectory_idx - 2`` — trajectory[0] (system) and trajectory[1] (user
# question) are framing, not rendered as steps. GT records ``step`` as
# the native trajectory index, so the scorer subtracts 2 to map onto the
# rendered coord. Agents are ``orchestrator`` (the LLM driving the loop)
# and ``frame_inspect_agent`` (the inner VLM the orchestrator delegates
# to via ``frame_inspect_tool`` calls).


_DVD_FRAMING_OFFSET = 2  # trajectory[0..1] are not rendered as steps


@register("dvd")
def _score_dvd(pred: Optional[dict], gt: dict) -> dict:
    if not pred:
        return dict(_MISS)

    ok_agents = _agents_from_gt(gt)
    canon_mode = _norm_mode(gt.get("mode"))
    raw_step = gt.get("step")
    canon_step = (
        str(int(raw_step) - _DVD_FRAMING_OFFSET) if raw_step is not None else ""
    )

    ok_steps = {canon_step} if canon_step else set()
    ok_modes = {canon_mode} if canon_mode else set()

    _accepted(gt, ok_agents, ok_steps, ok_modes)
    return _verdict(pred, ok_agents, ok_steps, ok_modes)


# ---------------------------------------------------------------------------
# EVA (Efficient Video Agent)
# ---------------------------------------------------------------------------
#
# Renderer at ``render/eva.py`` emits dense ``Step N`` (N = 0,1,2,…)
# where each rendered step corresponds to one assistant turn with its
# follow-up frame_select tool result folded in as a ``[tool_output]``
# observation. Native trajectory layout (post-merge):
#
#   trajectory[0]              = system prompt    (skipped)
#   trajectory[1]              = user question    (skipped)
#   trajectory[2 + 2k]         = assistant turn k (rendered as Step k)
#   trajectory[2 + 2k + 1]     = tool turn k      (folded into Step k)
#
# GT records ``step`` as the native trajectory idx of the injected
# assistant turn, so the rendered coord is ``(step - 2) // 2``. Single
# agent (``"agent"``) — same convention as smolagents.


_EVA_FRAMING_OFFSET = 2  # trajectory[0..1] are not rendered as steps


@register("eva")
def _score_eva(pred: Optional[dict], gt: dict) -> dict:
    if not pred:
        return dict(_MISS)

    canon_mode = _norm_mode(gt.get("mode"))
    raw_step = gt.get("step")
    canon_step = (
        str((int(raw_step) - _EVA_FRAMING_OFFSET) // 2)
        if raw_step is not None
        else ""
    )

    ok_agents = {"agent"}  # what the renderer emits
    ok_steps = {canon_step} if canon_step else set()
    ok_modes = {canon_mode} if canon_mode else set()

    _accepted(gt, ok_agents, ok_steps, ok_modes)
    return _verdict(pred, ok_agents, ok_steps, ok_modes)


# ---------------------------------------------------------------------------
# Debate / Dylan (round-only step match)
# ---------------------------------------------------------------------------
#
# Renderer emits ``Step R.P`` but GT records only ``round`` (no position)
# and ``agents`` as a list. So:
# - agent axis: pred is one of ``gt.agents`` (multi-correct for C.3).
# - step axis: pred's *round* component must equal ``gt.round`` — any P
#   within that round counts, since GT does not resolve to position.
# - mode axis: standard string compare.


def _round_of_step(s: Any) -> str:
    """Return the round component of a step coordinate string.

    ``"step 1.2"`` -> ``"1"``; ``"step 1"`` -> ``"1"``; ``"1.0"`` -> ``"1"``.
    """
    norm = _norm_step(s)
    return norm.split(".", 1)[0] if norm else ""


@register("debate")
@register("dylan")
def _score_round_only(pred: Optional[dict], gt: dict) -> dict:
    if not pred:
        return dict(_MISS)

    ok_agents = _agents_from_gt(gt)
    canon_round = str(gt.get("round")) if gt.get("round") is not None else ""
    canon_mode = _norm_mode(gt.get("mode"))

    ok_rounds = {canon_round} if canon_round else set()
    ok_modes = {canon_mode} if canon_mode else set()

    for ap in gt.get("accepted_predictions") or []:
        a = _norm_str(ap.get("agent_name"))
        if a:
            ok_agents.add(a)
        rd = _round_of_step(ap.get("step_coord"))
        if rd:
            ok_rounds.add(rd)
        m = _norm_mode(ap.get("mode"))
        if m:
            ok_modes.add(m)

    pa = _norm_str(pred.get("agent_name"))
    pred_round = _round_of_step(pred.get("step_coord"))
    pm = _norm_mode(pred.get("error_mode"))

    return {
        "agent": bool(pa) and pa in ok_agents,
        "step": bool(pred_round) and pred_round in ok_rounds,
        "mode": bool(pm) and pm in ok_modes,
    }
