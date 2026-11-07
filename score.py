"""Reads the call records and reports the four numbers that matter.

    python score.py recordings

What it reports, and why each one is here:

**Dead air.** How long after the caller stopped before the agent started.
Taken from the caller resuming rather than from the VAD's own end of speech,
so it does not grade the detector with the detector.

**Cut-ins.** An agent turn that starts inside a gap the caller then speaks
through, bucketed by how long that gap was. A gap of five seconds with an
agent turn in it is a handover; the same thing in a gap of three hundred
milliseconds took the floor mid-sentence. They need opposite fixes, so an
unbucketed interruption count tells you nothing.

**Replies that were generated and never spoken.** An assistant item entered
the chat context with no agent speech following it. This is the failure a
caller reports as "it just went quiet", and it is invisible in a transcript,
because the transcript has the sentence in it.

**Time to first audio.** From the TTS metrics. A silent agent with a healthy
ttfb is a different fault from one that never got audio at all.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import median
from typing import Any

# A reply is counted as spoken if speech starts within this long after the
# item is added. Generous on purpose: the question is whether it was ever
# spoken, not whether it was prompt.
SPOKEN_WITHIN_S = 10.0

BUCKETS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, float("inf")))


def load(path: Path) -> list[dict[str, Any]]:
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def gaps(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Every pause between two caller spans, and what the agent did in it."""
    user = record.get("user_spans") or []
    agent = record.get("agent_spans") or []
    out: list[dict[str, Any]] = []
    for (_, end), (next_start, _) in zip(user, user[1:]):
        length = next_start - end
        if length <= 0:
            continue
        started = next((s for s, _ in agent if end <= s < next_start), None)
        out.append(
            {
                "gap": length,
                "agent_started": started is not None,
                # How long the agent waited before taking the floor.
                "lead": None if started is None else started - end,
            }
        )
    return out


def unspoken_replies(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Assistant items with no agent speech after them."""
    agent_starts = [s for s, _ in (record.get("agent_spans") or [])]
    out = []
    for ev in record.get("events", []):
        if ev.get("type") != "conversation_item_added" or ev.get("role") != "assistant":
            continue
        if not (ev.get("text") or "").strip():
            continue
        t = ev["t"]
        if any(t - 1.0 <= s <= t + SPOKEN_WITHIN_S for s in agent_starts):
            continue
        out.append({"call": record["call_id"], "t": t, "text": ev["text"][:80]})
    return out


def _pick(records: list[dict], metric: str, field: str) -> list[float]:
    return [
        ev[field]
        for r in records
        for ev in r.get("events", [])
        if ev.get("metric") == metric and isinstance(ev.get(field), (int, float))
    ]


def _p90(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]


def _line(label: str, values: list[float]) -> str:
    if not values:
        return f"  {label:<34} no data"
    return (
        f"  {label:<34} median {median(values):6.2f}s"
        f"   p90 {_p90(values):6.2f}s   worst {max(values):6.2f}s   n={len(values)}"
    )


def report(records: list[dict[str, Any]]) -> str:
    all_gaps = [g for r in records for g in gaps(r)]
    unspoken = [u for r in records for u in unspoken_replies(r)]
    false_ints = [
        ev
        for r in records
        for ev in r.get("events", [])
        if ev.get("type") == "agent_false_interruption"
    ]

    lines = [f"{len(records)} call(s), {sum(r.get('duration', 0) for r in records):.0f}s of audio", ""]

    lines.append("Dead air before the agent takes the floor")
    handovers = [g["lead"] for g in all_gaps if g["agent_started"] and g["gap"] >= 2.0]
    lines.append(_line("after a pause of 2s or more", handovers))
    lines.append(_line("the framework's own eou delay", _pick(records, "eou", "end_of_utterance_delay")))
    lines.append("")

    lines.append("Cut-ins, by how long the caller actually paused")
    lines.append(f"  {'pause':<14}{'gaps':>6}{'agent spoke in it':>20}")
    for low, high in BUCKETS:
        chosen = [g for g in all_gaps if low <= g["gap"] < high]
        if not chosen:
            continue
        hit = sum(1 for g in chosen if g["agent_started"])
        name = f"{low:g} to {high:g}s" if high != float("inf") else f"over {low:g}s"
        lines.append(f"  {name:<14}{len(chosen):>6}{hit / len(chosen):>19.0%}")
    lines.append("")

    lines.append("Replies generated and never spoken")
    if unspoken:
        for u in unspoken[:10]:
            lines.append(f"  {u['call']} at {u['t']:7.2f}s  {u['text']!r}")
        if len(unspoken) > 10:
            lines.append(f"  ... and {len(unspoken) - 10} more")
    else:
        lines.append("  none")
    lines.append("")

    lines.append("Time to first audio")
    lines.append(_line("tts ttfb", _pick(records, "tts", "ttfb")))
    lines.append(_line("llm ttft", _pick(records, "llm", "ttft")))
    lines.append("")

    lost = sum(1 for ev in false_ints if not ev.get("resumed"))
    lines.append(
        f"False interruptions: {len(false_ints)}, of which {lost} did not resume"
    )
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    target = Path(argv[1]) if len(argv) > 1 else Path("recordings")
    if not target.exists():
        print(f"No such path: {target}", file=sys.stderr)
        return 1
    records = load(target)
    if not records:
        print(f"No records in {target}", file=sys.stderr)
        return 1
    print(report(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
