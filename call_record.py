"""One JSON record per call, built from the session's own events.

Nothing here patches the framework. Every number comes from an event
``AgentSession`` already emits, so this keeps working across upgrades and can
be dropped into an existing agent in three lines.

    recorder = CallRecorder(session, out_dir="recordings")
    recorder.attach()
    ...
    recorder.write()

The point of the record is that a bad call can be explained afterwards. A
caller who hears silence and asks "are you there?" leaves a shape in these
events: the agent entered ``thinking`` or ``speaking``, no audio followed,
and the caller started again. Without the record that call is unexplainable.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from livekit.agents import metrics as lk_metrics
from livekit.agents.voice import AgentSession

# Events are stamped with wall-clock ``created_at``; the record stores seconds
# from the first event instead, so two calls can be read side by side.
_EVENTS = (
    "user_state_changed",
    "agent_state_changed",
    "agent_false_interruption",
    "metrics_collected",
    "conversation_item_added",
    "user_input_transcribed",
    "eot_prediction",
    "speech_created",
    "error",
)


class CallRecorder:
    """Collects one call's events and writes them as a single JSON file."""

    def __init__(
        self,
        session: AgentSession,
        *,
        call_id: str | None = None,
        out_dir: str | Path = "recordings",
        keep_interim_transcripts: bool = False,
    ) -> None:
        self._session = session
        self.call_id = call_id or uuid.uuid4().hex[:12]
        self._out_dir = Path(out_dir)
        self._keep_interim = keep_interim_transcripts
        self._origin: float | None = None
        self._events: list[dict[str, Any]] = []
        self._attached = False

    # -- collection ------------------------------------------------------

    def attach(self) -> None:
        """Subscribes to the session. Safe to call more than once."""
        if self._attached:
            return
        self._attached = True
        for name in _EVENTS:
            self._session.on(name, self._record)

    def _at(self, created_at: float | None) -> float:
        now = created_at if created_at is not None else time.time()
        if self._origin is None:
            self._origin = now
        return round(now - self._origin, 4)

    def _record(self, ev: Any) -> None:
        kind = getattr(ev, "type", ev.__class__.__name__)
        row: dict[str, Any] = {
            "t": self._at(getattr(ev, "created_at", None)),
            "type": kind,
        }

        if kind in ("user_state_changed", "agent_state_changed"):
            row["from"] = str(getattr(ev, "old_state", ""))
            row["to"] = str(getattr(ev, "new_state", ""))

        elif kind == "user_input_transcribed":
            if not ev.is_final and not self._keep_interim:
                return
            row["final"] = bool(ev.is_final)
            row["text"] = ev.transcript
            row["item_id"] = ev.item_id
            row["language"] = ev.language

        elif kind == "agent_false_interruption":
            # The agent stopped for something that turned out not to be speech.
            # resumed=False means the reply was lost, not merely delayed.
            row["resumed"] = bool(getattr(ev, "resumed", False))

        elif kind == "eot_prediction":
            row["probability"] = ev.probability
            row["threshold"] = ev.threshold
            row["delay"] = ev.delay

        elif kind == "conversation_item_added":
            item = ev.item
            row["role"] = str(getattr(item, "role", ""))
            row["id"] = getattr(item, "id", None)
            row["interrupted"] = getattr(item, "interrupted", None)
            row["text"] = _text_of(item)

        elif kind == "speech_created":
            row["source"] = str(getattr(ev, "source", ""))
            row["user_initiated"] = bool(getattr(ev, "user_initiated", False))

        elif kind == "error":
            row["source"] = type(getattr(ev, "source", None)).__name__
            row["error"] = str(getattr(ev, "error", ""))

        elif kind == "metrics_collected":
            row.update(_metrics_row(ev.metrics))

        self._events.append(row)

    # -- output ----------------------------------------------------------

    def record(self) -> dict[str, Any]:
        """The call as a plain dict, with the spans already derived."""
        return {
            "call_id": self.call_id,
            "origin_unix": self._origin,
            "duration": self._events[-1]["t"] if self._events else 0.0,
            "user_spans": _spans(self._events, "user_state_changed", "speaking"),
            "agent_spans": _spans(self._events, "agent_state_changed", "speaking"),
            "events": self._events,
        }

    def write(self) -> Path:
        self._out_dir.mkdir(parents=True, exist_ok=True)
        path = self._out_dir / f"{self.call_id}.json"
        path.write_text(json.dumps(self.record(), indent=1), encoding="utf-8")
        return path


def _text_of(item: Any) -> str:
    content = getattr(item, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(c for c in content if isinstance(c, str))
    return ""


def _metrics_row(m: Any) -> dict[str, Any]:
    """Only the fields that say something about the caller's experience."""
    if isinstance(m, lk_metrics.TTSMetrics):
        # ttfb is request to first audio. A silent agent with a healthy ttfb
        # is a different fault from one that never got audio at all.
        return {
            "metric": "tts",
            "ttfb": m.ttfb,
            "audio_duration": m.audio_duration,
            "cancelled": m.cancelled,
            "speech_id": m.speech_id,
        }
    if isinstance(m, lk_metrics.LLMMetrics):
        return {
            "metric": "llm",
            "ttft": m.ttft,
            "duration": m.duration,
            "cancelled": m.cancelled,
            "speech_id": m.speech_id,
        }
    if isinstance(m, lk_metrics.STTMetrics):
        return {
            "metric": "stt",
            "duration": m.duration,
            "audio_duration": m.audio_duration,
        }
    if isinstance(m, lk_metrics.EOUMetrics):
        # end_of_utterance_delay is measured from the VAD's own end of speech,
        # so it grades the detector with the detector. Kept because it is the
        # framework's own number, and compared in score.py against the gaps,
        # which come from the caller resuming and do not depend on the VAD.
        return {
            "metric": "eou",
            "end_of_utterance_delay": m.end_of_utterance_delay,
            "transcription_delay": m.transcription_delay,
            "on_user_turn_completed_delay": m.on_user_turn_completed_delay,
            "speech_id": m.speech_id,
        }
    if isinstance(m, lk_metrics.InterruptionMetrics):
        return {
            "metric": "interruption",
            "num_interruptions": m.num_interruptions,
            "num_backchannels": m.num_backchannels,
            "detection_delay": m.detection_delay,
        }
    if isinstance(m, lk_metrics.VADMetrics):
        return {"metric": "vad", "idle_time": m.idle_time}
    return {"metric": type(m).__name__}


def _spans(events: list[dict[str, Any]], kind: str, state: str) -> list[list[float]]:
    """Closed [start, end] pairs for every run of ``state``.

    A run still open when the call ends is closed at the last event, so a call
    that drops mid-sentence still produces a usable span rather than none.
    """
    spans: list[list[float]] = []
    start: float | None = None
    for ev in events:
        if ev["type"] != kind:
            continue
        if ev.get("to") == state and start is None:
            start = ev["t"]
        elif ev.get("from") == state and start is not None:
            spans.append([start, ev["t"]])
            start = None
    if start is not None and events:
        spans.append([start, events[-1]["t"]])
    return spans
