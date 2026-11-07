# livekit-call-record

**When a voice agent goes quiet mid-call, the logs usually cannot tell you why.**

The transcript has the sentence in it. The LLM trace shows a completion. The
caller heard nothing and asked the agent to repeat. Nothing in the record says
which of those three things actually reached the phone.

This is a listener that writes one JSON file per call from events
`AgentSession` already emits, and a scorer that turns a pile of those files
into four numbers. No patched framework, no fork, no API keys.

```python
from call_record import CallRecorder

recorder = CallRecorder(session, call_id=ctx.room.name)
recorder.attach()
ctx.add_shutdown_callback(lambda: recorder.write())
```

Then:

```
python score.py recordings
```

## What it reports

```
Dead air before the agent takes the floor
  after a pause of 2s or more        median   4.00s   p90   4.00s   worst   4.00s   n=1
  the framework's own eou delay      median   0.55s   p90   0.55s   worst   0.55s   n=1

Cut-ins, by how long the caller actually paused
  pause           gaps   agent spoke in it
  0 to 0.5s          1                 0%
  0.5 to 1s          1               100%
  over 5s            1               100%

Replies generated and never spoken
  sample01 at   16.00s  'Sorry, could you say the street again?'

Time to first audio
  tts ttfb                           median   0.31s   p90   0.31s   worst   0.31s   n=1

False interruptions: 1, of which 1 did not resume
```

## The three ideas worth explaining

**Silence measured from the VAD's own speech-end is circular.** You are using
the detector to grade the detector. If endpointing runs late because
background noise keeps the VAD reporting speech, the silence you compute from
it is zero, because by its own account the caller never stopped. So dead air
here is measured from the caller resuming, which does not depend on the VAD.
The framework's own `end_of_utterance_delay` is reported next to it, and the
gap between the two lines is itself the finding.

**A single interruption count tells you nothing.** A decision landing in a
five second gap is a handover. The same decision in a 300 ms gap cut someone
off mid-sentence. They need opposite fixes, so cut-ins are bucketed by how
long the caller actually paused. The caller speaking again is what proves
they were not finished, so the call labels itself.

**A reply can be generated and never spoken.** An assistant item enters the
chat context and no agent speech follows it. The model believes it answered,
the transcript agrees, and the caller heard silence. This is invisible in
every log that reads the transcript, and it is the single most common thing
behind "the agent just went quiet".

## What it collects

| event | what it gives you |
| --- | --- |
| `user_state_changed` | caller speech spans |
| `agent_state_changed` | agent speech spans |
| `agent_false_interruption` | interruptions that were not speech, and whether the reply resumed |
| `metrics_collected` | STT, LLM and TTS timings, end-of-utterance delay, interruption counts |
| `conversation_item_added` | what was generated, and when it entered the context |
| `user_input_transcribed` | final transcripts |
| `eot_prediction` | turn-end probability against its threshold |
| `error` | failures, with the component that raised them |

Interim transcripts are dropped unless you pass
`keep_interim_transcripts=True`, because they are most of the volume and
rarely what you are looking for.

## Requirements

`livekit-agents`. Checked against 1.8.2 and 1.8.11; every event used here
exists in both. No other dependencies, and the scorer is standard library
only.

## Limits

Audio is never recorded or read, so nothing here can tell you how a call
*sounded*. Speech spans come from the VAD's view of the call, which is the
right view for turn-taking questions and the wrong one for asking whether the
VAD itself is correct. For that you still need labelled audio.
