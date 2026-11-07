"""A minimal LiveKit agent with the recorder attached.

Three lines: construct it, attach it, write it. Everything else here is the
smallest agent that will run.

    python example_agent.py console
"""

from __future__ import annotations

from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import deepgram, openai, silero

from call_record import CallRecorder

load_dotenv()


async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(),
    )

    recorder = CallRecorder(session, call_id=ctx.room.name, out_dir="recordings")
    recorder.attach()
    # Write even when the job ends badly, since a call that fell over is the
    # one worth reading.
    ctx.add_shutdown_callback(lambda: _write(recorder))

    await session.start(agent=Agent(instructions="You are a receptionist. Be brief."), room=ctx.room)


async def _write(recorder: CallRecorder) -> None:
    print(f"call record: {recorder.write()}")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
