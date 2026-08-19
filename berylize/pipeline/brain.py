"""
pipeline/brain.py
OpenAI GPT-4o brain with per-agent system prompts.
Supports barge-in cancellation via asyncio.Event.
"""

import asyncio
import logging
import os
import time
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 20


AGENT_PROMPTS = {
    "evedefault": (
        "You are Eve, a sharp, warm, and naturally expressive conversationalist. "
        "You speak like a real person — sometimes you pause, rephrase, or add a quick "
        "aside. You're curious, direct, and never robotic. "
        "You make people feel heard without being sycophantic. "
        "Keep responses under 60 words unless the question demands more."
    ),
    "jeff": (
        "You are Jeff, a confident and visionary founder archetype. "
        "You speak in clear, direct sentences with conviction. "
        "You distill complex ideas into simple truths. "
        "You're warm but never vague — every sentence moves the conversation forward. "
        "Keep responses under 80 words unless asked to elaborate."
    ),
    "nu": (
        "You are Nu, a precise analytical thinker. "
        "You lead with data and evidence, cite reasoning explicitly, "
        "and flag uncertainty when you see it. "
        "You're not cold — you're rigorous, and people trust you because of it. "
        "Keep responses under 80 words unless asked to elaborate."
    ),
    "india": (
        "You are India, a creative and expressive communicator. "
        "You use vivid language, analogies, and occasionally unexpected framings. "
        "You make abstract things tangible. "
        "You bring energy to the room without losing substance. "
        "Keep responses under 80 words unless asked to elaborate."
    ),
    "amanda": (
        "You are Amanda, warm, strategic, and deeply perceptive. "
        "You read between the lines of what people say and address the real question. "
        "You're the glue in the room — people feel heard when you speak. "
        "Keep responses under 80 words unless asked to elaborate."
    ),
}

VOICES = {
    "evedefault": "nova",
    "jeff":       "onyx",
    "nu":         "nova",
    "india":      "shimmer",
    "amanda":     "alloy",
}


class CliqueBrain:
    def __init__(self, active_agent: str = "evedefault"):
        self.active_agent  = active_agent
        self.history: list[dict] = []
        self._client       = None
        self._cancel_event = asyncio.Event()
        self._init_client()

    def _init_client(self):
        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            logger.info("OpenAI brain initialised.")
        except ImportError:
            logger.warning("openai not installed — brain in STUB mode.")

    def set_agent(self, name: str):
        if name not in AGENT_PROMPTS:
            raise ValueError(f"Unknown agent: {name}")
        self.active_agent = name
        self.history      = []
        logger.info(f"[BRAIN] agent → {name}")

    def cancel(self):
        """Signal barge-in: abort the current stream."""
        self._cancel_event.set()

    def reset_cancel(self):
        self._cancel_event.clear()

    def _trim_history(self):
        max_msgs = MAX_HISTORY_TURNS * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

    async def respond(self, user_text: str) -> AsyncGenerator[str, None]:
        if not user_text.strip():
            return

        self.reset_cancel()
        t0 = time.time()

        self.history.append({"role": "user", "content": user_text})
        self._trim_history()

        messages = [
            {"role": "system", "content": AGENT_PROMPTS[self.active_agent]},
            *self.history,
        ]

        if self._client is None:
            stub = f"I'm {self.active_agent.capitalize()}, I heard: {user_text}"
            for word in stub.split():
                if self._cancel_event.is_set():
                    break
                yield word + " "
                await asyncio.sleep(0.04)
            self.history.append({"role": "assistant", "content": stub})
            return

        full: list[str] = []
        first_token = True

        try:
            stream = await self._client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                stream=True,
                max_tokens=250,
                temperature=0.75,
            )
            async for chunk in stream:
                if self._cancel_event.is_set():
                    logger.info("[BRAIN] stream cancelled by barge-in")
                    break
                delta = chunk.choices[0].delta.content
                if delta:
                    if first_token:
                        logger.info(f"[BRAIN] first token: {(time.time()-t0)*1000:.0f}ms")
                        first_token = False
                    full.append(delta)
                    yield delta

        except Exception as e:
            logger.error(f"OpenAI stream error: {e}")
            err = "I'm having trouble right now."
            yield err
            full.append(err)

        assistant_text = "".join(full)
        if assistant_text:
            self.history.append({"role": "assistant", "content": assistant_text})
