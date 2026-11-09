"""Calls a local Ollama model. Start Ollama first: `ollama serve`."""

import os

import httpx

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


async def chat(messages: list[dict[str, str]]) -> str:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                # Low temperature keeps answers close to the notes without going flat.
                "options": {"temperature": 0.4},
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
