"""Community Compass chat API.

Run locally from the backend/ folder:
    uvicorn app.main:app --reload --port 8000
"""

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.llm import chat
from app.retrieve import search

SYSTEM_PROMPT = """You are Community Compass, a friendly advisor for AWS Builders Grant participants attending AWS re:Invent in Las Vegas.

You are given reference notes written by the community. Treat them as your most trusted source.

How to answer:
- Reply like a helpful chat assistant: direct, warm, and easy to scan. Use short paragraphs or simple dash lists when they help.
- Write plain text only. Do not use markdown: no asterisks, no bold, no headings, no tables.
- Never mention notes, files, sources, or "general context". Just answer the question.
- When the reference notes cover the question, build the answer on them and keep their specific details.
- You may add useful general knowledge to round out the answer, but never contradict the notes.
- For Las Vegas places and activities, name well-known attractions from general knowledge and describe them briefly. Do not state prices, opening hours or closures; suggest checking the official site for those.
- If you genuinely do not know the answer, reply exactly: "I don't know about that yet." Do not invent names, dates or prices.
- If the reference notes do not relate to the question, ignore them.
"""

app = FastAPI(title="Community Compass API")

# The Next.js app is on 3000 or 3001 during local development.
# This API should use a different port, such as 8000.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    chunks = search(request.question)
    notes = "\n\n---\n\n".join(c.text for c in chunks) or "(none)"
    user_message = f"Reference notes:\n{notes}\n\nQuestion: {request.question}"

    try:
        answer = await chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
        )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503,
            detail="Model is unavailable. Is `ollama serve` running?",
        ) from exc

    return AskResponse(answer=answer, sources=sorted({c.source for c in chunks}))
