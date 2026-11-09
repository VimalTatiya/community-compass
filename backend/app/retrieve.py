"""Keyword retrieval over the markdown notes in knowledge/.

Files are re-read on every call, so you can edit notes without restarting.
Swap `search` for embeddings later without changing the API.
"""

import re
from dataclasses import dataclass
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"

STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "your", "with", "this",
    "that", "from", "what", "when", "where", "how", "can", "does", "should",
    "will", "have", "has", "was", "were", "about", "there", "their", "they",
    "into", "any", "all", "use", "get", "which", "who", "why",
    # Generic words that appear in almost every note and cause false matches.
    "event", "events", "find", "exact", "catalog", "suitable", "list", "show",
    "tell", "give", "conference", "attendee", "attendees", "recommend",
}


@dataclass
class Chunk:
    source: str  # file name, e.g. "reserved-seating.md"
    text: str


def _stem(word: str) -> str:
    """Strip common suffixes so 'reserve', 'reserved' and 'reservations' can match."""
    for suffix in ("ing", "ed", "es", "s", "e"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {_stem(w) for w in words if len(w) > 2 and w not in STOPWORDS}


def _sections(text: str) -> list[str]:
    """Split markdown at '## ' headings so each question stays with its answer."""
    sections: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("# "):
            continue
        if line.startswith("## ") and current:
            sections.append("\n".join(current).strip())
            current = []
        current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    return [s for s in sections if s]


def load_chunks() -> list[Chunk]:
    """One chunk per section of each markdown file, keeping its file name as the source."""
    chunks: list[Chunk] = []
    if not KNOWLEDGE_DIR.exists():
        return chunks
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        for section in _sections(path.read_text(encoding="utf-8")):
            chunks.append(Chunk(source=path.name, text=section))
    return chunks


def search(question: str, k: int = 4) -> list[Chunk]:
    """Return the k paragraphs that share the most keywords with the question."""
    query = _words(question)
    if not query:
        return []
    scored = []
    for chunk in load_chunks():
        overlap = len(query & _words(chunk.text))
        if overlap:
            scored.append((overlap, chunk))
    if not scored:
        return []
    scored.sort(key=lambda pair: pair[0], reverse=True)
    # Drop weak matches (one shared word like "vegas") when a much better one exists.
    floor = max(1, scored[0][0] / 2)
    return [chunk for score, chunk in scored[:k] if score >= floor]
