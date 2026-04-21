import csv
import io
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from docx import Document
from pypdf import PdfReader

from .config import settings
from .schemas import DocumentSummary, RetrievedChunk


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".json", ".csv"}


@dataclass
class IndexedChunk:
    document_id: str
    file_name: str
    chunk_id: str
    content: str
    tokens: set[str]


@dataclass
class IndexedDocument:
    document_id: str
    file_name: str
    content_type: str
    chunk_count: int


class InMemoryRAGStore:
    def __init__(self) -> None:
        self.documents: dict[str, IndexedDocument] = {}
        self.chunks: list[IndexedChunk] = []

    def list_documents(self) -> list[DocumentSummary]:
        return [
            DocumentSummary(
                document_id=document.document_id,
                file_name=document.file_name,
                content_type=document.content_type,
                chunk_count=document.chunk_count,
            )
            for document in sorted(self.documents.values(), key=lambda item: item.file_name.lower())
        ]

    def ingest(self, file_name: str, content_type: str, payload: bytes) -> tuple[str, int]:
        extension = Path(file_name).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{extension or 'unknown'}'.")

        text = extract_text(file_name=file_name, extension=extension, payload=payload)
        normalized_text = normalize_whitespace(text)
        if not normalized_text:
            raise ValueError("The uploaded file did not contain readable text.")

        document_id = str(uuid.uuid4())
        chunk_texts = chunk_text(normalized_text, settings.chunk_size, settings.chunk_overlap)
        if not chunk_texts:
            raise ValueError("The uploaded file could not be chunked into retrievable text.")

        self.documents[document_id] = IndexedDocument(
            document_id=document_id,
            file_name=file_name,
            content_type=content_type or "application/octet-stream",
            chunk_count=len(chunk_texts),
        )
        self.chunks.extend(
            IndexedChunk(
                document_id=document_id,
                file_name=file_name,
                chunk_id=f"{document_id}:{index}",
                content=chunk,
                tokens=tokenize(chunk),
            )
            for index, chunk in enumerate(chunk_texts)
        )
        return document_id, len(chunk_texts)

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scored: list[RetrievedChunk] = []
        for chunk in self.chunks:
            overlap = query_tokens & chunk.tokens
            if not overlap:
                continue
            score = len(overlap) / max(len(query_tokens), 1)
            scored.append(
                RetrievedChunk(
                    document_id=chunk.document_id,
                    file_name=chunk.file_name,
                    chunk_id=chunk.chunk_id,
                    score=round(score, 4),
                    content=chunk.content,
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]


rag_store = InMemoryRAGStore()


def extract_text(file_name: str, extension: str, payload: bytes) -> str:
    if extension in {".txt", ".md"}:
        return payload.decode("utf-8", errors="ignore")
    if extension == ".json":
        data = json.loads(payload.decode("utf-8", errors="ignore"))
        return json.dumps(data, indent=2, ensure_ascii=True)
    if extension == ".csv":
        text = payload.decode("utf-8", errors="ignore")
        reader = csv.reader(io.StringIO(text))
        rows = [", ".join(cell.strip() for cell in row) for row in reader]
        return "\n".join(rows)
    if extension == ".pdf":
        reader = PdfReader(io.BytesIO(payload))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if extension == ".docx":
        document = Document(io.BytesIO(payload))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    raise ValueError(f"Unsupported file: {file_name}")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]{2,}", text.lower()))


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        return []
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)

    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(length, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks


def combine_retrieval_results(results: Iterable[RetrievedChunk]) -> str:
    rendered = []
    for index, item in enumerate(results, start=1):
        rendered.append(f"[Source {index}: {item.file_name} | score={item.score}]\n{item.content}")
    return "\n\n".join(rendered)
