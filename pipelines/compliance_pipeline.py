"""End-to-end requirements analysis pipeline.

The pipeline mirrors the flow described in ``Fine_Tuning_RAG.ipynb`` and the
system diagram (Requirements/Inquiry -> RAG -> Agent + Noncompliance AI).
It loads and chunks the requirements, retrieves the most relevant sections for
an inquiry, and then calls Gemini agents to produce a compliance matrix and
draft Jira tickets for noncompliant findings.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

try:  # Optional dependency when running against real Gemini models.
    from google import genai
    from google.genai import types as genai_types
except Exception:  # pragma: no cover - library optional for mock mode tests
    genai = None  # type: ignore
    genai_types = None  # type: ignore

from nltk import data as nltk_data
from nltk import download as nltk_download
from nltk.tokenize import sent_tokenize as nltk_sent_tokenize

try:  # Optional when ingesting PDFs.
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None  # type: ignore

try:  # Sentence transformers are ideal but optional for offline testing.
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


def _ensure_nltk_resource(resource: str) -> None:
    try:
        nltk_data.find(resource)
    except LookupError:  # pragma: no cover - executed during first run only
        try:
            nltk_download(resource)
        except Exception:
            pass


def _split_sentences(text: str) -> List[str]:
    try:
        _ensure_nltk_resource("tokenizers/punkt")
        return nltk_sent_tokenize(text)
    except Exception:  # pragma: no cover - fallback for offline runs
        return [sentence.strip() for sentence in text.split('.') if sentence.strip()]


@dataclass(slots=True)
class CompliancePipelineConfig:
    """Configuration for the compliance pipeline."""

    google_model: str = "gemini-2.0-flash-lite"
    google_api_key: Optional[str] = None
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    sentences_per_chunk: int = 3
    top_k: int = 5
    temperature: float = 0.2
    use_mock_llm: bool = False


@dataclass(slots=True)
class RequirementChunk:
    chunk_id: str
    text: str
    page: Optional[int] = None


@dataclass(slots=True)
class ComplianceMatrixEntry:
    requirement_id: str
    status: str
    summary: str


@dataclass(slots=True)
class JiraTicket:
    title: str
    description: str
    priority: str


@dataclass(slots=True)
class PipelineResult:
    retrieved_context: str
    compliance_matrix: List[ComplianceMatrixEntry]
    jira_tickets: List[JiraTicket]


class RequirementsCorpus:
    """Loads requirements (PDF/text) and prepares RAG chunks."""

    def __init__(self, chunks: Sequence[RequirementChunk]):
        self.chunks = list(chunks)

    @classmethod
    def from_pdf(cls, path: str, *, sentences_per_chunk: int = 3) -> "RequirementsCorpus":
        reader = PdfReader(path)
        sentences: List[str] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for sentence in sent_tokenize(text):
                clean = sentence.strip()
                if clean:
                    sentences.append(clean)

        chunks: List[RequirementChunk] = []
        counter = 1
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk_text = " ".join(sentences[i : i + sentences_per_chunk]).strip()
            if chunk_text:
                chunks.append(RequirementChunk(chunk_id=f"REQ-{counter:04d}", text=chunk_text))
                counter += 1
        return cls(chunks)


class SentenceTransformerRAG:
    """Simple semantic retriever inspired by the notebook workflow."""

    def __init__(self, corpus: RequirementsCorpus, embed_model_name: str):
        self.embed_model = SentenceTransformer(embed_model_name)
        self.chunks = corpus.chunks
        self.chunk_embeddings = self.embed_model.encode(
            [chunk.text for chunk in self.chunks], convert_to_numpy=True
        )

    def retrieve(self, query: str, top_k: int) -> List[RequirementChunk]:
        query_embedding = self.embed_model.encode(query, convert_to_numpy=True)
        chunk_norms = np.linalg.norm(self.chunk_embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        similarity = (self.chunk_embeddings @ query_embedding) / (chunk_norms * query_norm + 1e-10)
        ranked = np.argsort(-similarity)[:top_k]
        return [self.chunks[int(idx)] for idx in ranked]


class GeminiClient:
    """Minimal wrapper around the Google Generative AI SDK."""

    def __init__(self, api_key: Optional[str], model: str, temperature: float):
        key = api_key or os.getenv("GOOGLE_GENAI_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "Google Generative AI key not provided. Set GOOGLE_GENAI_KEY or GOOGLE_API_KEY."
            )
        self.client = genai.Client(api_key=key)
        self.model = model
        self.temperature = temperature

    def generate_json(self, prompt: str, schema: dict) -> dict:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=self.temperature,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        payload = response.candidates[0].content.parts[0].text
        return json.loads(payload)


class ComplianceAgent:
    """Produces a compliance matrix from RAG context."""

    SCHEMA = {
        "type": "object",
        "properties": {
            "matrix": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "requirement_id": {"type": "string"},
                        "status": {"type": "string"},
                        "summary": {"type": "string"},
                    },
                    "required": ["requirement_id", "status", "summary"],
                },
            }
        },
        "required": ["matrix"],
    }

    def __init__(self, client: GeminiClient):
        self.client = client

    def __call__(self, inquiry: str, context: str) -> List[ComplianceMatrixEntry]:
        prompt = f"""
        You are a compliance analyst. Use the provided requirements context to map each
        cited rule to the inquiry. Summaries must mention concrete obligations.

        Inquiry: {inquiry}

        Context:
        {context}
        """
        data = self.client.generate_json(prompt, self.SCHEMA)
        entries = [ComplianceMatrixEntry(**item) for item in data.get("matrix", [])]
        return entries


class NonComplianceExtractor:
    """Generates Jira ticket stubs for noncompliant findings."""

    SCHEMA = {
        "type": "object",
        "properties": {
            "tickets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "priority": {"type": "string"},
                    },
                    "required": ["title", "description", "priority"],
                },
            }
        },
        "required": ["tickets"],
    }

    def __init__(self, client: GeminiClient):
        self.client = client

    def __call__(self, inquiry: str, context: str) -> List[JiraTicket]:
        prompt = f"""
        Identify any potential noncompliance items between the requirements and the
        inquiry. Draft Jira-ready tickets. Use concise but actionable language.

        Inquiry: {inquiry}

        Context:
        {context}
        """
        data = self.client.generate_json(prompt, self.SCHEMA)
        return [JiraTicket(**item) for item in data.get("tickets", [])]


class CompliancePipeline:
    """High-level orchestrator covering RAG + dual-agent flow."""

    def __init__(
        self,
        requirements_path: str,
        config: Optional[CompliancePipelineConfig] = None,
    ) -> None:
        self.config = config or CompliancePipelineConfig()
        corpus = RequirementsCorpus.from_pdf(
            requirements_path, sentences_per_chunk=self.config.sentences_per_chunk
        )
        self.rag = SentenceTransformerRAG(corpus, self.config.embed_model_name)
        client = GeminiClient(
            api_key=self.config.google_api_key,
            model=self.config.google_model,
            temperature=self.config.temperature,
        )
        self.compliance_agent = ComplianceAgent(client)
        self.noncompliance_agent = NonComplianceExtractor(client)

    def run(self, inquiry: str) -> PipelineResult:
        retrieved_chunks = self.rag.retrieve(inquiry, top_k=self.config.top_k)
        context = "\n\n".join(f"[{chunk.chunk_id}] {chunk.text}" for chunk in retrieved_chunks)
        matrix = self.compliance_agent(inquiry, context)
        tickets = self.noncompliance_agent(inquiry, context)
        return PipelineResult(context, matrix, tickets)
