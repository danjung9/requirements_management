"""
Template pipeline that runs a simple RAG -> router -> agent flow and streams
text back in the shape expected by Gradio_Events.submit.
"""
import os
from dataclasses import dataclass
from typing import Iterator, Iterable, List, Dict, Any

from openai import OpenAI


# ---- Streaming message shape expected by app.py ----
@dataclass
class DeltaMessage:
    content: str | None = None
    reasoning_content: str | None = None  # leave None when you only stream text


@dataclass
class Choice:
    message: DeltaMessage


@dataclass
class Output:
    choices: list[Choice]


@dataclass
class Chunk:
    output: Output


# ---- Example RAG / Router / Agent stubs ----
class RAGModel:
    """Handles retrieval + requirement extraction."""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def extract_requirements(self, query: str) -> str:
        docs = self.retriever.search(query)
        # Replace with your own synthesis of requirements from docs.
        return self.llm.summarize(query=query, docs=docs)


class Router:
    """Chooses a target pipeline for the extracted requirements."""

    def route(self, requirements: str) -> str:
        # Replace with your own routing logic.
        if "jira" in requirements.lower():
            return "jira"
        return "matrix"


class JiraAgent:
    """Generates Jira ticket content using a Qwen model on OpenRouter and streams text."""

    def __init__(self,
                 model: str = "qwen/qwen3-4b:free",
                 api_key: str | None = None):
        resolved_key = api_key or os.getenv("OPENROUTER_API_KEY") \
            or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Missing OpenRouter API key: set OPENROUTER_API_KEY (preferred) "
                "or OPENAI_API_KEY in the environment, or pass api_key to JiraAgent"
            )

        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=resolved_key,
        )

    def stream(self, requirements: str) -> Iterable[str]:
        system_prompt = (
            "You are a Jira assistant. Draft a concise ticket with fields:\n"
            "- Summary: 1-line goal\n"
            "- Description: key context and expected behavior\n"
            "- Acceptance Criteria: 3-5 bullet points, clear and testable.\n"
            "Keep language direct and actionable.")
        user_prompt = (
            "Create a Jira ticket for these requirements:\n"
            f"{requirements}")

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "system",
                "content": system_prompt
            }, {
                "role": "user",
                "content": user_prompt
            }],
            max_tokens=512,
            temperature=0.3,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                # Yield raw text increments so the frontend can stream.
                yield delta.content


class ComplianceMatrixAgent:
    """Creates a compliance matrix CSV using a Qwen model on OpenRouter and streams CSV text."""

    def __init__(self,
                 model: str = "qwen/qwen3-4b:free",
                 api_key: str | None = None):
        resolved_key = api_key or os.getenv("OPENROUTER_API_KEY") \
            or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Missing OpenRouter API key: set OPENROUTER_API_KEY (preferred) "
                "or OPENAI_API_KEY in the environment, or pass api_key to ComplianceMatrixAgent"
            )

        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=resolved_key,
        )

    def stream(self, requirements: str) -> Iterable[str]:
        system_prompt = (
            "You are a compliance analyst. Produce a CSV with headers:\n"
            "Requirement,Control,Status,Notes\n"
            "Map the given requirements to likely controls; set Status to Pending; "
            "provide concise Notes. Output only CSV text.")
        user_prompt = (
            "Create a compliance matrix CSV for these requirements:\n"
            f"{requirements}")

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "system",
                "content": system_prompt
            }, {
                "role": "user",
                "content": user_prompt
            }],
            max_tokens=512,
            temperature=0.3,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                # Yield CSV text increments so the frontend can stream.
                yield delta.content


# ---- Pipeline wrapper ----
class RequirementsPipeline:
    """
    Wraps RAG -> router -> agent into a streaming interface compatible with
    Gradio_Events.submit.
    """

    def __init__(self, rag_model: RAGModel, router: Router,
                 jira_agent: JiraAgent, matrix_agent: ComplianceMatrixAgent):
        self.rag_model = rag_model
        self.router = router
        self.agents = {
            "jira": jira_agent,
            "matrix": matrix_agent,
        }

    def _extract_user_query(self, messages: List[Dict[str, Any]]) -> str:
        # Grab the last user message; adjust if you need a different strategy.
        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get("content", "")
        return ""

    def stream(self, *, messages: list[dict]) -> Iterator[Chunk]:
        """Run RAG -> route -> agent and stream tokens as Chunk objects."""
        query = self._extract_user_query(messages)
        requirements = self.rag_model.extract_requirements(query)
        target = self.router.route(requirements)

        agent = self.agents.get(target)
        if not agent:
            raise ValueError(f"No agent configured for route '{target}'")

        # Each agent streams plain text; front end accumulates it.
        for token in agent.stream(requirements=requirements):
            yield Chunk(
                output=Output(
                    choices=[
                        Choice(
                            message=DeltaMessage(
                                content=token,
                                reasoning_content=None,
                            ))
                    ]))

    def run(self, *, messages: list[dict]) -> str:
        """Non-streaming helper that collects the full text response."""
        parts: list[str] = []
        for chunk in self.stream(messages=messages):
            parts.append(chunk.output.choices[0].message.content or "")
        return "".join(parts)
