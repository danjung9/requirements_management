"""
Template pipeline that runs a simple RAG -> router -> agent flow and streams
text back in the shape expected by Gradio_Events.submit.
"""
from dataclasses import dataclass
from typing import Iterator, Iterable, List, Dict, Any


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
    """Creates Jira tickets from requirements and streams text tokens."""

    def stream(self, requirements: str) -> Iterable[str]:
        # Replace with your Jira ticket creation logic and yield updates/tokens.
        result = f"[JIRA] Ticket(s) created for: {requirements}"
        for token in result.split():
            yield token + " "


class ComplianceMatrixAgent:
    """Creates a compliance matrix CSV and streams text tokens."""

    def stream(self, requirements: str) -> Iterable[str]:
        # Replace with your CSV generation logic and yield CSV lines or tokens.
        csv_lines = [
            "Requirement,Control,Status",
            f'"{requirements}",ExampleControl,Pending',
        ]
        for line in csv_lines:
            yield line + "\n"


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
