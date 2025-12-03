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

    def extract_requirements(self, query: str) -> dict:
        docs = self.retriever.search(query)
        # Replace with your own synthesis and compliance assessment.
        requirements = self.llm.summarize(query=query, docs=docs)
        compliant = self._assess_compliance(requirements)
        return {
            "requirements": requirements,
            "compliant": compliant,
        }

    def _assess_compliance(self, requirements: str) -> bool:
        text = requirements.lower()
        non_compliant_markers = [
            "gap",
            "missing",
            "non-compliant",
            "not compliant",
            "fail",
        ]
        return not any(marker in text for marker in non_compliant_markers)


class Router:
    """Chooses a target pipeline for the extracted requirements."""

    def route(self, *, compliant: bool, requirements: str, user_query: str = "") -> str:
        """
        Route based on explicit intent first, otherwise by compliance status.

        - If the user asks for a Jira ticket (keywords like "jira", "ticket",
          "issue", "bug", "story", "epic"), route to Jira.
        - Else: non-compliant -> Jira; compliant -> matrix.
        """
        intent = user_query.lower()
        wants_jira = any(
            keyword in intent
            for keyword in ("jira", "ticket", "issue", "bug", "story", "epic")
        )
        if wants_jira:
            return "jira"
        return "matrix" if compliant else "jira"


class JiraAgent:
    """Generates Jira ticket content using a Qwen model on OpenRouter and streams text."""

    def __init__(self,
                 model: str = "qwen/qwen3-vl-8b-instruct", #"qwen/qwen3-4b:free",
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
            "You are a Jira assistant. Respond ONLY with valid JSON (no markdown) in this exact shape:\n"
            '{\n'
            '  "fields": {\n'
            '    "project": { "key": "SERVICEDESK" },\n'
            '    "summary": "User-submitted issue summary",\n'
            '    "description": "Detailed problem description",\n'
            '    "issuetype": { "name": "Incident" },\n'
            '    "reporter": { "name": "username@email.com" },\n'
            '    "priority": { "name": "High" },\n'
            '    "labels": ["mobile-app", "user-submitted"],\n'
            '    "customfield_10002": { "value": "Customer" }\n'
            '  }\n'
            '}\n'
            "Do not add, remove, or rename keys. Fill the values using the user's request. Keep it to valid JSON only.")
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
                 model: str = "qwen/qwen3-vl-8b-instruct",#"qwen/qwen3-4b:free",
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
            "You are a compliance analyst. Produce ONLY a markdown table (no code fences, no prose) "
            "with these exact headers:\n"
            "| Requirement ID | Requirement Text | Source (Spec/Std) | Verification Method | Evidence / Link | Status | Notes |\n"
            "Use realistic IDs (e.g., SYS-REQ-001…), short requirement text, cite a plausible source, "
            "choose a verification method (Test, Inspection, Analysis, or combo), provide a concise evidence/link label, "
            "set Status to one of: Compliant, Partially compliant, Non-compliant, or Pending, and keep Notes brief.")
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
        extraction = self.rag_model.extract_requirements(query)
        requirements = extraction["requirements"]
        compliant = extraction["compliant"]

        target = self.router.route(
            compliant=compliant,
            requirements=requirements,
            user_query=query,
        )

        agent = self.agents.get(target)
        if not agent:
            raise ValueError(f"No agent configured for route '{target}'")

        # Console visibility for debugging which agent is used.
        print(f"[pipeline] routing to '{target}' (compliant={compliant})")

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
