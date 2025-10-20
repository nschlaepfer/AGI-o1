import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4
import math

from openai import OpenAI


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    denom_a = math.sqrt(sum(x * x for x in a))
    denom_b = math.sqrt(sum(x * x for x in b))
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return dot / (denom_a * denom_b)


@dataclass
class MemoryItem:
    id: str
    title: str
    description: str
    content: str
    outcome: str
    tags: List[str]
    created_at: str
    source: Optional[str]
    embedding: List[float]
    metadata: Dict[str, Any]

    def as_prompt(self) -> str:
        outcome_label = f"[{self.outcome.upper()}]"
        return f"{outcome_label} {self.title} — {self.description}\n{self.content}"


class ReasoningBank:
    def __init__(
        self,
        client: OpenAI,
        storage_path: Path,
        embedding_model: str,
        reasoning_model: str,
        max_items: int = 500,
    ) -> None:
        self._client = client
        self._storage_path = Path(storage_path)
        self._embedding_model = embedding_model
        self._reasoning_model = reasoning_model
        self._max_items = max_items
        self._memories: List[MemoryItem] = []
        self._load()

    # Persistence ----------------------------------------------------------------
    def _load(self) -> None:
        if not self._storage_path.exists():
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            self._storage_path.write_text(json.dumps({"memories": []}, indent=2))
        try:
            data = json.loads(self._storage_path.read_text())
            self._memories = [
                MemoryItem(
                    id=item["id"],
                    title=item["title"],
                    description=item["description"],
                    content=item["content"],
                    outcome=item.get("outcome", "success"),
                    tags=item.get("tags", []),
                    created_at=item.get("created_at", datetime.utcnow().isoformat() + "Z"),
                    source=item.get("source"),
                    embedding=item.get("embedding", []),
                    metadata=item.get("metadata", {}),
                )
                for item in data.get("memories", [])
            ]
        except Exception as exc:
            logging.error("Failed to load ReasoningBank storage: %s", exc)
            self._memories = []

    def _save(self) -> None:
        data = {"memories": [asdict(m) for m in self._memories]}
        self._storage_path.write_text(json.dumps(data, indent=2))

    # Retrieval -------------------------------------------------------------------
    def retrieve(self, query: Optional[str], top_k: int = 3) -> List[Dict[str, Any]]:
        if not self._memories:
            return []
        if not query:
            recent = sorted(self._memories, key=lambda m: m.created_at, reverse=True)[:top_k]
            return [{"item": m, "score": None} for m in recent]
        embedding = self._embed_text(query)
        scored = [
            {"item": memory, "score": _cosine_similarity(memory.embedding, embedding)}
            for memory in self._memories
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def format_memory_prompt(self, retrieved: List[Dict[str, Any]]) -> Optional[str]:
        if not retrieved:
            return None
        lines = [
            "ReasoningBank Memory Hints:",
        ]
        for idx, item in enumerate(retrieved, start=1):
            memory = item["item"]
            score = item.get("score")
            score_label = f"(score={score:.2f})" if score is not None else ""
            lines.append(f"{idx}. {memory.title} {score_label}")
            lines.append(f"   {memory.description}")
            lines.append(f"   {memory.content}")
        return "\n".join(lines)

    # Interaction ingestion -------------------------------------------------------
    def process_interaction(
        self,
        task: str,
        agent_output: str,
        transcript: str,
        metadata: Optional[Dict[str, Any]] = None,
        outcome: Optional[str] = None,
    ) -> None:
        try:
            response = self._client.responses.create(
                model=self._reasoning_model,
                input=self._build_memory_prompt(task, agent_output, transcript),
                max_output_tokens=700,
                temperature=0.2,
                response_format={
                    "type": "json_schema",
                    "json_schema": self._memory_schema(),
                },
            )
        except Exception as exc:
            logging.warning("ReasoningBank distillation failed: %s", exc)
            return

        payload = None
        for item in getattr(response, "output", []) or []:
            for block in item.get("content", []):
                if block.get("type") in {"json", "json_schema"} and "json" in block:
                    payload = block["json"]
                    break
            if payload is not None:
                break
        if payload is None:
            try:
                payload = json.loads(response.output_text)
            except Exception as exc:
                logging.warning("ReasoningBank distillation JSON parse error: %s", exc)
                return

        confidence = payload.get("confidence", 0)
        overall_outcome = outcome or payload.get("outcome", "unknown")

        if confidence < 0.4:
            logging.info("Skipping ReasoningBank update due to low confidence.")
            return

        memories = payload.get("memories", [])
        if not memories:
            return

        items: List[MemoryItem] = []
        for memory in memories:
            try:
                embedding = self._embed_text(memory["content"])
                inferred_outcome = memory.get("outcome", overall_outcome or "unknown")
                item = MemoryItem(
                    id=str(uuid4()),
                    title=memory["title"],
                    description=memory["description"],
                    content=memory["content"],
                    outcome=inferred_outcome,
                    tags=memory.get("tags", []),
                    created_at=datetime.utcnow().isoformat() + "Z",
                    source=memory.get("source"),
                    embedding=embedding,
                    metadata={
                        **(metadata or {}),
                        "task": task,
                        "memory_outcome": inferred_outcome,
                        "overall_outcome": overall_outcome,
                        "confidence": confidence,
                    },
                )
                items.append(item)
            except Exception as exc:
                logging.warning("Skipping malformed memory item: %s", exc)

        if not items:
            return

        self._ingest(items)

    # Internal helpers ------------------------------------------------------------
    def _ingest(self, items: List[MemoryItem]) -> None:
        self._memories.extend(items)
        if len(self._memories) > self._max_items:
            self._memories = sorted(self._memories, key=lambda m: m.created_at, reverse=True)[
                : self._max_items
            ]
        self._save()
        logging.info("ReasoningBank stored %d new memory items.", len(items))

    def _embed_text(self, text: str) -> List[float]:
        response = self._client.embeddings.create(
            model=self._embedding_model,
            input=text,
        )
        return list(response.data[0].embedding)

    def _build_memory_prompt(self, task: str, agent_output: str, transcript: str) -> str:
        return (
            "You are a ReasoningBank memory curator. Extract transferable reasoning strategies "
            "and pitfalls from the interaction below.\n\n"
            f"Task:\n{task}\n\n"
            f"Agent output:\n{agent_output}\n\n"
            f"Recent transcript:\n{transcript}\n\n"
            "Label overall outcome as success or failure. Highlight reusable strategies or cautionary lessons."
        )

    @staticmethod
    def _memory_schema() -> Dict[str, Any]:
        return {
            "name": "ReasoningBankUpdate",
            "schema": {
                "type": "object",
                "properties": {
                    "confidence": {
                        "type": "number",
                        "description": "Confidence (0-1) that the extracted memories are correct.",
                    },
                    "outcome": {
                        "type": "string",
                        "enum": ["success", "failure", "mixed"],
                        "description": "Overall outcome of the interaction.",
                    },
                    "memories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "content": {"type": "string"},
                                "outcome": {
                                    "type": "string",
                                    "enum": ["success", "failure", "guardrail"],
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "source": {"type": "string"},
                            },
                            "required": ["title", "description", "content"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["confidence", "outcome", "memories"],
                "additionalProperties": False,
            },
        }

    # Introspection ----------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        return {
            "count": len(self._memories),
            "storage_path": str(self._storage_path),
            "embedding_model": self._embedding_model,
            "reasoning_model": self._reasoning_model,
        }

    def export(self) -> List[Dict[str, Any]]:
        return [asdict(m) for m in self._memories]
