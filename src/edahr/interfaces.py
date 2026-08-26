from __future__ import annotations

from typing import Protocol, Sequence

from .schemas import ContextBlock, Generation, Hit, Node


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Hit]: ...


class Reranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class Generator(Protocol):
    def generate(self, query: str, context: Sequence[ContextBlock]) -> Generation: ...


class Verifier(Protocol):
    def support_score(self, claim: str, evidence: str) -> float: ...

