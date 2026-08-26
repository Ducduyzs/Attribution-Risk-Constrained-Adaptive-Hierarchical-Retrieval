from __future__ import annotations

import json
import os
from typing import Sequence

from .schemas import Claim, ContextBlock, Generation


class BGEM3Encoder:
    """BGE-M3 adapter exposing dense, learned-sparse and ColBERT vectors."""

    def __init__(self, model_name: str, device: str = "cuda", use_fp16: bool = True):
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise RuntimeError("Install FlagEmbedding to use BGE-M3 retrieval") from exc
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16, devices=device)

    def encode(self, texts: Sequence[str], batch_size: int = 12) -> dict:
        return self.model.encode(
            list(texts),
            batch_size=batch_size,
            max_length=8192,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )


class BGEReranker:
    def __init__(self, model_name: str, device: str = "cuda", use_fp16: bool = True):
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install FlagEmbedding to use the neural reranker") from exc
        self.model = FlagReranker(model_name, use_fp16=use_fp16, devices=device)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        values = self.model.compute_score([[query, text] for text in texts], normalize=True)
        if isinstance(values, float):
            return [values]
        return [float(value) for value in values]


class NliVerifier:
    """Entailment scorer used to reject unsupported generated claims."""

    def __init__(self, model_name: str, device: str = "cuda"):
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install transformers to use NLI verification") from exc
        pipeline_device = 0 if device.startswith("cuda") else -1
        self.classifier = pipeline(
            "text-classification", model=model_name, device=pipeline_device, top_k=None
        )

    def support_score(self, claim: str, evidence: str) -> float:
        result = self.classifier({"text": evidence, "text_pair": claim})
        by_label = {
            str(item["label"]).lower(): float(item["score"])
            for item in self._label_items(result)
        }
        for label, score in by_label.items():
            if "entail" in label:
                return score
        return max(by_label.values(), default=0.0)

    @staticmethod
    def _label_items(result):
        # transformers >=5 returns a flat list of {label, score} dicts for a
        # single dict input; some 4.x versions nest it one level deeper.
        if isinstance(result, dict):
            return [result]
        first = result[0] if result else None
        if isinstance(first, list):
            return first
        return list(result)


class OpenAIStructuredGenerator:
    """Grounded generator that requires claim-level context identifiers."""

    def __init__(self, model_name: str, api_key: str | None = None, temperature: float = 0.0):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install openai to use OpenAI generation") from exc
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, query: str, context: Sequence[ContextBlock]) -> Generation:
        evidence = "\n\n".join(
            f"[{block.context_id}] {block.source}, pages {block.page_start}-{block.page_end}\n{block.text}"
            for block in context
        )
        prompt = f"""Answer the scientific question only from the supplied evidence.
Return JSON with keys answerable (boolean), reason (string), and claims (array).
Each claim must contain text, citations (context IDs), and confidence from 0 to 1.
Do not cite an ID that is absent from the evidence. If evidence is insufficient,
set answerable=false and return no claims.

Question: {query}

Evidence:
{evidence}
"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=self.temperature,
        )
        return _generation_from_payload(json.loads(response.choices[0].message.content))


class GeminiStructuredGenerator:
    """Grounded generator that requires claim-level context identifiers."""

    def __init__(self, model_name: str, api_key: str | None = None):
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install google-genai to use Gemini generation") from exc
        self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        self.model_name = model_name
        self.temperature = 0.0

    def generate(self, query: str, context: Sequence[ContextBlock]) -> Generation:
        evidence = "\n\n".join(
            f"[{block.context_id}] {block.source}, pages {block.page_start}-{block.page_end}\n{block.text}"
            for block in context
        )
        prompt = f"""Answer the scientific question only from the supplied evidence.
Return JSON with keys answerable (boolean), reason (string), and claims (array).
Each claim must contain text, citations (context IDs), and confidence from 0 to 1.
Do not cite an ID that is absent from the evidence. If evidence is insufficient,
set answerable=false and return no claims.

Question: {query}

Evidence:
{evidence}
"""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": self.temperature,
            },
        )
        return _generation_from_payload(json.loads(response.text))


def _generation_from_payload(payload: dict) -> Generation:
    claims = tuple(
        Claim(
            text=str(item["text"]),
            citations=tuple(str(value) for value in item.get("citations", [])),
            confidence=float(item.get("confidence", 0.0)),
        )
        for item in payload.get("claims", [])
    )
    return Generation(
        answerable=bool(payload.get("answerable", bool(claims))),
        claims=claims,
        reason=str(payload.get("reason", "")),
    )
