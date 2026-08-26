from __future__ import annotations

from pathlib import Path

from .config import Settings
from .hierarchy import HierarchyBuilder
from .index import MultiRepresentationIndex
from .ingestion import DoclingScientificLoader
from .models import (
    BGEM3Encoder,
    BGEReranker,
    GeminiStructuredGenerator,
    NliVerifier,
    OpenAIStructuredGenerator,
)
from .pipeline import AdaptiveHierarchicalPipeline


def build_pipeline(
    pdf_paths: list[str | Path], settings: Settings | None = None
) -> AdaptiveHierarchicalPipeline:
    settings = settings or Settings()
    documents = DoclingScientificLoader().load(pdf_paths)
    hierarchy = HierarchyBuilder(settings).build(documents)
    encoder = BGEM3Encoder(settings.embedding_model, settings.device, settings.use_fp16)
    index = MultiRepresentationIndex(hierarchy, encoder, settings)
    reranker = BGEReranker(settings.reranker_model, settings.device, settings.use_fp16)
    if settings.llm_provider == "gemini":
        generator = GeminiStructuredGenerator(settings.llm_model, settings.gemini_api_key)
    else:
        generator = OpenAIStructuredGenerator(settings.llm_model, settings.openai_api_key)
    verifier = NliVerifier(settings.nli_model, settings.device)
    return AdaptiveHierarchicalPipeline(
        hierarchy=hierarchy,
        retriever=index,
        reranker=reranker,
        generator=generator,
        verifier=verifier,
        settings=settings,
    )
