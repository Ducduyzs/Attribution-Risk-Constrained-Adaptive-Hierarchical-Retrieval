"""Small real-weight smoke checks for the GPU model stack.

Each component is intentionally run as a separate process by ``all`` so GPU
memory is released between checks.  No API credentials or benchmark artifacts
are read or written.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

COMPONENTS = ("bge_m3", "reranker", "nli", "raptor", "longrag", "bart", "qwen")


def run_component(name: str) -> None:
    if name == "bge_m3":
        from edahr.models import BGEM3Encoder

        result = BGEM3Encoder("BAAI/bge-m3").encode(["scientific evidence retrieval"])
        assert len(result["dense_vecs"]) == 1
    elif name == "reranker":
        from edahr.models import BGEReranker

        scores = BGEReranker("BAAI/bge-reranker-v2-m3").score(
            "What is adaptive retrieval?", ["Adaptive retrieval changes its search depth."]
        )
        assert len(scores) == 1
    elif name == "nli":
        from edahr.models import NliVerifier

        support, contradiction = NliVerifier(
            "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        ).score_details("The method is adaptive.", "The method adapts retrieval depth.")
        assert 0.0 <= support <= 1.0 and 0.0 <= contradiction <= 1.0
    elif name == "raptor":
        from edahr.baselines_raptor import _embed_texts

        vectors = _embed_texts(
            ["RAPTOR recursively clusters and summarizes text."],
            "sentence-transformers/multi-qa-mpnet-base-cos-v1",
            "cuda",
        )
        assert len(vectors) == 1 and vectors[0]
    elif name == "longrag":
        from edahr.baselines_longrag import _embed_texts

        vectors = _embed_texts(
            ["LongRAG retrieves long semantic units."],
            "BAAI/bge-large-en-v1.5",
            "cuda",
            1,
        )
        assert len(vectors) == 1 and vectors[0]
    elif name == "bart":
        from edahr.baselines_raptor import LocalAbstractiveSummarizer

        summarizer = LocalAbstractiveSummarizer(
            "facebook/bart-large-cnn", max_length=24, min_length=5
        )
        assert summarizer(
            "Adaptive retrieval selects evidence at multiple hierarchy levels. "
            "It aims to preserve evidence while controlling context cost."
        )
    elif name == "qwen":
        from edahr.baselines_longrag import LocalFreeReader, LongRagUnit

        reader = LocalFreeReader("Qwen/Qwen2.5-7B-Instruct", max_new_tokens=16)
        unit = LongRagUnit(
            unit_id="smoke", source="smoke", text="The reported value is 42.",
            paragraphs=((0, 25, "p1", "The reported value is 42."),),
            member_child_ids=("c1",),
        )
        assert reader.generate_answer("What value is reported?", [unit])
    else:  # pragma: no cover - argparse prevents this
        raise ValueError(name)
    print(f"PASS {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=("all", *COMPONENTS), default="all")
    args = parser.parse_args()
    if args.component != "all":
        run_component(args.component)
        return

    env = os.environ.copy()
    for component in COMPONENTS:
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--component", component],
            check=True,
            cwd=PROJECT_ROOT,
            env=env,
        )
    print("PASS all", flush=True)


if __name__ == "__main__":
    main()
