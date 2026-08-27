from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from edahr.models import NliVerifier, _generation_from_payload, _generation_schema, _invalid_generation


class CitationContractTests(unittest.TestCase):
    def test_schema_enumerates_only_request_context_ids(self):
        schema = _generation_schema(["C1", "C2"])
        citations = schema["schema"]["properties"]["claims"]["items"]["properties"]["citations"]
        self.assertEqual(citations["items"]["enum"], ["C1", "C2"])
        self.assertEqual(citations["minItems"], 1)

    def test_payload_contract_cases_are_retained_for_telemetry(self):
        cases = {
            "numeric": (["1"], "invalid_citation:1"),
            "bracketed": (["[C1]"], "invalid_citation:[C1]"),
            "blank": ([""], "empty_citation"),
            "unknown": (["C9"], "invalid_citation:C9"),
        }
        for name, (citations, expected_error) in cases.items():
            with self.subTest(name=name):
                generation = _generation_from_payload({
                    "answerable": True, "reason": "x",
                    "claims": [{"text": "claim", "citations": citations, "confidence": 0.9}],
                }, ["C1"])
                self.assertEqual(generation.claims[0].citations, tuple(citations))
                self.assertTrue(any(expected_error in error for error in generation.validation_errors))

    def test_valid_refusal_and_empty_answerable_output(self):
        refusal = _generation_from_payload({
            "answerable": False, "reason": "insufficient", "claims": [],
        }, ["C1"])
        self.assertFalse(refusal.validation_errors)
        empty = _generation_from_payload({
            "answerable": True, "reason": "x", "claims": [],
        }, ["C1"])
        self.assertIn("answerable_without_claims", empty.validation_errors)
        provider_refusal = _invalid_generation("provider_refusal", "refused")
        self.assertFalse(provider_refusal.answerable)
        self.assertEqual(provider_refusal.validation_errors, ("provider_refusal",))

    def test_valid_bare_context_id_is_accepted(self):
        generation = _generation_from_payload({
            "answerable": True, "reason": "x",
            "claims": [{"text": "claim", "citations": ["C1"], "confidence": 0.9}],
        }, ["C1"])
        self.assertEqual(generation.validation_errors, ())


class NliLabelTests(unittest.TestCase):
    def test_entailment_index_comes_from_model_config_labels(self):
        labels = {2: "contradiction", 0: "neutral", 1: "entailment"}
        self.assertEqual(NliVerifier._find_label_index(labels, "entail"), 1)
        self.assertEqual(NliVerifier._find_label_index(labels, "contradict"), 2)

    def test_entailment_is_selected_regardless_of_response_order(self):
        verifier = object.__new__(NliVerifier)
        verifier.entailment_label = "entailment"
        verifier.contradiction_label = "contradiction"
        verifier.classifier = lambda _: [
            {"label": "contradiction", "score": 0.97},
            {"label": "neutral", "score": 0.02},
            {"label": "entailment", "score": 0.01},
        ]
        self.assertEqual(verifier.score_details("claim", "evidence"), (0.01, 0.97))

    def test_missing_expected_response_label_fails_fast(self):
        verifier = object.__new__(NliVerifier)
        verifier.entailment_label = "entailment"
        verifier.contradiction_label = None
        verifier.classifier = lambda _: [{"label": "neutral", "score": 0.99}]
        with self.assertRaisesRegex(ValueError, "lacks configured label"):
            verifier.support_score("claim", "evidence")


if __name__ == "__main__":
    unittest.main()
