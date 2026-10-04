from dataclasses import replace

from edahr.config import Settings
from edahr.experimental_v9 import (Unit, blocks_from_units, dependency_groups,
    diagnostics, restrict_to_visible, select_units, sentence_units, verify_visible)
from edahr.schemas import Claim, Generation, Hierarchy, Level, Node


def hierarchy():
    text = 'Model X improves accuracy. This result holds only on dataset A. Other data show no gain.'
    node = Node('l1', Level.CHILD, 'd', 'paper', text, text, 1, 1,
                section_id='s', char_start=0, char_end=len(text),
                metadata={'paragraph_texts': {'p1': text}})
    return Hierarchy({'l1': node}, ('l1',))


def test_sentence_offsets_and_overlap_deduplication():
    h = hierarchy()
    h.nodes['l2'] = replace(h.node('l1'), node_id='l2')
    units = sentence_units(h, ['l1', 'l2'])
    assert len(units) == 3
    for unit in units:
        assert h.node(unit.leaf_id).text[unit.start:unit.end] == unit.text


def test_group_selection_is_atomic_and_budgeted():
    units = [Unit('a', 'answer', 'l', 's', 0, 6, 1),
             Unit('b', 'condition', 'l', 's', 7, 16, .1)]
    groups = {'a': ('a', 'b'), 'b': ('a', 'b')}
    assert select_units(units, 'answer', 49, lambda t: 1, groups=groups) == []
    chosen = select_units(units, 'answer', 50, lambda t: 1, groups=groups)
    assert {u.uid for u in chosen} == {'a', 'b'}


def test_dependency_anaphora_and_section_boundary():
    h = hierarchy()
    units = sentence_units(h, ['l1'])
    groups = dependency_groups(units, 'dependency')
    assert units[0].uid in groups[units[1].uid]
    isolated = replace(units[1], section_id='other')
    groups = dependency_groups([units[0], isolated], 'dependency')
    assert groups[isolated.uid] == (isolated.uid,)


def test_verifier_never_reads_hidden_or_uncited_same_leaf_text():
    h = hierarchy()
    units = sentence_units(h, ['l1'])
    blocks = blocks_from_units(units[:2], h, lambda s: len(s.split()))
    calls = []
    class Verifier:
        def score_details(self, claim, evidence):
            calls.append(evidence)
            return .95, .01
    raw = Generation(True, (Claim('Model X improves accuracy.', ('C1',), .99),))
    verified, evidence, trace = verify_visible(raw, blocks, h, Verifier(), Settings(), {'l1'})
    assert len(verified.claims) == 1
    assert calls == [units[0].text]
    assert list(evidence.values())[0].quote == units[0].text
    assert h.node('l1').text.endswith('no gain.')


def test_partial_leaf_does_not_count_as_full_evidence_retention():
    h = hierarchy()
    blocks = blocks_from_units(sentence_units(h, ['l1'])[:1], h, len)
    record = {'reference_evidence_sets': [[h.node('l1').text]]}
    raw = Generation(True, (Claim('claim', ('C1',), .9),))
    result = diagnostics(record, h, ['l1'], blocks, raw, {}, {'l1'})
    assert result['packed_leaf_touch_recall'] == 1
    assert result['packed_full_leaf_recall'] == 0
    assert result['complete_evidence_set_visible'] == 0
    assert 0 < result['gold_paragraph_char_coverage_best_ref'] < 1


def test_mmr_penalizes_duplicate_candidate():
    units = [Unit('a', 'method improves accuracy', 'a', 's', 0, 25, .95),
             Unit('b', 'method improves accuracy', 'b', 's', 30, 55, .94),
             Unit('c', 'dataset contains documents', 'c', 's', 60, 85, .90)]
    chosen = select_units(units, 'method dataset', 50, lambda text: 1, method='mmr')
    assert [u.uid for u in chosen] == ['a', 'c']
