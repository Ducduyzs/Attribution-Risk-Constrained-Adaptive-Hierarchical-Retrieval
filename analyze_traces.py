import json
from collections import Counter

# Analyze B_flat
flat_statuses = []
flat_invalid = 0
flat_total_claims = 0

with open('data/artifacts/v6_dev_diagnostic_rerun_fixed/artifacts_B_flat.jsonl') as f:
    for line in f:
        rec = json.loads(line)
        flat_total_claims += rec['generated_claim_count']
        for trace in rec.get('verification_trace', []):
            flat_statuses.append(trace['status'])
            if trace['status'] == 'invalid_or_missing_context_citation':
                flat_invalid += 1

# Analyze learned_v5
learned_statuses = []
learned_invalid = 0
learned_total_claims = 0

with open('data/artifacts/v6_dev_diagnostic_rerun_fixed/artifacts_learned_v5.jsonl') as f:
    for line in f:
        rec = json.loads(line)
        learned_total_claims += rec['generated_claim_count']
        for trace in rec.get('verification_trace', []):
            learned_statuses.append(trace['status'])
            if trace['status'] == 'invalid_or_missing_context_citation':
                learned_invalid += 1

print('=== B_flat ===')
print('Total claims:', flat_total_claims)
print('Status distribution:', Counter(flat_statuses))
print('Invalid citation rate:', flat_invalid, '/', flat_total_claims, '=', flat_invalid/flat_total_claims)

print()
print('=== learned_v5 ===')
print('Total claims:', learned_total_claims)
print('Status distribution:', Counter(learned_statuses))
print('Invalid citation rate:', learned_invalid, '/', learned_total_claims, '=', learned_invalid/learned_total_claims)

# Check the one accepted claim in learned_v5
with open('data/artifacts/v6_dev_diagnostic_rerun_fixed/artifacts_learned_v5.jsonl') as f:
    for line in f:
        rec = json.loads(line)
        for trace in rec.get('verification_trace', []):
            if trace['status'] == 'accepted':
                print()
                print('=== Accepted claim example (learned_v5) ===')
                print('Query:', rec['query'])
                print('Claim:', trace['claim_text'])
                print('Cited context IDs:', trace['cited_context_ids'])
                print('Candidates:', trace['candidates'])
                print('Selected:', trace['selected_child_ids'])