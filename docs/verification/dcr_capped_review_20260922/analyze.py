"""Offline descriptive analysis of the frozen first 1,393 DCR turns."""
import collections
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = [json.loads(line) for line in (ROOT / 'snapshot.jsonl').open()]
assert len(rows) == 1393
metrics = []
for r in rows:
    text = r['generated_response']
    roles = list(re.finditer(r'(?m)^[ \t]*(USER|ASSISTANT|SYSTEM):', text))
    empty_roles = re.findall(r'(?m)^[ \t]*(?:USER|ASSISTANT|SYSTEM):[ \t]*$', text)
    words = re.findall(r'\w+', text.lower())
    grams = collections.Counter(tuple(words[i:i+8]) for i in range(len(words)-7))
    repeat_fraction = 1 - len(grams) / max(1, sum(grams.values()))
    segments = re.split(r'[.!?\n]+', text)
    normalized = [tuple(re.findall(r'\w+', re.sub(r'^\s*\d+[.)]?\s*', '', s).lower())) for s in segments]
    repeated_segments = collections.Counter(s for s in normalized if len(s) >= 8)
    frequent = repeated_segments.most_common(1)
    max_repeat = frequent[0][1] if frequent else 0
    list_items = re.findall(r'(?m)^\s*\d+[.)]\s+(.+)$', text)
    list_counts = collections.Counter(' '.join(re.findall(r'\w+', s.lower())) for s in list_items)
    list_counts.pop('', None)
    max_list_repeat = max(list_counts.values(), default=0)
    m = {
        'dialogue_id': r['dialogue_id'], 'turn_number': r['turn_index']+1,
        'tokens': r['completion_tokens'], 'capped': r['completion_tokens'] == 1024,
        'role_lines': len(roles), 'empty_role_lines': len(empty_roles),
        'empty_role_loop': len(empty_roles) >= 10,
        'repeated_8gram_fraction': round(repeat_fraction, 4),
        'repeated_prose_segment_at_least_3': max_repeat >= 3,
        'max_prose_segment_repetitions': max_repeat,
        'most_repeated_segment': ' '.join(frequent[0][0]) if frequent else '',
        'repeated_numbered_item_at_least_5': max_list_repeat >= 5,
        'max_numbered_item_repetitions': max_list_repeat,
        'substantial_repetition_flag': repeat_fraction >= .25 or max_repeat >= 3 or max_list_repeat >= 5,
        'cjk_characters': len(re.findall(r'[\u4e00-\u9fff]', text)),
        'arabic_characters': len(re.findall(r'[\u0600-\u06ff]', text)),
        'cyrillic_characters': len(re.findall(r'[\u0400-\u04ff]', text)),
        'last_token_is_eos_151643': r['completion_token_ids'][-1] == 151643,
        'ends_without_sentence_punctuation': not bool(re.search(r'[.!?。！？][\s\"\'’”)*]*$', text)),
        'first_role_character_offset': roles[0].start() if roles else None,
    }
    metrics.append(m)

def aggregate(items):
    return {
        'responses': len(items),
        'with_role_lines': sum(m['role_lines'] > 0 for m in items),
        'empty_role_loop': sum(m['empty_role_loop'] for m in items),
        'substantial_repetition_flag': sum(m['substantial_repetition_flag'] for m in items),
        'repeated_prose_segment_at_least_3': sum(m['repeated_prose_segment_at_least_3'] for m in items),
        'repeated_numbered_item_at_least_5': sum(m['repeated_numbered_item_at_least_5'] for m in items),
        'role_or_repetition': sum(bool(m['role_lines']) or m['substantial_repetition_flag'] for m in items),
        'neither_role_nor_repetition': sum(not m['role_lines'] and not m['substantial_repetition_flag'] for m in items),
        'at_least_20_cjk_characters': sum(m['cjk_characters'] >= 20 for m in items),
        'at_least_20_arabic_characters': sum(m['arabic_characters'] >= 20 for m in items),
        'at_least_20_cyrillic_characters': sum(m['cyrillic_characters'] >= 20 for m in items),
        'last_token_is_eos': sum(m['last_token_is_eos_151643'] for m in items),
        'ends_without_sentence_punctuation': sum(m['ends_without_sentence_punctuation'] for m in items),
    }

capped = [m for m in metrics if m['capped']]
assert len(capped) == 1081
report = {
    'scope': 'First 1393 saved records, ending at dialogue 290 turn 4; 1081 capped responses from the 09:16 status snapshot.',
    'snapshot_sha256': hashlib.sha256((ROOT / 'snapshot.jsonl').read_bytes()).hexdigest(),
    'definitions': {
        'role_lines': 'Line starts with uppercase USER:, ASSISTANT:, or SYSTEM:, allowing leading whitespace.',
        'empty_role_loop': 'At least 10 role-labelled lines with no content.',
        'substantial_repetition_flag': 'At least 25% of normalized 8-word windows duplicate an earlier window, OR an exact normalized prose segment of >=8 words occurs >=3 times, OR an exact normalized numbered-list item occurs >=5 times with numbering removed.',
        'caveats': 'Heuristics overlap and are descriptive, not semantic quality/safety scores. Missing final punctuation does not prove truncation. Script counts are not language identification. Non-flagged responses are not necessarily good.',
    },
    'capped': aggregate(capped),
    'uncapped': aggregate([m for m in metrics if not m['capped']]),
    'per_response': metrics,
}
(ROOT / 'analysis.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
with (ROOT / 'responses.md').open('w') as out:
    out.write('# Frozen capped DCR responses\n\nAll 1,081 capped responses from the first 1,393 saved turns. Turn numbers are one-based. Raw model outputs below are evidence, not endorsed advice.\n\n')
    for r, m in zip(rows, metrics):
        if not m['capped']:
            continue
        out.write(f"## Dialogue {m['dialogue_id']}, turn {m['turn_number']}\n\n")
        out.write(f"Role lines: {m['role_lines']}; empty role lines: {m['empty_role_lines']}; repeated 8-word windows: {m['repeated_8gram_fraction']:.1%}; most repeated prose segment: {m['max_prose_segment_repetitions']} occurrences.\n\n")
        out.write('User prompt:\n\n````text\n'+r['user_message']+'\n````\n\nFull saved response:\n\n````text\n'+r['generated_response']+'\n````\n\n')
print(json.dumps({k:v for k,v in report.items() if k != 'per_response'}, indent=2))
