"""Quick look at the `shared` family (practice set). Run from the repo root:

    python3 analysis/shared/00_peek.py
"""
import collections
import csv
import json

rows = list(csv.DictReader(open('datasets/practice_puzzles.csv')))
print(list(rows[0].keys()))
c = collections.Counter((r['family'], r['design_tier']) for r in rows)
for k, v in sorted(c.items()):
    print(k, v)
sh = [r for r in rows if r['family'] == 'shared']
for r in sh[:6]:
    print(r['id'], r['starting_values'], r['value_limit'], r['lower_bound'], r['design_tier'])
    print(r['targets'])
