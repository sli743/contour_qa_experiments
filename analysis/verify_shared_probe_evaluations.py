#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

KEYS = ['patient_id', 'probe_type', 'level']
CHECKS = ['input_sha256', 'target_sha256']


def main() -> None:
    ap = argparse.ArgumentParser(description='Verify multiple model CSVs used exactly the same deterministic probe masks')
    ap.add_argument('--csv', action='append', required=True, help='NAME=path/to/probe_eval_sweep_detailed.csv')
    args = ap.parse_args()

    loaded = []
    for spec in args.csv:
        if '=' not in spec:
            raise ValueError(f'Expected NAME=PATH, got {spec!r}')
        name, raw = spec.split('=', 1)
        path = Path(raw).expanduser().resolve()
        df = pd.read_csv(path, dtype={'patient_id': str})
        missing = [c for c in KEYS + CHECKS if c not in df.columns]
        if missing:
            raise ValueError(f'{name}: {path} missing shared-bank verification columns: {missing}')
        if df.duplicated(KEYS).any():
            raise ValueError(f'{name}: duplicate patient/probe/level keys in {path}')
        loaded.append((name, path, df[KEYS + CHECKS].copy()))

    ref_name, ref_path, ref = loaded[0]
    ref = ref.rename(columns={c: f'{c}__ref' for c in CHECKS})
    for name, path, cur in loaded[1:]:
        merged = ref.merge(cur, on=KEYS, how='outer', indicator=True)
        if not merged['_merge'].eq('both').all():
            bad = merged.loc[~merged['_merge'].eq('both'), KEYS + ['_merge']].head(10)
            raise RuntimeError(f'{name} and {ref_name} do not contain identical probe keys. Examples:\n{bad}')
        for c in CHECKS:
            same = merged[f'{c}__ref'].astype(str).eq(merged[c].astype(str))
            if not same.all():
                bad = merged.loc[~same, KEYS + [f'{c}__ref', c]].head(10)
                raise RuntimeError(f'{name} and {ref_name} differ in {c}; probes are not identical. Examples:\n{bad}')
        print(f'OK: {name} matches {ref_name} for {len(merged)} deterministic probe rows')
    print(f'All {len(loaded)} evaluations share identical probe inputs and targets.')


if __name__ == '__main__':
    main()
