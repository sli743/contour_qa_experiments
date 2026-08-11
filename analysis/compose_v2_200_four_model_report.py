#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def strip_first_h1(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith('# '):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    return '\n'.join(lines).strip()


def rewrite_relative_links(text: str, prefix: str) -> str:
    # Rebase ordinary relative markdown links/images while leaving anchors/URLs untouched.
    pat = re.compile(r'(!?\[[^\]]*\]\()([^\)]+)(\))')
    def repl(m: re.Match[str]) -> str:
        target = m.group(2)
        if target.startswith(('http://', 'https://', '#', '/', 'mailto:')):
            return m.group(0)
        return f"{m.group(1)}{prefix.rstrip('/')}/{target}{m.group(3)}"
    return pat.sub(repl, text)


def main() -> None:
    ap = argparse.ArgumentParser(description='Compose pooled + per-probe v2-200 four-model markdown report')
    ap.add_argument('--pooled-dir', type=Path, required=True)
    ap.add_argument('--by-probe-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--roi', default='Lung_L')
    args = ap.parse_args()

    pooled_md = args.pooled_dir / 'all_probes_four_model_comparison.md'
    index_md = args.by_probe_dir / 'index.md'
    if not pooled_md.is_file():
        raise FileNotFoundError(pooled_md)
    if not index_md.is_file():
        raise FileNotFoundError(index_md)

    parts = [
        f'# {args.roi} four-model QA comparison — stochastic v2-200',
        '',
        'This report compares **Pilot AE**, **Benchmark-leakage DAE**, **Random-morphology DAE**, and '
        '**Stochastic-region-dropout DAE v2-200** on the same deterministic shared probe bank.',
        '',
        'The pooled and probe-separated analyses use the project\'s two-criterion definition: global errors are '
        'classified by absolute whole-ROI volume change, local bite/blob errors by focal defect depth. Shift probes '
        'are excluded from these two-criterion ROC summaries and should be analyzed separately as positional errors.',
        '',
        '## Pooled across probe families',
        '',
        rewrite_relative_links(strip_first_h1(pooled_md.read_text(encoding='utf-8')), 'pooled'),
        '',
        '## Probe-separated results',
        '',
    ]

    # index lines look like: - [dilated](dilated/report.md) — criterion
    rx = re.compile(r'^- \[(?P<name>[^\]]+)\]\((?P<rel>[^\)]+)\)(?P<suffix>.*)$')
    for line in index_md.read_text(encoding='utf-8').splitlines():
        m = rx.match(line.strip())
        if not m:
            continue
        report_path = args.by_probe_dir / m.group('rel')
        if not report_path.is_file():
            continue
        probe_dir = report_path.parent.name
        body = strip_first_h1(report_path.read_text(encoding='utf-8'))
        body = rewrite_relative_links(body, f'by_probe/{probe_dir}')
        parts += [f"### {m.group('name')}", '', body, '']

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(parts).rstrip() + '\n', encoding='utf-8')
    print(f'Wrote combined report: {args.output}')


if __name__ == '__main__':
    main()
