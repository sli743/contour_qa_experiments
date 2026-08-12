#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def strip_first_h1(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    return "\n".join(lines).strip()


def rewrite_relative_links(text: str, prefix: str) -> str:
    pat = re.compile(r"(!?\[[^\]]*\]\()([^\)]+)(\))")

    def repl(m: re.Match[str]) -> str:
        target = m.group(2)
        if target.startswith(("http://", "https://", "#", "/", "mailto:")):
            return m.group(0)
        return f"{m.group(1)}{prefix.rstrip('/')}/{target}{m.group(3)}"

    return pat.sub(repl, text)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compose pooled and per-probe shared-bank four-model Markdown report"
    )
    ap.add_argument("--pooled-dir", type=Path, required=True)
    ap.add_argument("--by-probe-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--roi", default="Lung_L")
    ap.add_argument("--comparison-name", required=True)
    ap.add_argument("--model-label", action="append", required=True)
    args = ap.parse_args()

    pooled_md = args.pooled_dir / "all_probes_four_model_comparison.md"
    index_md = args.by_probe_dir / "index.md"
    if not pooled_md.is_file():
        raise FileNotFoundError(pooled_md)
    if not index_md.is_file():
        raise FileNotFoundError(index_md)
    if len(args.model_label) != 4:
        raise ValueError(f"Expected exactly four --model-label values, got {len(args.model_label)}")

    labels = ", ".join(f"**{x}**" for x in args.model_label[:-1])
    labels += f", and **{args.model_label[-1]}**"
    parts = [
        f"# {args.roi} four-model QA comparison — {args.comparison_name}",
        "",
        f"This report compares {labels} on the same deterministic shared probe bank.",
        "",
        "The pooled and probe-separated analyses use the project's two-criterion definition: global errors are "
        "classified by absolute whole-ROI volume change, while local bite/blob errors are classified by focal "
        "defect depth. Shift probes are excluded from these two-criterion ROC summaries and should be interpreted "
        "separately as positional errors.",
        "",
        "## Pooled across probe families",
        "",
        rewrite_relative_links(strip_first_h1(pooled_md.read_text(encoding="utf-8")), "pooled"),
        "",
        "## Probe-separated results",
        "",
    ]

    rx = re.compile(r"^- \[(?P<name>[^\]]+)\]\((?P<rel>[^\)]+)\)(?P<suffix>.*)$")
    for line in index_md.read_text(encoding="utf-8").splitlines():
        m = rx.match(line.strip())
        if not m:
            continue
        report_path = args.by_probe_dir / m.group("rel")
        if not report_path.is_file():
            continue
        probe_dir = report_path.parent.name
        body = strip_first_h1(report_path.read_text(encoding="utf-8"))
        body = rewrite_relative_links(body, f"by_probe/{probe_dir}")
        parts += [f"### {m.group('name')}", "", body, ""]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote combined report: {args.output}")


if __name__ == "__main__":
    main()
