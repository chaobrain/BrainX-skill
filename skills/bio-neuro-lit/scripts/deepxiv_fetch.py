#!/usr/bin/env python3
"""Thin progressive-reading adapter around the installed DeepXiv CLI.

Adapted from wanshuiyin/Auto-claude-code-research-in-sleep at commit
e59008d7a42eea50a2797e55dd0d85bbbf6572f5 (MIT). See
../THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from typing import Sequence


INSTALL_MESSAGE = "deepxiv CLI not found. Install it with: pip install deepxiv-sdk"


def ensure_deepxiv_installed() -> dict[str, object]:
    """Return the installed DeepXiv binary, if available."""
    binary = shutil.which("deepxiv")
    if binary:
        return {"ok": True, "binary": binary, "message": ""}
    return {"ok": False, "binary": None, "message": INSTALL_MESSAGE}


def run_cli_json(args: Sequence[str]) -> dict | list:
    """Run DeepXiv and decode its JSON output."""
    install = ensure_deepxiv_installed()
    if not install["ok"]:
        raise RuntimeError(str(install["message"]))

    process = subprocess.run(
        [str(install["binary"]), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if process.returncode != 0:
        message = (process.stderr or process.stdout or "deepxiv command failed").strip()
        raise RuntimeError(message)

    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("deepxiv returned invalid JSON output") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Progressively read a known arXiv paper through DeepXiv."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    brief = subparsers.add_parser(
        "paper-brief", help="Fetch brief paper metadata and TLDR."
    )
    brief.add_argument("arxiv_id")

    head = subparsers.add_parser(
        "paper-head", help="Fetch paper metadata and the section overview."
    )
    head.add_argument("arxiv_id")

    section = subparsers.add_parser(
        "paper-section", help="Fetch one named paper section."
    )
    section.add_argument("arxiv_id")
    section.add_argument("section_name")

    return parser


def dispatch(args: argparse.Namespace) -> dict | list:
    if args.command == "paper-brief":
        return run_cli_json(["paper", args.arxiv_id, "--brief", "--format", "json"])
    if args.command == "paper-head":
        return run_cli_json(["paper", args.arxiv_id, "--head", "--format", "json"])
    if args.command == "paper-section":
        return run_cli_json(
            [
                "paper",
                args.arxiv_id,
                "--section",
                args.section_name,
                "--format",
                "json",
            ]
        )
    raise RuntimeError(f"Unsupported command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = dispatch(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
