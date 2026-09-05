#!/usr/bin/env python3
"""Thin Exa recall-expansion helper for research-paper search.

Adapted from wanshuiyin/Auto-claude-code-research-in-sleep at commit
e59008d7a42eea50a2797e55dd0d85bbbf6572f5 (MIT). See
../THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


INSTALL_MESSAGE = "exa-py not found. Install it with: pip install exa-py"


def get_client() -> Any:
    """Create an Exa client without making Exa a required dependency."""
    api_key = os.getenv("EXA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "EXA_API_KEY environment variable is required. Get a key from: https://exa.ai"
        )

    try:
        from exa_py import Exa
    except ImportError as exc:
        raise RuntimeError(INSTALL_MESSAGE) from exc

    client = Exa(api_key=api_key)
    client.headers["x-exa-integration"] = "brainx-skill-bio-neuro-lit"
    return client


def parse_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def content_kwargs(content_mode: str, max_chars: int) -> dict[str, Any]:
    if content_mode == "none":
        return {}
    if content_mode == "text":
        return {"text": {"max_characters": max_chars}}
    if content_mode == "summary":
        return {"summary": True}
    return {"highlights": {"max_characters": max_chars}}


def process_result(result: Any, content_mode: str) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "title": getattr(result, "title", None) or "No Title",
        "url": getattr(result, "url", None) or "",
    }

    for source_field, target_field in (
        ("published_date", "published_date"),
        ("author", "author"),
    ):
        value = getattr(result, source_field, None)
        if value:
            entry[target_field] = value

    if content_mode == "highlights":
        value = getattr(result, "highlights", None)
    elif content_mode == "text":
        value = getattr(result, "text", None)
    elif content_mode == "summary":
        value = getattr(result, "summary", None)
    else:
        value = None

    if value:
        entry[content_mode] = value
    return entry


def search(
    query: str,
    max_results: int = 10,
    search_type: str = "auto",
    content_mode: str = "highlights",
    max_chars: int = 4000,
    category: str = "research paper",
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    include_text: list[str] | None = None,
    exclude_text: list[str] | None = None,
    start_published_date: str | None = None,
    end_published_date: str | None = None,
) -> dict[str, Any]:
    """Search Exa and return lightweight recall-expansion records."""
    client = get_client()
    kwargs: dict[str, Any] = {
        "query": query,
        "num_results": max_results,
        "type": search_type,
        "category": category,
    }
    kwargs.update(content_kwargs(content_mode, max_chars))

    optional = {
        "include_domains": include_domains,
        "exclude_domains": exclude_domains,
        "include_text": include_text,
        "exclude_text": exclude_text,
        "start_published_date": start_published_date,
        "end_published_date": end_published_date,
    }
    kwargs.update({key: value for key, value in optional.items() if value})

    response = client.search_and_contents(**kwargs)
    return {
        "mode": "search",
        "query": query,
        "type": search_type,
        "category": category,
        "returned": len(response.results),
        "data": [process_result(result, content_mode) for result in response.results],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Expand biology or neuroscience paper recall through Exa."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    search_parser = subparsers.add_parser("search", help="Search Exa.")
    search_parser.add_argument("query")
    search_parser.add_argument("--max", type=int, default=10, metavar="N")
    search_parser.add_argument(
        "--type",
        default="auto",
        dest="search_type",
        choices=("auto", "neural", "fast", "instant"),
    )
    search_parser.add_argument(
        "--category", default="research paper", help="Exa category filter."
    )
    search_parser.add_argument(
        "--content",
        default="highlights",
        dest="content_mode",
        choices=("highlights", "text", "summary", "none"),
    )
    search_parser.add_argument("--max-chars", type=int, default=4000, metavar="N")
    search_parser.add_argument("--include-domains")
    search_parser.add_argument("--exclude-domains")
    search_parser.add_argument("--include-text")
    search_parser.add_argument("--exclude-text")
    search_parser.add_argument("--start-date")
    search_parser.add_argument("--end-date")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = search(
            query=args.query,
            max_results=args.max,
            search_type=args.search_type,
            content_mode=args.content_mode,
            max_chars=args.max_chars,
            category=args.category,
            include_domains=parse_list(args.include_domains),
            exclude_domains=parse_list(args.exclude_domains),
            include_text=parse_list(args.include_text),
            exclude_text=parse_list(args.exclude_text),
            start_published_date=args.start_date,
            end_published_date=args.end_date,
        )
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
