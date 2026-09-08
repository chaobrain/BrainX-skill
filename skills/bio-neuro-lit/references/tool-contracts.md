# Literature tool contracts

Open this reference before invoking the literature tools or when diagnosing availability. MCP clients may prefix a registered server name, such as `mcp__europepmc__search_articles`; the leaf tool names and input fields below are the server contracts. Inspect the available tool list rather than inventing a different leaf name.

Install and activate the bundled MCP set in Codex with `npx brainx-skill mcp install`. This registers Europe PMC as `europepmc`, the Full-Text Resolver as `fulltext_resolver`, and the BrainX reviewer as `codex`. Restart Codex or reload MCP configuration after installation. If registration fails because one of those names already exists, do not overwrite it automatically; inspect the existing user-owned configuration first.

## Europe PMC MCP

The skill uses the existing CoChatAI Europe PMC MCP interface. It requires no API key.

| Tool | Input schema | Use |
|---|---|---|
| `search_articles` | `query: str`; `result_type: str = "lite"`; `page_size: int = 25`; `cursor_mark: str = "*"`; `sort: str | null = null` | Search Lucene terms and field qualifiers. Valid result types are `lite`, `core`, and `idlist`; page size is clamped to 1-1000. Returns `hitCount`, `nextCursorMark`, and `resultList.result[]`. |
| `get_article` | `source: str`; `ext_id: str` | Fetch one core article record. Common sources are `MED` for PMID, `PMC` for PMCID, and `PPR` for preprint records. |
| `get_references` | `source: str`; `ext_id: str`; `page_size: int = 25`; `page: int = 1` | Snowball backward from a selected paper when its cited work is relevant to the question. |
| `get_citations` | `source: str`; `ext_id: str`; `page_size: int = 25`; `page: int = 1` | Snowball forward from a selected paper when later evidence is relevant. |
| `get_database_links` | `source: str`; `ext_id: str`; `db: str | null = null` | Retrieve linked biological databases only when the review needs those cross-references. |
| `get_fulltext_xml` | `pmcid: str` | Existing Europe PMC JATS access. In this skill, prefer the Full-Text Resolver because it normalizes Markdown and preserves cross-source version provenance. |
| `get_annotations` | `article_ids: list[str]`; `type: str | null = null`; `section: str | null = null` | Retrieve text-mined entities for up to eight `SOURCE:ID` values only when entity annotations help the review. |

Useful `search_articles` query fields include `TITLE_ABS`, `AUTH`, `JOURNAL`, `PUB_YEAR`, `OPEN_ACCESS`, and `HAS_FT`. Use `TITLE_ABS` to constrain decisive mechanistic or computational terms after a broad query drifts into clinical material. Use `sort: "P_PDATE_D desc"` only when the question requires newest-first retrieval; otherwise keep relevance order.

## Full-Text Resolver MCP

The bundled server registers the following exact tools.

Shared work fields:

| Field | Type | Meaning |
|---|---|---|
| `doi` | optional string | Published or preprint DOI; prefer it when available unless a PMCID is already known. |
| `pmid` | optional string | PubMed identifier. |
| `pmcid` | optional string | PubMed Central identifier, with or without `PMC`. |
| `title` | optional string | Exact title fallback; resolution is conservative. |
| `authors` | optional string or string array | Narrows title lookup. |
| `year` | optional integer, 1600-3000 | Narrows title lookup. |

At least one of `doi`, `pmid`, `pmcid`, or `title` is required.

| Tool | Additional input | Use |
|---|---|---|
| `resolve_fulltext` | none | Resolve the best accessible version and return provenance without downloading the document. |
| `list_versions` | none | List accessible `publishedVersion`, `acceptedVersion`, `submittedVersion`, or `unknown` candidates. |
| `get_fulltext` | `version?: publishedVersion | acceptedVersion | submittedVersion | unknown`; `source?: PMC | OpenAlex | bioRxiv | medRxiv`; `include_raw: bool = false`; `max_chars?: int` from 1000 to 2,000,000 | Retrieve JATS or HTML as normalized Markdown. PDF-only versions remain provenance-preserving links and are not parsed by resolver v1. |

For normal reading, call `get_fulltext` directly. Use `list_versions` first when version selection changes the scientific interpretation. Omit `max_chars` for a complete converted document; use it only when a bounded first pass is needed.

If the server is registered in Codex as `fulltext_resolver`, its rendered tool names are commonly `mcp__fulltext_resolver__resolve_fulltext`, `mcp__fulltext_resolver__list_versions`, and `mcp__fulltext_resolver__get_fulltext`.

## Exa helper

Resolve `scripts/exa_search.py` relative to this skill directory. Exa is optional and requires Python 3, `exa-py`, and `EXA_API_KEY`.

```bash
python3 <skill-directory>/scripts/exa_search.py search \
  "QUERY" --max 10 --category "research paper" --content highlights
```

Useful optional filters are `--type`, `--include-domains`, `--exclude-domains`, `--include-text`, `--exclude-text`, `--start-date`, `--end-date`, and `--max-chars`. Prefer research-paper searches. Treat lab pages, project pages, and technical reports as leads unless their claims are supported by a paper.

When `exa-py` or `EXA_API_KEY` is missing, record that recall expansion was unavailable and continue with Europe PMC.

## DeepXiv helper

Resolve `scripts/deepxiv_fetch.py` relative to this skill directory. DeepXiv is optional and requires the `deepxiv` CLI from `deepxiv-sdk`.

```bash
python3 <skill-directory>/scripts/deepxiv_fetch.py paper-brief ARXIV_ID
python3 <skill-directory>/scripts/deepxiv_fetch.py paper-head ARXIV_ID
python3 <skill-directory>/scripts/deepxiv_fetch.py paper-section ARXIV_ID "Methods"
```

Run the commands in that order and stop as soon as the evidence is sufficient. If DeepXiv is unavailable or a section cannot be retrieved, use the Full-Text Resolver for the selected work when possible; otherwise retain abstract-only evidence.
