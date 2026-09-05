# Reuse audit

Audit completed on September 4, 2026. Forks were created under `Wilsonnijc-bot` before implementation.

| Component | Source repository and audited revision | Reused files or modules | License | Changes required |
|---|---|---|---|---|
| Europe PMC metadata and JATS retrieval | `CoChatAI/europepmc-mcp-server` at `a1af9d06ec7f5f2caef56005fdc1d9d4f12cb115`; fork `Wilsonnijc-bot/europepmc-mcp-server` | `server.py`: REST base URL, search contract, full-text XML endpoint, timeout and user-agent behavior | MIT | Adapted to a Node source adapter; retained only lookup and JATS retrieval needed by the resolver. Added accepted-manuscript detection and normalized provenance. |
| OpenAlex DOI and title resolution | `DeyangLiu123/openalex-mcp` at `e139b22ced22e0b724ad84ce43bacb470017161a`; fork `Wilsonnijc-bot/openalex-mcp` | `ids.py`: DOI normalization, conservative exact-title matching, identifier parsing; `formats.py`: OA location access patterns | MIT | Ported pure normalization and conservative matching behavior to Node. Removed search/harvesting and paid content-download functionality. |
| OpenAlex transport and location fields | `cyanheads/openalex-mcp-server` at `513f1158f03bd1dda4a180941c75d7506e6723c5`; fork `Wilsonnijc-bot/openalex-mcp-server` | `openalex-service.ts`, `types.ts`, `server-config.ts`, `field-catalog.json`: anonymous/keyed configuration, retries, `best_oa_location`, `primary_location`, `locations[]`, `version`, license, and URL fields | Apache-2.0 | Adapted the works-only subset to Node 18 and the official MCP SDK. Kept API key and `mailto` optional. Did not use the paid OpenAlex content endpoint. |
| bioRxiv/medRxiv crosswalk and full text | `cyanheads/biorxiv-mcp-server` at `2b406943815192328481797332cc706b386887ff`; fork `Wilsonnijc-bot/biorxiv-mcp-server` | `biorxiv-service.ts`, `biorxiv-fulltext-service.ts`, `types.ts`, `shared.ts`: `/pubs`, `/details`, revision selection, JATS/HTML URLs, markup cleanup, retries and error isolation | Apache-2.0 | Added the official reverse lookup from published DOI to preprint DOI, selected the latest revision, and normalized the result into the common provenance model. Removed framework-specific storage and transport dependencies. |
| JATS conversion fallback | `yogsoth-ai/biorxiv-mcp` at `c17747a4a635003d692d4e05b6eda093536908dd`; fork `Wilsonnijc-bot/biorxiv-mcp` | `convert.py`: JATS-first conversion and degraded text fallback; `fetch.py`: structured-source preference | Apache-2.0 | Reimplemented the converter in Node with `linkedom`, preserving structured abstract, sections, figure captions, tables, and references. The AWS requester-pays MECA backend remains optional and is not required in v1. |
| MCP stdio server | Official `@modelcontextprotocol/sdk` package, version range `^1.30.0` | `McpServer`, `StdioServerTransport`, Zod input validation, tool annotations | MIT | Registered only `resolve_fulltext`, `get_fulltext`, and `list_versions`; no custom JSON-RPC transport was written. |

## Selection decision

Use Node.js because the host repository already requires Node 18 and the richest reusable bioRxiv implementation is TypeScript. This avoids requiring Python, Pandoc, FastMCP, Bun, Node 24, AWS, or a second environment for the default resolver.

The upstream MCP servers were not vendored whole because they expose broad search and analysis interfaces outside this resolver's boundary. The implementation adapts only the source-client, normalization, retry, conversion, and provenance behavior needed by the three public tools.

## Reuse boundary

New code is limited to:

```text
common resolver orchestration
normalized version/provenance model
source ranking and fallback
MCP tool composition
end-to-end tests
```

No Nature, Cell Press, Neuron, PNAS, arXiv, Crossref, Unpaywall, CORE, Semantic Scholar, or paywall-specific client was added.
