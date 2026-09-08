# Full-text resolver MCP

This stdio MCP resolves biology and neuroscience papers to the best accessible full-text version without duplicating general literature search.

Source priority is:

```text
published PMC or publisher OA
accepted repository manuscript
submitted bioRxiv/medRxiv or repository preprint
unresolved
```

Every result preserves the requested work separately from the retrieved document. A preprint is never labeled as the final journal article.

## Tools

| Tool | Use |
|---|---|
| `resolve_fulltext` | Resolve the best accessible version and return provenance without downloading the document. |
| `get_fulltext` | Retrieve JATS or HTML as Markdown. Preserve PDF-only versions as source links. |
| `list_versions` | List every known accessible published, accepted, and submitted version. |

All tools accept any useful combination of `doi`, `pmid`, `pmcid`, `title`, `authors`, and `year`. Prefer DOI. Title resolution requires a normalized exact title match; the resolver does not confirm relationships from title similarity alone.

`get_fulltext` also accepts:

| Input | Meaning |
|---|---|
| `version` | Restrict retrieval to `publishedVersion`, `acceptedVersion`, `submittedVersion`, or `unknown`. |
| `source` | Restrict retrieval to `PMC`, `OpenAlex`, `bioRxiv`, or `medRxiv`. |
| `include_raw` | Include retrieved JATS or HTML in `structuredContent`. Defaults to `false`. |
| `max_chars` | Optionally truncate returned Markdown. Omit for the complete converted document. |

## Install and register

Install this resolver together with Europe PMC search and the BrainX Codex reviewer:

```bash
npx brainx-skill mcp install
```

The installer creates a durable runtime under `~/.brainx/mcp`, registers the resolver as `fulltext_resolver`, and refuses to overwrite an existing user-owned registration. Restart Codex or reload MCP configuration after installation. Use `npx brainx-skill mcp remove` to remove the managed registrations and runtime.

For repository development, install dependencies and start the stdio server directly:

```bash
npm install
node mcp-servers/fulltext-resolver/server.mjs
```

The second command waits for an MCP client and does not print protocol data to stderr during normal startup.

For manual Codex registration, add the server to `~/.codex/config.toml`, or to `.codex/config.toml` in a trusted project:

```toml
[mcp_servers.fulltext_resolver]
command = "node"
args = ["/absolute/path/to/brainx-skill-bundle/mcp-servers/fulltext-resolver/server.mjs"]
startup_timeout_sec = 20
tool_timeout_sec = 120
```

Optional environment configuration:

```toml
[mcp_servers.fulltext_resolver.env]
OPENALEX_MAILTO = "researcher@example.edu"
OPENALEX_API_KEY = "optional-key"
FULLTEXT_RESOLVER_USER_AGENT = "lab-fulltext-resolver/1.0 (mailto:researcher@example.edu)"
FULLTEXT_HTTP_TIMEOUT_MS = "30000"
FULLTEXT_MAX_BYTES = "8000000"
```

OpenAlex works anonymously by default. Europe PMC and the bioRxiv/medRxiv metadata and publication-crosswalk APIs require no key. No paid publisher API or AWS credential is required.

## Example calls

Resolve by DOI:

```json
{
  "doi": "10.1038/s41586-020-2649-2"
}
```

Retrieve a known preprint version of a journal paper:

```json
{
  "doi": "10.1371/journal.pone.0256482",
  "source": "medRxiv"
}
```

A successful result identifies at least:

```text
requested DOI
retrieved DOI or identifier
source
version
relationship
license when known
source location
original format
normalized format
```

## Retrieval behavior

- PMC: fetch Europe PMC JATS and convert the article body, abstract, figures, tables, and references to Markdown.
- OpenAlex: inspect `best_oa_location`, `primary_location`, and `locations[]`; preserve each location's version and license.
- bioRxiv/medRxiv: query the official published-DOI crosswalk, fetch the latest preprint revision, and prefer its JATS URL.
- HTML repositories: extract the main article container and convert it to Markdown.
- PDFs: return the accessible PDF URL and provenance. PDF parsing is an explicit v1 non-goal.

Source outages are isolated. A Europe PMC, OpenAlex, or bioRxiv failure is recorded in `source_errors` and does not prevent another source from resolving the work.

## Security and correctness boundaries

- The resolver follows only public HTTP(S) URLs and rejects localhost and private literal IP addresses.
- It does not authenticate to institutions, bypass paywalls, automate browser logins, or call publisher-specific APIs.
- It does not fabricate content when retrieval fails.
- OpenAlex repository locations can point to landing pages rather than direct manuscripts. `get_fulltext` reports a fetch failure and tries the next ranked version when the page cannot yield article text.
- Raw JATS/HTML is returned only when `include_raw=true`.

## Test

```bash
node --test mcp-servers/fulltext-resolver/test/*.test.mjs
```

The default tests use fixtures and mocked source clients. Live smoke checks are intentionally separate from the deterministic suite.

See `REUSE.md` for the reuse audit and `THIRD_PARTY_NOTICES.md` for attribution.
