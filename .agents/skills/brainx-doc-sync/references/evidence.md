# Evidence for documentation changes

## Read the inventory

`scripts/audit_links.py` scans Markdown under `skills/`,
`source_html_references/`, `braintools-references/`, this development skill,
and the maintained root guides. It groups public HTTP(S) candidates by URL without fragments and
preserves each source file, line, and fragment. It writes JSON to stdout and
does not edit the checkout. Redirects and status describe retrieval only.

| Result | Next action |
|---|---|
| `reachable` | Open the content; verify the symbol, relevant section, and release. Check redirects for lost anchors or unrelated landing pages. |
| `missing_candidate` | Confirm the 404/410 through the official index or versioned source; replace only with a matching source. |
| `access_limited`, `rate_limited`, `server_error`, `transport_error` | Preserve the distinction from missing pages. Retry a small, bounded sample or consult a versioned official source. |
| `redirect_blocked`, `http_error` | Inspect the destination or HTTP response before changing the link. |
| `not_checked` | Keep it unresolved. The reason distinguishes inventory-only, host scope, and probe budget. |

Only allowlisted hosts are probed. Use repeated `--host` flags to select public
official hosts. Review `skipped_locations` for URLs with credentials, queries,
unsupported ports, or invalid syntax. The report keeps only their source
locations. `--limit` caps initial URL probes, `--jobs` caps concurrency, and
`--timeout` sets the socket timeout in seconds.

Review `missing_local_targets` for relative inline Markdown links whose
in-checkout file or directory is absent. Confirm each candidate in context.
Reference-style links, local heading fragments, and embedded assets are
outside this check. Add `--check --anchors` to test percent-decoded fragments
against static HTML IDs and named anchors in the first 2 MiB of each selected
page. A missing candidate can still require manual verification for a
client-rendered page or a redirect that changes the fragment. Exit code zero
means the scan completed.

Review `missing_skill_routes` for backtick paths in published skill Markdown.
Use skill-root-relative `references/...` and `scripts/...` within a skill.
Prefix a cross-skill path with the owning skill name and an arrow. Keep wrapped
routes in one paragraph. The check uses manifest skill roots, skips fenced code,
flags checkout-only paths, and reports the target line.

## Record one claim with its evidence

For each substantive edit, record the file and claim, target release, official
source and relevant section, and execution evidence when needed. For example,
use the projection's `delay` parameter reference to support accepted delay
shapes. Use a pulse probe through that same projection to support measured
arrival time. A manual buffer probe is evidence for a different path.

Prefer a release-tagged official API or published documentation for contracts.
Use an official example for composition; reconcile it with the target API
before copying. Use release metadata and migration notes to establish the
version boundary. If an unversioned page is all that is available, label it
as such and retain only the claims verified for the target. Do not infer API
removal from a URL failure or numerical equivalence from a renamed function.
