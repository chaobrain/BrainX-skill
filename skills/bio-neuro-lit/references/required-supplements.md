# Recover a required original supplement

Use this route when reproduction needs a scientific definition in an original supplement or linked source artifact that the default reader did not expose. Keep the normal Full-Text Resolver route for article reading and the ordinary abstract-only fallback for literature reviews; a failed conversion or legacy URL does not establish that a required original artifact is unavailable.

## Locate the artifact through its publication

Inspect the article's supplementary links and official publisher or repository metadata. Record the DOI/PMCID, requested publication version and exact artifact named by the source. Check whether an available reader already supports that artifact before adding a direct retrieval boundary. Do not substitute a preprint supplement for the published one without verifying and reporting the relationship.

If a linked endpoint fails, consult the provider's current distribution documentation rather than repeatedly guessing old URLs. For PMC, use the [official cloud-service documentation](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/) and its linked [dataset README](https://pmc-oa-opendata.s3.amazonaws.com/README.txt). The public dataset provides article-version metadata and supplementary objects:

| Interface | Use |
|---|---|
| Official article inventory or PMCID prefix listing | Locate available version prefixes; do not assume version 1 exists or that the largest version is the requested publication. |
| Article JSON metadata | Verify identifiers, publication/manuscript status and the linked `media_urls`. |
| Referenced supplementary media object | Retrieve the specifically required file from its advertised location; preserve its bytes and verify the metadata digest when supplied. |

Treat this as targeted access to a named primary artifact, not a second discovery engine or an assumption that every article is present in a public dataset. Preserve the failure and bound claims when no authorized official copy can be obtained.

## Verify and read the original

Save the URL, retrieval time, response status/type, file size and SHA-256 alongside the original bytes. Inspect the actual format: an HTTP success or a `.pdf` name can contain challenge HTML or an XML error. Match identifiers and title/version against official metadata; do not accept filename similarity alone as provenance.

Inspect complete relevant pages and captions. Render a PDF when extraction loses equation grouping, signs, subscripts or figure-specific settings. Record the exact page, equation, section or code location supporting each consequential definition. A main-text summary does not establish a missing supplement's formula, analysis windows, normalization set or control distribution.

## Hand off the evidence

Distinguish verified source values from unstated implementation choices and conflicts between the article, supplement and executable author code. For an active reproduction, hand these records to `brainx-modeling-loop` → `references/paper-reproduction.md` before implementation or revised analysis. Keep case-specific parameters and findings in the project evidence, not in this skill. If a decisive definition remains missing, state the unsupported claim and the evidence needed; do not fill the gap and call it verified reproduction.
