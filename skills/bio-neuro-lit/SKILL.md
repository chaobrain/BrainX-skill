---
name: bio-neuro-lit
description: Review biology, neuroscience, and computational-neuroscience literature with Europe PMC as the primary discovery source, the BrainX Full-Text Resolver as the default reader, Exa for optional recall expansion, and DeepXiv for optional progressive reading of arXiv-compatible computational papers. Use for finding papers, related-work reviews, evidence synthesis, or explaining a bio/neuro research area; do not use as a generic clinical-medicine or computer-science search workflow.
---

# Bio-neuro literature review

## Purpose and boundary

Use this skill to answer a biology, neuroscience, or computational-neuroscience question from literature evidence. Keep the canonical path:

`interpret question -> decompose terminology -> search Europe PMC -> optionally expand with Exa -> normalize, deduplicate, and rank -> deep-read selected papers -> compare evidence -> persist modeling-relevant evidence when applicable -> answer the question`

Use Europe PMC as the primary academic discovery database and the existing Full-Text Resolver as the default reader. Use Exa only to expand recall and DeepXiv only when progressive reading of a known arXiv-compatible paper is useful. Do not add generic arXiv, Semantic Scholar, Gemini, standalone OpenAlex, Zotero, Obsidian, or local-PDF workflows.

## Underlying mental model

Discovery and reading are separate stages. Search titles, abstracts, and metadata broadly enough to find the field, then retrieve full text only for the papers that can materially change the answer.

| Role | Interface | Decision boundary |
|---|---|---|
| Primary discovery | Europe PMC MCP | Start here for normal biology, neuroscience, and computational-neuroscience questions. Treat its structured identifiers and metadata as canonical when sources conflict. |
| Recall expansion | `scripts/exa_search.py` | Use when recent, preprint, project, technical-report, or cross-disciplinary work may use terminology that Europe PMC misses. Do not treat Exa as authoritative bibliographic metadata. |
| Default reader | Full-Text Resolver MCP | Use for selected papers by PMCID, DOI, PMID, or conservative metadata fallback. Preserve the retrieved version and relationship to the requested work. |
| Progressive reader | `scripts/deepxiv_fetch.py` | Use for a selected paper with an arXiv ID when `brief -> head -> section` is more efficient than loading the complete document. |

Open `references/tool-contracts.md` before the first tool call, when a tool is unavailable, or when exact arguments, namespaces, setup, or fallback behavior matter.

## Interpret and decompose the question

Translate the user's question into a small set of purposeful searches; query variation should recover domain synonyms without broadening into an unrelated medical field.

1. State the biological or computational question actually being answered.
2. Identify only the dimensions that change retrieval: biological entity or cell type, brain region or circuit, channel or receptor, physiological phenomenon, model, computational method, recording modality, experimental paradigm, and important abbreviation or synonym.
3. Generate a few complementary formulations. Keep each formulation close enough to the question that a matching paper could contribute evidence.
4. For mechanistic or computational questions, favor terms such as neuron, circuit, electrophysiology, neural dynamics, neural coding, biophysical model, or experimental neuroscience. Do not let diagnosis, treatment, epidemiology, patient outcome, or hospital studies dominate unless the user asks for them.

Example decomposition:

```text
Question: parameter fitting of conductance-based models from electrophysiology recordings

- conductance-based neuron model parameter estimation
- Hodgkin-Huxley parameter fitting electrophysiology
- ion channel conductance optimization patch clamp
- biophysical single-neuron model parameter inference
```

## Discover papers

Europe PMC supplies the initial candidate set; Exa may add recall but never replaces a failed primary academic search.

| API | Description |
|---|---|
| Europe PMC `search_articles` | Use for each targeted formulation. Request `result_type="core"` when identifiers, MeSH terms, and full-text metadata affect selection; keep relevance ordering unless recency is part of the question. Use `TITLE_ABS` on the decisive mechanistic or modeling terms when an unqualified query drifts into broad clinical results. |
| Europe PMC `get_article` | Use after discovery when one candidate needs complete structured metadata before normalization or identifier resolution. |
| `scripts/exa_search.py search` | Use optionally with `--category "research paper"` for recall expansion. Continue without Exa when its SDK or `EXA_API_KEY` is unavailable. |

```json
{
  "query": "TITLE_ABS:(Hodgkin-Huxley OR conductance-based) AND TITLE_ABS:(parameter estimation OR parameter fitting) AND TITLE_ABS:(electrophysiology OR patch clamp)",
  "result_type": "core",
  "page_size": 25
}
```

If a tightly qualified query becomes too narrow, loosen one dimension at a time and compare the added candidates; do not discard the neuro-specific relevance judgment.

Use `--sources europepmc`, `--sources europepmc,exa`, or `--sources europepmc,exa,deepxiv` only when the user explicitly overrides sources. The resolver remains the reading stage and is not a discovery-source switch. Enabling `deepxiv` permits progressive reading of known arXiv IDs; it does not make DeepXiv the primary search database.

If Europe PMC is unavailable, report that primary academic discovery failed. Exa-only material may be shown as supplemental leads, but do not silently present it as an equivalent literature review.

## Normalize, deduplicate, and select

Keep one lightweight record per work and preserve provenance from every contributing source.

```text
title
authors
year
journal_or_venue
abstract
doi
pmid
pmcid
arxiv_id
source
url
full_text_status
```

Merge duplicates in this order:

`PMCID -> PMID -> DOI -> arXiv ID -> normalized title`

Normalize identifier prefixes, DOI URLs, whitespace, punctuation, and title case before comparison. A title match must be conservative enough to avoid merging different works. When Europe PMC and Exa overlap, keep Europe PMC metadata and add Exa to source provenance; retain a useful Exa URL or snippet only when it adds information.

Rank by ability to answer the question, not keyword count alone. Consider biological match, represented scale, method or experiment match, data modality, directness of evidence, publication version, and recency when relevant. Distinguish foundational work from newer evidence. Search broadly, but choose only the strongest papers for deep reading.

## Read selected papers

Full text is evidence depth, not a requirement for retaining a relevant paper.

| API | Description |
|---|---|
| Full-Text Resolver `get_fulltext` | Use first for selected papers. Supply the strongest available identifier in the order `PMCID -> DOI -> PMID -> exact metadata`; it returns normalized Markdown plus source, version, relationship, and location provenance. |
| Full-Text Resolver `list_versions` | Use when the requested publication may have published, accepted, and submitted versions whose distinction affects interpretation. |
| `scripts/deepxiv_fetch.py paper-brief` | Use for a known arXiv ID to establish relevance before section reading. |
| `scripts/deepxiv_fetch.py paper-head` | Use after the brief to inspect the section map. |
| `scripts/deepxiv_fetch.py paper-section` | Use only for the section needed to answer the question, such as Methods, Results, Experiments, or Limitations. |

```json
{
  "pmcid": "PMC1234567",
  "include_raw": false
}
```

Prefer the Full-Text Resolver for ordinary PMC and journal biology papers. Use DeepXiv mainly for computational neuroscience or ML-neuroscience papers with arXiv IDs. If the resolver returns no usable Markdown, including a PDF-only location, retain the paper's abstract and metadata, set `full_text_status` to `abstract-only`, preserve the retrieval note or source URL, and limit claims to what the abstract supports.

## Extract and compare evidence

Extract only fields that help answer the question. For experimental work, these may include biological system, species, brain region, cell type, recording modality, protocol, dataset or sample, and quantitative result. For computational work, also inspect model type or equations, represented scale, state variables, fitted parameters, objective or loss, optimizer or inference method, experimental data, validation, and modeling assumptions.

Do not force molecular, cellular, systems, and computational papers into one fixed schema. Record whether each claim comes from full text or abstract only, and distinguish a final article from an accepted manuscript or preprint.

Synthesize across papers:

1. Answer the user's research question directly.
2. Group evidence by mechanism, method, result, or explanatory theme.
3. Identify consensus, disagreement, methodological differences, important experiments or datasets, modeling assumptions, limitations, and unresolved gaps.
4. Compare papers rather than writing one disconnected summary per paper.
5. Do not infer a stronger conclusion than the retrieved evidence supports, and do not fabricate missing identifiers, metadata, results, or full text.

## Persist modeling-relevant literature

A literature review informs the modeling loop only when selected papers change a concrete modeling or validation decision.

| Artifact | Description |
|---|---|
| `brainmodeling-memory.md` | When an active BrainX project already has this file, append one `## Literature evidence` block containing only papers that materially affect the specification, mechanism, equations, parameters, inputs, controls, observables, validation, or allowed claims. Preserve identifiers, evidence depth, retrieved version, implications, and limitations. |

```markdown
## Literature evidence: <topic> - <YYYY-MM-DD>
- Research question: <question this search addressed>
- Review artifact: <path or stable identifier, when one exists>

### Essential papers
- <citation plus DOI, PMID, PMCID, or arXiv ID>
  - Evidence: full text | abstract only; <published, accepted, or submitted version>
  - Essential finding: <question-relevant result>
  - Modeling implication: <specific decision this supports, constrains, or challenges>
  - Limitation or non-claim: <what the paper does not establish>

### Cross-paper synthesis
- Supported decisions: <decisions supported across the selected evidence>
- Disagreement or uncertainty: <conflicting evidence and unresolved choices>
- Required tests or controls: <model checks implied by the literature>
```

Append this block without editing earlier memory and without adding or changing a `## Checkpoint`, iteration, or step. Deduplicate against earlier literature blocks by stable identifier and append only new evidence or a correction that names the earlier record it supersedes. Keep complete article summaries in the literature-review artifact; memory receives only the essential modeling consequences and evidence pointers.

Do not create `brainmodeling-memory.md` for a standalone literature request. When no active modeling memory exists, keep the synthesis in the review output and state no modeling-memory handoff was required.

## Report the review

Use this structure unless the user requests another format:

```markdown
# Literature review: <topic>

## Search interpretation
<What question and terminology were searched.>

## Main findings
<Evidence synthesis that answers the question.>

## Key papers
| Paper | Year | Why it matters | Evidence used |
|---|---:|---|---|
| ... | ... | ... | full text / abstract only |

## Comparison
<Methods, findings, assumptions, and disagreements.>

## Gaps and unresolved questions
<What remains uncertain or poorly tested.>

## Retrieval notes
<Only material limitations: Exa-only lead, unavailable full text, version substitution, or DeepXiv section reading.>
```

Include enough bibliographic information to identify every key paper, preferably DOI, PMID, PMCID, or canonical URL when available. Keep retrieval logs and raw API responses out of the normal answer.

## Boundaries and common failures

- Do not use broad clinical volume as a proxy for relevance to a mechanistic or computational neuroscience question.
- Do not retrieve full text for every candidate.
- Do not substitute Exa metadata for conflicting Europe PMC metadata.
- Do not call generic arXiv as an independent discovery source.
- Do not rebuild PMC, OpenAlex, bioRxiv, repository, or publisher resolution already owned by the Full-Text Resolver.
- Do not label a preprint or accepted manuscript as the requested final publication.
- Do not drop a relevant paper solely because only its abstract is accessible.
- Do not turn the final answer into a bibliography dump or implementation log.
- Do not write every relevant paper into modeling memory; persist only evidence that changes a modeling or validation decision.
- Do not create or advance a modeling checkpoint while appending literature evidence.
