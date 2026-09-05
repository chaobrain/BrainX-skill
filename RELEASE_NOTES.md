BrainX Skill 1.0.14 records 9 commits merged into `main`.

## Release record

- **Version:** `1.0.14`
- **Previous release:** `v1.0.13`
- **Commits included:** 9

## New features

### feat: persist MCP reviews as Markdown ([`6420427`](https://github.com/chaobrain/BrainX-skill/commit/6420427c126a8e5e0794a85a2abd131370353bfa))

Require the BrainX reviewer to return a standalone Markdown report with stable verdict fields.

Have the modeling-loop caller save the response verbatim per iteration while keeping the reviewer read-only and preserving its thread for follow-up.

**Changed files:** `mcp-servers/codex/README.md`, `mcp-servers/codex/system-prompt.md`, `mcp-servers/codex/test/server.test.mjs`, `plan.md`, `skills/brainx-modeling-loop/SKILL.md`

### feat: add BrainX visualization skill ([`889cc0a`](https://github.com/chaobrain/BrainX-skill/commit/889cc0a71d110c9fe0094f0d946ee899f8789354))

No additional commit description was provided.

**Changed files:** `.github/scripts/bundle-layout.test.mjs`, `brainx-closed-loop-plan.md`, `manifest.json`, `package.json`, `plan.md`, `skills/brainx-visualization/SKILL.md`, `skills/brainx-visualization/agents/openai.yaml`, `skills/brainx-visualization/references/interactive-and-visualization-styling.md`, `skills/brainx-visualization/references/neural-data-visualization.md`, `skills/brainx-visualization/references/statistical-and-model-visualization.md`

### feat: add literature research and bundled MCP setup ([`cc89053`](https://github.com/chaobrain/BrainX-skill/commit/cc89053860a6e5a2dad3df951ee3c6ec2bfcf305))

Add the bio-neuro-lit workflow with Europe PMC discovery, provenance-aware full-text retrieval, optional Exa and DeepXiv helpers, and an append-only handoff of essential modeling evidence into brainmodeling-memory.md.

Bundle the Full-Text Resolver and BrainX Codex reviewer, install the pinned Europe PMC server, and register all three Codex MCP servers through one idempotent command with ownership, conflict, rollback, and removal coverage.

**Changed files:** `.github/scripts/bundle-layout.test.mjs`, `.github/scripts/literature-mcp.test.mjs`, `README.md`, `installation/lib/cli.js`, `installation/lib/literature-mcp.js`, `manifest.json`, `mcp-servers/codex/README.md`, `mcp-servers/fulltext-resolver/README.md`, `mcp-servers/fulltext-resolver/REUSE.md`, `mcp-servers/fulltext-resolver/THIRD_PARTY_NOTICES.md`, `mcp-servers/fulltext-resolver/server.mjs`, `mcp-servers/fulltext-resolver/src/document.mjs`, and 20 more

## Fixes

### fix: strengthen BrainX fitting workflow ([`7e67cdd`](https://github.com/chaobrain/BrainX-skill/commit/7e67cdd6838cb756c314a9a720f082ea8152c93d))

No additional commit description was provided.

**Changed files:** `brainx-fitting-skill-diagnosis.md`, `how-to-refine-skill.md`, `mcp-servers/codex/README.md`, `mcp-servers/codex/UPSTREAM.md`, `mcp-servers/codex/server.mjs`, `mcp-servers/codex/system-prompt.md`, `mcp-servers/codex/test/server.test.mjs`, `plan.md`, `skills/brainx-general-guard/SKILL.md`, `skills/brainx-modeling-loop/SKILL.md`, `skills/brainx-modeling-loop/references/parameter-fitting-workflow.md`, `skills/package-skills/braincell/SKILL.md`, and 3 more

### fix(ci): install dependencies before release tests ([`5f3ce65`](https://github.com/chaobrain/BrainX-skill/commit/5f3ce65fd2060533ab11c2f1a05cdedce528994e))

Install package dependencies on the clean release runner before executing the full MCP and skill test suite, and cover the required workflow ordering.

**Changed files:** `.github/scripts/release-workflow.test.mjs`, `.github/workflows/release-notes.yml`

## Other changes

### test: add C. elegans fitting comparison ([`66a0a74`](https://github.com/chaobrain/BrainX-skill/commit/66a0a74a56e7d7dd1b09299da1bb7dc399057d06))

No additional commit description was provided.

**Changed files:** `brainx-display-cases/16-C.elegans-muscle-cell/Fig4A-D.txt`, `brainx-display-cases/16-C.elegans-muscle-cell/README.md`, `brainx-display-cases/16-C.elegans-muscle-cell/celegans_muscle_inference.py`, `brainx-display-cases/16-C.elegans-muscle-cell/results/Fig4A-D.txt`, `brainx-display-cases/16-C.elegans-muscle-cell/results/README.md`, `brainx-display-cases/16-C.elegans-muscle-cell/results/celegans_muscle_inference.py`, `brainx-display-cases/16-C.elegans-muscle-cell/results/test_celegans_muscle_inference.py`, `brainx-display-cases/16-C.elegans-muscle-cell/run0/BrainXStudyRecord.md`, `brainx-display-cases/16-C.elegans-muscle-cell/run0/Fig4A-D.txt`, `brainx-display-cases/16-C.elegans-muscle-cell/run0/NeuroSpecification.md`, `brainx-display-cases/16-C.elegans-muscle-cell/run0/agent-final.md`, `brainx-display-cases/16-C.elegans-muscle-cell/run0/artifacts/acceleration_iteration1.md`, and 133 more

### test: add display cases 17 and 18 ([`8ad4c7e`](https://github.com/chaobrain/BrainX-skill/commit/8ad4c7e3b6e003a3e9ed162928da805267f27bc3))

Import the sparse E/I network-state reproduction and the C. elegans SHK-1/EGL-19 channel-fitting reproduction from yixinliu commit bf8518895281b1f282a0a56555ec26bd8d36a0e7.

**Changed files:** `brainx-display-cases/17-sparse-EI-network-states/NeuroSpecification.md`, `brainx-display-cases/17-sparse-EI-network-states/acceleration-and-parity.md`, `brainx-display-cases/17-sparse-EI-network-states/brainmodeling-memory.md`, `brainx-display-cases/17-sparse-EI-network-states/brainx-study-record.md`, `brainx-display-cases/17-sparse-EI-network-states/prompt.md`, `brainx-display-cases/17-sparse-EI-network-states/runs/20260824T151959+0800-production-brunel-seeds5/RUN_SPEC.md`, `brainx-display-cases/17-sparse-EI-network-states/runs/20260824T151959+0800-production-brunel-seeds5/code.diff`, `brainx-display-cases/17-sparse-EI-network-states/runs/20260824T151959+0800-production-brunel-seeds5/command.txt`, `brainx-display-cases/17-sparse-EI-network-states/runs/20260824T151959+0800-production-brunel-seeds5/config.json`, `brainx-display-cases/17-sparse-EI-network-states/runs/20260824T151959+0800-production-brunel-seeds5/environment.json`, `brainx-display-cases/17-sparse-EI-network-states/runs/20260824T151959+0800-production-brunel-seeds5/launch.sh`, `brainx-display-cases/17-sparse-EI-network-states/runs/20260824T151959+0800-production-brunel-seeds5/results/raw/repeat-00_asynchronous_irregular.npz`, and 56 more

### test: sync C. elegans fitting case prompt ([`1767125`](https://github.com/chaobrain/BrainX-skill/commit/17671255728b0e0aaee8ab4c18d8621552fb9e73))

No additional commit description was provided.

**Changed files:** `brainx-display-cases/18-C.elegans-fit channels/prompt.md`

### test: import new Brunel run cases ([`ced8bb5`](https://github.com/chaobrain/BrainX-skill/commit/ced8bb5c64bb377d59e1db9804945cf65bb52b90))

No additional commit description was provided.

**Changed files:** `brainx-display-cases/17-sparse-EI-network-states/NeuroSpecification.md`, `brainx-display-cases/17-sparse-EI-network-states/acceleration-and-parity.md`, `brainx-display-cases/17-sparse-EI-network-states/benchmark_parity.py`, `brainx-display-cases/17-sparse-EI-network-states/brainmodeling-memory.md`, `brainx-display-cases/17-sparse-EI-network-states/brainx-study-record.md`, `brainx-display-cases/17-sparse-EI-network-states/fresh-brunel-fig8/NeuroSpecification.md`, `brainx-display-cases/17-sparse-EI-network-states/fresh-brunel-fig8/acceleration-and-parity.md`, `brainx-display-cases/17-sparse-EI-network-states/fresh-brunel-fig8/benchmark_parity.py`, `brainx-display-cases/17-sparse-EI-network-states/fresh-brunel-fig8/brainmodeling-memory.md`, `brainx-display-cases/17-sparse-EI-network-states/fresh-brunel-fig8/brainx-study-record.md`, `brainx-display-cases/17-sparse-EI-network-states/fresh-brunel-fig8/brunel_fig8.py`, `brainx-display-cases/17-sparse-EI-network-states/fresh-brunel-fig8/figures/brunel_fig8_reproduction.png`, and 235 more

**Full Changelog:** https://github.com/chaobrain/BrainX-skill/compare/v1.0.13...v1.0.14
