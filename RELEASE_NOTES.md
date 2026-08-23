BrainX Skill 1.0.13 records 25 commits merged into `main`.

## Release record

- **Version:** `1.0.13`
- **Previous release:** `v1.0.12`
- **Commits included:** 25

## New features

### feat(docs): deploy the documentation site to brainx.chaobrain.com/skills ([`7cc4561`](https://github.com/chaobrain/BrainX-skill/commit/7cc45617dbfb7adc5a38ba5813fe71233b48cb97))

Add the zero-downtime Deploy Docs workflow used by every other BrainX
package site: build with Sphinx, upload an artifact, ship it to a
timestamped release directory, then swap the current symlink atomically
and reload nginx.

Document the two areas the site did not cover: the per-agent installation
path matrix with the receipt-driven update semantics, and the release and
deployment process including the required secrets and nginx blocks.

Point html_baseurl at /skills/ and use the published BrainX-skill logo,
which drops the 1.8 MB unreferenced docs/images/image.png.

**Changed files:** `.github/workflows/deploy-docs.yml`, `docs/conf.py`, `docs/images/image.png`, `docs/index.rst`, `docs/releasing.rst`, `docs/supported-agents.rst`

### feat: add BrainX closed-loop modeling workflows ([`5790da4`](https://github.com/chaobrain/BrainX-skill/commit/5790da429c0b81b27b6f065308c8e65fd97444d7))

No additional commit description was provided.

**Changed files:** `.github/scripts/agent-skills-workflow.test.mjs`, `.github/scripts/bundle-layout.test.mjs`, `.github/scripts/prepare-release.test.mjs`, `.github/workflows/agent-skills-validation.yml`, `AGENTS.md`, `braintools-references/braintools-input-current.md`, `brainx-closed-loop-plan.md`, `installation/lib/bundle.js`, `manifest.json`, `mcp-servers/codex/README.md`, `mcp-servers/codex/UPSTREAM.md`, `mcp-servers/codex/server.mjs`, and 350 more

### feat: strengthen BrainX MCP review ([`046c89a`](https://github.com/chaobrain/BrainX-skill/commit/046c89aee4ab03491c7457f06a50aef9a2163d68))

Condense the reviewer contract around scientific support, minimal BrainX-native code, and optimization adequacy.

Expose the training and parameter-fitting references directly to fresh MCP review sessions, preserve responses in the caller context, and document a longer tool timeout for complete reviews.

Include regression coverage for the reviewer criteria and reference injection.

**Changed files:** `how-to-refine-skill.md`, `mcp-servers/codex/README.md`, `mcp-servers/codex/server.mjs`, `mcp-servers/codex/system-prompt.md`, `mcp-servers/codex/test/server.test.mjs`, `plan.md`, `skills/brainx-modeling-loop/SKILL.md`

## Documentation

### docs(readme): show the BrainX Skill logo at the top ([`3773de8`](https://github.com/chaobrain/BrainX-skill/commit/3773de8c204abf5ea193a875c79259a9c89af3d3))

No additional commit description was provided.

**Changed files:** `README.md`

### docs: add BrainX closed-loop modeling plan ([`244ebc1`](https://github.com/chaobrain/BrainX-skill/commit/244ebc128da0551e05a2c3932653e9767d1dc4f7))

No additional commit description was provided.

**Changed files:** `brainx-closed-loop-plan.md`

## Other changes

### Refine alpha rhythm BrainX guidance ([`e865a07`](https://github.com/chaobrain/BrainX-skill/commit/e865a0732a3f551197684acc64cbb3e72f9464ee))

No additional commit description was provided.

**Changed files:** `brainx-display-cases/05-alpha-rhythm/control-unisolated/README.md`, `brainx-display-cases/05-alpha-rhythm/control-unisolated/agent-final.md`, `brainx-display-cases/05-alpha-rhythm/control-unisolated/alpha_rhythm.png`, `brainx-display-cases/05-alpha-rhythm/control-unisolated/alpha_rhythm.py`, `brainx-display-cases/05-alpha-rhythm/control-unisolated/brainmass-api-coverage-diagnosis.md`, `brainx-display-cases/05-alpha-rhythm/control-unisolated/codex-events.jsonl`, `brainx-display-cases/05-alpha-rhythm/control-unisolated/harness-metadata.txt`, `brainx-display-cases/05-alpha-rhythm/run0/.agents/skills/braincell/SKILL.md`, `brainx-display-cases/05-alpha-rhythm/run0/.agents/skills/braincell/references/area-scaled-hh-pattern.md`, `brainx-display-cases/05-alpha-rhythm/run0/.agents/skills/braincell/references/braincell-custom-ion-channel-authoring.md`, `brainx-display-cases/05-alpha-rhythm/run0/.agents/skills/braincell/references/channel-library.md`, `brainx-display-cases/05-alpha-rhythm/run0/.agents/skills/braincell/references/ion-library.md`, and 357 more

### Refine seizure recruitment BrainX guidance ([`54310e0`](https://github.com/chaobrain/BrainX-skill/commit/54310e01a3f92b518bc2415e09bc19885a412b07))

No additional commit description was provided.

**Changed files:** `brainx-display-cases/06-seizure-recruitment/failed-launch-run7-config-parse-path/codex-events.jsonl`, `brainx-display-cases/06-seizure-recruitment/failed-launch-run7-config-parse-path/harness-metadata.txt`, `brainx-display-cases/06-seizure-recruitment/failed-launch-run7-env-chdir-option/codex-events.jsonl`, `brainx-display-cases/06-seizure-recruitment/failed-launch-run7-env-chdir-option/harness-metadata.txt`, `brainx-display-cases/06-seizure-recruitment/failed-launch-run7-repo-cwd/codex-events.jsonl`, `brainx-display-cases/06-seizure-recruitment/failed-launch-run7-repo-cwd/harness-metadata.txt`, `brainx-display-cases/06-seizure-recruitment/failed-launch-run7-sandbox-syntax/codex-events.jsonl`, `brainx-display-cases/06-seizure-recruitment/failed-launch-run7-sandbox-syntax/harness-metadata.txt`, `brainx-display-cases/06-seizure-recruitment/failed-launch-run7-seatbelt-trust-path/codex-events.jsonl`, `brainx-display-cases/06-seizure-recruitment/failed-launch-run7-seatbelt-trust-path/harness-metadata.txt`, `brainx-display-cases/06-seizure-recruitment/invalid-run7-nested-seatbelt/agent-final.md`, `brainx-display-cases/06-seizure-recruitment/invalid-run7-nested-seatbelt/codex-events.jsonl`, and 1255 more

### Refine binocular rivalry BrainX guidance ([`bd26968`](https://github.com/chaobrain/BrainX-skill/commit/bd269684c16465384ae51b8e9960237a367dbbe1))

No additional commit description was provided.

**Changed files:** `brainx-display-cases/08-binocular-rivalry/run0/README.md`, `brainx-display-cases/08-binocular-rivalry/run0/agent-final.md`, `brainx-display-cases/08-binocular-rivalry/run0/binocular_rivalry.py`, `brainx-display-cases/08-binocular-rivalry/run0/brainmass-api-coverage-diagnosis.md`, `brainx-display-cases/08-binocular-rivalry/run0/codex-events.jsonl`, `brainx-display-cases/08-binocular-rivalry/run0/codex-stderr.log`, `brainx-display-cases/08-binocular-rivalry/run0/harness-metadata.txt`, `brainx-display-cases/08-binocular-rivalry/run0/results/binocular_rivalry.png`, `brainx-display-cases/08-binocular-rivalry/run0/results/binocular_rivalry_results.npz`, `brainx-display-cases/08-binocular-rivalry/run1/README.md`, `brainx-display-cases/08-binocular-rivalry/run1/__pycache__/binocular_rivalry.cpython-312.pyc`, `brainx-display-cases/08-binocular-rivalry/run1/agent-final.md`, and 8 more

### Refine online working memory BrainTrace guidance ([`5882ec2`](https://github.com/chaobrain/BrainX-skill/commit/5882ec23c384ad40a2ebb8f89399a9e8a91e418a))

No additional commit description was provided.

**Changed files:** `brainx-display-cases/04-online-working-memory/run0/.agents/skills/braincell/SKILL.md`, `brainx-display-cases/04-online-working-memory/run0/.agents/skills/braincell/references/area-scaled-hh-pattern.md`, `brainx-display-cases/04-online-working-memory/run0/.agents/skills/braincell/references/braincell-custom-ion-channel-authoring.md`, `brainx-display-cases/04-online-working-memory/run0/.agents/skills/braincell/references/channel-library.md`, `brainx-display-cases/04-online-working-memory/run0/.agents/skills/braincell/references/ion-library.md`, `brainx-display-cases/04-online-working-memory/run0/.agents/skills/braincell/references/mixions-for-adaptation.md`, `brainx-display-cases/04-online-working-memory/run0/.agents/skills/braincell/references/multicompartment/braincell-manual-morphology-construction.md`, `brainx-display-cases/04-online-working-memory/run0/.agents/skills/braincell/references/multicompartment/cv-policy-reference.md`, `brainx-display-cases/04-online-working-memory/run0/.agents/skills/braincell/references/multicompartment/filter-function-library.md`, `brainx-display-cases/04-online-working-memory/run0/.agents/skills/braincell/references/multicompartment/morphology-io-loading-validation.md`, `brainx-display-cases/04-online-working-memory/run0/.agents/skills/braincell/references/multicompartment/multicompartment-cell-workflow.md`, `brainx-display-cases/04-online-working-memory/run0/.agents/skills/braincell/references/multicompartment/probe-reference.md`, and 348 more

### Add BrainX skill documentation showcase ([`67dadcf`](https://github.com/chaobrain/BrainX-skill/commit/67dadcfd7e04795538580d1ed87a09aadc062092))

No additional commit description was provided.

**Changed files:** `.gitignore`, `docs/_static/brainx-skill.css`, `docs/_static/brainx-skill.js`, `docs/_static/cases/01-spike-frequency-adaptation/README.md`, `docs/_static/cases/01-spike-frequency-adaptation/agent-final.md`, `docs/_static/cases/01-spike-frequency-adaptation/agent-log.jsonl`, `docs/_static/cases/01-spike-frequency-adaptation/experiment.py`, `docs/_static/cases/01-spike-frequency-adaptation/prompt.md`, `docs/_static/cases/01-spike-frequency-adaptation/spike-frequency-adaptation.png`, `docs/_static/cases/02-learning-temporal-order/README.md`, `docs/_static/cases/02-learning-temporal-order/agent-final.md`, `docs/_static/cases/02-learning-temporal-order/agent-log.jsonl`, and 81 more

### Update BrainX docs and perturbation workflows ([`78c01b6`](https://github.com/chaobrain/BrainX-skill/commit/78c01b6820556d8c1bf01458b51bbad76ade1126))

No additional commit description was provided.

**Changed files:** `BrainX doc style.md`, `PRODUCT.md`, `brainx-display-cases/13-single-cell-perturbation-bio/prompt.md`, `brainx-display-cases/13-single-cell-perturbation-bio/run0/README.md`, `brainx-display-cases/13-single-cell-perturbation-bio/run0/results/broad_inhibition_observables.npz`, `brainx-display-cases/13-single-cell-perturbation-bio/run0/results/influence_curves.csv`, `brainx-display-cases/13-single-cell-perturbation-bio/run0/results/influence_vs_signal_correlation.png`, `brainx-display-cases/13-single-cell-perturbation-bio/run0/results/inhibition_dominant_observables.npz`, `brainx-display-cases/13-single-cell-perturbation-bio/run0/results/parameters.json`, `brainx-display-cases/13-single-cell-perturbation-bio/run0/results/specific_strong_ei_observables.npz`, `brainx-display-cases/13-single-cell-perturbation-bio/run0/results/summary.json`, `brainx-display-cases/13-single-cell-perturbation-bio/run0/v1_single_cell_perturbation.py`, and 129 more

### Restore top-level quick start navigation ([`2728a34`](https://github.com/chaobrain/BrainX-skill/commit/2728a34f68dae4efa39541e3c94ddcda4cc259b8))

No additional commit description was provided.

**Changed files:** `docs/index.rst`

### Group docs into quickstart and examples ([`72378fe`](https://github.com/chaobrain/BrainX-skill/commit/72378fe2cc8b62f1f89cafbd71482c92cf018e06))

No additional commit description was provided.

**Changed files:** `docs/index.rst`, `docs/installation.rst`

### Separate docs overview and quickstart video ([`5fabee5`](https://github.com/chaobrain/BrainX-skill/commit/5fabee5c1075140b6768ddfcccdf60a4f7ba5342))

No additional commit description was provided.

**Changed files:** `docs/_static/brainx-skill.css`, `docs/index.rst`, `docs/quickstart.rst`

### Expand the quickstart workflow ([`d61a67d`](https://github.com/chaobrain/BrainX-skill/commit/d61a67d988de3324d50c70b992503b7314459787))

No additional commit description was provided.

**Changed files:** `docs/_static/brainx-skill.css`, `docs/quickstart.rst`

### Use the quickstart video prompt ([`9b70985`](https://github.com/chaobrain/BrainX-skill/commit/9b709850984c61d3b4285022cb64dfae8efe3a17))

No additional commit description was provided.

**Changed files:** `docs/quickstart.rst`

### Add skill reference documentation ([`e3244fb`](https://github.com/chaobrain/BrainX-skill/commit/e3244fb5fa52a74a796278a129fc403643a80cea))

No additional commit description was provided.

**Changed files:** `docs/_static/brainx-skill.css`, `docs/conf.py`, `docs/creative-experiment-verification.rst`, `docs/index.rst`, `docs/paper-reproduction.rst`, `docs/skill-reference.rst`, `docs/skill-reference/braincell.rst`, `docs/skill-reference/brainevent.rst`, `docs/skill-reference/brainmass.rst`, `docs/skill-reference/brainpy-state.rst`, `docs/skill-reference/brainstate.rst`, `docs/skill-reference/braintrace.rst`, and 6 more

### add reference for case09 ([`5a0c0eb`](https://github.com/chaobrain/BrainX-skill/commit/5a0c0ebce570e39c43a779c6dce1cd968f039179))

No additional commit description was provided.

**Changed files:** `brainx-display-cases/09-neural-compass/prompt.md`, `brainx-display-cases/15-decision-making/prompt.md`, `brainx-display-cases/16 place cell navigation/prompt.md`

### add case 15 ([`bc6240e`](https://github.com/chaobrain/BrainX-skill/commit/bc6240e204a642edf8ea6e3f4a25541da5ea1e99))

No additional commit description was provided.

**Changed files:** `brainx-display-cases/09-neural-compass/prompt.md`, `brainx-display-cases/15-decision-making/prompt.md`, `brainx-display-cases/15-grid-cell-theta-sweep/prompt.md`, `brainx-display-cases/15-grid-cell-theta-sweep/run0/theta_sweeps.py`, `brainx-display-cases/15-grid-cell-theta-sweep/run0/theta_sweeps_evidence.npz`, `brainx-display-cases/15-grid-cell-theta-sweep/run0/theta_sweeps_main.png`, `brainx-display-cases/15-grid-cell-theta-sweep/run0/theta_sweeps_mechanisms.png`, `brainx-display-cases/15-grid-cell-theta-sweep/run0/theta_sweeps_metrics.json`, `brainx-display-cases/15-grid-cell-theta-sweep/run1/README.md`, `brainx-display-cases/15-grid-cell-theta-sweep/run1/results/baseline_cycle_metrics.csv`, `brainx-display-cases/15-grid-cell-theta-sweep/run1/results/population_and_alignment.png`, `brainx-display-cases/15-grid-cell-theta-sweep/run1/results/protocols_and_mechanisms.png`, and 5 more

### Reorganize display cases and add case 15 results ([`32baafe`](https://github.com/chaobrain/BrainX-skill/commit/32baafe44b320b26dad07c559d2a751bf394920a))

No additional commit description was provided.

**Changed files:** `brainx-display-cases/01-spike-frequency-adaptation/prompt.md`, `brainx-display-cases/01-spike-frequency-adaptation/run1/README.md`, `brainx-display-cases/01-spike-frequency-adaptation/run1/agent-final.md`, `brainx-display-cases/01-spike-frequency-adaptation/run1/braincell-api-coverage-diagnosis.md`, `brainx-display-cases/01-spike-frequency-adaptation/run1/codex-events.jsonl`, `brainx-display-cases/01-spike-frequency-adaptation/run1/spike_frequency_adaptation.png`, `brainx-display-cases/01-spike-frequency-adaptation/run1/spike_frequency_adaptation.py`, `brainx-display-cases/01-spike-frequency-adaptation/run2/README.md`, `brainx-display-cases/01-spike-frequency-adaptation/run2/agent-final.md`, `brainx-display-cases/01-spike-frequency-adaptation/run2/braincell-api-coverage-diagnosis.md`, `brainx-display-cases/01-spike-frequency-adaptation/run2/codex-events.jsonl`, `brainx-display-cases/01-spike-frequency-adaptation/run2/spike_frequency_adaptation.png`, and 4404 more

### chore(ci): use the organization deploy credentials ([`adaa499`](https://github.com/chaobrain/BrainX-skill/commit/adaa499b5b6e7fc1be452f7cecd318e4215a70b6))

The organization publishes DEPLOY_*_G, and BrainX-skill already has
access to them, so the workflow needs no repository-level secrets.

Drop the releasing page; the release and deployment process is internal
and does not belong on the public documentation site.

**Changed files:** `.github/workflows/deploy-docs.yml`, `docs/index.rst`, `docs/releasing.rst`

### Refine documentation experience ([`9b5e8a9`](https://github.com/chaobrain/BrainX-skill/commit/9b5e8a9c4820ff257cadf9732954c5990cdc343f))

No additional commit description was provided.

**Changed files:** `docs/_static/brainx-skill.css`, `docs/_static/brainx-skill.js`, `docs/cases/01-spike-frequency-adaptation.rst`, `docs/cases/02-learning-temporal-order.rst`, `docs/cases/03-sound-localization.rst`, `docs/cases/05-alpha-rhythm.rst`, `docs/cases/06-seizure-recruitment.rst`, `docs/cases/07-cortical-wave-obstacle.rst`, `docs/cases/08-binocular-rivalry.rst`, `docs/cases/09-neural-compass.rst`, `docs/cases/10-prior-bias.rst`, `docs/cases/11-sleep-memory-replay.rst`, and 19 more

### Remove supported agents documentation ([`74aa7ec`](https://github.com/chaobrain/BrainX-skill/commit/74aa7ec463f0a25b81e29a876b64a92ca2d48f20))

No additional commit description was provided.

**Changed files:** `docs/index.rst`, `docs/supported-agents.rst`

### Expand homepage installation section ([`da04a86`](https://github.com/chaobrain/BrainX-skill/commit/da04a86390216c513e3998bf6588353ef38ac50c))

No additional commit description was provided.

**Changed files:** `docs/index.rst`

### case 16,successfullly reproduced ([`af32161`](https://github.com/chaobrain/BrainX-skill/commit/af3216120476e7c87ea0ef3d5e275406791c840d))

No additional commit description was provided.

**Changed files:** `brainx-display-cases/16-C.elegans-muscle-cell/Fig4A-D.txt`, `brainx-display-cases/16-C.elegans-muscle-cell/README.md`, `brainx-display-cases/16-C.elegans-muscle-cell/celegans_muscle_inference.py`, `brainx-display-cases/16-C.elegans-muscle-cell/prompt.md`, `brainx-display-cases/16-C.elegans-muscle-cell/results/held_out_validation.png`, `brainx-display-cases/16-C.elegans-muscle-cell/results/posterior_samples.csv`, `brainx-display-cases/16-C.elegans-muscle-cell/results/report.json`, `brainx-display-cases/16-C.elegans-muscle-cell/results/trace_predictions.csv`, `brainx-display-cases/16-C.elegans-muscle-cell/test_celegans_muscle_inference.py`

**Full Changelog:** https://github.com/chaobrain/BrainX-skill/compare/v1.0.12...v1.0.13
