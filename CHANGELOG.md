# Changelog

All notable changes to BrainX Skill are recorded here. The release workflow
updates this file from every unreleased commit on `main`.

## 1.0.13 - 2026-08-23

BrainX Skill 1.0.13 records 25 commits merged into `main`.

### Release record

- **Version:** `1.0.13`
- **Previous release:** `v1.0.12`
- **Commits included:** 25

### New features

#### feat(docs): deploy the documentation site to brainx.chaobrain.com/skills ([`7cc4561`](https://github.com/chaobrain/BrainX-skill/commit/7cc45617dbfb7adc5a38ba5813fe71233b48cb97))

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

#### feat: add BrainX closed-loop modeling workflows ([`5790da4`](https://github.com/chaobrain/BrainX-skill/commit/5790da429c0b81b27b6f065308c8e65fd97444d7))

No additional commit description was provided.

**Changed files:** `.github/scripts/agent-skills-workflow.test.mjs`, `.github/scripts/bundle-layout.test.mjs`, `.github/scripts/prepare-release.test.mjs`, `.github/workflows/agent-skills-validation.yml`, `AGENTS.md`, `braintools-references/braintools-input-current.md`, `brainx-closed-loop-plan.md`, `installation/lib/bundle.js`, `manifest.json`, `mcp-servers/codex/README.md`, `mcp-servers/codex/UPSTREAM.md`, `mcp-servers/codex/server.mjs`, and 350 more

#### feat: strengthen BrainX MCP review ([`046c89a`](https://github.com/chaobrain/BrainX-skill/commit/046c89aee4ab03491c7457f06a50aef9a2163d68))

Condense the reviewer contract around scientific support, minimal BrainX-native code, and optimization adequacy.

Expose the training and parameter-fitting references directly to fresh MCP review sessions, preserve responses in the caller context, and document a longer tool timeout for complete reviews.

Include regression coverage for the reviewer criteria and reference injection.

**Changed files:** `how-to-refine-skill.md`, `mcp-servers/codex/README.md`, `mcp-servers/codex/server.mjs`, `mcp-servers/codex/system-prompt.md`, `mcp-servers/codex/test/server.test.mjs`, `plan.md`, `skills/brainx-modeling-loop/SKILL.md`

### Documentation

#### docs(readme): show the BrainX Skill logo at the top ([`3773de8`](https://github.com/chaobrain/BrainX-skill/commit/3773de8c204abf5ea193a875c79259a9c89af3d3))

No additional commit description was provided.

**Changed files:** `README.md`

#### docs: add BrainX closed-loop modeling plan ([`244ebc1`](https://github.com/chaobrain/BrainX-skill/commit/244ebc128da0551e05a2c3932653e9767d1dc4f7))

No additional commit description was provided.

**Changed files:** `brainx-closed-loop-plan.md`

### Other changes

#### Refine alpha rhythm BrainX guidance ([`e865a07`](https://github.com/chaobrain/BrainX-skill/commit/e865a0732a3f551197684acc64cbb3e72f9464ee))

No additional commit description was provided.

**Changed files:** `brainx-display-cases/05-alpha-rhythm/control-unisolated/README.md`, `brainx-display-cases/05-alpha-rhythm/control-unisolated/agent-final.md`, `brainx-display-cases/05-alpha-rhythm/control-unisolated/alpha_rhythm.png`, `brainx-display-cases/05-alpha-rhythm/control-unisolated/alpha_rhythm.py`, `brainx-display-cases/05-alpha-rhythm/control-unisolated/brainmass-api-coverage-diagnosis.md`, `brainx-display-cases/05-alpha-rhythm/control-unisolated/codex-events.jsonl`, `brainx-display-cases/05-alpha-rhythm/control-unisolated/harness-metadata.txt`, `brainx-display-cases/05-alpha-rhythm/run0/.agents/skills/braincell/SKILL.md`, `brainx-display-cases/05-alpha-rhythm/run0/.agents/skills/braincell/references/area-scaled-hh-pattern.md`, `brainx-display-cases/05-alpha-rhythm/run0/.agents/skills/braincell/references/braincell-custom-ion-channel-authoring.md`, `brainx-display-cases/05-alpha-rhythm/run0/.agents/skills/braincell/references/channel-library.md`, `brainx-display-cases/05-alpha-rhythm/run0/.agents/skills/braincell/references/ion-library.md`, and 357 more

#### Refine seizure recruitment BrainX guidance ([`54310e0`](https://github.com/chaobrain/BrainX-skill/commit/54310e01a3f92b518bc2415e09bc19885a412b07))

No additional commit description was provided.

**Changed files:** `brainx-display-cases/06-seizure-recruitment/failed-launch-run7-config-parse-path/codex-events.jsonl`, `brainx-display-cases/06-seizure-recruitment/failed-launch-run7-config-parse-path/harness-metadata.txt`, `brainx-display-cases/06-seizure-recruitment/failed-launch-run7-env-chdir-option/codex-events.jsonl`, `brainx-display-cases/06-seizure-recruitment/failed-launch-run7-env-chdir-option/harness-metadata.txt`, `brainx-display-cases/06-seizure-recruitment/failed-launch-run7-repo-cwd/codex-events.jsonl`, `brainx-display-cases/06-seizure-recruitment/failed-launch-run7-repo-cwd/harness-metadata.txt`, `brainx-display-cases/06-seizure-recruitment/failed-launch-run7-sandbox-syntax/codex-events.jsonl`, `brainx-display-cases/06-seizure-recruitment/failed-launch-run7-sandbox-syntax/harness-metadata.txt`, `brainx-display-cases/06-seizure-recruitment/failed-launch-run7-seatbelt-trust-path/codex-events.jsonl`, `brainx-display-cases/06-seizure-recruitment/failed-launch-run7-seatbelt-trust-path/harness-metadata.txt`, `brainx-display-cases/06-seizure-recruitment/invalid-run7-nested-seatbelt/agent-final.md`, `brainx-display-cases/06-seizure-recruitment/invalid-run7-nested-seatbelt/codex-events.jsonl`, and 1255 more

#### Refine binocular rivalry BrainX guidance ([`bd26968`](https://github.com/chaobrain/BrainX-skill/commit/bd269684c16465384ae51b8e9960237a367dbbe1))

No additional commit description was provided.

**Changed files:** `brainx-display-cases/08-binocular-rivalry/run0/README.md`, `brainx-display-cases/08-binocular-rivalry/run0/agent-final.md`, `brainx-display-cases/08-binocular-rivalry/run0/binocular_rivalry.py`, `brainx-display-cases/08-binocular-rivalry/run0/brainmass-api-coverage-diagnosis.md`, `brainx-display-cases/08-binocular-rivalry/run0/codex-events.jsonl`, `brainx-display-cases/08-binocular-rivalry/run0/codex-stderr.log`, `brainx-display-cases/08-binocular-rivalry/run0/harness-metadata.txt`, `brainx-display-cases/08-binocular-rivalry/run0/results/binocular_rivalry.png`, `brainx-display-cases/08-binocular-rivalry/run0/results/binocular_rivalry_results.npz`, `brainx-display-cases/08-binocular-rivalry/run1/README.md`, `brainx-display-cases/08-binocular-rivalry/run1/__pycache__/binocular_rivalry.cpython-312.pyc`, `brainx-display-cases/08-binocular-rivalry/run1/agent-final.md`, and 8 more

#### Refine online working memory BrainTrace guidance ([`5882ec2`](https://github.com/chaobrain/BrainX-skill/commit/5882ec23c384ad40a2ebb8f89399a9e8a91e418a))

No additional commit description was provided.

**Changed files:** `brainx-display-cases/04-online-working-memory/run0/.agents/skills/braincell/SKILL.md`, `brainx-display-cases/04-online-working-memory/run0/.agents/skills/braincell/references/area-scaled-hh-pattern.md`, `brainx-display-cases/04-online-working-memory/run0/.agents/skills/braincell/references/braincell-custom-ion-channel-authoring.md`, `brainx-display-cases/04-online-working-memory/run0/.agents/skills/braincell/references/channel-library.md`, `brainx-display-cases/04-online-working-memory/run0/.agents/skills/braincell/references/ion-library.md`, `brainx-display-cases/04-online-working-memory/run0/.agents/skills/braincell/references/mixions-for-adaptation.md`, `brainx-display-cases/04-online-working-memory/run0/.agents/skills/braincell/references/multicompartment/braincell-manual-morphology-construction.md`, `brainx-display-cases/04-online-working-memory/run0/.agents/skills/braincell/references/multicompartment/cv-policy-reference.md`, `brainx-display-cases/04-online-working-memory/run0/.agents/skills/braincell/references/multicompartment/filter-function-library.md`, `brainx-display-cases/04-online-working-memory/run0/.agents/skills/braincell/references/multicompartment/morphology-io-loading-validation.md`, `brainx-display-cases/04-online-working-memory/run0/.agents/skills/braincell/references/multicompartment/multicompartment-cell-workflow.md`, `brainx-display-cases/04-online-working-memory/run0/.agents/skills/braincell/references/multicompartment/probe-reference.md`, and 348 more

#### Add BrainX skill documentation showcase ([`67dadcf`](https://github.com/chaobrain/BrainX-skill/commit/67dadcfd7e04795538580d1ed87a09aadc062092))

No additional commit description was provided.

**Changed files:** `.gitignore`, `docs/_static/brainx-skill.css`, `docs/_static/brainx-skill.js`, `docs/_static/cases/01-spike-frequency-adaptation/README.md`, `docs/_static/cases/01-spike-frequency-adaptation/agent-final.md`, `docs/_static/cases/01-spike-frequency-adaptation/agent-log.jsonl`, `docs/_static/cases/01-spike-frequency-adaptation/experiment.py`, `docs/_static/cases/01-spike-frequency-adaptation/prompt.md`, `docs/_static/cases/01-spike-frequency-adaptation/spike-frequency-adaptation.png`, `docs/_static/cases/02-learning-temporal-order/README.md`, `docs/_static/cases/02-learning-temporal-order/agent-final.md`, `docs/_static/cases/02-learning-temporal-order/agent-log.jsonl`, and 81 more

#### Update BrainX docs and perturbation workflows ([`78c01b6`](https://github.com/chaobrain/BrainX-skill/commit/78c01b6820556d8c1bf01458b51bbad76ade1126))

No additional commit description was provided.

**Changed files:** `BrainX doc style.md`, `PRODUCT.md`, `brainx-display-cases/13-single-cell-perturbation-bio/prompt.md`, `brainx-display-cases/13-single-cell-perturbation-bio/run0/README.md`, `brainx-display-cases/13-single-cell-perturbation-bio/run0/results/broad_inhibition_observables.npz`, `brainx-display-cases/13-single-cell-perturbation-bio/run0/results/influence_curves.csv`, `brainx-display-cases/13-single-cell-perturbation-bio/run0/results/influence_vs_signal_correlation.png`, `brainx-display-cases/13-single-cell-perturbation-bio/run0/results/inhibition_dominant_observables.npz`, `brainx-display-cases/13-single-cell-perturbation-bio/run0/results/parameters.json`, `brainx-display-cases/13-single-cell-perturbation-bio/run0/results/specific_strong_ei_observables.npz`, `brainx-display-cases/13-single-cell-perturbation-bio/run0/results/summary.json`, `brainx-display-cases/13-single-cell-perturbation-bio/run0/v1_single_cell_perturbation.py`, and 129 more

#### Restore top-level quick start navigation ([`2728a34`](https://github.com/chaobrain/BrainX-skill/commit/2728a34f68dae4efa39541e3c94ddcda4cc259b8))

No additional commit description was provided.

**Changed files:** `docs/index.rst`

#### Group docs into quickstart and examples ([`72378fe`](https://github.com/chaobrain/BrainX-skill/commit/72378fe2cc8b62f1f89cafbd71482c92cf018e06))

No additional commit description was provided.

**Changed files:** `docs/index.rst`, `docs/installation.rst`

#### Separate docs overview and quickstart video ([`5fabee5`](https://github.com/chaobrain/BrainX-skill/commit/5fabee5c1075140b6768ddfcccdf60a4f7ba5342))

No additional commit description was provided.

**Changed files:** `docs/_static/brainx-skill.css`, `docs/index.rst`, `docs/quickstart.rst`

#### Expand the quickstart workflow ([`d61a67d`](https://github.com/chaobrain/BrainX-skill/commit/d61a67d988de3324d50c70b992503b7314459787))

No additional commit description was provided.

**Changed files:** `docs/_static/brainx-skill.css`, `docs/quickstart.rst`

#### Use the quickstart video prompt ([`9b70985`](https://github.com/chaobrain/BrainX-skill/commit/9b709850984c61d3b4285022cb64dfae8efe3a17))

No additional commit description was provided.

**Changed files:** `docs/quickstart.rst`

#### Add skill reference documentation ([`e3244fb`](https://github.com/chaobrain/BrainX-skill/commit/e3244fb5fa52a74a796278a129fc403643a80cea))

No additional commit description was provided.

**Changed files:** `docs/_static/brainx-skill.css`, `docs/conf.py`, `docs/creative-experiment-verification.rst`, `docs/index.rst`, `docs/paper-reproduction.rst`, `docs/skill-reference.rst`, `docs/skill-reference/braincell.rst`, `docs/skill-reference/brainevent.rst`, `docs/skill-reference/brainmass.rst`, `docs/skill-reference/brainpy-state.rst`, `docs/skill-reference/brainstate.rst`, `docs/skill-reference/braintrace.rst`, and 6 more

#### add reference for case09 ([`5a0c0eb`](https://github.com/chaobrain/BrainX-skill/commit/5a0c0ebce570e39c43a779c6dce1cd968f039179))

No additional commit description was provided.

**Changed files:** `brainx-display-cases/09-neural-compass/prompt.md`, `brainx-display-cases/15-decision-making/prompt.md`, `brainx-display-cases/16 place cell navigation/prompt.md`

#### add case 15 ([`bc6240e`](https://github.com/chaobrain/BrainX-skill/commit/bc6240e204a642edf8ea6e3f4a25541da5ea1e99))

No additional commit description was provided.

**Changed files:** `brainx-display-cases/09-neural-compass/prompt.md`, `brainx-display-cases/15-decision-making/prompt.md`, `brainx-display-cases/15-grid-cell-theta-sweep/prompt.md`, `brainx-display-cases/15-grid-cell-theta-sweep/run0/theta_sweeps.py`, `brainx-display-cases/15-grid-cell-theta-sweep/run0/theta_sweeps_evidence.npz`, `brainx-display-cases/15-grid-cell-theta-sweep/run0/theta_sweeps_main.png`, `brainx-display-cases/15-grid-cell-theta-sweep/run0/theta_sweeps_mechanisms.png`, `brainx-display-cases/15-grid-cell-theta-sweep/run0/theta_sweeps_metrics.json`, `brainx-display-cases/15-grid-cell-theta-sweep/run1/README.md`, `brainx-display-cases/15-grid-cell-theta-sweep/run1/results/baseline_cycle_metrics.csv`, `brainx-display-cases/15-grid-cell-theta-sweep/run1/results/population_and_alignment.png`, `brainx-display-cases/15-grid-cell-theta-sweep/run1/results/protocols_and_mechanisms.png`, and 5 more

#### Reorganize display cases and add case 15 results ([`32baafe`](https://github.com/chaobrain/BrainX-skill/commit/32baafe44b320b26dad07c559d2a751bf394920a))

No additional commit description was provided.

**Changed files:** `brainx-display-cases/01-spike-frequency-adaptation/prompt.md`, `brainx-display-cases/01-spike-frequency-adaptation/run1/README.md`, `brainx-display-cases/01-spike-frequency-adaptation/run1/agent-final.md`, `brainx-display-cases/01-spike-frequency-adaptation/run1/braincell-api-coverage-diagnosis.md`, `brainx-display-cases/01-spike-frequency-adaptation/run1/codex-events.jsonl`, `brainx-display-cases/01-spike-frequency-adaptation/run1/spike_frequency_adaptation.png`, `brainx-display-cases/01-spike-frequency-adaptation/run1/spike_frequency_adaptation.py`, `brainx-display-cases/01-spike-frequency-adaptation/run2/README.md`, `brainx-display-cases/01-spike-frequency-adaptation/run2/agent-final.md`, `brainx-display-cases/01-spike-frequency-adaptation/run2/braincell-api-coverage-diagnosis.md`, `brainx-display-cases/01-spike-frequency-adaptation/run2/codex-events.jsonl`, `brainx-display-cases/01-spike-frequency-adaptation/run2/spike_frequency_adaptation.png`, and 4404 more

#### chore(ci): use the organization deploy credentials ([`adaa499`](https://github.com/chaobrain/BrainX-skill/commit/adaa499b5b6e7fc1be452f7cecd318e4215a70b6))

The organization publishes DEPLOY_*_G, and BrainX-skill already has
access to them, so the workflow needs no repository-level secrets.

Drop the releasing page; the release and deployment process is internal
and does not belong on the public documentation site.

**Changed files:** `.github/workflows/deploy-docs.yml`, `docs/index.rst`, `docs/releasing.rst`

#### Refine documentation experience ([`9b5e8a9`](https://github.com/chaobrain/BrainX-skill/commit/9b5e8a9c4820ff257cadf9732954c5990cdc343f))

No additional commit description was provided.

**Changed files:** `docs/_static/brainx-skill.css`, `docs/_static/brainx-skill.js`, `docs/cases/01-spike-frequency-adaptation.rst`, `docs/cases/02-learning-temporal-order.rst`, `docs/cases/03-sound-localization.rst`, `docs/cases/05-alpha-rhythm.rst`, `docs/cases/06-seizure-recruitment.rst`, `docs/cases/07-cortical-wave-obstacle.rst`, `docs/cases/08-binocular-rivalry.rst`, `docs/cases/09-neural-compass.rst`, `docs/cases/10-prior-bias.rst`, `docs/cases/11-sleep-memory-replay.rst`, and 19 more

#### Remove supported agents documentation ([`74aa7ec`](https://github.com/chaobrain/BrainX-skill/commit/74aa7ec463f0a25b81e29a876b64a92ca2d48f20))

No additional commit description was provided.

**Changed files:** `docs/index.rst`, `docs/supported-agents.rst`

#### Expand homepage installation section ([`da04a86`](https://github.com/chaobrain/BrainX-skill/commit/da04a86390216c513e3998bf6588353ef38ac50c))

No additional commit description was provided.

**Changed files:** `docs/index.rst`

#### case 16,successfullly reproduced ([`af32161`](https://github.com/chaobrain/BrainX-skill/commit/af3216120476e7c87ea0ef3d5e275406791c840d))

No additional commit description was provided.

**Changed files:** `brainx-display-cases/16-C.elegans-muscle-cell/Fig4A-D.txt`, `brainx-display-cases/16-C.elegans-muscle-cell/README.md`, `brainx-display-cases/16-C.elegans-muscle-cell/celegans_muscle_inference.py`, `brainx-display-cases/16-C.elegans-muscle-cell/prompt.md`, `brainx-display-cases/16-C.elegans-muscle-cell/results/held_out_validation.png`, `brainx-display-cases/16-C.elegans-muscle-cell/results/posterior_samples.csv`, `brainx-display-cases/16-C.elegans-muscle-cell/results/report.json`, `brainx-display-cases/16-C.elegans-muscle-cell/results/trace_predictions.csv`, `brainx-display-cases/16-C.elegans-muscle-cell/test_celegans_muscle_inference.py`

**Full Changelog:** https://github.com/chaobrain/BrainX-skill/compare/v1.0.12...v1.0.13

## 1.0.12 - 2026-08-12

BrainX Skill 1.0.12 records 10 commits merged into `main`.

### Release record

- **Version:** `1.0.12`
- **Previous release:** `v1.0.11`
- **Commits included:** 10

### Documentation

#### docs(braincell): refine spike-frequency adaptation workflow ([`8155f69`](https://github.com/chaobrain/BrainX-skill/commit/8155f69483636df8dc545778ed2825ed8d3fb75c))

No additional commit description was provided.

**Changed files:** `brainx-display-cases/01-spike-frequency-adaptation/prompt.md`, `brainx-display-cases/01-spike-frequency-adaptation/run1/README.md`, `brainx-display-cases/01-spike-frequency-adaptation/run1/agent-final.md`, `brainx-display-cases/01-spike-frequency-adaptation/run1/braincell-api-coverage-diagnosis.md`, `brainx-display-cases/01-spike-frequency-adaptation/run1/codex-events.jsonl`, `brainx-display-cases/01-spike-frequency-adaptation/run1/spike_frequency_adaptation.png`, `brainx-display-cases/01-spike-frequency-adaptation/run1/spike_frequency_adaptation.py`, `brainx-display-cases/01-spike-frequency-adaptation/run2/README.md`, `brainx-display-cases/01-spike-frequency-adaptation/run2/agent-final.md`, `brainx-display-cases/01-spike-frequency-adaptation/run2/braincell-api-coverage-diagnosis.md`, `brainx-display-cases/01-spike-frequency-adaptation/run2/codex-events.jsonl`, `brainx-display-cases/01-spike-frequency-adaptation/run2/spike_frequency_adaptation.png`, and 9 more

#### docs(skills): collect remaining refinement work ([`78ee3b7`](https://github.com/chaobrain/BrainX-skill/commit/78ee3b7f15587a1484811b0d76ba6238d902ebf7))

No additional commit description was provided.

**Changed files:** `batches.meta`, `brainx-display-cases/02-learning-temporal-order/prompt.md`, `brainx-display-cases/03-sound-localization/prompt.md`, `brainx-display-cases/04-online-working-memory/prompt.md`, `brainx-display-cases/05-alpha-rhythm/prompt.md`, `brainx-display-cases/06-seizure-recruitment/prompt.md`, `brainx-display-cases/07-cortical-wave-obstacle/prompt.md`, `brainx-display-cases/08-binocular-rivalry/prompt.md`, `brainx-display-cases/09-neural-compass/prompt.md`, `brainx-display-cases/10-prior-bias-ambiguous-evidence/prompt.md`, `brainx-display-cases/10-prior-bias-ambiguous-evidence/run1/brainmass-api-coverage-diagnosis.md`, `brainx-display-cases/10-prior-bias-ambiguous-evidence/run1/build_decision_bias_data.py`, and 47 more

#### docs(skills): refine temporal-order learning workflow ([`dd8ce61`](https://github.com/chaobrain/BrainX-skill/commit/dd8ce61fd26de4d8aac8fe2fc442ed1bac5bdb05))

No additional commit description was provided.

**Changed files:** `brainx-display-cases/02-learning-temporal-order/pre-refinement-isolated-control/.matplotlib/fontlist-v3.11.0.json`, `brainx-display-cases/02-learning-temporal-order/pre-refinement-isolated-control/CONTROL-NOTE.md`, `brainx-display-cases/02-learning-temporal-order/pre-refinement-isolated-control/README.md`, `brainx-display-cases/02-learning-temporal-order/pre-refinement-isolated-control/agent-final.md`, `brainx-display-cases/02-learning-temporal-order/pre-refinement-isolated-control/codex-events.jsonl`, `brainx-display-cases/02-learning-temporal-order/pre-refinement-isolated-control/harness-metadata.txt`, `brainx-display-cases/02-learning-temporal-order/pre-refinement-isolated-control/temporal_order_learning.py`, `brainx-display-cases/02-learning-temporal-order/pre-refinement-isolated-control/temporal_order_relearning.png`, `brainx-display-cases/02-learning-temporal-order/pre-refinement-isolated-control/test_temporal_order_learning.py`, `brainx-display-cases/02-learning-temporal-order/run0-invalid-auth/codex-events.jsonl`, `brainx-display-cases/02-learning-temporal-order/run0-invalid-auth/harness-metadata.txt`, `brainx-display-cases/02-learning-temporal-order/run0/README.md`, and 41 more

#### docs(skills): refine sound localization workflow ([`3e687ab`](https://github.com/chaobrain/BrainX-skill/commit/3e687ab9808d53c57b7a57c0feda94c6222451c0))

No additional commit description was provided.

**Changed files:** `brainx-display-cases/03-sound-localization/run0/README.md`, `brainx-display-cases/03-sound-localization/run0/agent-final.md`, `brainx-display-cases/03-sound-localization/run0/brainpy-state-api-coverage-diagnosis.md`, `brainx-display-cases/03-sound-localization/run0/codex-events.jsonl`, `brainx-display-cases/03-sound-localization/run0/codex-stderr.log`, `brainx-display-cases/03-sound-localization/run0/harness-metadata.txt`, `brainx-display-cases/03-sound-localization/run0/sound_localization.py`, `brainx-display-cases/03-sound-localization/run0/test_sound_localization.py`, `brainx-display-cases/03-sound-localization/run1/agent-final.md`, `brainx-display-cases/03-sound-localization/run1/brainpy-state-api-coverage-diagnosis.md`, `brainx-display-cases/03-sound-localization/run1/codex-events.jsonl`, `brainx-display-cases/03-sound-localization/run1/codex-stderr.log`, and 17 more

### Other changes

#### chore: remove tests directory ([`201e506`](https://github.com/chaobrain/BrainX-skill/commit/201e5061238a04634f0051121d32d132bfe77a1a))

No additional commit description was provided.

**Changed files:** `tests/thalamic_rebound.py`, `tests/thalamic_rebound/simulate_rebound.py`, `tests/thalamic_rebound/thalamic_rebound_comparison.png`

#### Refine prior-bias modeling guidance ([`46b07d9`](https://github.com/chaobrain/BrainX-skill/commit/46b07d9dea82b81cd51d0f3775ff3449700d3483))

No additional commit description was provided.

**Changed files:** `brainx-display-cases/10-prior-bias-ambiguous-evidence/run2-invalid-auth/codex-events.jsonl`, `brainx-display-cases/10-prior-bias-ambiguous-evidence/run2-invalid-auth/codex-stderr.log`, `brainx-display-cases/10-prior-bias-ambiguous-evidence/run2-invalid-auth/harness-metadata.txt`, `brainx-display-cases/10-prior-bias-ambiguous-evidence/run2/README.md`, `brainx-display-cases/10-prior-bias-ambiguous-evidence/run2/agent-final.md`, `brainx-display-cases/10-prior-bias-ambiguous-evidence/run2/brainpy-state-api-coverage-diagnosis.md`, `brainx-display-cases/10-prior-bias-ambiguous-evidence/run2/codex-events.jsonl`, `brainx-display-cases/10-prior-bias-ambiguous-evidence/run2/codex-stderr.log`, `brainx-display-cases/10-prior-bias-ambiguous-evidence/run2/harness-metadata.txt`, `brainx-display-cases/10-prior-bias-ambiguous-evidence/run2/prior_bias_decision.py`, `brainx-display-cases/10-prior-bias-ambiguous-evidence/run2/prior_bias_results.png`, `brainx-display-cases/10-prior-bias-ambiguous-evidence/run3/README.md`, and 22 more

#### Refine cortical wave BrainX guidance ([`e9c8dee`](https://github.com/chaobrain/BrainX-skill/commit/e9c8deee4e3801174c37616d69dfa2c4e4e60303))

No additional commit description was provided.

**Changed files:** `.gitattributes`, `braintools-references/braintools-input-current.md`, `brainx-display-cases/07-cortical-wave-obstacle/run0/README.md`, `brainx-display-cases/07-cortical-wave-obstacle/run0/agent-final.md`, `brainx-display-cases/07-cortical-wave-obstacle/run0/brainpy-state-api-coverage-diagnosis.md`, `brainx-display-cases/07-cortical-wave-obstacle/run0/codex-events.jsonl`, `brainx-display-cases/07-cortical-wave-obstacle/run0/codex-stderr.log`, `brainx-display-cases/07-cortical-wave-obstacle/run0/cortical_wave.py`, `brainx-display-cases/07-cortical-wave-obstacle/run0/harness-metadata.txt`, `brainx-display-cases/07-cortical-wave-obstacle/run0/pyproject.toml`, `brainx-display-cases/07-cortical-wave-obstacle/run0/results/outcomes.csv`, `brainx-display-cases/07-cortical-wave-obstacle/run0/results/phase_map.png`, and 38 more

#### Refine neural compass validation guidance ([`914ab23`](https://github.com/chaobrain/BrainX-skill/commit/914ab233dc532633254ca5cfcd1d42983dd1890e))

No additional commit description was provided.

**Changed files:** `brainx-display-cases/09-neural-compass/run0/README.md`, `brainx-display-cases/09-neural-compass/run0/agent-final.md`, `brainx-display-cases/09-neural-compass/run0/brainpy-state-api-coverage-diagnosis.md`, `brainx-display-cases/09-neural-compass/run0/codex-events.jsonl`, `brainx-display-cases/09-neural-compass/run0/codex-stderr.log`, `brainx-display-cases/09-neural-compass/run0/harness-metadata.txt`, `brainx-display-cases/09-neural-compass/run0/head_direction_compass.py`, `brainx-display-cases/09-neural-compass/run0/results/head_direction_compass.png`, `brainx-display-cases/09-neural-compass/run0/results/lesion_sweep.csv`, `brainx-display-cases/09-neural-compass/run0/results/summary.json`, `brainx-display-cases/09-neural-compass/run0/test_head_direction_compass.py`, `brainx-display-cases/09-neural-compass/run1/README.md`, and 31 more

#### Refine sleep replay validation guidance ([`a656735`](https://github.com/chaobrain/BrainX-skill/commit/a6567351f842edb0950fae6be416b994b51ec53a))

No additional commit description was provided.

**Changed files:** `brainx-display-cases/11-sleep-memory-replay/attempt-network-failure-0/attempt-metadata.txt`, `brainx-display-cases/11-sleep-memory-replay/attempt-network-failure-0/codex-events.jsonl`, `brainx-display-cases/11-sleep-memory-replay/run0/README.md`, `brainx-display-cases/11-sleep-memory-replay/run0/agent-final.md`, `brainx-display-cases/11-sleep-memory-replay/run0/brainpy-state-api-coverage-diagnosis.md`, `brainx-display-cases/11-sleep-memory-replay/run0/codex-events.jsonl`, `brainx-display-cases/11-sleep-memory-replay/run0/harness-metadata.txt`, `brainx-display-cases/11-sleep-memory-replay/run0/sleep_replay.py`, `brainx-display-cases/11-sleep-memory-replay/run0/sleep_replay_results.png`, `brainx-display-cases/11-sleep-memory-replay/run0/test_sleep_replay.py`, `brainx-display-cases/11-sleep-memory-replay/run1/README.md`, `brainx-display-cases/11-sleep-memory-replay/run1/agent-final.md`, and 25 more

#### Refine cumulative scientific validation guidance ([`7ef18ea`](https://github.com/chaobrain/BrainX-skill/commit/7ef18eae0ca2b93fc1a71847fa77210cd28e8000))

No additional commit description was provided.

**Changed files:** `brainx-display-cases/12-edge-of-criticality/run0/README.md`, `brainx-display-cases/12-edge-of-criticality/run0/agent-final.md`, `brainx-display-cases/12-edge-of-criticality/run0/brainpy-state-api-coverage-diagnosis.md`, `brainx-display-cases/12-edge-of-criticality/run0/codex-events.jsonl`, `brainx-display-cases/12-edge-of-criticality/run0/criticality_scan.py`, `brainx-display-cases/12-edge-of-criticality/run0/harness-metadata.txt`, `brainx-display-cases/12-edge-of-criticality/run0/quick-results/criticality_scan.csv`, `brainx-display-cases/12-edge-of-criticality/run0/quick-results/criticality_scan.png`, `brainx-display-cases/12-edge-of-criticality/run0/quick-results/summary.json`, `brainx-display-cases/12-edge-of-criticality/run0/results/criticality_scan.csv`, `brainx-display-cases/12-edge-of-criticality/run0/results/criticality_scan.png`, `brainx-display-cases/12-edge-of-criticality/run0/results/summary.json`, and 15 more

**Full Changelog:** https://github.com/chaobrain/BrainX-skill/compare/v1.0.11...v1.0.12

## 1.0.11 - 2026-08-10

BrainX Skill 1.0.11 records 6 commits merged into `main`.

### Release record

- **Version:** `1.0.11`
- **Previous release:** `v1.0.10`
- **Commits included:** 6

### New features

#### feat(skills): add gated legacy BrainPy workflow ([`3706cbe`](https://github.com/chaobrain/BrainX-skill/commit/3706cbed7d0a17a332872d1481ccc35e8250c72f))

No additional commit description was provided.

**Changed files:** `plan.md`, `skills/brainpy-state/SKILL.md`, `skills/brainpy-state/references/brainPy(legacy)/analysis.md`, `skills/brainpy-state/references/brainPy(legacy)/brainpy legacy workflow.md`, `skills/brainpy-state/references/brainPy(legacy)/built-in dynamic neuron model.md`, `skills/brainpy-state/references/brainPy(legacy)/connecting neurons.md`, `skills/brainpy-state/references/brainPy(legacy)/customize neuron and synpase.md`, `skills/brainpy-state/references/brainPy(legacy)/infrastructure/Input generation.md`, `skills/brainpy-state/references/brainPy(legacy)/infrastructure/More about simulation.md`, `skills/brainpy-state/references/brainPy(legacy)/infrastructure/Multi-device array sharding.md`, `skills/brainpy-state/references/brainPy(legacy)/infrastructure/Parallel experiment execution.md`, `skills/brainpy-state/references/brainPy(legacy)/infrastructure/array creation and mechanics.md`, and 14 more

#### feat(adapters): support Google Antigravity and repair skills validation CI (#10) ([`5abe23f`](https://github.com/chaobrain/BrainX-skill/commit/5abe23f6b67d02ebe5c70f95218e06120a8a53a5))

* feat(adapters): support Google Antigravity

Antigravity reads global skills from ~/.gemini/config/skills and workspace
skills from <cwd>/.agents/skills, so an adapter needs a scope-dependent
destination for the first time.

Add an optional projectPath to the adapter shape, resolve it for project
scope, and accept either path when validating a receipt destination.

In project scope Antigravity shares <cwd>/.agents/skills with Codex.
Install once per resolved destination and record that ownership for every
harness in the group, and fall back to any receipt record at the same
destination so adding Antigravity after Codex proves ownership instead of
failing.

* fix(ci): register braintrace and close validation gaps

skills/braintrace was shipped without a manifest.json or package.json files
entry, which broke Agent Skills validation on main.

Register it, and close the gaps that let the drift through: verify the
published files list against the skill directories, reject loose files
directly under skills/, check manifest names against the installer name
pattern, and compare the lists with diff so mismatches are readable and
names containing spaces cannot mask a difference.

**Changed files:** `.github/scripts/adapters.test.mjs`, `.github/workflows/agent-skills-validation.yml`, `README.md`, `adapters/antigravity.js`, `installation/lib/installer.js`, `installation/lib/paths.js`, `installation/lib/prompts.js`, `manifest.json`, `package.json`

### Documentation

#### docs(braintrace): add planned source references ([`7830770`](https://github.com/chaobrain/BrainX-skill/commit/7830770d8b759a044435357bcc389fbae0d2e129))

No additional commit description was provided.

**Changed files:** `skills/braintrace/SKILL.md`, `skills/braintrace/references/Drtrl.md`, `skills/braintrace/references/ETP operators.md`, `skills/braintrace/references/algorithm selection.md`, `skills/braintrace/references/batching.md`, `skills/braintrace/references/compiler_internal.md`, `skills/braintrace/references/custom ETP primitives.md`, `skills/braintrace/references/custom algorithms.md`, `skills/braintrace/references/customizing_primitive_transforms.md`, `skills/braintrace/references/pp_pprop workflow.md`, `skills/braintrace/references/pre-built-braintrace-layer.md`

#### docs(braintrace): complete online learning guidance ([`d92f3e9`](https://github.com/chaobrain/BrainX-skill/commit/d92f3e98c7efb56b7a84bfc623f9b9f670e5b98d))

No additional commit description was provided.

**Changed files:** `skills/braintrace/SKILL.md`, `skills/braintrace/references/Drtrl.md`, `skills/braintrace/references/ETP operators.md`, `skills/braintrace/references/algorithm selection.md`, `skills/braintrace/references/batching.md`, `skills/braintrace/references/compiler_internal.md`, `skills/braintrace/references/custom ETP primitives.md`, `skills/braintrace/references/custom algorithms.md`, `skills/braintrace/references/customizing_primitive_transforms.md`, `skills/braintrace/references/pp_pprop workflow.md`, `skills/braintrace/references/pre-built-braintrace-layer.md`

#### docs: repair broken skill routing and resolve AGENTS.md contradictions (#11) ([`16eed45`](https://github.com/chaobrain/BrainX-skill/commit/16eed450d87f915a8320863ed93a76dd0312a573))

* docs(skills): repair broken routing and align skills with AGENTS.md

An audit of all nine skills found defects that CI does not catch: skills-ref
validation, the node test suite, and the manifest/package checks were all green
while four routing targets pointed at files that do not exist and seven of
braincell's eight shipped scripts were unreachable.

Dangling routing targets:

- brainstate cited references/braintools-optimizer-reference.md from SKILL.md
  and transformation-grad-expansion.md, but the file was never written. Add
  references/braintools/optimizer.md, derived from the verified brainmass and
  brainpy-state copies and scoped to BrainState training, and retarget both
  citations. No API surface is invented.
- brainstate's only inbound route to parameter-transforms-regularizers-catalog.md
  used bare filenames that resolve against the skill root and miss. Rewrite both
  as references/brainstate/ paths.
- brainpy-state's array-creation.md routed array_split(), backend extraction, and
  as_numpy() to array-mechanics.md, which exists only in brainunit. Retarget to
  skills/brainunit/references/array-mechanics.md, matching the cross-skill path
  form already used elsewhere.

Unreachable content:

- braincell had no script table at all. Add Application script examples routing
  its seven previously unroutable scripts, each row derived from the script's own
  purpose docstring. cell_multicompartment_reference.py keeps its single existing
  route from the multicompartment workflow.

Rule 20 form:

- brainx-install's Reference Routing was three prose paragraphs; convert to the
  required two-column table.
- brainstate's Script references was a bullet list placed before the routing
  table; rename to Application script examples, convert to a table, and move it
  after the routing table.

Rules 17 and 23:

- brainx-install and brainx-general-guard had no Purpose and boundary section.
  Add one to each; all nine skills now open with it. Sentence-case the fourteen
  Title Case headings in brainx-install, and repair the guard heading that ran
  two sentences together with a shouted ONLY.
- brainx-install's description was the only quoted one in the repo and began with
  a literal leading space and a lowercase sentence start.
- Fix the # Brainstate H1 casing, a double space in brainmass, and two missing
  blank lines before script tables.

The guard routed only the three scale skills, leaving braintrace, brainevent,
brainunit, and brainstate with no guard-level entry point. Add a cross-cutting
concern table rather than forcing them into the deliberately scale-based one.

Verified: skills-ref 0.1.1 reports Valid skill for all nine; node --test passes
26/26; a link scan reports zero dangling targets and zero unrouted shipped files
(was four and eight); all 40 scripts compile.

Noted but not changed: skills/brainstate/agents/openai.yaml is valid and Rule
1-compliant but nothing in adapters/ or installation/ reads it, and it is the
only agents/ directory in the repo. AGENTS.md Rule 17 orders reference routing
before boundaries and common failures while Rule 20 requires the file to end on
the routing table; the two contradict and skills follow both readings. Neither
is resolved here.

* docs(skills): drop unconsumed brainstate openai.yaml

skills/brainstate/agents/openai.yaml declared display_name, short_description,
and default_prompt, but nothing read it: no adapter in adapters/ and no code in
installation/ references agents/ or openai.yaml, and it was the only agents/
directory in the repo. Remove it rather than leave unreplicated dead metadata.

Rule 1 still permits agents/ for per-platform interface metadata; this deletes
one unused instance, not the allowance.

* docs: resolve contradictions in AGENTS.md

Six rules contradicted each other or contradicted what the nine shipped skills
actually do. In every case the rule was wrong, not the skills: after these edits
all nine pass unchanged.

Rule 17 vs Rule 20 — the tail order. Rule 17 ordered reference routing before
boundaries and common failures; Rule 20 told you to close the file with the
routing table. Rule 20 then contradicted itself by requiring a separate script
table, which has to sit somewhere. Skills split across all three readings: four
end on a script table, three on the routing table, one on boundaries. Rule 17 now
states the full tail — reference routing, application script examples, boundaries
and common failures — keeps them in that relative order, and marks each optional.
Rule 20 is renamed from "End with a reference routing table" accordingly.

Rule 17's first heading. It required Purpose and boundary to be the first heading
after the frontmatter, which seven of nine skills violate by carrying an H1 title.
Now: first `##` section, H1 title permitted.

Rule 20's routing invariant. Read strictly it required every shipped file to have
a row in SKILL.md, which braincell, brainpy-state, and brainmass all break on
purpose by delegating a grouped family to its workflow reference. The invariant is
now exactly one inbound row anywhere in the skill, with delegation described and
braincell named as the pattern. Also states the path form — skill-root-relative,
never bare, full repository path across skills — which is what the four dangling
targets fixed in the previous commit got wrong.

Rule 20 vs Rule 1. Rule 1 permits agents/ while Rule 20 demanded every shipped
file appear in a routing row. Scope the invariant to references/ and script files.

Rule 1's nesting limit. It allowed one level below references/; brainpy-state
ships 22 files at two levels under nest-compatible/scripts/ and
brainPy(legacy)/infrastructure/. Raised to two.

Rule 4's stated range. Descriptions were said to run 240-580 characters; they
actually run 227-426. Corrected.

Rule 10 said "run both checks" above three commands and validated a single skill
while CI validates all of them. It now validates every skill, and says plainly
that neither check reads routing tables — so neither would have caught any of the
routing defects fixed in the previous commit.

**Changed files:** `AGENTS.md`, `skills/braincell/SKILL.md`, `skills/brainevent/SKILL.md`, `skills/brainmass/SKILL.md`, `skills/brainpy-state/SKILL.md`, `skills/brainpy-state/references/array-creation.md`, `skills/brainstate/SKILL.md`, `skills/brainstate/agents/openai.yaml`, `skills/brainstate/references/brainstate/transformation-grad-expansion.md`, `skills/brainstate/references/braintools/optimizer.md`, `skills/brainx-general-guard/SKILL.md`, `skills/brainx-install/SKILL.md`

#### docs(braintrace): center memory-efficient training ([`c379905`](https://github.com/chaobrain/BrainX-skill/commit/c37990583c27b3080131cab420ccffe50565ad75))

No additional commit description was provided.

**Changed files:** `plan.md`, `skills/braintrace/SKILL.md`

**Full Changelog:** https://github.com/chaobrain/BrainX-skill/compare/v1.0.10...v1.0.11

## 1.0.10 - 2026-07-31

BrainX Skill 1.0.10 records 1 commit merged into `main`.

### Release record

- **Version:** `1.0.10`
- **Previous release:** `v1.0.9`
- **Commits included:** 1

### New features

#### feat(skills): tighten BrainX guard and install handoff ([`4df5ebc`](https://github.com/chaobrain/BrainX-skill/commit/4df5ebc94063433bf2dc88a2b12d9c2a212942d8))

No additional commit description was provided.

**Changed files:** `plan.md`, `skills/brainx-general-guard/SKILL.md`, `skills/brainx-install/SKILL.md`

**Full Changelog:** https://github.com/chaobrain/BrainX-skill/compare/v1.0.9...v1.0.10

## 1.0.9 - 2026-07-30

BrainX Skill 1.0.9 records 3 commits merged into `main`.

### Release record

- **Version:** `1.0.9`
- **Previous release:** `v1.0.8`
- **Commits included:** 3

### New features

#### feat(skills): add BrainMass and refine model routing ([`f5426d6`](https://github.com/chaobrain/BrainX-skill/commit/f5426d646b09c6f159fb5b44cceb63ddda165c41))

No additional commit description was provided.

**Changed files:** `README.md`, `manifest.json`, `package.json`, `plan.md`, `skills/braincell/SKILL.md`, `skills/brainmass/SKILL.md`, `skills/brainmass/references/batch-transform-acceleration.md`, `skills/brainmass/references/brainstate/parameter-constraints-regularization.md`, `skills/brainmass/references/brainstate/parameter-transforms-regularizers-catalog.md`, `skills/brainmass/references/braintools/cogtask.md`, `skills/brainmass/references/braintools/data-preprocessing.md`, `skills/brainmass/references/braintools/metric.md`, and 28 more

### Documentation

#### docs(braincell): refine authoring and visualization guides ([`54dee95`](https://github.com/chaobrain/BrainX-skill/commit/54dee95b94f0d3af23a837dc7cd36180d9396148))

No additional commit description was provided.

**Changed files:** `skills/braincell/SKILL.md`, `skills/braincell/references/braincell-custom-ion-channel-authoring.md`, `skills/braincell/references/channel-library.md`, `skills/braincell/references/ion-library.md`, `skills/braincell/references/multicompartment/filter-function-library.md`, `skills/braincell/references/multicompartment/topology-building-and-visualization.md`

#### docs(brainmass): refine reference routing categories ([`909833a`](https://github.com/chaobrain/BrainX-skill/commit/909833a67c35199a045f68b666e1c5c487e3a1de))

No additional commit description was provided.

**Changed files:** `plan.md`, `skills/brainmass/SKILL.md`

**Full Changelog:** https://github.com/chaobrain/BrainX-skill/compare/v1.0.8...v1.0.9

## 1.0.8 - 2026-07-29

BrainX Skill 1.0.8 records 16 commits merged into `main`.

### Release record

- **Version:** `1.0.8`
- **Previous release:** `v1.0.6`
- **Commits included:** 16

### New features

#### feat(brainevent): add event-driven connectivity skill ([`67ea94b`](https://github.com/chaobrain/BrainX-skill/commit/67ea94b429665cbc3e1c417176d567efc99fe0af))

No additional commit description was provided.

**Changed files:** `manifest.json`, `package.json`, `plan.md`, `skills/brainevent/SKILL.md`, `skills/brainevent/references/connectivity-variants.md`, `skills/brainevent/references/custom-operators.md`, `skills/brainevent/references/scripts/102_EI_net_1996.py`, `skills/brainevent/references/scripts/204_joglekar_2018_propagation.py`, `skills/brainevent/references/sparse-formats.md`, `skills/brainevent/references/synaptic-plasticity.md`

#### feat(installer): add Windsurf, Gemini CLI, and OpenCode adapters (#8) ([`9fb1c03`](https://github.com/chaobrain/BrainX-skill/commit/9fb1c03dc4defee657bcb502dcc4689097281a16))

Extend the install-destination adapter registry beyond Claude Code, Codex,
and Cursor. The consumer pipeline (installer, paths, prompts, receipt) is
already generic over the adapters array, so this is purely additive plus
docs and the first test coverage for the adapter registry.

**Changed files:** `.github/scripts/adapters.test.mjs`, `README.md`, `adapters/gemini.js`, `adapters/opencode.js`, `adapters/windsurf.js`, `installation/lib/installer.js`, `package.json`

### Fixes

#### fix(release): stop after recovering a tagged release ([`211ff21`](https://github.com/chaobrain/BrainX-skill/commit/211ff214951566a50dca02c9b56ef18c6bfc8ee5))

Mark successful recovery in the workflow outputs and skip version preparation, tagging, and publishing steps after the missing npm or GitHub release has been restored.

**Changed files:** `.github/scripts/release-workflow.test.mjs`, `.github/workflows/release-notes.yml`

#### fix(release): remove accidental v1.0.7 ([`231d749`](https://github.com/chaobrain/BrainX-skill/commit/231d7493ccd4ea2e83c7db1aa8056fb100e57676))

No additional commit description was provided.

**Changed files:** `.github/scripts/prepare-release.mjs`, `.github/scripts/prepare-release.test.mjs`, `CHANGELOG.md`, `RELEASE_NOTES.md`, `package.json`, `release-notes.config.json`

#### fix(manifest): register brainpy-state and brainx-general-guard skills (#7) ([`46a51c4`](https://github.com/chaobrain/BrainX-skill/commit/46a51c4fca20038f8248989f1dc9562719889d55))

Register skills/brainpy-state/ and skills/brainx-general-guard/ in manifest.json
and package.json files, fixing the Agent Skills validation workflow, which failed
on every run because the sorted manifest array did not match the sorted skill
directories.

Relocate skills/brainpy-state/NEST-compatible/ to references/nest-compatible/ to
satisfy AGENTS.md rule 1, and normalize the branch's routing links to
skill-root-relative paths.

Add the npmmirror update command to the low-bandwidth README section.

**Changed files:** `README.md`, `manifest.json`, `package.json`, `plan.md`, `skills/brainpy-state/NEST-compatible/nest-workflow.md`, `skills/brainpy-state/NEST-compatible/references/devices.md`, `skills/brainpy-state/NEST-compatible/references/divergence-and-parity.md`, `skills/brainpy-state/NEST-compatible/references/integration-categories.md`, `skills/brainpy-state/NEST-compatible/references/model-library.md`, `skills/brainpy-state/NEST-compatible/references/network-building.md`, `skills/brainpy-state/NEST-compatible/references/synapse-and-connectivity.md`, `skills/brainpy-state/NEST-compatible/scripts/brette_et_al_2007.py`, and 22 more

### Documentation

#### docs: refresh BrainX skill guidance ([`acefb6d`](https://github.com/chaobrain/BrainX-skill/commit/acefb6dfd97a584c17e6f9727a4a7aa41fdb8290))

No additional commit description was provided.

**Changed files:** `AGENTS.md`, `archive/solver-library-with-effects.md`, `plan.md`, `skills/brainevent/SKILL.md`, `skills/brainevent/references/custom-operators-cpu.md`, `skills/brainevent/references/custom-operators-gpu.md`, `skills/brainevent/references/custom-operators.md`, `skills/brainevent/references/scripts/coba_ei_teaching.py`, `skills/brainpy-state/NEST-compatible/nest-workflow.md`, `skills/brainpy-state/NEST-compatible/references/devices.md`, `skills/brainpy-state/NEST-compatible/references/divergence-and-parity.md`, `skills/brainpy-state/NEST-compatible/references/integration-categories.md`, and 36 more

#### docs: add application script table headers ([`b508f5d`](https://github.com/chaobrain/BrainX-skill/commit/b508f5d2e71e1572707db4ed1e2201bf18932d19))

No additional commit description was provided.

**Changed files:** `skills/brainevent/SKILL.md`, `skills/brainpy-state/SKILL.md`

#### docs: refine BrainEvent API structure ([`097d2be`](https://github.com/chaobrain/BrainX-skill/commit/097d2be61ccb0330c47cf3ac776f296fdb034bd7))

No additional commit description was provided.

**Changed files:** `plan.md`, `skills/brainevent/SKILL.md`, `skills/brainevent/references/sparse-formats.md`

#### docs: sharpen BrainEvent neuroscience mappings ([`06138a5`](https://github.com/chaobrain/BrainX-skill/commit/06138a567db0e47f6766250fbbc18e31e46994b8))

No additional commit description was provided.

**Changed files:** `skills/brainevent/SKILL.md`

#### docs: clarify BrainEvent plasticity mechanism ([`4903635`](https://github.com/chaobrain/BrainX-skill/commit/4903635d390733b1f51a3a719054d8bca4f54d2d))

No additional commit description was provided.

**Changed files:** `skills/brainevent/SKILL.md`

#### docs: add AGENTS.md skill-creation rules and CLAUDE.md pointer (#5) ([`90b1a23`](https://github.com/chaobrain/BrainX-skill/commit/90b1a23139337857c8c39831ad22457bb1634f31))

Give agents a concrete, numbered rulebook for authoring BrainX skills,
covering layout, frontmatter, manifest/package registration, the
progressive-disclosure authoring principles, SKILL.md structure, and
verification steps. CLAUDE.md imports it via @AGENTS.md.

**Changed files:** `AGENTS.md`, `CLAUDE.md`

#### docs(brainpy-state): remove unrouted event-driven operators stub (#9) ([`39d271a`](https://github.com/chaobrain/BrainX-skill/commit/39d271a382ed23dcfe31098b7b022f44093d4f24))

references/brainstate-dynamics/brain-dynamics-event-driven-operators.md was
draft scaffolding that shipped: its body is "Should eventually cover",
"Common mistakes to document", and "Placeholder examples" rather than a
reference.

Nothing routed to it, making it the only shipped file under skills/brainpy-state
absent from every routing row. Its own "Used by" block pointed at two files
that have never existed in this repository. Its subject is owned by
skills/brainevent/, which SKILL.md already routes to.

plan.md places this reference under skills/brainstate/, not skills/brainpy-state/,
which is why no brainpy-state routing row ever pointed at it. That planned
brainstate reference is untouched and remains unbuilt.

**Changed files:** `skills/brainpy-state/references/brainstate-dynamics/brain-dynamics-event-driven-operators.md`

#### docs: add BrainCell and reorganize skill references ([`ef3a39d`](https://github.com/chaobrain/BrainX-skill/commit/ef3a39d35edbd7e9b4920a19833ed66f13145354))

No additional commit description was provided.

**Changed files:** `archive/braincell-multicompartment-parent-2026-07-29.md`, `archive/braincell-skill-single-and-multicompartment-2026-07-29.md`, `braintools-references/braintools-cogtask.md`, `braintools-references/braintools-connectivity.md`, `braintools-references/braintools-data-preprocessing.md`, `braintools-references/braintools-input-current.md`, `braintools-references/braintools-metric.md`, `braintools-references/braintools-optimizer.md`, `braintools-references/braintools-parameter-initializer.md`, `braintools-references/braintools-surrogate.md`, `braintools-references/braintools-visualize.md`, `manifest.json`, and 44 more

### Other changes

#### put draft ([`7ed22b3`](https://github.com/chaobrain/BrainX-skill/commit/7ed22b3bb6d0166f11bb6139d6cf98bc26693b04))

No additional commit description was provided.

**Changed files:** `AGENTS.md`, `skills/brainpy-state/NEST-compatible/nest-workflow.md`, `skills/brainpy-state/SKILL.md`, `skills/brainpy-state/references/brainstate-dynamics/brain-dynamics-delay-protocol.md`, `skills/brainpy-state/references/brainstate-dynamics/brain-dynamics-event-driven-operators.md`, `skills/brainpy-state/references/brainstate-dynamics/scripts/training-snn.py`, `skills/brainpy-state/references/braintools-optimizer.md`, `skills/brainx-general-guard/SKILL.md`, `skills/brainx-general-guard/references/brainstate-randomness-reproducibility/advanced-randomness.md`, `skills/brainx-general-guard/references/brainstate-randomness-reproducibility/randomness-and-reproducibility.md`

#### updates ([`44f811e`](https://github.com/chaobrain/BrainX-skill/commit/44f811e8d97edb11356ed382496f566b694f8e57))

No additional commit description was provided.

**Changed files:** `AGENTS.md`, `skills/brainpy-state/SKILL.md`, `skills/brainpy-state/references/component-selection.md`, `skills/brainpy-state/references/projection-patterns.md`, `skills/brainpy-state/references/training-variations.md`

#### chore: stop ignoring AGENTS.md and CLAUDE.md (#6) ([`8173882`](https://github.com/chaobrain/BrainX-skill/commit/8173882d9ddebdc36a3232270ad003ce54d0f2a1))

Both files are now tracked and committed (#5); the ignore rule was
vestigial and only obscured that they're shared, not local-only.

**Changed files:** `.gitignore`

**Full Changelog:** https://github.com/chaobrain/BrainX-skill/compare/v1.0.6...v1.0.8

## 1.0.6 - 2026-07-25

BrainX Skill 1.0.6 records 13 commits merged into `main`.

### Release record

- **Version:** `1.0.6`
- **Previous release:** `e2b723e489a645fab5d42044a1b73ab21f69b38e`
- **Commits included:** 13

### New features

#### feat(skills): expand BrainUnit and BrainState guidance ([`cc434f6`](https://github.com/chaobrain/BrainX-skill/commit/cc434f69150cc49fc8cbb79571a286bccd70f0af))

Improve BrainUnit unit-aware transformations with explicit guidance for JAX transforms, state-aware gradients, custom JVP/VJP rules, and unit-boundary validation.

Rebuild the BrainState core workflow around state roles, environments, exp_euler_step, state-aware transforms, randomness, parameter constraints, diagnostics, and canonical reference routing. Consolidate randomness and diagnostics, move runnable examples into scripts, and remove dynamics and training references now owned by other skills.

Add the main-branch release-note workflow. It increments patch versions, records package, changelog, and latest notes, and publishes detailed commit bodies and changed files only to chaobrain/BrainX-skill.

Verification:
- 39 Node tests passed (35 existing and 4 release tests)
- 19 BrainState Markdown files have valid local links
- 4 Python scripts parse successfully
- workflow YAML and JSON configuration parse successfully

**Changed files:** `.github/scripts/prepare-release.mjs`, `.github/scripts/prepare-release.test.mjs`, `.github/workflows/release-notes.yml`, `CHANGELOG.md`, `README.md`, `RELEASE_NOTES.md`, `plan.md`, `release-notes.config.json`, `skills/brainstate/SKILL.md`, `skills/brainstate/references/brainstate/brainstate-control-flow-patterns.md`, `skills/brainstate/references/brainstate/brainstate-transformed-diagnostics.md`, `skills/brainstate/references/brainstate/parameter-constraints-regularization.md`, and 16 more

#### feat(release): synchronize npm and GitHub publishing ([`ddaa37a`](https://github.com/chaobrain/BrainX-skill/commit/ddaa37a2ae06b96b11d1aee4ee3802f6cc1f7bca))

No additional commit description was provided.

**Changed files:** `.github/scripts/npm-registry.mjs`, `.github/scripts/npm-registry.test.mjs`, `.github/scripts/prepare-release.integration.test.mjs`, `.github/scripts/prepare-release.mjs`, `.github/scripts/prepare-release.test.mjs`, `.github/scripts/release-workflow.test.mjs`, `.github/workflows/release-notes.yml`, `package.json`, `release-notes.config.json`

#### feat(skills): refine BrainState and BrainUnit guidance ([`a9025df`](https://github.com/chaobrain/BrainX-skill/commit/a9025df8c8f961796e500b353a7e611955cfe313))

Condense BrainState reference routing, restructure its reproducible randomness guidance, and identify BrainState and BrainUnit as the central stateful and physical-quantity infrastructure for BrainX modeling and simulation.

**Changed files:** `skills/brainstate/SKILL.md`, `skills/brainunit/SKILL.md`

### Fixes

#### fix(ci): invoke Agent Skills validator through Python ([`e1705e6`](https://github.com/chaobrain/BrainX-skill/commit/e1705e66f3a4009bd5947013d3db16dbac746b25))

No additional commit description was provided.

**Changed files:** `.github/scripts/agent-skills-workflow.test.mjs`, `.github/workflows/agent-skills-validation.yml`

### Documentation

#### docs: add trust and compatibility badges ([`8fde8af`](https://github.com/chaobrain/BrainX-skill/commit/8fde8af1dea8d9f9c56585d46eb7a424b4c73480))

No additional commit description was provided.

**Changed files:** `.github/workflows/agent-skills-validation.yml`, `.github/workflows/scorecard.yml`, `README.md`

#### docs: polish README badge styling ([`75e5320`](https://github.com/chaobrain/BrainX-skill/commit/75e5320452c7ca635ed417946b764e162f5fa502))

No additional commit description was provided.

**Changed files:** `.impeccable.md`, `README.md`

#### docs: add agent-specific install commands ([`9e8ef51`](https://github.com/chaobrain/BrainX-skill/commit/9e8ef51bf3ced7a7567dcd3ca4ba2fcadb06b4e2))

No additional commit description was provided.

**Changed files:** `README.md`

### Other changes

#### refactor: rename installation-code/ to installation/ (#3) ([`44192b6`](https://github.com/chaobrain/BrainX-skill/commit/44192b68e0f5b5c582b9d55ea3d1c039c69f9be9))

- git mv installation-code -> installation (history preserved for all 10 files)
- update package.json bin entry and files[] to the new path
- update AGENTS.md CLI-location reference
- set Apache LICENSE copyright to 2026 BrainX Team

**Changed files:** `AGENTS.md`, `LICENSE`, `installation-code/bin/brainx.js`, `installation-code/lib/adapter-transaction.js`, `installation-code/lib/bundle.js`, `installation-code/lib/cli.js`, `installation-code/lib/constants.js`, `installation-code/lib/hash.js`, `installation-code/lib/installer.js`, `installation-code/lib/paths.js`, `installation-code/lib/prompts.js`, `installation-code/lib/receipt.js`, and 11 more

#### ci: make release-notes workflow manual-only via workflow_dispatch ([`06677b8`](https://github.com/chaobrain/BrainX-skill/commit/06677b8b25953d6cc924d26c1ff69bac72c7d163))

Releases no longer fire on every push/merge to main. The workflow now
runs only when triggered manually from the Actions tab. The head_commit
guard (which existed only to prevent the release commit from
re-triggering the push workflow) is dropped since manual dispatch cannot
recurse; the fork-repository guard is kept.

**Changed files:** `.github/workflows/release-notes.yml`

#### Merge pull request #4 from chaobrain/codex/npm-publish-sync ([`447a47b`](https://github.com/chaobrain/BrainX-skill/commit/447a47b9abe8641211c5e1febd6976b552e6cb50))

Synchronize npm and GitHub releases

#### small update on gitignore ([`7517ac0`](https://github.com/chaobrain/BrainX-skill/commit/7517ac0efb0e3501670429f690ea3697d6e285b2))

No additional commit description was provided.

**Changed files:** `.gitignore`, `.impeccable.md`, `AGENTS.md`

#### chore: ignore local agent instruction files ([`30ef962`](https://github.com/chaobrain/BrainX-skill/commit/30ef96236581ea11b18fa135f15e8ac2bea45eeb))

No additional commit description was provided.

**Changed files:** `.gitignore`, `CLAUDE.md`

#### refine brainstate skill guidance ([`08f3336`](https://github.com/chaobrain/BrainX-skill/commit/08f33364a74758b8bca38843365f1faf1d975b84))

No additional commit description was provided.

**Changed files:** `plan.md`, `skills/brainstate/SKILL.md`, `skills/brainstate/references/simulation-environment.md`, `skills/brainstate/references/size-inference-variations.md`

**Full Changelog:** https://github.com/chaobrain/BrainX-skill/compare/e2b723e489a645fab5d42044a1b73ab21f69b38e...v1.0.6

## 1.0.5 - 2026-07-24

BrainX Skill 1.0.5 records 1 commit merged into `main`.

### Release record

- **Version:** `1.0.5`
- **Previous release:** `v1.0.4`
- **Commits included:** 1

### Other changes

#### refactor: rename installation-code/ to installation/ (#3) ([`44192b6`](https://github.com/chaobrain/BrainX-skill/commit/44192b68e0f5b5c582b9d55ea3d1c039c69f9be9))

- git mv installation-code -> installation (history preserved for all 10 files)
- update package.json bin entry and files[] to the new path
- update AGENTS.md CLI-location reference
- set Apache LICENSE copyright to 2026 BrainX Team

**Changed files:** `AGENTS.md`, `LICENSE`, `installation-code/bin/brainx.js`, `installation-code/lib/adapter-transaction.js`, `installation-code/lib/bundle.js`, `installation-code/lib/cli.js`, `installation-code/lib/constants.js`, `installation-code/lib/hash.js`, `installation-code/lib/installer.js`, `installation-code/lib/paths.js`, `installation-code/lib/prompts.js`, `installation-code/lib/receipt.js`, and 11 more

**Full Changelog:** https://github.com/chaobrain/BrainX-skill/compare/v1.0.4...v1.0.5

## 1.0.4 - 2026-07-24

BrainX Skill 1.0.4 records 1 commit merged into `main`.

### Release record

- **Version:** `1.0.4`
- **Previous release:** `e2b723e489a645fab5d42044a1b73ab21f69b38e`
- **Commits included:** 1

### New features

#### feat(skills): expand BrainUnit and BrainState guidance ([`cc434f6`](https://github.com/chaobrain/BrainX-skill/commit/cc434f69150cc49fc8cbb79571a286bccd70f0af))

Improve BrainUnit unit-aware transformations with explicit guidance for JAX transforms, state-aware gradients, custom JVP/VJP rules, and unit-boundary validation.

Rebuild the BrainState core workflow around state roles, environments, exp_euler_step, state-aware transforms, randomness, parameter constraints, diagnostics, and canonical reference routing. Consolidate randomness and diagnostics, move runnable examples into scripts, and remove dynamics and training references now owned by other skills.

Add the main-branch release-note workflow. It increments patch versions, records package, changelog, and latest notes, and publishes detailed commit bodies and changed files only to chaobrain/BrainX-skill.

Verification:
- 39 Node tests passed (35 existing and 4 release tests)
- 19 BrainState Markdown files have valid local links
- 4 Python scripts parse successfully
- workflow YAML and JSON configuration parse successfully

**Changed files:** `.github/scripts/prepare-release.mjs`, `.github/scripts/prepare-release.test.mjs`, `.github/workflows/release-notes.yml`, `CHANGELOG.md`, `README.md`, `RELEASE_NOTES.md`, `plan.md`, `release-notes.config.json`, `skills/brainstate/SKILL.md`, `skills/brainstate/references/brainstate/brainstate-control-flow-patterns.md`, `skills/brainstate/references/brainstate/brainstate-transformed-diagnostics.md`, `skills/brainstate/references/brainstate/parameter-constraints-regularization.md`, and 16 more

**Full Changelog:** https://github.com/chaobrain/BrainX-skill/compare/e2b723e489a645fab5d42044a1b73ab21f69b38e...v1.0.4

## 1.0.3

Version `1.0.3` is the baseline for automated release tracking. Earlier changes
were released before this changelog was introduced.
