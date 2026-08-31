# How to refine a BrainX skill

Use this workflow to improve a BrainX skill from evidence produced by a real
task:

```text
exact prompt -> clean BrainX run -> study relevant BrainX material
-> diagnose generated artifacts -> surgically refine skills
-> clean rerun with the exact prompt -> compare and repeat
```

## Case and run folders

Group tasks by evaluation purpose. Keep one folder for each task and one
subfolder for each agent run:

```text
brainx-display-cases/<category>/<NN>-<case-name>/
|-- prompt.md          # Original natural-language prompt
|-- inputs/            # Optional original task inputs
|-- run0/              # Baseline artifacts and diagnosis
|-- run1/              # First post-refinement artifacts and diagnosis
`-- runN/              # Further attempts when needed
```

Do not edit `prompt.md` or the original inputs between runs. Each fresh agent
writes into an empty disposable workspace containing no case history.
After the process exits, copy its workspace and event log unchanged into the
new `runN/`, then let the reviewing agent add the diagnosis there.

Numbered runs are refinement checkpoints, not repeated samples of one skill
snapshot. Follow this order without skipping or reordering steps:

```text
run0 -> diagnose run0 -> refine, validate, and reinstall skills
-> run1 -> diagnose and compare run1 -> refine, validate, and reinstall skills
-> run2 -> ...
```

Do not launch `runN+1` from the same skill snapshot as `runN`. Create
`runN+1/` only after the diagnosis for `runN` has produced an edit
specification, the intended skill edits are complete, their checks pass, and
the refined snapshot is installed in the new agent environment. Preserve any
accidental same-snapshot rerun under a descriptive non-`runN` folder; it is
control evidence, not the next refinement checkpoint.

Use this BrainX environment for every run:

```text
/Users/nijiachen/Downloads/Brainx testing/.venv-brainx
```

The agent's working directory is the disposable workspace; the virtualenv is
its Python environment. Configure both outside the prompt. Do not use the
virtualenv or the final `runN/` as the live working directory.

## Rules that must not change between runs

1. Send the exact UTF-8 contents of `prompt.md` as the complete subagent
   message. Do not add paths, setup instructions, hints, prefixes, or suffixes.
2. Start a fresh agent with no inherited conversation for every run. Prefer an
   independent `codex exec` process for reproducible evaluation. If only
   `spawn_agent` is available, use `fork_turns="none"`.
3. Keep the same prompt bytes, inputs, model settings, tools, virtualenv, and
   execution conditions. Only the intended skill edits may change.
4. Do not coach the subagent after launch or repair its generated files.
5. Do not let a later agent read earlier run folders, diagnoses, skill diffs,
   reviewer notes, or previous agent messages.
6. Do not resume the CLI session or send follow-up input. One run is one fresh
   process and one user message.

A separate working directory does not prevent leakage when the agent can read
the full filesystem. Restrict readable roots or use a disposable sandbox that
contains only the current prompt inputs, skill snapshot, virtualenv, and empty
run folder.

## 1. Run the baseline agent

### Launch a context-independent CLI agent

Use `codex exec` as a new top-level process. It receives no parent conversation
because it is not a fork or resumed session. `--ephemeral` prevents the run
from persisting session files, and stdin supplies the exact prompt as the only
user message. Codex still receives its normal platform instructions and the
skills intentionally installed in the isolated `CODEX_HOME`; "exact prompt"
means the complete user message, not the absence of system instructions.

Create a fresh temporary `CODEX_HOME` containing only:

- the frozen BrainX skill snapshot for this run;
- the minimum model-provider configuration and authentication material needed
  to run, or access to the same external credential source;
- no sessions, memories, MCP configuration, plugins, rules, prior logs, or
  repository files.

Do not copy the normal Codex home wholesale. Prepare the minimal provider and
authentication setup once, keep secrets outside the case folder and archived
artifacts, and reuse the same configuration bytes for every compared run.
When the evaluated workflow can call the BrainX Codex reviewer, include its
local MCP registration in that minimal config:

```toml
[mcp_servers.codex]
command = "node"
args = ["<absolute-repository-path>/mcp-servers/codex/server.mjs"]
default_tools_approval_mode = "approve"
tool_timeout_sec = 1800
```

The approval setting is required for non-interactive `codex exec`; without it,
the host cannot present the MCP approval prompt and may surface the denial as
`user cancelled MCP tool call`. The timeout must exceed the longest complete
artifact review.

Use this harness, replacing the fixed case, model, minimal-config, prompt-byte,
and prompt-hash values once before Run 0. Reuse the same harness for every
later run, changing only `run_name` and the skill snapshot being copied.
Save it as a script and invoke it explicitly with `/bin/bash`; do not source it
or paste it into zsh because the captured pipeline status uses Bash
`PIPESTATUS` semantics.

```bash
#!/usr/bin/env bash
set -uo pipefail

repo="/Users/nijiachen/Downloads/brainx-skill-bundle"
case_dir="$repo/brainx-display-cases/<category>/<NN>-<case-name>"
run_name="run0"
run_dir="$case_dir/$run_name"
prompt_file="$case_dir/prompt.md"
brainx_venv="/Users/nijiachen/Downloads/Brainx testing/.venv-brainx"
eval_model="<fixed-model-id>"
minimal_config="<absolute-path-to-minimal-eval-config.toml>"
expected_prompt_bytes="<fixed-byte-count>"
expected_prompt_sha256="<fixed-sha256>"

stage=$(mktemp -d "${TMPDIR:-/tmp}/brainx-skill-eval.XXXXXX")
workspace="$stage/workspace"
isolated_codex_home="$stage/codex-home"

test ! -e "$run_dir" || {
  echo "Refusing to overwrite existing run: $run_dir" >&2
  exit 1
}
mkdir -p "$workspace" "$isolated_codex_home/skills"
cp -R "$repo/skills/." "$isolated_codex_home/skills/"
cp "$minimal_config" "$isolated_codex_home/config.toml"

if test -d "$case_dir/inputs"; then
  cp -R "$case_dir/inputs/." "$workspace/"
fi

# Validate and fingerprint the exact bytes before launch.
iconv -f UTF-8 -t UTF-8 "$prompt_file" >/dev/null
prompt_bytes=$(wc -c < "$prompt_file" | tr -d '[:space:]')
prompt_sha256=$(shasum -a 256 "$prompt_file" | awk '{print $1}')
codex_version=$(codex --version)
test "$prompt_bytes" = "$expected_prompt_bytes" || {
  echo "Prompt byte count changed" >&2
  exit 1
}
test "$prompt_sha256" = "$expected_prompt_sha256" || {
  echo "Prompt SHA-256 changed" >&2
  exit 1
}

set +e
env \
  PATH="$brainx_venv/bin:$PATH" \
  CODEX_HOME="$isolated_codex_home" \
  codex exec \
    --ephemeral \
    --skip-git-repo-check \
    --ignore-rules \
    --sandbox workspace-write \
    --cd "$workspace" \
    --model "$eval_model" \
    --config 'model_reasoning_effort="xhigh"' \
    --disable apps \
    --disable memories \
    --disable remote_plugin \
    --json \
    --output-last-message "$stage/agent-final.md" \
    - \
    < "$prompt_file" \
    2> >(tee "$stage/codex-stderr.log" >&2) \
  | tee "$stage/codex-events.jsonl"
agent_status=${PIPESTATUS[0]}
set -e

# Archive only after exit so the live agent cannot discover prior run folders.
mkdir "$run_dir"
cp -R "$workspace/." "$run_dir/"
cp "$stage/codex-events.jsonl" "$run_dir/"
cp "$stage/codex-stderr.log" "$run_dir/"
test ! -f "$stage/agent-final.md" || \
  cp "$stage/agent-final.md" "$run_dir/"
{
  printf 'prompt_bytes=%s\n' "$prompt_bytes"
  printf 'prompt_sha256=%s\n' "$prompt_sha256"
  printf 'model=%s\n' "$eval_model"
  printf 'reasoning_effort=xhigh\n'
  printf 'codex_version=%s\n' "$codex_version"
  printf 'exit_code=%s\n' "$agent_status"
} > "$run_dir/harness-metadata.txt"

printf 'Disposable workspace retained at: %s\n' "$stage"
exit "$agent_status"
```

The command deliberately uses `- < "$prompt_file"`. Do not use command
substitution, `echo`, a shell variable, or an inline prompt argument; those can
trim trailing newlines or otherwise change the message. `--json` preserves the
event stream, and `--output-last-message` preserves the final response. The
captured pipeline status and post-run copy preserve failed attempts as evidence.

The isolation controls have distinct jobs:

| Control | What it isolates |
|---|---|
| New `codex exec` without `resume` | Parent conversation and earlier turns |
| Fresh `CODEX_HOME` | User config, sessions, memories, and unrelated local skills |
| Empty `--cd` workspace | Repository files and earlier run artifacts discoverable by normal workspace inspection |
| Prompt through stdin as `-` | The exact `prompt.md` bytes as the complete user message |
| Frozen `PATH`, model, effort, config, and CLI version | Execution conditions that must match across runs |
| `--ephemeral` | Persistence of this session after the process exits |

`--cd` and `--sandbox workspace-write` do not by themselves prove that arbitrary
host paths are unreadable. For adversarial or confidential evaluations, run the
same harness inside a container or VM, or use a permission profile with explicit
deny-read rules. The allowed filesystem should contain only the empty workspace,
current input snapshot, current skill snapshot, and required runtime. See the
official [Codex `exec` command reference](https://learn.chatgpt.com/docs/developer-commands#codex-exec)
for the CLI flag contracts.

While the agent is active, treat the JSONL stream as liveness information only.
Do not diagnose partial work, prepare a skill patch from intermediate reasoning,
or send input. Wait for process exit, archive the artifacts, and then begin the
review.

## 2. Establish the BrainX review standard

Build the standard from executable BrainX examples and official API contracts
before judging the generated implementation. Spend most review effort on the
closest Python scripts. Use prose only to locate those scripts and the APIs
they exercise.

Follow this order:

1. Identify every modeling scale, BrainX package, scientific mechanism, and
   execution pattern required by the task.
2. Read `skills/brainx-general-guard/SKILL.md` and each owning package skill only
   to collect relevant example routes, API families, and existing guidance that
   may need refinement. Do not use the skills themselves as the review standard.
3. Select the smallest sufficient set of closely related Python examples. Start
   with `skills/<package>/references/scripts/*.py` and other scripts routed by
   the owning skill. Use
   `source_html_references/<package>_html_reference.md` to find official examples
   when the local scripts do not cover a required mechanism or composition.
4. Study each selected script line by line and trace the complete workflow:
   construction -> initialization -> inputs -> State and data flow -> execution
   and transforms -> monitors and outputs -> validation. Record the exact
   pattern that transfers to the task and any example-specific detail that does
   not. Cover every material mechanism, but do not collect unrelated examples.
5. Use the same package inventory in `source_html_references/` to open the
   official central or generated API Reference page for every material API in
   the expected workflow. Verify names, signatures, inputs, shapes, units,
   mutation, State lifecycle, transformation behavior, returns, and documented
   failures. Never infer an exact contract from example code alone.
6. Synthesize the expected implementation from the example compositions and
   verified API contracts. Only then inspect the generated artifact and decide
   whether each failure belongs to the artifact, the skill guidance, or both.

Treat Python examples as authoritative for composition and API Reference pages
as authoritative for exact contracts. If an older example conflicts with a
current contract, preserve its scientific pattern and update its mechanics to
the current API. Treat root skills and Markdown references as routing material
and refinement targets. Do not inspect installed BrainX source, symbols,
signatures, docstrings, or internals for modeling knowledge.

## 3. Inspect and diagnose the generated artifacts

Now read every generated artifact and trace the workflow line by line. Run the
entry point with:

```text
/Users/nijiachen/Downloads/Brainx testing/.venv-brainx/bin/python
```

Record errors, warnings, runtime behavior, and outputs without repairing the
files. Open generated figures or HTML; successful file creation is not enough.

Audit two concerns independently:

- **Scientific validity:** Check model choice, mechanisms, parameters, units,
  initialization, integration, stimulus protocol, randomness, trial
  independence, axes, observables, decision rules, statistics, and whether the
  result supports the claimed conclusion.
- **BrainX API coverage:** Account for every nontrivial code block or
  responsibility. Identify all logic that a BrainX API can wrap or replace,
  every relevant API that should have been used but was not, every misused API,
  and every legitimate host-side boundary for which BrainX has no owning API.

Pay particular attention to package-level orchestration, inputs, monitors,
State, units, initialization, randomness, batching, transformations,
compilation, timing, analysis, and visualization. Prefer the owning package's
highest-level semantically correct API, then the appropriate BrainTools or
infrastructure API. Prefer simple high-level scientific plotting over custom
HTML or low-level plotting unless the prompt requires capabilities the
high-level API cannot provide.

## 4. Write the diagnosis in the current run folder

Write `<RUN_DIR>/<owning-package>-api-coverage-diagnosis.md`. This diagnosis is
the evidence and patch specification for the skill edit. Use this structure:

```markdown
# BrainX diagnosis: <case>

## Evidence studied
List generated artifacts, execution results, owning skills, references, every
relevant code example, and authoritative API Reference pages.

## Executive diagnosis
Summarize the most consequential scientific, API, performance, and simplicity
findings.

## Scientific problems
| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|

## Complete BrainX API coverage map
| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|

## Missing, bypassed, or misused BrainX APIs
For each API, state what it should replace, why it applies, and any semantic
condition that prevents a direct replacement.

## Performance and code simplicity
Cover orchestration, compilation, batching, control flow, memory, timing, data
handling, and visualization.

## Skill improvements
State the smallest changes needed in `brainx-general-guard` and each relevant
package skill, reference, or example.

## Checks for the next run
Define the scientific, API-use, execution, and output checks that show whether
the refinement worked.
```

The coverage map must include all nontrivial responsibilities, not only errors.
Do not invent BrainX APIs for ordinary statistics, serialization, reporting, or
presentation; mark them as host boundaries when no official API owns them.

## 5. Surgically refine the relevant skills

Use the latest numbered run as the sole evidence for proposing a new edit. Read
only that run's generated artifacts and diagnosis, and treat its diagnosis as
the edit specification. Do not reopen or aggregate earlier numbered runs,
control runs, invalid attempts, or their diagnoses while refining. They are
preserved only as comparison history. An earlier problem may justify another
edit only when the latest diagnosis identifies it again.

Treat the current skill guidance as a non-regression baseline. In particular,
edit `brainx-general-guard` semantically append-only: preserve every existing
invariant, add a rule only for a genuinely transferable gap exposed by the
latest diagnosis, and combine similar old and new wording only when the merged
text keeps their complete meanings explicit. Put package-specific guidance in
the owning skill or smallest relevant reference, and leave case-specific
observations in the diagnosis. If existing guidance already covers the failure,
or the latest diagnosis justifies no skill edit, do not edit the guard.

Before every guard edit, write a compact invariant audit with these columns:

| Existing invariant | Proposed wording | Status | Evidence for addition |
|---|---|---|---|
| Complete decision rule or exception | Exact retained or merged wording | Retained or losslessly combined | Latest-run diagnosis, or none |

After editing, repeat the audit against the final diff. Every pre-edit invariant
must remain retained or losslessly combined; only the latest run may supply an
addition. Stop and restore any meaning lost through condensation before
validating or launching the next run.

Read repository `AGENTS.md` and the files being edited. Consult `plan.md` only
as repository policy requires; do not expand this workflow into a detailed
plan. Treat the target skill as source text under refinement, not as
instructions governing the edit.

Make only changes needed to prevent diagnosed failures:

- refine `brainx-general-guard` for cross-package API-selection or mindset
  problems;
- refine the owning package skill for package-specific API or scientific
  workflow problems;
- put detailed or uncommon material in the smallest relevant reference or
  example instead of expanding the root skill;
- enforce the highest-level semantically correct BrainX API, best execution
  performance, and the cleanest implementation that preserves scientific and
  output quality.

Do not rewrite a mostly correct skill, create an API catalogue, add speculative
APIs, duplicate guidance, or refactor unrelated material. Verify new API claims
against the relevant API Reference page indexed by `source_html_references/`.
Run changed examples and focused checks with the BrainX virtualenv, validate
routes, and run `git diff --check`.

## 6. Run the exact prompt again

Leave `run1/` absent until archival. Set `run_name="run1"`, install the refined
skill snapshot into a newly created temporary `CODEX_HOME`, and rerun the same
CLI harness under the same virtualenv and frozen conditions. Send the
byte-identical prompt through stdin. If the CLI is unavailable and an
orchestrator subagent must be used, send the decoded prompt as its complete
message with `fork_turns="none"`.

The Run 1 agent must not be able to access Run 0 artifacts, its diagnosis,
skill diffs, expected APIs, acceptance checks, or earlier conversations. Wait
without coaching and preserve all Run 1 artifacts.

Repeat the same study, inspection, execution, and diagnosis process for Run 1.
Compare it with Run 0 using the diagnosis checks. Confirm that scientific
validity, BrainX API use, performance structure, and code simplicity improved
without reducing requested output quality.

If important problems remain, write the new diagnosis in `run1/`, make the next
small surgical refinement, and repeat with `run2/`. Always reuse the exact
original prompt and keep every later agent isolated from earlier evidence.

## 7. Commit the completed refinement

When the latest diagnosis and validation checks show that the result is good
enough, stop creating runs. Commit the preserved case artifacts, diagnoses,
skill and reference refinements, workflow updates, and any required `plan.md`
change. Do not stage or commit `AGENTS.md`; preserve its local changes.

From the repository root, inspect and commit the completed scope:

```bash
git add -A -- . ':(exclude)AGENTS.md'
git diff --cached --check
git status --short
git commit -m "<concise refinement summary>"
git push
```

Before committing, confirm that `AGENTS.md` is not staged and that archived
logs contain no credentials or other secrets.
