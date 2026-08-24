#!/usr/bin/env bash
set -uo pipefail

repo="/Users/nijiachen/Downloads/brainx-skill-bundle"
case_name="${1:?usage: $0 <case-name> <run-name>}"
run_name="${2:?usage: $0 <case-name> <run-name>}"
case_dir=$(find "$repo/brainx-display-cases" \
  -mindepth 2 \
  -maxdepth 2 \
  -type d \
  -name "$case_name" \
  -print \
  -quit)
test -n "$case_dir" || {
  echo "Unknown refinement case: $case_name" >&2
  exit 1
}
run_dir="$case_dir/$run_name"
prompt_file="$case_dir/prompt.md"
brainx_venv="/Users/nijiachen/Downloads/Brainx testing/.venv-brainx"
eval_model="gpt-5.6-sol"

case "$case_name" in
  04-online-working-memory)
    expected_prompt_bytes="733"
    expected_prompt_sha256="629e6b6f36083f51bf808a1e8c0014a272222b993511097c7068261728e00119"
    ;;
  05-alpha-rhythm)
    expected_prompt_bytes="577"
    expected_prompt_sha256="c6b42ffbed8f7f18ed64b1184e8c68ffb9491e63d8967f0234264075964a3f41"
    ;;
  06-seizure-recruitment)
    expected_prompt_bytes="560"
    expected_prompt_sha256="eb726d96bf1cb1baac11f39113fce2faf2dab096fa97e01f71c89938f13bd139"
    ;;
  08-binocular-rivalry)
    expected_prompt_bytes="718"
    expected_prompt_sha256="b8e98eb70a623270c82a1894d6b8229c9b8ca2f71034ca336433c41c24f477b1"
    ;;
  *)
    echo "Unsupported refinement case: $case_name" >&2
    exit 1
    ;;
esac

stage=$(mktemp -d "${TMPDIR:-/tmp}/brainx-skill-eval.XXXXXX")
stage=$(realpath "$stage")
workspace="$stage/workspace"
isolated_codex_home="$stage/codex-home"
minimal_config="$isolated_codex_home/config.toml"
sandbox_profile="$stage/eval.sb"

test ! -e "$run_dir" || {
  echo "Refusing to overwrite existing run: $run_dir" >&2
  exit 1
}
mkdir -p "$workspace/.agents/skills" "$isolated_codex_home/skills"
cp -R "$repo/skills/." "$isolated_codex_home/skills/"
cp -R "$repo/skills/." "$workspace/.agents/skills/"
cp "$repo/brainx-display-cases/minimal-eval-config.toml" "$minimal_config"

sed \
  -e "s|__REPO__|$repo|g" \
  -e "s|__NORMAL_CODEX_HOME__|/Users/nijiachen/.codex|g" \
  -e "s|__NORMAL_AGENTS_HOME__|/Users/nijiachen/.agents|g" \
  "$repo/brainx-display-cases/refinement-eval.sb.in" > "$sandbox_profile"

find "$(dirname "$stage")" \
  -maxdepth 1 \
  -type d \
  -name 'brainx-skill-*' \
  ! -path "$stage" \
  -print > "$stage/prior-stages.txt"
while IFS= read -r prior_stage; do
      printf '(deny file-read* (subpath "%s"))\n' "$prior_stage"
done < "$stage/prior-stages.txt" >> "$sandbox_profile"

(cd "$workspace" && sandbox-exec -f "$sandbox_profile" test -r "$isolated_codex_home/skills/brainmass/SKILL.md") || {
  echo "Isolated skill snapshot is not readable" >&2
  exit 1
}
if (cd "$workspace" && sandbox-exec -f "$sandbox_profile" test -r "$prompt_file"); then
  echo "Isolation failure: repository is readable inside evaluator sandbox" >&2
  exit 1
fi
if (cd "$workspace" && sandbox-exec -f "$sandbox_profile" test -r "/Users/nijiachen/.codex/auth.json"); then
  echo "Isolation failure: normal Codex home is readable inside evaluator sandbox" >&2
  exit 1
fi
if (cd "$workspace" && sandbox-exec -f "$sandbox_profile" test -r "/Users/nijiachen/.agents/skills/brainmass/SKILL.md"); then
  echo "Isolation failure: normal shared skills are readable inside evaluator sandbox" >&2
  exit 1
fi
while IFS= read -r prior_stage; do
      if (cd "$workspace" && sandbox-exec -f "$sandbox_profile" test -r "$prior_stage"); then
        echo "Isolation failure: prior evaluator stage is readable: $prior_stage" >&2
        exit 1
      fi
done < "$stage/prior-stages.txt"
(cd "$workspace" && sandbox-exec -f "$sandbox_profile" /usr/bin/touch "$workspace/.isolation-write-probe") || {
  echo "Evaluator workspace is not writable inside host sandbox" >&2
  exit 1
}
rm "$workspace/.isolation-write-probe"

if test -d "$case_dir/inputs"; then
  cp -R "$case_dir/inputs/." "$workspace/"
fi

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
(
  cd "$workspace" &&
  env \
    PATH="$brainx_venv/bin:$PATH" \
    CODEX_HOME="$isolated_codex_home" \
    sandbox-exec -f "$sandbox_profile" codex exec \
      --ephemeral \
      --skip-git-repo-check \
      --ignore-rules \
      --sandbox danger-full-access \
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
      2> >(tee "$stage/codex-stderr.log" >&2)
) \
  | tee "$stage/codex-events.jsonl"
agent_status=${PIPESTATUS[0]}
set -e

mkdir "$run_dir"
cp -R "$workspace/." "$run_dir/"
cp "$stage/codex-events.jsonl" "$run_dir/"
cp "$stage/codex-stderr.log" "$run_dir/"
test ! -f "$stage/agent-final.md" || cp "$stage/agent-final.md" "$run_dir/"
{
  printf 'prompt_bytes=%s\n' "$prompt_bytes"
  printf 'prompt_sha256=%s\n' "$prompt_sha256"
  printf 'model=%s\n' "$eval_model"
  printf 'reasoning_effort=xhigh\n'
  printf 'codex_version=%s\n' "$codex_version"
  printf 'host_read_isolation=macos-seatbelt\n'
  printf 'exit_code=%s\n' "$agent_status"
} > "$run_dir/harness-metadata.txt"

printf 'Disposable workspace retained at: %s\n' "$stage"
exit "$agent_status"
