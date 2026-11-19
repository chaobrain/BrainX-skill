BrainX Skill 1.0.17 restores the Codex reviewer on current Codex CLI versions.

## Fixes

- Run the reviewer through Codex App Server. The `codex` and `codex-reply` tools keep the read-only review policy and can resume the same Codex thread after the reviewer restarts.
- Fix Windows MCP installation when npm or Codex is provided as a command shim.
- Validate the installer and run a real Codex CLI MCP smoke test on Windows and Linux in GitHub Actions.

## Upgrade notes

- Use a Codex CLI with App Server stdio support. The reviewer was tested with Codex CLI 0.166.0 and 0.170.0.
- If a reviewer call supplies the legacy `profile` argument, replace it with explicit `model`, `cwd`, or `config` settings as needed.

Changes: [PR #16](https://github.com/chaobrain/BrainX-skill/pull/16) · [v1.0.16 to the verified baseline](https://github.com/chaobrain/BrainX-skill/compare/v1.0.16...88055b75aa3cfcc9ec97cccd023f7b21a30228bc)
