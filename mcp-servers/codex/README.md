# BrainX Codex MCP server

This server preserves the open-source Codex CLI MCP implementation while specializing fresh sessions for independent BrainX scientific review. It delegates execution and persistence to `codex mcp-server`, so `codex-reply` continues the same native thread with its existing context.

The proxy changes only fresh `codex` calls:

- `base-instructions` receives `system-prompt.md`;
- `developer-instructions` identifies the allowed BrainX domain skills and the
  exact training and fitting review-reference paths;
- `skills.config` exposes the skill directories listed in `skills.json`.

The modeling-loop skill is not exposed to the reviewer. The reviewer opens only
`training-workflow.md` and/or `parameter-fitting-workflow.md` when the supplied
`ACTIVE_COVERAGE` requires them.

All `codex-reply` calls and other MCP messages pass through unchanged.

## Requirements

- Node.js 18 or later
- Codex CLI with `codex mcp-server` support
- a working Codex login

## Register

From this repository, register the proxy under the `codex` server name:

```bash
claude mcp remove codex -s user
claude mcp add codex -s user -- node "$PWD/mcp-servers/codex/server.mjs"
```

For a project-scoped Codex host, add the equivalent STDIO server to `.codex/config.toml`:

```toml
[mcp_servers.codex]
command = "node"
args = ["/absolute/path/to/brainx-skill-bundle/mcp-servers/codex/server.mjs"]
default_tools_approval_mode = "approve"
tool_timeout_sec = 1800
```

Set `default_tools_approval_mode = "approve"` when a non-interactive Codex host
must call the reviewer. Without it, Codex cannot present the MCP approval prompt,
terminates the server before dispatching `tools/call`, and may report the denial
as `user cancelled MCP tool call`.

Keep `tool_timeout_sec` above the longest expected review turn. A full artifact
review commonly exceeds Codex's default MCP tool-call budget; when the host
cancels that call, the caller may report `user cancelled MCP tool call` even
though no user cancelled it.

The proxy forces fresh reviews to `sandbox = "read-only"` and
`approval-policy = "never"` before forwarding them upstream. Auto-approving the
MCP tool therefore does not grant the reviewer write access or command approval.

Set `BRAINX_CODEX_BIN` only when `codex` is not on `PATH` for the MCP host.

## Use

Start a review with `mcp__codex__codex`. Its returned `content` is the reviewer
response available to the calling agent in the same turn. The response is a
complete Markdown document; the reviewer remains read-only and does not create
the file. The calling modeling agent saves `content` verbatim as
`reviews/iteration-<N>.md`, records that path and the returned `threadId`, and
uses `codex-reply` only for a follow-up that must continue the reviewer's context.

Edit `system-prompt.md` to change the reviewer contract. Edit `skills.json` to change the BrainX skills exposed to fresh reviewer sessions.

## Verify

```bash
node --test mcp-servers/codex/test/*.test.mjs
```
