# BrainX Codex MCP server

This adapter exposes the `codex` and `codex-reply` MCP tools for independent
BrainX scientific review and runs them through `codex app-server --listen
stdio://`. Codex owns native thread history and persistence; the adapter returns
the completed Markdown report and native thread ID to the MCP caller.

Use `codex` for a fresh thread and `codex-reply` for `thread/resume` followed by
another turn in that thread, including after the adapter restarts. Both paths
enforce the BrainX system prompt, allowed skills, `read-only` sandbox and
`never` approval policy. Caller settings cannot replace these constraints.
The reviewer disables the MCP registrations in the effective workspace
configuration to prevent recursive startup. Tool callers cannot add MCP
registrations through `config`. When the host uses a custom `CODEX_HOME` or
environment credentials, list those variable names in the MCP registration's
`env_vars`; the Codex MCP host filters unlisted environment variables.

`system-prompt.md` supplies the scientific review contract. `skills.json` selects
the exposed domain skills. The developer instructions identify the exact
training and fitting review references; the modeling-loop workflow itself is
not exposed as a reviewer skill.

The adapter waits for native turn completion, excludes commentary from the
report, surfaces failed/interrupted/empty turns as MCP tool errors, and
translates MCP cancellation to `turn/interrupt`. Interactive approval requests
are declined; unsupported interactive requests return protocol errors.

## Requirements

- Node.js 18 or later
- Codex CLI with App Server stdio support (tested on 0.166.0 and 0.170.0)
- a working Codex login

## Register

Install and register this reviewer together with Europe PMC search and the Full-Text Resolver:

```bash
npx brainx-skill mcp install
```

The installer copies a durable reviewer bundle and its required BrainX skills under `~/.brainx/mcp`, registers it as `codex`, and refuses to overwrite an unrelated registration. An exact reviewer registration from this repository is retained so its existing approval and timeout settings remain intact. Restart Codex or reload MCP configuration after installation. Use `npx brainx-skill mcp remove` to remove the managed registrations and runtime.

For repository development or another MCP host, register the proxy manually under the `codex` server name:

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

The adapter forces fresh and resumed reviews to `sandbox = "read-only"` and
`approvalPolicy = "never"`. Auto-approving the MCP tool does not grant the
reviewer write access or command approval.

Set `BRAINX_CODEX_BIN` only when `codex` is not on `PATH` for the MCP host.
On Windows, npm `.cmd`/`.ps1` shims resolve to the package JavaScript entry and
run through Node without a shell. Custom wrappers should expose an executable.

## Use

Start a review with `mcp__codex__codex`. Its returned `content` is the reviewer
response available to the calling agent in the same turn. The response is a
complete Markdown document; the reviewer remains read-only and does not create
the file. The calling modeling agent saves `content` verbatim as
`reviews/iteration-<N>.md`, records that path and `structuredContent.threadId`, and
uses `codex-reply` only for a follow-up that must continue the reviewer's context.

The successful tool result contains `content: [{type: "text", text: report}]`
and `structuredContent: {threadId, content: report}`. `codex-reply` accepts
`{threadId, prompt}`. Fresh calls accept `prompt`, optional `model`, `cwd`, and
`config`; the former `profile` option now returns an explicit unsupported-option
error. Review constraints always override legacy instruction/permission options.

Edit `system-prompt.md` to change the reviewer contract. Edit `skills.json` to change the BrainX skills exposed to fresh reviewer sessions.

## Verify

```bash
npm test
node .github/scripts/smoke-reviewer.mjs
```

The transport is dependency-free and uses JSONL stdio on both sides. MCP
`initialize` initializes the real App Server before advertising the tools.
The smoke test uses a real Codex CLI, an isolated Codex home and a loopback
Responses fixture. It verifies the handshake, a fresh review, and continued
history after restarting the reviewer. Run `npm install` first. Pass a path to
an extracted package's `server.mjs` to verify the shipped files.
