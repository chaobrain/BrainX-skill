# Upstream provenance

The original reviewer forwarded the now-removed `codex mcp-server` command.
The current implementation exposes the same MCP tool names through a BrainX
adapter backed by the native Codex App Server stdio protocol.

Verified against CLI 0.166.0 and 0.170.0:

- Official protocol: https://learn.chatgpt.com/docs/app-server
- Removed command: https://learn.chatgpt.com/docs/developer-commands
- Version-specific schema: `codex app-server generate-json-schema --out <directory>`
- Upstream implementation: https://github.com/openai/codex/tree/main/codex-rs/app-server
- License: Apache-2.0, the same license used by this repository

`app-server.mjs` implements JSONL requests and native turn-event handling.
`server.mjs` owns the MCP tool schemas, maps fresh reviews to `thread/start`,
maps continuation to `thread/resume`, and maps review prompts to `turn/start`.
Codex retains responsibility for model execution, thread persistence and
history. The adapter injects BrainX instructions and skill configuration,
enforces read-only/no-approval execution, prevents recursive reviewer startup,
returns completed assistant reports, and forwards cancellation to
`turn/interrupt`. It does not copy or recreate Codex's execution engine.

The repository smoke test checks native-thread behavior against the real App
Server with a local Responses fixture. It verifies persisted-thread resume
after restarting the adapter without calling a hosted model.
