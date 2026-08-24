# Upstream provenance

The linked ARIS repository registers the Codex CLI directly:

```bash
claude mcp add codex -s user -- codex mcp-server
```

This proxy delegates to that same command. Its tool schemas, native thread creation, persistence, event handling, and `codex-reply` continuation remain implemented by OpenAI Codex.

Upstream inspected on 2026-08-23:

- Repository: <https://github.com/openai/codex>
- Revision: `83d1fe0e67b1323f71febc2925817732b449f1d9`
- MCP configuration and schemas: `codex-rs/mcp-server/src/codex_tool_config.rs`
- Session and reply execution: `codex-rs/mcp-server/src/codex_tool_runner.rs`
- JSON-RPC dispatch: `codex-rs/mcp-server/src/message_processor.rs`
- License: Apache-2.0, the same license used by this repository

Modified behavior is confined to `server.mjs`: fresh `codex` tool calls receive the BrainX system prompt and skill configuration before being forwarded upstream.
