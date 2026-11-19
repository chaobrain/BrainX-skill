import { AppServerClient } from '../../app-server.mjs';
import { serve } from '../../server.mjs';
import { fileURLToPath } from 'node:url';
serve(new AppServerClient({ binary: process.execPath,
  prefixArgs: [fileURLToPath(new URL('./fake-codex.mjs', import.meta.url))],
  timeoutMs: Number(process.env.TEST_REVIEW_TIMEOUT_MS || 5000),
}));
