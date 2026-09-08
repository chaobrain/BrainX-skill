#!/usr/bin/env node

import { pathToFileURL } from 'node:url';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import { FullTextResolver } from './src/resolver.mjs';

const workInputSchema = {
  doi: z.string().optional().describe('Published or preprint DOI. Preferred when available.'),
  pmid: z.string().optional().describe('PubMed identifier.'),
  pmcid: z.string().optional().describe('PubMed Central identifier, with or without the PMC prefix.'),
  title: z.string().optional().describe('Exact publication title. Used conservatively when no stronger identifier is available.'),
  authors: z.union([z.string(), z.array(z.string())]).optional().describe('Author names used only to narrow title lookup.'),
  year: z.number().int().min(1600).max(3000).optional().describe('Publication year used only to narrow title lookup.'),
};

const annotations = {
  readOnlyHint: true,
  destructiveHint: false,
  idempotentHint: true,
  openWorldHint: true,
};

function jsonContent(value) {
  return [{ type: 'text', text: JSON.stringify(value, null, 2) }];
}

function fulltextContent(result) {
  const resolved = result.resolved_document;
  const lines = [
    '# Full-text resolution',
    '',
    `- Requested DOI: ${result.requested_work.doi || 'not supplied'}`,
    `- Retrieved identifier: ${resolved?.resolved_doi || resolved?.identifier || 'none'}`,
    `- Source: ${result.source || 'none'}`,
    `- Version: ${result.version || 'none'}`,
    `- Relationship: ${result.relationship || 'none'}`,
    `- Location: ${result.location || 'none'}`,
    '',
  ];
  if (result.document?.markdown) {
    lines.push(result.document.markdown);
  } else {
    lines.push(result.document?.retrieval_note || result.unresolved_reason || 'No full text was retrieved.');
  }
  return [{ type: 'text', text: lines.join('\n') }];
}

export function createMcpServer(options = {}) {
  const resolver = options.resolver ?? new FullTextResolver(options);
  const server = new McpServer({
    name: 'brainx-fulltext-resolver',
    version: '1.0.0',
  });

  server.registerTool('resolve_fulltext', {
    title: 'Resolve full text',
    description: 'Resolve the best accessible version of a biology or neuroscience work through PMC, OpenAlex, then bioRxiv/medRxiv. Returns provenance without downloading the document.',
    inputSchema: workInputSchema,
    annotations,
  }, async (input, extra) => {
    const result = await resolver.resolve(input, { signal: extra.signal });
    return {
      content: jsonContent(result),
      structuredContent: result,
    };
  });

  server.registerTool('list_versions', {
    title: 'List full-text versions',
    description: 'List accessible published, accepted-manuscript, and submitted/preprint versions while preserving source and identifier relationships.',
    inputSchema: workInputSchema,
    annotations,
  }, async (input, extra) => {
    const result = await resolver.listVersions(input, { signal: extra.signal });
    return {
      content: jsonContent(result),
      structuredContent: result,
    };
  });

  server.registerTool('get_fulltext', {
    title: 'Get full text',
    description: 'Resolve and retrieve the best accessible document as Markdown when JATS or HTML is available. PDF-only versions are returned as provenance-preserving source links and are not parsed in v1.',
    inputSchema: {
      ...workInputSchema,
      version: z.enum(['publishedVersion', 'acceptedVersion', 'submittedVersion', 'unknown'])
        .optional()
        .describe('Optional version selector.'),
      source: z.enum(['PMC', 'OpenAlex', 'bioRxiv', 'medRxiv'])
        .optional()
        .describe('Optional source selector.'),
      include_raw: z.boolean().default(false).describe('Include source JATS/HTML in structuredContent.'),
      max_chars: z.number().int().min(1000).max(2_000_000)
        .optional()
        .describe('Optionally truncate returned Markdown to this many characters.'),
    },
    annotations,
  }, async ({ version, source, include_raw, max_chars, ...work }, extra) => {
    const result = await resolver.getFulltext(
      work,
      { version, source, include_raw, max_chars },
      { signal: extra.signal },
    );
    return {
      content: fulltextContent(result),
      structuredContent: result,
      isError: !result.resolved,
    };
  });

  return server;
}

export async function main() {
  const server = createMcpServer();
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

const invokedPath = process.argv[1] ? pathToFileURL(process.argv[1]).href : '';
if (import.meta.url === invokedPath) {
  main().catch((error) => {
    process.stderr.write(`Full-text resolver MCP failed: ${error.stack || error.message}\n`);
    process.exitCode = 1;
  });
}
