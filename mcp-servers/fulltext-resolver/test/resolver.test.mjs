import assert from 'node:assert/strict';
import test from 'node:test';
import { FullTextResolver } from '../src/resolver.mjs';

function candidate(overrides = {}) {
  const source = overrides.source || 'OpenAlex';
  const version = overrides.version || 'acceptedVersion';
  const url = overrides.url || 'https://repository.example/article';
  return {
    identifier: overrides.identifier || 'W1',
    title: 'Resolved paper',
    authors: ['Researcher A'],
    requested_doi: '10.1000/requested',
    resolved_doi: overrides.resolved_doi ?? '10.1000/requested',
    pmid: null,
    pmcid: overrides.pmcid || null,
    journal: 'Test Journal',
    year: 2026,
    source,
    provider_name: overrides.provider_name || source,
    version,
    relationship: overrides.relationship || 'same_work',
    license: 'cc-by',
    source_location: url,
    location: { landing_page_url: url, fulltext_url: url, pdf_url: null },
    original_format: overrides.original_format || 'html',
    normalized_format: overrides.original_format === 'pdf' ? null : 'markdown',
    retrieval: { kind: overrides.kind || 'html', url, alternatives: overrides.alternatives || [] },
  };
}

class StaticSource {
  constructor(result, error) {
    this.result = result || { metadata: {}, candidates: [] };
    this.error = error;
  }

  async lookup() {
    if (this.error) throw this.error;
    return this.result;
  }
}

function resolverWith({ pmc = [], openalex = [], biorxiv = [], http } = {}) {
  return new FullTextResolver({
    europePmc: new StaticSource({
      metadata: { title: 'Resolved paper', doi: '10.1000/requested', pmid: '42' },
      candidates: pmc,
    }),
    openAlex: new StaticSource({ metadata: {}, candidates: openalex }),
    biorxiv: new StaticSource({ metadata: {}, candidates: biorxiv }),
    http: http || { text: async () => ({ body: '<html><article><h1>Text</h1><p>Body.</p></article></html>', contentType: 'text/html', url: 'https://repository.example/article' }) },
  });
}

test('prefers published PMC over accepted and submitted versions', async () => {
  const pmc = candidate({
    source: 'PMC',
    provider_name: 'Europe PMC',
    version: 'publishedVersion',
    original_format: 'jats_xml',
    kind: 'jats',
    pmcid: 'PMC123',
    url: 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC123/fullTextXML',
  });
  const accepted = candidate({ version: 'acceptedVersion' });
  const submitted = candidate({
    source: 'bioRxiv',
    version: 'submittedVersion',
    relationship: 'preprint_of_requested_work',
    resolved_doi: '10.1101/example',
    url: 'https://www.biorxiv.org/content/10.1101/examplev1.full',
  });
  const resolver = resolverWith({ pmc: [pmc], openalex: [accepted], biorxiv: [submitted] });
  const result = await resolver.resolve({ doi: '10.1000/requested' });
  assert.equal(result.source, 'PMC');
  assert.equal(result.version, 'publishedVersion');
  assert.deepEqual(result.versions.map((version) => version.version), [
    'publishedVersion',
    'acceptedVersion',
    'submittedVersion',
  ]);
});

test('prefers authoritative preprint JATS over an OpenAlex PDF for the same version', async () => {
  const openAlexPdf = candidate({
    source: 'OpenAlex',
    version: 'submittedVersion',
    original_format: 'pdf',
    kind: 'pdf',
    url: 'https://repository.example/preprint.pdf',
  });
  const biorxivJats = candidate({
    source: 'bioRxiv',
    version: 'submittedVersion',
    original_format: 'jats_xml',
    kind: 'jats',
    url: 'https://www.biorxiv.org/content/example.source.xml',
  });
  const result = await resolverWith({ openalex: [openAlexPdf], biorxiv: [biorxivJats] })
    .resolve({ doi: '10.1000/requested' });
  assert.equal(result.source, 'bioRxiv');
});

test('returns an accepted manuscript when no published copy is accessible', async () => {
  const resolver = resolverWith({ openalex: [candidate({ version: 'acceptedVersion' })] });
  const result = await resolver.resolve({ doi: '10.1000/requested' });
  assert.equal(result.resolved, true);
  assert.equal(result.version, 'acceptedVersion');
});

test('preserves preprint provenance and identifiers', async () => {
  const preprint = candidate({
    source: 'medRxiv',
    version: 'submittedVersion',
    relationship: 'preprint_of_requested_work',
    resolved_doi: '10.1101/2026.01.01.123456',
  });
  const resolver = resolverWith({ biorxiv: [preprint] });
  const result = await resolver.resolve({ doi: '10.1000/requested' });
  assert.equal(result.relationship, 'preprint_of_requested_work');
  assert.equal(result.resolved_document.requested_doi, '10.1000/requested');
  assert.equal(result.resolved_document.resolved_doi, '10.1101/2026.01.01.123456');
});

test('returns a structured unresolved result without fabricated content', async () => {
  const result = await resolverWith().getFulltext({ doi: '10.1000/requested' });
  assert.equal(result.resolved, false);
  assert.equal(result.document, null);
  assert.match(result.unresolved_reason, /No accessible full-text version/);
});

test('continues when one source fails', async () => {
  const transient = Object.assign(new Error('OpenAlex unavailable'), { retryable: true, status: 503 });
  const resolver = new FullTextResolver({
    europePmc: new StaticSource({ metadata: { doi: '10.1000/requested' }, candidates: [] }),
    openAlex: new StaticSource(null, transient),
    biorxiv: new StaticSource({ metadata: {}, candidates: [candidate({ source: 'bioRxiv', version: 'submittedVersion' })] }),
    http: { text: async () => { throw new Error('not used'); } },
  });
  const result = await resolver.resolve({ doi: '10.1000/requested' });
  assert.equal(result.resolved, true);
  assert.equal(result.source_errors[0].source, 'OpenAlex');
  assert.equal(result.source_errors[0].retryable, true);
});

test('retrieves HTML Markdown and falls back after a failed higher-ranked location', async () => {
  const primary = candidate({ source: 'PMC', version: 'publishedVersion', kind: 'jats', original_format: 'jats_xml', url: 'https://example.org/fail.xml' });
  const fallback = candidate({ source: 'OpenAlex', version: 'acceptedVersion', url: 'https://repository.example/article' });
  const http = {
    async text(url) {
      if (url.includes('fail')) throw new Error('origin unavailable');
      return {
        body: '<html><article><h1>Repository manuscript</h1><p>Usable text.</p></article></html>',
        contentType: 'text/html',
        url,
      };
    },
  };
  const result = await resolverWith({ pmc: [primary], openalex: [fallback], http })
    .getFulltext({ doi: '10.1000/requested' });
  assert.equal(result.source, 'OpenAlex');
  assert.match(result.document.markdown, /Usable text/);
  assert.equal(result.fetch_errors.length, 1);
});

test('falls back from unavailable Europe PMC JATS to official PMC HTML', async () => {
  const jatsUrl = 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC123/fullTextXML';
  const htmlUrl = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC123/';
  const manuscript = candidate({
    source: 'PMC',
    provider_name: 'Europe PMC',
    version: 'acceptedVersion',
    original_format: 'jats_xml',
    kind: 'jats',
    pmcid: 'PMC123',
    url: jatsUrl,
    alternatives: [{ kind: 'html', url: htmlUrl }],
  });
  const calls = [];
  const http = {
    async text(url) {
      calls.push(url);
      if (url === jatsUrl) throw Object.assign(new Error('JATS unavailable'), { status: 404 });
      return {
        body: '<html><article><h1>Author manuscript</h1><p>Accepted full text.</p></article></html>',
        contentType: 'text/html',
        url,
      };
    },
  };
  const result = await resolverWith({ pmc: [manuscript], http }).getFulltext({ doi: '10.1000/requested' });

  assert.deepEqual(calls, [jatsUrl, htmlUrl]);
  assert.equal(result.version, 'acceptedVersion');
  assert.match(result.document.markdown, /Accepted full text/);
});

test('keeps PDF-only documents as links instead of claiming Markdown extraction', async () => {
  const pdf = candidate({ original_format: 'pdf', kind: 'pdf', url: 'https://publisher.example/paper.pdf' });
  const result = await resolverWith({ openalex: [pdf] }).getFulltext({ doi: '10.1000/requested' });
  assert.equal(result.resolved, true);
  assert.equal(result.document.content_retrieved, false);
  assert.equal(result.document.markdown, null);
  assert.match(result.document.retrieval_note, /does not parse PDFs/);
});

test('cached retrievals include raw source only when each call requests it', async () => {
  const body = '<html><article><h1>Cached text</h1><p>Raw marker.</p></article></html>';
  for (const order of [[false, true], [true, false]]) {
    let calls = 0;
    const http = {
      async text(url) {
        calls += 1;
        return { body, contentType: 'text/html', url };
      },
    };
    const resolver = resolverWith({ openalex: [candidate()], http });
    const first = await resolver.getFulltext({ doi: '10.1000/requested' }, { include_raw: order[0] });
    const second = await resolver.getFulltext({ doi: '10.1000/requested' }, { include_raw: order[1] });

    assert.equal(calls, 1);
    assert.equal(Object.hasOwn(first.document, 'raw'), order[0]);
    assert.equal(Object.hasOwn(second.document, 'raw'), order[1]);
    if (order[0]) assert.equal(first.document.raw, body);
    if (order[1]) assert.equal(second.document.raw, body);
  }
});

test('truncating a retrieval does not mutate the cached document', async () => {
  const body = `<html><article><h1>Long text</h1><p>${'A'.repeat(200)}</p></article></html>`;
  let calls = 0;
  const http = {
    async text(url) {
      calls += 1;
      return { body, contentType: 'text/html', url };
    },
  };
  const resolver = resolverWith({ openalex: [candidate()], http });
  const truncated = await resolver.getFulltext({ doi: '10.1000/requested' }, { max_chars: 40 });
  const complete = await resolver.getFulltext({ doi: '10.1000/requested' });

  assert.equal(truncated.document.markdown.length, 40);
  assert.equal(truncated.document.truncated, true);
  assert.ok(truncated.document.original_character_count > 40);
  assert.equal(complete.document.truncated, false);
  assert.ok(complete.document.markdown.length > 40);
  assert.equal(calls, 1);
});

test('uses JATS metadata to refine an unknown candidate version', async () => {
  const manuscript = candidate({
    source: 'PMC',
    provider_name: 'Europe PMC',
    version: 'unknown',
    original_format: 'jats_xml',
    kind: 'jats',
    pmcid: 'PMC123',
    url: 'https://example.org/PMC123.xml',
  });
  const xml = '<article><front><article-meta><title-group><article-title>Manuscript</article-title></title-group><custom-meta-group><custom-meta><meta-name>is-manuscript</meta-name><meta-value>yes</meta-value></custom-meta></custom-meta-group></article-meta></front><body><p>Full text.</p></body></article>';
  const http = {
    async text(url) {
      return { body: xml, contentType: 'application/xml', url };
    },
  };
  const result = await resolverWith({ pmc: [manuscript], http })
    .getFulltext({ doi: '10.1000/requested' });

  assert.equal(result.version, 'acceptedVersion');
  assert.equal(result.resolved_document.version, 'acceptedVersion');
  assert.equal(result.versions[0].version, 'acceptedVersion');
  assert.equal(result.document.version, 'acceptedVersion');
});

test('clears resolution metadata when every candidate retrieval fails', async () => {
  const unavailable = candidate({ source: 'PMC', version: 'publishedVersion' });
  const resolver = resolverWith({
    pmc: [unavailable],
    http: { text: async () => { throw new Error('origin unavailable'); } },
  });
  const result = await resolver.getFulltext({ doi: '10.1000/requested' });

  assert.equal(result.resolved, false);
  for (const field of ['resolved_document', 'source', 'version', 'relationship', 'format', 'location', 'document']) assert.equal(result[field], null);
  assert.equal(result.fetch_errors.length, 1);
});
