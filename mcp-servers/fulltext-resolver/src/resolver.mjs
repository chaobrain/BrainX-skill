import { htmlToDocument, jatsToDocument } from './document.mjs';
import { HttpClient } from './http.mjs';
import { normalizeRequest } from './identifiers.mjs';
import {
  deduplicateCandidates,
  publicCandidate,
  relationshipFor,
  requestedWorkFrom,
  sortCandidates,
} from './model.mjs';
import { BiorxivSource } from './sources/biorxiv.mjs';
import { EuropePmcSource } from './sources/europe-pmc.mjs';
import { OpenAlexSource } from './sources/openalex.mjs';

function errorRecord(source, error) {
  return {
    source,
    message: error instanceof Error ? error.message : String(error),
    status: error?.status ?? null,
    retryable: Boolean(error?.retryable),
    retry_after_seconds: error?.retryAfter ?? null,
  };
}

function mergeMetadata(...records) {
  const result = {};
  for (const record of records) {
    if (!record) continue;
    for (const [key, value] of Object.entries(record)) {
      if (result[key] === undefined || result[key] === null || result[key] === '' || result[key]?.length === 0) {
        if (value !== undefined && value !== null && value !== '' && value?.length !== 0) result[key] = value;
      }
    }
  }
  return result;
}

function enrichRequest(request, metadata) {
  return {
    ...request,
    doi: request.doi || metadata.doi || '',
    pmid: request.pmid || metadata.pmid || '',
    pmcid: request.pmcid || metadata.pmcid || '',
    title: request.title || metadata.title || '',
    authors: request.authors.length ? request.authors : metadata.authors || [],
    year: request.year || metadata.year,
  };
}

function resolutionResult(request, metadata, candidates, sourceErrors) {
  const versions = candidates.map(publicCandidate);
  const resolvedDocument = versions[0] || null;
  return {
    resolved: Boolean(resolvedDocument),
    requested_work: requestedWorkFrom(request, metadata),
    resolved_document: resolvedDocument,
    source: resolvedDocument?.source ?? null,
    version: resolvedDocument?.version ?? null,
    relationship: resolvedDocument?.relationship ?? null,
    format: resolvedDocument?.normalized_format ?? resolvedDocument?.original_format ?? null,
    location: resolvedDocument?.source_location ?? null,
    versions,
    source_errors: sourceErrors,
    unresolved_reason: resolvedDocument ? null : 'No accessible full-text version was found.',
  };
}

export class FullTextResolver {
  constructor(options = {}) {
    this.http = options.http ?? new HttpClient(options.httpOptions);
    this.europePmc = options.europePmc ?? new EuropePmcSource(this.http, options.europePmcOptions);
    this.openAlex = options.openAlex ?? new OpenAlexSource(this.http, options.openAlexOptions);
    this.biorxiv = options.biorxiv ?? new BiorxivSource(this.http, options.biorxivOptions);
    this.cache = new Map();
    this.cacheSize = options.cacheSize ?? 16;
  }

  async resolveInternal(input, options = {}) {
    const request = normalizeRequest(input);
    const sourceErrors = [];
    let europePmcResult = { metadata: {}, candidates: [] };
    try {
      europePmcResult = await this.europePmc.lookup(request, options);
      for (const error of europePmcResult.sourceErrors || []) {
        sourceErrors.push(errorRecord('PMC', error));
      }
    } catch (error) {
      sourceErrors.push(errorRecord('PMC', error));
    }

    const enriched = enrichRequest(request, europePmcResult.metadata);
    const secondary = await Promise.allSettled([
      this.openAlex.lookup(enriched, options),
      this.biorxiv.lookup(enriched, options),
    ]);
    const [openAlexResult, biorxivResult] = secondary.map((result, index) => {
      if (result.status === 'fulfilled') return result.value;
      sourceErrors.push(errorRecord(index === 0 ? 'OpenAlex' : 'bioRxiv/medRxiv', result.reason));
      return { metadata: {}, candidates: [] };
    });

    const metadata = mergeMetadata(
      europePmcResult.metadata,
      openAlexResult.metadata,
      biorxivResult.metadata,
    );
    const candidates = deduplicateCandidates([
      ...europePmcResult.candidates,
      ...openAlexResult.candidates,
      ...biorxivResult.candidates,
    ]);
    return { request, metadata, candidates, sourceErrors };
  }

  async resolve(input, options = {}) {
    const internal = await this.resolveInternal(input, options);
    return resolutionResult(
      internal.request,
      internal.metadata,
      internal.candidates,
      internal.sourceErrors,
    );
  }

  async listVersions(input, options = {}) {
    const result = await this.resolve(input, options);
    return {
      resolved: result.resolved,
      requested_work: result.requested_work,
      versions: result.versions,
      source_errors: result.source_errors,
      unresolved_reason: result.unresolved_reason,
    };
  }

  selectCandidates(candidates, selection = {}) {
    const source = String(selection.source || '').toLowerCase();
    return sortCandidates(candidates).filter((candidate) => {
      if (selection.version && candidate.version !== selection.version) return false;
      if (source && candidate.source.toLowerCase() !== source && candidate.provider_name.toLowerCase() !== source) {
        return false;
      }
      return true;
    });
  }

  remember(key, value) {
    if (this.cache.has(key)) this.cache.delete(key);
    this.cache.set(key, value);
    while (this.cache.size > this.cacheSize) {
      this.cache.delete(this.cache.keys().next().value);
    }
  }

  async fetchCandidate(candidate, options = {}) {
    const retrievals = [candidate.retrieval, ...(candidate.retrieval.alternatives || [])];
    let lastError;
    for (const retrieval of retrievals) {
      try {
        return await this.fetchRetrieval(candidate, retrieval, options);
      } catch (error) {
        lastError = error;
      }
    }
    throw lastError;
  }

  async fetchRetrieval(candidate, retrieval, options = {}) {
    if (retrieval.kind === 'pdf') {
      return {
        title: candidate.title,
        authors: candidate.authors,
        abstract: candidate.abstract ?? null,
        sections: [],
        figure_captions: [],
        tables: [],
        references: [],
        markdown: null,
        content_retrieved: false,
        retrieval_note: 'This accessible version is PDF-only. The resolver preserves the PDF URL but does not parse PDFs in v1.',
        source_url: retrieval.url,
        original_format: 'pdf',
        normalized_format: null,
      };
    }

    const cacheKey = `${retrieval.kind}:${retrieval.url}`;
    if (this.cache.has(cacheKey)) {
      const cached = this.cache.get(cacheKey);
      const document = structuredClone(cached.document);
      if (options.includeRaw) document.raw = cached.raw;
      return document;
    }
    const response = await this.http.text(retrieval.url, {
      signal: options.signal,
      accept: retrieval.kind === 'jats'
        ? 'application/xml,text/xml'
        : 'text/html,application/xhtml+xml',
    });
    let document;
    if (retrieval.kind === 'jats' || /(?:xml|jats)/i.test(response.contentType)) {
      document = jatsToDocument(response.body);
    } else if (/pdf/i.test(response.contentType)) {
      document = {
        title: candidate.title,
        authors: candidate.authors,
        abstract: candidate.abstract ?? null,
        sections: [],
        figure_captions: [],
        tables: [],
        references: [],
        markdown: null,
        content_retrieved: false,
        retrieval_note: 'The source returned a PDF. The resolver preserves its URL but does not parse PDFs in v1.',
        source_url: response.url,
        original_format: 'pdf',
        normalized_format: null,
      };
    } else {
      document = htmlToDocument(response.body, response.url);
    }
    document.content_retrieved = Boolean(document.markdown);
    document.source_url = response.url;
    this.remember(cacheKey, {
      document: structuredClone(document),
      raw: response.body,
    });
    const result = structuredClone(document);
    if (options.includeRaw) result.raw = response.body;
    return result;
  }

  async getFulltext(input, selection = {}, options = {}) {
    const internal = await this.resolveInternal(input, options);
    const candidates = this.selectCandidates(internal.candidates, selection);
    const fetchErrors = [];
    for (const candidate of candidates) {
      try {
        const document = await this.fetchCandidate(candidate, {
          signal: options.signal,
          includeRaw: selection.include_raw,
        });
        if (selection.max_chars && document.markdown?.length > selection.max_chars) {
          const originalCharacterCount = document.markdown.length;
          document.markdown = document.markdown.slice(0, selection.max_chars);
          document.truncated = true;
          document.original_character_count = originalCharacterCount;
        } else {
          document.truncated = false;
        }
        const resolvedCandidate = candidate.version === 'unknown' && document.version
          ? {
            ...candidate,
            version: document.version,
            relationship: candidate.requested_doi || candidate.resolved_doi
              ? relationshipFor(document.version, candidate.requested_doi, candidate.resolved_doi)
              : candidate.relationship,
          }
          : candidate;
        const resolvedCandidates = internal.candidates.map((item) => (
          item === candidate ? resolvedCandidate : item
        ));
        return {
          ...resolutionResult(
            internal.request,
            internal.metadata,
            resolvedCandidates,
            internal.sourceErrors,
          ),
          resolved_document: publicCandidate(resolvedCandidate),
          source: resolvedCandidate.source,
          version: resolvedCandidate.version,
          relationship: resolvedCandidate.relationship,
          format: document.normalized_format || document.original_format,
          location: resolvedCandidate.source_location,
          document,
          fetch_errors: fetchErrors,
        };
      } catch (error) {
        fetchErrors.push(errorRecord(candidate.source, error));
      }
    }

    return {
      ...resolutionResult(
        internal.request,
        internal.metadata,
        internal.candidates,
        internal.sourceErrors,
      ),
      resolved: false,
      resolved_document: null,
      source: null,
      version: null,
      relationship: null,
      format: null,
      location: null,
      document: null,
      fetch_errors: fetchErrors,
      unresolved_reason: candidates.length
        ? 'Accessible versions were identified, but none could be retrieved.'
        : 'No accessible full-text version matched the requested selection.',
    };
  }
}
