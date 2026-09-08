// Adapted from DeyangLiu123/openalex-mcp (MIT) and
// cyanheads/openalex-mcp-server (Apache-2.0); modified into a location/version
// resolver adapter. See ../../REUSE.md and ../../THIRD_PARTY_NOTICES.md.

import { normalizeDoi, normalizeTitle } from '../identifiers.mjs';
import {
  inferFormat,
  normalizeVersion,
  relationshipFor,
  resolvedDoiFor,
} from '../model.mjs';

function plainText(value = '') {
  return String(value).replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
}

function authorsFromWork(work) {
  if (!Array.isArray(work.authorships)) return [];
  return work.authorships
    .map((authorship) => authorship.author?.display_name)
    .filter(Boolean);
}

function venueFromWork(work) {
  return work.primary_location?.source?.display_name
    || work.primary_location?.raw_source_name
    || '';
}

function openAlexId(work) {
  return String(work.id || '').split('/').pop() || null;
}

function isPmcLocation(location) {
  const sourceName = String(location.source?.display_name || '').toLowerCase();
  const urls = `${location.landing_page_url || ''} ${location.pdf_url || ''}`.toLowerCase();
  return sourceName.includes('pubmed central')
    || sourceName.includes('europe pmc')
    || /\/pmc\/|\/articles\/pmc/.test(urls);
}

function locationKey(location) {
  return location.pdf_url || location.landing_page_url || location.id || '';
}

function allLocations(work) {
  const values = [work.best_oa_location, work.primary_location, ...(work.locations || [])]
    .filter((location) => location && typeof location === 'object');
  const seen = new Set();
  return values.filter((location) => {
    const key = locationKey(location);
    if (!key || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export class OpenAlexSource {
  constructor(http, options = {}) {
    this.http = http;
    this.baseUrl = (options.baseUrl
      ?? process.env.OPENALEX_BASE_URL
      ?? 'https://api.openalex.org').replace(/\/$/, '');
    this.apiKey = options.apiKey ?? process.env.OPENALEX_API_KEY ?? '';
    this.mailto = options.mailto ?? process.env.OPENALEX_MAILTO ?? '';
  }

  params(values = {}) {
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(values)) {
      if (value !== undefined && value !== null && value !== '') params.set(key, String(value));
    }
    if (this.apiKey) params.set('api_key', this.apiKey);
    if (this.mailto) params.set('mailto', this.mailto);
    return params;
  }

  async resolveWork(request, options = {}) {
    if (request.doi) {
      const identifier = `https://doi.org/${normalizeDoi(request.doi)}`;
      return this.http.json(`${this.baseUrl}/works/${encodeURIComponent(identifier)}?${this.params()}`, options);
    }
    if (request.pmid) {
      return this.http.json(`${this.baseUrl}/works/pmid:${encodeURIComponent(request.pmid)}?${this.params()}`, options);
    }
    if (!request.title) return null;

    const params = this.params({ search: request.title, 'per-page': 3 });
    const data = await this.http.json(`${this.baseUrl}/works?${params}`, options);
    const expected = normalizeTitle(request.title);
    return (data.results || []).find((work) => normalizeTitle(work.display_name) === expected) || null;
  }

  async lookup(request, options = {}) {
    let work;
    try {
      work = await this.resolveWork(request, options);
    } catch (error) {
      if (error.status === 404) return { metadata: {}, candidates: [] };
      throw error;
    }
    if (!work) return { metadata: {}, candidates: [] };

    const workDoi = normalizeDoi(work.doi || request.doi);
    // A preprint DOI namespace overrides a conflicting version label from an aggregator.
    const isPreprintDoi = workDoi.startsWith('10.1101/') || workDoi.startsWith('10.48550/arxiv.');
    const metadata = {
      title: plainText(work.display_name),
      authors: authorsFromWork(work),
      doi: workDoi,
      pmid: String(work.ids?.pmid || '').split('/').pop() || request.pmid || '',
      pmcid: request.pmcid || '',
      journal: venueFromWork(work),
      year: Number(work.publication_year) || undefined,
      openalex_id: openAlexId(work),
    };

    const candidates = [];
    for (const locationRecord of allLocations(work)) {
      if (isPmcLocation(locationRecord)) continue;
      const location = {
        landing_page_url: locationRecord.landing_page_url || null,
        pdf_url: locationRecord.pdf_url || null,
        fulltext_url: locationRecord.pdf_url || locationRecord.landing_page_url || null,
      };
      if (!location.fulltext_url) continue;
      const version = isPreprintDoi
        ? 'submittedVersion'
        : normalizeVersion(locationRecord.version, locationRecord);
      // Keep direct repository manuscripts when OpenAlex's aggregate OA flag is stale.
      const directRepositoryManuscript = locationRecord.source?.type === 'repository'
        && Boolean(locationRecord.pdf_url)
        && ['acceptedVersion', 'submittedVersion'].includes(version);
      if (!locationRecord.is_oa && !directRepositoryManuscript) continue;
      const originalFormat = inferFormat(location);
      const resolvedDoi = resolvedDoiFor(location, version === 'submittedVersion' ? '' : workDoi);
      candidates.push({
        identifier: openAlexId(work),
        title: metadata.title || request.title || null,
        authors: metadata.authors,
        requested_doi: request.doi || workDoi || null,
        resolved_doi: resolvedDoi || null,
        pmid: metadata.pmid || null,
        pmcid: null,
        journal: metadata.journal || null,
        year: metadata.year || request.year || null,
        source: 'OpenAlex',
        provider_name: locationRecord.source?.display_name || 'OpenAlex location',
        provider_type: locationRecord.source?.type || null,
        version,
        relationship: relationshipFor(version, request.doi || workDoi, resolvedDoi),
        license: locationRecord.license || null,
        source_location: location.landing_page_url || location.fulltext_url,
        location,
        original_format: originalFormat,
        normalized_format: originalFormat === 'pdf' ? null : 'markdown',
        retrieval: {
          kind: originalFormat === 'jats_xml' ? 'jats' : originalFormat,
          url: location.fulltext_url,
        },
      });
    }
    return { metadata, candidates };
  }
}
