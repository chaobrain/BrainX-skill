// Adapted from CoChatAI/europepmc-mcp-server server.py (MIT); modified into a
// resolver adapter with version provenance. See ../../REUSE.md.

import { normalizeTitle } from '../identifiers.mjs';
import { inferFormat, normalizeVersion, relationshipFor } from '../model.mjs';

function escapeLucene(value) {
  return String(value).replace(/([+\-&|!(){}[\]^"~*?:\\/])/g, '\\$1');
}

// Europe PMC embeds inline tags such as <i> in otherwise plain metadata titles.
function plainText(value = '') {
  return String(value).replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
}

function authorsFromRecord(record) {
  const authors = record.authorList?.author;
  if (Array.isArray(authors)) {
    return authors.map((author) => author.fullName || author.lastName).filter(Boolean);
  }
  return String(record.authorString || '').split(/\s*,\s*/).filter(Boolean);
}

function metadataFromRecord(record) {
  return {
    title: plainText(record.title),
    authors: authorsFromRecord(record),
    doi: record.doi || '',
    pmid: record.pmid || (record.source === 'MED' ? record.id : ''),
    pmcid: record.pmcid || '',
    journal: record.journalInfo?.journal?.title || record.journalTitle || '',
    year: Number(record.pubYear || record.journalInfo?.printPublicationDate?.slice(0, 4)) || undefined,
    license: record.license || null,
  };
}

function queryFor(request) {
  if (request.doi) return `DOI:"${escapeLucene(request.doi)}"`;
  if (request.pmcid) return `PMCID:${escapeLucene(request.pmcid)}`;
  if (request.pmid) return `EXT_ID:${escapeLucene(request.pmid)} AND SRC:MED`;
  const clauses = [`TITLE:"${escapeLucene(request.title)}"`];
  if (request.authors[0]) clauses.push(`AUTH:"${escapeLucene(request.authors[0])}"`);
  if (request.year) clauses.push(`PUB_YEAR:${request.year}`);
  return clauses.join(' AND ');
}

export class EuropePmcSource {
  constructor(http, options = {}) {
    this.http = http;
    this.baseUrl = (options.baseUrl
      ?? process.env.EUROPE_PMC_BASE_URL
      ?? 'https://www.ebi.ac.uk/europepmc/webservices/rest').replace(/\/$/, '');
  }

  async lookup(request, options = {}) {
    const params = new URLSearchParams({
      query: queryFor(request),
      format: 'json',
      resultType: 'core',
      pageSize: '10',
    });
    let records = [];
    let lookupError;
    try {
      const data = await this.http.json(`${this.baseUrl}/search?${params}`, options);
      records = Array.isArray(data.resultList?.result) ? data.resultList.result : [];
    } catch (error) {
      if (!request.pmcid) throw error;
      lookupError = error;
    }
    let record = records[0];
    if (request.title && !request.doi && !request.pmid && !request.pmcid) {
      const normalizedRequested = normalizeTitle(request.title);
      record = records.find((candidate) => normalizeTitle(candidate.title) === normalizedRequested);
    }

    const metadata = record ? metadataFromRecord(record) : {};
    const pmcid = metadata.pmcid || request.pmcid;
    const candidates = [];
    if (pmcid) {
      const version = record
        ? normalizeVersion(
          ['authMan', 'epmcAuthMan', 'nihAuthMan'].some((field) => record[field] === 'Y')
            ? 'acceptedVersion'
            : 'publishedVersion',
        )
        : 'unknown';
      const location = {
        landing_page_url: `https://europepmc.org/articles/${pmcid}`,
        fulltext_url: `${this.baseUrl}/${pmcid}/fullTextXML`,
      };
      const resolvedDoi = metadata.doi || request.doi || '';
      candidates.push({
        identifier: pmcid,
        title: metadata.title || request.title || null,
        authors: metadata.authors || request.authors,
        requested_doi: request.doi || metadata.doi || null,
        resolved_doi: resolvedDoi || null,
        pmid: metadata.pmid || request.pmid || null,
        pmcid,
        journal: metadata.journal || null,
        year: metadata.year || request.year || null,
        source: 'PMC',
        provider_name: 'Europe PMC',
        version,
        relationship: relationshipFor(version, request.doi || metadata.doi, resolvedDoi),
        license: metadata.license,
        source_location: location.landing_page_url,
        location,
        original_format: inferFormat(location),
        normalized_format: 'markdown',
        retrieval: {
          kind: 'jats',
          url: location.fulltext_url,
          // Some NIH author manuscripts have a PMC page but no Europe PMC JATS endpoint.
          alternatives: [{
            kind: 'html',
            url: `https://pmc.ncbi.nlm.nih.gov/articles/${pmcid}/`,
          }],
        },
      });
    }
    return { metadata, candidates, sourceErrors: lookupError ? [lookupError] : [] };
  }

  async fetchJats(pmcid, options = {}) {
    return this.http.text(`${this.baseUrl}/${encodeURIComponent(pmcid)}/fullTextXML`, {
      ...options,
      accept: 'application/xml,text/xml',
    });
  }
}
