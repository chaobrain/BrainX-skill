// Adapted from cyanheads/biorxiv-mcp-server services (Apache-2.0); modified to
// support published-DOI crosswalk resolution. See ../../REUSE.md.

import { encodeDoiPath, normalizeDoi } from '../identifiers.mjs';
import { inferFormat, relationshipFor } from '../model.mjs';

function normalizeUpstreamText(text) {
  if (!text) return '';
  return String(text)
    .replace(/O_FIG[\s\S]*?C_FIG/g, ' ')
    .replace(/AO_SCPLOWBSTRACTC_SCPLOW/g, ' ')
    .replace(/[OC]_SCPLOW/g, '')
    .replace(/org\.highwire\.dtl\.\S+/g, ' ')
    .replace(/[OCM]_FIG\b/g, ' ')
    .replace(/\bSRC=\S*/gi, ' ')
    .replace(/Graphical Abstract/gi, ' ')
    .replace(/<\/?(?:i|b|u|em|strong|sub|sup|small|br|p|span|div|a|h[1-6])\b[^>]*>/gi, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function latestRevision(revisions) {
  return [...revisions].sort((left, right) => {
    const versionDifference = Number(right.version || 0) - Number(left.version || 0);
    if (versionDifference !== 0) return versionDifference;
    return String(right.date || '').localeCompare(String(left.date || ''));
  })[0];
}

function authorsFromString(value) {
  return String(value || '').split(/\s*;\s*/).map((author) => author.trim()).filter(Boolean);
}

export class BiorxivSource {
  constructor(http, options = {}) {
    this.http = http;
    this.baseUrl = (options.baseUrl
      ?? process.env.BIORXIV_API_BASE_URL
      ?? 'https://api.biorxiv.org').replace(/\/$/, '');
    this.webBaseUrls = {
      biorxiv: options.biorxivWebBaseUrl ?? 'https://www.biorxiv.org',
      medrxiv: options.medrxivWebBaseUrl ?? 'https://www.medrxiv.org',
    };
  }

  async getCrosswalk(doi, server, options = {}) {
    const url = `${this.baseUrl}/pubs/${server}/${encodeDoiPath(doi)}/na/json`;
    const data = await this.http.json(url, options);
    return data.collection?.[0] || null;
  }

  async getDetails(doi, server, options = {}) {
    const url = `${this.baseUrl}/details/${server}/${encodeDoiPath(doi)}/0/json`;
    const data = await this.http.json(url, options);
    return Array.isArray(data.collection) ? data.collection : [];
  }

  async lookup(request, options = {}) {
    if (!request.doi) return { metadata: {}, candidates: [] };

    const requestedDoi = normalizeDoi(request.doi);
    const directPreprint = requestedDoi.startsWith('10.1101/');
    const serverResults = await Promise.allSettled(['biorxiv', 'medrxiv'].map(async (server) => {
      if (directPreprint) {
        const revisions = await this.getDetails(requestedDoi, server, options);
        return revisions.length ? { server, revisions, crosswalk: null } : null;
      }

      const crosswalk = await this.getCrosswalk(requestedDoi, server, options);
      const preprintDoi = normalizeDoi(crosswalk?.preprint_doi || crosswalk?.biorxiv_doi);
      const publishedDoi = normalizeDoi(crosswalk?.published_doi);
      if (!preprintDoi || publishedDoi !== requestedDoi) return null;
      const revisions = await this.getDetails(preprintDoi, server, options);
      return { server, revisions, crosswalk };
    }));

    const candidates = [];
    let metadata = {};
    for (const result of serverResults) {
      if (result.status !== 'fulfilled' || !result.value) continue;
      const { server, revisions, crosswalk } = result.value;
      const revision = latestRevision(revisions);
      if (!revision) continue;
      const preprintDoi = normalizeDoi(revision.doi || crosswalk?.preprint_doi);
      const versionNumber = String(revision.version || '1');
      const fallbackHtml = `${this.webBaseUrls[server]}/content/${preprintDoi}v${versionNumber}.full`;
      const location = {
        landing_page_url: `${this.webBaseUrls[server]}/content/${preprintDoi}v${versionNumber}`,
        fulltext_url: revision.jatsxml || fallbackHtml,
        pdf_url: `${this.webBaseUrls[server]}/content/${preprintDoi}v${versionNumber}.full.pdf`,
      };
      const title = normalizeUpstreamText(revision.title || crosswalk?.preprint_title);
      const authors = authorsFromString(revision.authors || crosswalk?.preprint_authors);
      const source = server === 'biorxiv' ? 'bioRxiv' : 'medRxiv';
      const originalFormat = inferFormat(location);
      metadata = {
        title: title || request.title || '',
        authors,
        doi: crosswalk?.published_doi || (directPreprint ? preprintDoi : requestedDoi),
        journal: crosswalk?.published_journal || '',
        year: Number(String(crosswalk?.published_date || revision.date || '').slice(0, 4)) || undefined,
      };
      candidates.push({
        identifier: preprintDoi,
        title: title || request.title || null,
        authors,
        requested_doi: requestedDoi,
        resolved_doi: preprintDoi,
        pmid: null,
        pmcid: null,
        journal: crosswalk?.published_journal || null,
        year: Number(String(revision.date || '').slice(0, 4)) || request.year || null,
        source,
        provider_name: source,
        version: 'submittedVersion',
        relationship: relationshipFor('submittedVersion', requestedDoi, preprintDoi),
        license: revision.license && revision.license !== 'NA' ? revision.license : null,
        source_location: location.landing_page_url,
        location,
        original_format: originalFormat,
        normalized_format: 'markdown',
        preprint_revision: versionNumber,
        published_doi: normalizeDoi(crosswalk?.published_doi || revision.published) || null,
        abstract: normalizeUpstreamText(revision.abstract || crosswalk?.preprint_abstract) || null,
        retrieval: {
          kind: originalFormat === 'jats_xml' ? 'jats' : 'html',
          url: location.fulltext_url,
        },
      });
    }
    return { metadata, candidates };
  }
}

export { normalizeUpstreamText };
