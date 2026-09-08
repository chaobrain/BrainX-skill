import { doiFromUrl, normalizeDoi } from './identifiers.mjs';

export const VERSION_PRIORITY = {
  publishedVersion: 3,
  acceptedVersion: 2,
  submittedVersion: 1,
  unknown: 0,
};

const SOURCE_PRIORITY = {
  PMC: 3,
  OpenAlex: 1,
  bioRxiv: 1,
  medRxiv: 1,
};

const FORMAT_PRIORITY = {
  jats_xml: 3,
  html: 2,
  pdf: 1,
  unknown: 0,
};

export function normalizeVersion(value, flags = {}) {
  if (Object.hasOwn(VERSION_PRIORITY, value)) return value;
  if (flags.is_published) return 'publishedVersion';
  if (flags.is_accepted) return 'acceptedVersion';
  if (flags.is_preprint) return 'submittedVersion';
  return 'unknown';
}

export function relationshipFor(version, requestedDoi, resolvedDoi) {
  const requested = normalizeDoi(requestedDoi);
  const resolved = normalizeDoi(resolvedDoi);
  if (version === 'submittedVersion') {
    const requestedPreprint = requested.startsWith('10.1101/') || requested.startsWith('10.48550/arxiv.');
    return requestedPreprint && requested === resolved ? 'same_work' : 'preprint_of_requested_work';
  }
  if (version === 'acceptedVersion') return 'accepted_manuscript_of_requested_work';
  if (requested && resolved && requested !== resolved) {
    return 'version_of_requested_work';
  }
  return 'same_work';
}

export function inferFormat(location = {}) {
  const values = [location.fulltext_url, location.pdf_url, location.landing_page_url]
    .filter(Boolean)
    .map(String);
  if (values.some((value) => /(?:fulltextxml|\.xml(?:$|[?#])|source\.xml)/i.test(value))) return 'jats_xml';
  if (values.some((value) => /\.pdf(?:$|[?#])|[?&]pdf=render/i.test(value))) return 'pdf';
  if (values.length > 0) return 'html';
  return 'unknown';
}

export function resolvedDoiFor(location, fallbackDoi = '') {
  return doiFromUrl(location.fulltext_url)
    || doiFromUrl(location.pdf_url)
    || doiFromUrl(location.landing_page_url)
    || normalizeDoi(fallbackDoi);
}

export function candidateScore(candidate) {
  return (VERSION_PRIORITY[candidate.version] ?? 0) * 100
    + (SOURCE_PRIORITY[candidate.source] ?? 0) * 10
    + (FORMAT_PRIORITY[candidate.original_format] ?? 0);
}

export function sortCandidates(candidates) {
  return [...candidates].sort((left, right) => {
    const scoreDifference = candidateScore(right) - candidateScore(left);
    if (scoreDifference !== 0) return scoreDifference;
    return String(left.source_location).localeCompare(String(right.source_location));
  });
}

export function deduplicateCandidates(candidates) {
  const seen = new Set();
  const result = [];
  for (const candidate of sortCandidates(candidates)) {
    const location = String(candidate.source_location || '').replace(/^http:/, 'https:').replace(/\/$/, '');
    const key = location || `${candidate.source}:${candidate.version}:${candidate.resolved_doi || candidate.identifier || ''}`;
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(candidate);
  }
  return result;
}

export function requestedWorkFrom(request, metadata = {}) {
  return {
    title: metadata.title || request.title || null,
    authors: metadata.authors?.length ? metadata.authors : request.authors,
    doi: normalizeDoi(metadata.doi || request.doi) || null,
    pmid: metadata.pmid || request.pmid || null,
    pmcid: metadata.pmcid || request.pmcid || null,
    journal: metadata.journal || null,
    year: metadata.year || request.year || null,
  };
}

export function publicCandidate(candidate) {
  const { retrieval, ...publicFields } = candidate;
  return publicFields;
}
