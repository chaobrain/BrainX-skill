// Adapted from DeyangLiu123/openalex-mcp ids.py (MIT); modified for Node.js.
// See ../REUSE.md and ../THIRD_PARTY_NOTICES.md.

import { domainToASCII } from 'node:url';

const DOI_PATTERN = /^10\.\d{4,9}\/\S+$/i;
const PMCID_PATTERN = /^PMC\d+$/i;
const PMID_PATTERN = /^\d+$/;

export function normalizeDoi(value = '') {
  let normalized = decodeURIComponent(String(value).trim());
  normalized = normalized.replace(/^doi:\s*/i, '');
  normalized = normalized.replace(/^https?:\/\/(?:dx\.)?doi\.org\//i, '');
  return normalized.replace(/[\s.]+$/, '').toLowerCase();
}

export function isValidDoi(value) {
  return DOI_PATTERN.test(normalizeDoi(value));
}

export function normalizePmcid(value = '') {
  const normalized = String(value).trim().toUpperCase();
  if (!normalized) return '';
  return normalized.startsWith('PMC') ? normalized : `PMC${normalized}`;
}

export function normalizePmid(value = '') {
  return String(value).trim();
}

export function normalizeTitle(value = '') {
  return String(value)
    .normalize('NFKC')
    .toLocaleLowerCase('en-US')
    .replace(/[\p{P}\p{S}\s]+/gu, '');
}

export function normalizeAuthors(authors) {
  if (Array.isArray(authors)) {
    return authors.map((author) => String(author).trim()).filter(Boolean);
  }
  if (typeof authors === 'string') {
    return authors.split(/\s*;\s*/).map((author) => author.trim()).filter(Boolean);
  }
  return [];
}

export function normalizeRequest(input = {}) {
  const request = {
    doi: normalizeDoi(input.doi),
    pmid: normalizePmid(input.pmid),
    pmcid: normalizePmcid(input.pmcid),
    title: String(input.title ?? '').trim(),
    authors: normalizeAuthors(input.authors),
    year: input.year === undefined ? undefined : Number(input.year),
  };

  if (!request.doi && !request.pmid && !request.pmcid && !request.title) {
    throw new Error('Provide at least one of doi, pmid, pmcid, or title.');
  }
  if (request.doi && !DOI_PATTERN.test(request.doi)) {
    throw new Error(`Malformed DOI: ${input.doi}`);
  }
  if (request.pmid && !PMID_PATTERN.test(request.pmid)) {
    throw new Error(`Malformed PMID: ${input.pmid}`);
  }
  if (request.pmcid && !PMCID_PATTERN.test(request.pmcid)) {
    throw new Error(`Malformed PMCID: ${input.pmcid}`);
  }
  if (request.year !== undefined && (!Number.isInteger(request.year) || request.year < 1600 || request.year > 3000)) {
    throw new Error(`Malformed publication year: ${input.year}`);
  }
  return request;
}

export function doiFromUrl(value = '') {
  const text = String(value);
  const doiUrlMatch = text.match(/doi\.org\/(10\.\d{4,9}\/[^?#\s]+)/i);
  if (doiUrlMatch) return normalizeDoi(doiUrlMatch[1]);

  const preprintMatch = text.match(/\/(10\.1101\/[^/?#]+)/i);
  if (preprintMatch) {
    // Remove bioRxiv revision and representation suffixes before comparing work identity.
    const doi = preprintMatch[1].replace(/(?:v\d+)?(?:\.(?:full|source))?(?:\.(?:pdf|xml))?$/i, '');
    return normalizeDoi(doi);
  }

  const arxivMatch = text.match(/arxiv\.org\/(?:abs|pdf|html)\/([^?#]+?)(?:\.pdf)?$/i);
  if (arxivMatch) return `10.48550/arxiv.${arxivMatch[1].replace(/v\d+$/i, '').toLowerCase()}`;
  return '';
}

function isPrivateIpv4(hostname) {
  const parts = hostname.split('.').map(Number);
  if (parts.length !== 4 || parts.some((part) => !Number.isInteger(part) || part < 0 || part > 255)) {
    return false;
  }
  return parts[0] === 10
    || parts[0] === 127
    || (parts[0] === 169 && parts[1] === 254)
    || (parts[0] === 172 && parts[1] >= 16 && parts[1] <= 31)
    || (parts[0] === 192 && parts[1] === 168)
    || parts[0] === 0;
}

export function assertPublicHttpUrl(value) {
  const url = new URL(value);
  if (!['http:', 'https:'].includes(url.protocol)) {
    throw new Error(`Unsupported source URL protocol: ${url.protocol}`);
  }
  const hostname = domainToASCII(url.hostname).toLowerCase();
  if (!hostname || hostname === 'localhost' || hostname.endsWith('.localhost') || isPrivateIpv4(hostname)) {
    throw new Error(`Refusing non-public source URL: ${value}`);
  }
  if (hostname === '::1' || hostname.startsWith('fc') || hostname.startsWith('fd') || hostname.startsWith('fe80:')) {
    throw new Error(`Refusing non-public source URL: ${value}`);
  }
  return url;
}

export function encodeDoiPath(doi) {
  return normalizeDoi(doi).split('/').map(encodeURIComponent).join('/');
}
