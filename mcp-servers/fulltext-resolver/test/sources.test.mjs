import assert from 'node:assert/strict';
import test from 'node:test';
import { FullTextResolver } from '../src/resolver.mjs';
import { BiorxivSource } from '../src/sources/biorxiv.mjs';
import { EuropePmcSource } from '../src/sources/europe-pmc.mjs';
import { OpenAlexSource } from '../src/sources/openalex.mjs';

const request = {
  doi: '10.1000/requested',
  pmid: '',
  pmcid: '',
  title: '',
  authors: [],
  year: undefined,
};

test('Europe PMC maps an author manuscript to an accepted JATS candidate', async () => {
  let calledUrl = '';
  const http = {
    async json(url) {
      calledUrl = url;
      return {
        resultList: {
          result: [{
            title: 'Accepted <i>paper</i>',
            doi: '10.1000/requested',
            pmid: '123',
            pmcid: 'PMC123',
            pubYear: '2026',
            epmcAuthMan: 'Y',
            authorString: 'Researcher A, Researcher B',
            journalInfo: { journal: { title: 'Neuron' } },
          }],
        },
      };
    },
  };
  const result = await new EuropePmcSource(http).lookup(request);
  assert.match(decodeURIComponent(calledUrl), /DOI:"10\.1000\\\/requested"/);
  assert.equal(result.metadata.title, 'Accepted paper');
  assert.equal(result.candidates[0].version, 'acceptedVersion');
  assert.equal(result.candidates[0].original_format, 'jats_xml');
  assert.equal(result.candidates[0].pmcid, 'PMC123');
});

test('direct PMCID fallback survives Europe PMC metadata failure and records the error', async () => {
  const metadataError = Object.assign(new Error('Europe PMC metadata unavailable'), {
    status: 503,
    retryable: true,
  });
  let calledUrl = '';
  const http = {
    async json(url) {
      calledUrl = url;
      throw metadataError;
    },
  };
  const emptySource = { lookup: async () => ({ metadata: {}, candidates: [] }) };
  const resolver = new FullTextResolver({
    http,
    europePmc: new EuropePmcSource(http),
    openAlex: emptySource,
    biorxiv: emptySource,
  });

  const result = await resolver.resolve({ pmcid: '123' });

  assert.equal(result.resolved, true);
  assert.equal(result.resolved_document.pmcid, 'PMC123');
  assert.equal(result.resolved_document.version, 'unknown');
  assert.equal(result.source_errors[0].source, 'PMC');
  assert.equal(result.source_errors[0].status, 503);
  assert.match(decodeURIComponent(calledUrl), /PMCID:PMC123/);
});

test('OpenAlex preserves location versions and excludes duplicate PMC locations', async () => {
  const http = {
    async json() {
      return {
        id: 'https://openalex.org/W123',
        doi: 'https://doi.org/10.1000/requested',
        display_name: 'Resolved work',
        publication_year: 2026,
        authorships: [{ author: { display_name: 'Researcher A' } }],
        primary_location: { source: { display_name: 'Neuron' } },
        locations: [
          {
            is_oa: true,
            version: 'publishedVersion',
            landing_page_url: 'https://publisher.example/article',
            pdf_url: 'https://publisher.example/article.pdf',
            license: 'cc-by',
            source: { display_name: 'Neuron', type: 'journal' },
          },
          {
            is_oa: true,
            version: 'acceptedVersion',
            landing_page_url: 'https://repository.example/manuscript',
            source: { display_name: 'University Repository', type: 'repository' },
          },
          {
            is_oa: true,
            version: 'publishedVersion',
            landing_page_url: 'https://europepmc.org/articles/PMC123',
            source: { display_name: 'Europe PMC (PubMed Central)', type: 'repository' },
          },
        ],
      };
    },
  };
  const result = await new OpenAlexSource(http).lookup(request);
  assert.deepEqual(result.candidates.map((candidate) => candidate.version), [
    'publishedVersion',
    'acceptedVersion',
  ]);
  assert.equal(result.candidates[0].original_format, 'pdf');
  assert.equal(result.candidates[1].provider_name, 'University Repository');
});
test('OpenAlex retains a direct accepted repository PDF when its OA flag is stale', async () => {
  const http = {
    async json() {
      return {
        id: 'https://openalex.org/W2551595439',
        doi: 'https://doi.org/10.1016/j.neuron.2016.10.030',
        display_name: 'Neural Architecture of Hunger-Dependent Multisensory Decision Making in C. elegans',
        publication_year: 2016,
        locations: [{
          is_oa: false,
          version: 'acceptedVersion',
          landing_page_url: 'http://eprints.whiterose.ac.uk/109026/1/NEURON-Ghosh2016--pre-proof.pdf',
          pdf_url: 'http://eprints.whiterose.ac.uk/109026/1/NEURON-Ghosh2016--pre-proof.pdf',
          source: {
            display_name: 'White Rose Research Online',
            type: 'repository',
          },
        }],
      };
    },
  };
  const result = await new OpenAlexSource(http).lookup({
    ...request,
    doi: '10.1016/j.neuron.2016.10.030',
  });

  assert.equal(result.candidates[0].provider_name, 'White Rose Research Online');
  assert.equal(result.candidates[0].version, 'acceptedVersion');
  assert.equal(result.candidates[0].original_format, 'pdf');
});

test('OpenAlex keeps bioRxiv DOI locations as submitted versions', async () => {
  const preprintRequest = { ...request, doi: '10.1101/2024.07.15.603498' };
  const http = {
    async json() {
      return {
        id: 'https://openalex.org/WPREPRINT',
        doi: 'https://doi.org/10.1101/2024.07.15.603498',
        display_name: 'Modeling <i>C. elegans</i> muscle cells',
        publication_year: 2024,
        locations: [{
          is_oa: true,
          version: 'acceptedVersion',
          landing_page_url: 'https://doi.org/10.1101/2024.07.15.603498',
          pdf_url: 'https://www.biorxiv.org/content/10.1101/2024.07.15.603498v1.full.pdf',
          source: { display_name: 'bioRxiv', type: 'repository' },
        }],
      };
    },
  };
  const result = await new OpenAlexSource(http).lookup(preprintRequest);

  assert.equal(result.metadata.title, 'Modeling C. elegans muscle cells');
  assert.equal(result.candidates[0].version, 'submittedVersion');
  assert.equal(result.candidates[0].resolved_doi, '10.1101/2024.07.15.603498');
  assert.equal(result.candidates[0].relationship, 'same_work');
});

test('bioRxiv crosswalk resolves a published DOI to a submitted JATS manuscript', async () => {
  const calls = [];
  const http = {
    async json(url) {
      calls.push(url);
      if (url.includes('/pubs/biorxiv/')) {
        return {
          collection: [{
            preprint_doi: '10.1101/2026.01.01.123456',
            published_doi: '10.1000/requested',
            published_journal: 'Nature Neuroscience',
          }],
        };
      }
      if (url.includes('/details/biorxiv/')) {
        return {
          collection: [{
            doi: '10.1101/2026.01.01.123456',
            title: 'Preprint title',
            authors: 'Researcher A; Researcher B',
            version: '2',
            date: '2026-02-03',
            license: 'cc_by',
            jatsxml: 'https://www.biorxiv.org/content/early/2026/02/03/2026.01.01.123456.source.xml',
            published: '10.1000/requested',
          }],
        };
      }
      return { collection: [] };
    },
  };
  const result = await new BiorxivSource(http).lookup(request);
  assert.equal(result.candidates.length, 1);
  assert.equal(result.candidates[0].source, 'bioRxiv');
  assert.equal(result.candidates[0].version, 'submittedVersion');
  assert.equal(result.candidates[0].relationship, 'preprint_of_requested_work');
  assert.equal(result.candidates[0].resolved_doi, '10.1101/2026.01.01.123456');
  assert.equal(result.candidates[0].preprint_revision, '2');
  assert.ok(calls.some((url) => url.includes('/pubs/biorxiv/10.1000/requested/na/json')));
});

test('bioRxiv rejects a crosswalk whose published DOI does not match', async () => {
  const http = {
    async json(url) {
      if (url.includes('/pubs/biorxiv/')) {
        return { collection: [{ preprint_doi: '10.1101/example', published_doi: '10.1000/other' }] };
      }
      return { collection: [] };
    },
  };
  const result = await new BiorxivSource(http).lookup(request);
  assert.deepEqual(result.candidates, []);
});
