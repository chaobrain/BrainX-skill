import assert from 'node:assert/strict';
import test from 'node:test';
import { htmlToDocument, jatsToDocument } from '../src/document.mjs';
import { doiFromUrl, normalizeDoi, normalizeRequest } from '../src/identifiers.mjs';

const JATS = `<?xml version="1.0" encoding="UTF-8"?>
<article>
  <front>
    <journal-meta><journal-title-group><journal-title>Neuron</journal-title></journal-title-group></journal-meta>
    <article-meta>
      <article-id pub-id-type="pmcid">PMC1234567</article-id>
      <article-id pub-id-type="doi">10.1000/example</article-id>
      <title-group><article-title>A test article</article-title></title-group>
      <contrib-group content-type="author">
        <contrib><name><given-names>Ada</given-names><surname>Lovelace</surname></name></contrib>
      </contrib-group>
      <pub-date><year>2026</year></pub-date>
      <abstract><p>Structured <italic>abstract</italic> text.</p></abstract>
    </article-meta>
  </front>
  <body>
    <sec><title>Results</title>
      <p>The result contains <bold>evidence</bold>.</p>
      <fig><label>Figure 1</label><caption><title>Activity</title><p>A neural trace.</p></caption></fig>
      <table-wrap><label>Table 1</label><caption><p>Measurements</p></caption>
        <table><tr><th>Region</th><th>Value</th></tr><tr><td>V1</td><td>4</td></tr></table>
      </table-wrap>
    </sec>
  </body>
  <back><ref-list><ref><label>1</label><mixed-citation>Example reference.</mixed-citation></ref></ref-list></back>
</article>`;

test('normalizes DOI forms and rejects malformed identifiers', () => {
  assert.equal(normalizeDoi('https://doi.org/10.1000/ABC.'), '10.1000/abc');
  assert.equal(
    doiFromUrl('https://www.biorxiv.org/content/10.1101/2024.07.15.603498v1.full.pdf'),
    '10.1101/2024.07.15.603498',
  );
  assert.throws(() => normalizeRequest({ doi: 'not-a-doi' }), /Malformed DOI/);
  assert.throws(() => normalizeRequest({}), /Provide at least one/);
});

test('converts JATS into structured Markdown', () => {
  const document = jatsToDocument(JATS);
  assert.equal(document.title, 'A test article');
  assert.deepEqual(document.authors, ['Ada Lovelace']);
  assert.equal(document.doi, '10.1000/example');
  assert.equal(document.pmcid, 'PMC1234567');
  assert.equal(document.version, 'publishedVersion');
  assert.match(document.abstract, /Structured \*abstract\* text/);
  assert.match(document.markdown, /## Results/);
  assert.match(document.markdown, /\*\*Figure 1\.\*\* Activity A neural trace\./);
  assert.match(document.markdown, /\| Region \| Value \|/);
  assert.deepEqual(document.references, ['1 Example reference.']);
});

test('infers submitted and accepted versions from JATS custom metadata', () => {
  const withCustomMeta = (name) => JATS.replace(
    '<abstract>',
    `<custom-meta-group><custom-meta><meta-name>${name}</meta-name><meta-value>yes</meta-value></custom-meta></custom-meta-group><abstract>`,
  );

  assert.equal(
    jatsToDocument(withCustomMeta('is-preprint')).version,
    'submittedVersion',
  );
  assert.equal(
    jatsToDocument(withCustomMeta('is-manuscript')).version,
    'acceptedVersion',
  );
});

test('converts the main HTML article without navigation or scripts', () => {
  const document = htmlToDocument(`
    <html><head><title>Fallback title</title><script>bad()</script></head>
    <body><nav>Menu</nav><article><h1>Repository copy</h1><p>Full text.</p></article></body></html>
  `, 'https://repository.example/article');
  assert.equal(document.title, 'Repository copy');
  assert.match(document.markdown, /# Repository copy/);
  assert.match(document.markdown, /Full text\./);
  assert.doesNotMatch(document.markdown, /Menu|bad\(\)/);
});
