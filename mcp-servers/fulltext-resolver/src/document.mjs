// JATS conversion behavior adapts yogsoth-ai/biorxiv-mcp convert.py
// (Apache-2.0); modified into a structured Node.js converter. See ../REUSE.md.

import { DOMParser, parseHTML } from 'linkedom';
import TurndownService from 'turndown';

function cleanText(value = '') {
  return String(value).replace(/\s+/g, ' ').trim();
}

function elementName(node) {
  return String(node.localName || node.nodeName || '').toLowerCase();
}

function childElements(node) {
  return Array.from(node?.childNodes || []).filter((child) => child.nodeType === 1);
}

function renderInline(node) {
  if (!node) return '';
  if (node.nodeType === 3) return node.nodeValue || '';
  if (node.nodeType !== 1) return '';
  const name = elementName(node);
  const content = Array.from(node.childNodes || []).map(renderInline).join('');
  if (['italic', 'i', 'em'].includes(name)) return `*${content}*`;
  if (['bold', 'b', 'strong'].includes(name)) return `**${content}**`;
  if (name === 'sup') return `<sup>${content}</sup>`;
  if (name === 'sub') return `<sub>${content}</sub>`;
  if (name === 'break' || name === 'br') return '\n';
  if (name === 'ext-link') {
    const href = node.getAttribute('xlink:href') || node.getAttribute('href');
    return href ? `[${content || href}](${href})` : content;
  }
  return content;
}

function paragraphMarkdown(node) {
  return cleanText(renderInline(node));
}

function tableMarkdown(table) {
  if (!table) return '';
  const rows = Array.from(table.querySelectorAll('tr')).map((row) =>
    Array.from(row.querySelectorAll('th, td')).map((cell) => cleanText(renderInline(cell))),
  ).filter((row) => row.length > 0);
  if (!rows.length) return '';
  const width = Math.max(...rows.map((row) => row.length));
  const normalized = rows.map((row) => [...row, ...Array(Math.max(0, width - row.length)).fill('')]);
  const header = normalized[0];
  const body = normalized.slice(1);
  return [
    `| ${header.join(' | ')} |`,
    `| ${header.map(() => '---').join(' | ')} |`,
    ...body.map((row) => `| ${row.join(' | ')} |`),
  ].join('\n');
}

function figureMarkdown(figure) {
  const label = cleanText(figure.querySelector(':scope > label')?.textContent || 'Figure');
  const title = paragraphMarkdown(figure.querySelector(':scope > caption > title'));
  const paragraphs = Array.from(figure.querySelectorAll(':scope > caption > p'))
    .map(paragraphMarkdown)
    .filter(Boolean);
  const caption = [title, ...paragraphs].filter(Boolean).join(' ');
  return caption ? `**${label}.** ${caption}` : '';
}

function listMarkdown(list, depth = 0) {
  const ordered = list.getAttribute('list-type') === 'order';
  return childElements(list)
    .filter((child) => elementName(child) === 'list-item')
    .map((item, index) => {
      const prefix = ordered ? `${index + 1}.` : '-';
      const paragraphs = childElements(item)
        .filter((child) => elementName(child) === 'p')
        .map(paragraphMarkdown)
        .filter(Boolean);
      const nested = childElements(item).find((child) => elementName(child) === 'list');
      const line = `${'  '.repeat(depth)}${prefix} ${paragraphs.join(' ')}`.trimEnd();
      return nested ? `${line}\n${listMarkdown(nested, depth + 1)}` : line;
    })
    .join('\n');
}

function renderBlocks(parent, headingLevel, collectors) {
  const blocks = [];
  for (const child of childElements(parent)) {
    const name = elementName(child);
    if (name === 'title') continue;
    if (name === 'p') {
      const paragraph = paragraphMarkdown(child);
      if (paragraph) blocks.push(paragraph);
    } else if (name === 'sec') {
      blocks.push(renderSection(child, headingLevel, collectors));
    } else if (name === 'fig') {
      const figure = figureMarkdown(child);
      if (figure) {
        collectors.figureCaptions.push(figure);
        blocks.push(figure);
      }
    } else if (name === 'table-wrap') {
      const label = cleanText(child.querySelector(':scope > label')?.textContent || 'Table');
      const caption = cleanText(child.querySelector(':scope > caption')?.textContent || '');
      const table = tableMarkdown(child.querySelector('table'));
      const rendered = [`**${label}.**${caption ? ` ${caption}` : ''}`, table].filter(Boolean).join('\n\n');
      if (rendered) {
        collectors.tables.push(rendered);
        blocks.push(rendered);
      }
    } else if (name === 'list') {
      const list = listMarkdown(child);
      if (list) blocks.push(list);
    } else if (['disp-quote', 'boxed-text', 'statement'].includes(name)) {
      const quote = cleanText(child.textContent);
      if (quote) blocks.push(quote.split('\n').map((line) => `> ${line}`).join('\n'));
    }
  }
  return blocks.filter(Boolean).join('\n\n');
}

function renderSection(section, headingLevel, collectors) {
  const title = cleanText(section.querySelector(':scope > title')?.textContent || 'Untitled section');
  const level = Math.min(Math.max(headingLevel, 2), 6);
  const body = renderBlocks(section, level + 1, collectors);
  collectors.sections.push({ title, markdown: body });
  return `${'#'.repeat(level)} ${title}${body ? `\n\n${body}` : ''}`;
}

function jatsAuthors(document) {
  return Array.from(document.querySelectorAll('article-meta > contrib-group[content-type="author"] > contrib, article-meta > contrib-group > contrib'))
    .map((contributor) => {
      const given = cleanText(contributor.querySelector('given-names')?.textContent);
      const surname = cleanText(contributor.querySelector('surname')?.textContent);
      const collaborative = cleanText(contributor.querySelector('collab')?.textContent);
      return collaborative || [given, surname].filter(Boolean).join(' ');
    })
    .filter(Boolean);
}

function jatsVersion(document) {
  const customMetadata = new Map(
    Array.from(document.querySelectorAll('custom-meta')).map((entry) => [
      cleanText(entry.querySelector('meta-name')?.textContent).toLowerCase(),
      cleanText(entry.querySelector('meta-value')?.textContent).toLowerCase(),
    ]),
  );
  const enabled = (name) => ['yes', 'true', '1'].includes(customMetadata.get(name));
  if (enabled('is-preprint')) return 'submittedVersion';
  if (enabled('is-manuscript')) return 'acceptedVersion';
  return 'publishedVersion';
}

export function jatsToDocument(xml) {
  const document = new DOMParser().parseFromString(xml, 'text/xml');
  const parserError = document.querySelector('parsererror');
  if (parserError) throw new Error(`Invalid JATS XML: ${cleanText(parserError.textContent)}`);

  const title = cleanText(document.querySelector('article-meta article-title')?.textContent);
  const authors = jatsAuthors(document);
  const abstractNodes = Array.from(document.querySelectorAll('article-meta > abstract'));
  const abstract = abstractNodes
    .map((node) => Array.from(node.querySelectorAll(':scope > p')).map(paragraphMarkdown).filter(Boolean).join('\n\n'))
    .filter(Boolean)
    .join('\n\n');
  const collectors = { sections: [], figureCaptions: [], tables: [] };
  const body = document.querySelector('article > body') || document.querySelector('body');
  const bodyMarkdown = body ? renderBlocks(body, 2, collectors) : '';
  const references = Array.from(document.querySelectorAll('ref-list > ref'))
    .map((reference) => {
      const label = cleanText(reference.querySelector(':scope > label')?.textContent);
      const citation = cleanText(Array.from(reference.childNodes || [])
        .filter((child) => child !== reference.querySelector(':scope > label'))
        .map((child) => child.textContent || child.nodeValue || '')
        .join(' '));
      return [label, citation].filter(Boolean).join(' ');
    })
    .filter(Boolean);
  const journal = cleanText(document.querySelector('journal-title')?.textContent);
  const year = Number(cleanText(document.querySelector('article-meta > pub-date > year')?.textContent)) || null;
  const doi = cleanText(document.querySelector('article-id[pub-id-type="doi"]')?.textContent) || null;
  const pmcid = cleanText(document.querySelector('article-id[pub-id-type="pmcid"]')?.textContent) || null;
  const version = jatsVersion(document);

  const header = [
    title ? `# ${title}` : '',
    authors.length ? `**Authors:** ${authors.join(', ')}` : '',
    journal ? `**Journal:** ${journal}${year ? ` (${year})` : ''}` : '',
    abstract ? `## Abstract\n\n${abstract}` : '',
  ].filter(Boolean).join('\n\n');
  const referenceMarkdown = references.length
    ? `## References\n\n${references.map((reference) => `- ${reference}`).join('\n')}`
    : '';
  const markdown = [header, bodyMarkdown, referenceMarkdown].filter(Boolean).join('\n\n').trim();
  if (!markdown) throw new Error('JATS document contains no extractable text.');

  return {
    title: title || null,
    authors,
    doi,
    pmcid,
    version,
    journal: journal || null,
    year,
    abstract: abstract || null,
    sections: collectors.sections,
    figure_captions: collectors.figureCaptions,
    tables: collectors.tables,
    references,
    markdown,
    original_format: 'jats_xml',
    normalized_format: 'markdown',
  };
}

function createTurndown() {
  const turndown = new TurndownService({
    bulletListMarker: '-',
    codeBlockStyle: 'fenced',
    headingStyle: 'atx',
  });
  turndown.remove(['script', 'style', 'noscript', 'nav', 'form', 'button', 'svg']);
  turndown.addRule('table', {
    filter: 'table',
    replacement(_content, node) {
      const table = tableMarkdown(node);
      return table ? `\n\n${table}\n\n` : '';
    },
  });
  return turndown;
}

export function htmlToDocument(html, sourceUrl) {
  const { document } = parseHTML(html);
  for (const node of document.querySelectorAll('script, style, noscript, nav, form, button, svg, header, footer, aside')) {
    node.remove();
  }
  const root = document.querySelector('.fulltext-view, article, main, [role="main"], .article') || document.body;
  const markdown = createTurndown().turndown(root?.innerHTML || '').trim();
  if (!markdown) throw new Error(`HTML page contains no extractable article text: ${sourceUrl}`);
  return {
    title: cleanText(root?.querySelector('h1')?.textContent || document.querySelector('title')?.textContent) || null,
    authors: [],
    abstract: null,
    sections: [],
    figure_captions: [],
    tables: [],
    references: [],
    markdown,
    original_format: 'html',
    normalized_format: 'markdown',
    source_url: sourceUrl,
  };
}
