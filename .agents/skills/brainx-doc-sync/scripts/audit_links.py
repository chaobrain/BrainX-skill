"""Inventory maintained BrainX Markdown links and optionally probe their status."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from http.client import HTTPException
import json
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


DOC_DIRS = ('skills', 'source_html_references', 'braintools-references',
            '.agents/skills/brainx-doc-sync')
DOC_FILES = ('README.md', 'plan.md', 'how-to-refine-skill.md')
DEFAULT_HOSTS = {
    'brainx.chaobrain.com', 'github.com', 'raw.githubusercontent.com', 'pypi.org',
    *(f'{name}.readthedocs.io' for name in (
        'brainpy', 'brainunit', 'brainstate', 'brainevent', 'braincell',
        'brainmass', 'braintrace', 'braintools',
    )),
}
URL = re.compile(r'''https?://[^\s<>`"'|]+''')
INLINE_LINK = re.compile(
    r'(?<!!)\[[^\]\n]*\]\(\s*(?:<([^>\n]+)>|((?:[^\s()<>]|\([^\s()]*\))+))(?=[\s)])'
)
SOFT_WRAP = r'[ \t]*(?:\n[ \t]*)?'
SKILL_ROUTE = re.compile(
    r'`(?P<checkout>skills/(?:package-skills/)?[^/`\r\n]+/[^`\r\n]+\.(?:md|py|sh))`'
    rf'|(?:`(?P<skill>[a-z][a-z0-9-]*)`{SOFT_WRAP}(?:→|->){SOFT_WRAP})?'
    r'`(?P<target>(?:references/|scripts/)[^`\r\n]+\.(?:md|py|sh)|SKILL\.md)`'
)
FENCE = re.compile(r'^ {0,3}(`{3,}|~{3,})(.*)$')
MAX_HTML_BYTES = 2 * 1024 * 1024


def markdown_files(root):
    candidates = [root / name for name in DOC_FILES]
    for directory in DOC_DIRS:
        candidates.extend((root / directory).rglob('*.md'))
    return sorted(p for p in candidates if p.is_file() and p.resolve().is_relative_to(root))


def safe_url(url):
    try:
        parts = urlsplit(url)
        return (parts.scheme in {'http', 'https'} and parts.hostname
                and parts.username is None and parts.password is None
                and not parts.query and parts.port in {None, 80, 443})
    except ValueError:
        return False


def inventory(root, files):
    refs = defaultdict(list)
    skipped = []
    for path in files:
        for line, text in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
            for match in URL.finditer(text):
                candidate = match[0].rstrip('.,;:!]?')
                while candidate.endswith(')') and candidate.count(')') > candidate.count('('):
                    candidate = candidate[:-1]
                url, _, fragment = candidate.partition('#')
                if not safe_url(url):
                    skipped.append({'file': path.relative_to(root).as_posix(), 'line': line})
                    continue
                refs[url].append({'file': path.relative_to(root).as_posix(),
                                  'line': line, 'fragment': fragment})
    return refs, skipped


def markdown_prose_lines(text):
    """Blank fenced code lines while preserving source line numbers."""
    fenced = None
    for line in text.splitlines(keepends=True):
        fence = FENCE.match(line)
        if fence:
            marker, rest = fence.groups()
            if fenced is None:
                fenced = marker
            elif marker.startswith(fenced) and not rest.strip():
                fenced = None
        if fence or fenced is not None:
            yield '\n' if line.endswith('\n') else ''
        else:
            yield line


def missing_local_targets(root, files):
    missing = []
    for path in files:
        lines = markdown_prose_lines(path.read_text(encoding='utf-8-sig'))
        for line, text in enumerate(lines, 1):
            prose = re.sub(r'(`+).*?\1', '', text)
            for match in INLINE_LINK.finditer(prose):
                target = match[1] or match[2]
                try:
                    parts = urlsplit(target)
                except ValueError:
                    continue
                if parts.scheme or parts.netloc or not parts.path or parts.path.startswith('/'):
                    continue
                try:
                    resolved = (path.parent / unquote(parts.path)).resolve()
                    if resolved.is_relative_to(root) and not resolved.exists():
                        missing.append({'file': path.relative_to(root).as_posix(),
                                        'line': line, 'target': parts.path})
                except (OSError, ValueError, RuntimeError):
                    continue
    return missing


def missing_skill_routes(root, files):
    """Resolve backtick routes against the skill roots used by the installer."""
    names = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))['skills']
    skills = {p.parent.name: p.parent for p in (root / 'skills').rglob('SKILL.md')
              if p.parent.name in names}
    missing = []
    for path in files:
        owner = next((parent for parent in path.parents if parent in skills.values()), None)
        if owner is None or path.name == 'THIRD_PARTY_NOTICES.md':
            continue
        text = ''.join(markdown_prose_lines(path.read_text(encoding='utf-8-sig')))
        file = path.relative_to(root).as_posix()
        for match in SKILL_ROUTE.finditer(text):
            name = match['skill'] or owner.name
            group = 'checkout' if match['checkout'] else 'target'
            target = match[group]
            if group == 'target' and name in skills:
                base = skills[name].resolve()
                resolved = (base / target).resolve()
                if resolved.is_relative_to(base) and resolved.is_file():
                    continue
            line = text.count('\n', 0, match.start(group)) + 1
            missing.append({'file': file, 'line': line, 'skill': name, 'target': target})
    return missing


class AnchorParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.anchors = set()

    def handle_starttag(self, tag, attrs):
        self.anchors.update(value for key, value in attrs
                            if value and (key == 'id' or (tag == 'a' and key == 'name')))


def check_anchors(response, fragments):
    if response.headers.get_content_type() != 'text/html':
        return {fragment: 'unsupported_content_type' for fragment in fragments}
    body = response.read(MAX_HTML_BYTES + 1)
    parser = AnchorParser()
    try:
        parser.feed(body[:MAX_HTML_BYTES].decode(response.headers.get_content_charset() or 'utf-8',
                                                  errors='replace'))
        parser.close()
    except LookupError:
        return {fragment: 'unsupported_encoding' for fragment in fragments}
    return {fragment: ('found' if unquote(fragment) in parser.anchors else
                       'body_limit' if len(body) > MAX_HTML_BYTES else 'missing_candidate')
            for fragment in sorted(fragments)}


class RedirectBlocked(Exception):
    pass


class AllowedRedirects(HTTPRedirectHandler):
    def __init__(self, hosts):
        self.hosts = hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not safe_url(newurl) or urlsplit(newurl).hostname not in self.hosts:
            raise RedirectBlocked
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def http_result(code):
    if 200 <= code < 300:
        return 'reachable'
    if code in {404, 410}:
        return 'missing_candidate'
    if code in {401, 403, 407}:
        return 'access_limited'
    if code == 429:
        return 'rate_limited'
    if 500 <= code < 600:
        return 'server_error'
    return 'http_error'


def probe(url, hosts, timeout, fragments=()):
    opener = build_opener(AllowedRedirects(hosts))
    request = Request(url, headers={'User-Agent': 'BrainX-skill-doc-audit/1.0'})
    try:
        with opener.open(request, timeout=timeout) as response:
            result = {'status': http_result(response.status), 'http_status': response.status,
                      'final_url': response.geturl()}
            if fragments and result['status'] == 'reachable':
                try:
                    result['anchors'] = check_anchors(response, fragments)
                except (HTTPException, OSError, ValueError):
                    result['anchors'] = {fragment: 'body_read_error' for fragment in fragments}
            return result
    except HTTPError as error:
        with error:
            return {'status': http_result(error.code), 'http_status': error.code,
                    'final_url': error.geturl()}
    except RedirectBlocked:
        return {'status': 'redirect_blocked', 'reason': 'destination_outside_safe_host_scope'}
    except (HTTPException, URLError, OSError, ValueError) as error:
        reason = error.reason if isinstance(error, URLError) else error
        return {'status': 'transport_error', 'reason': type(reason).__name__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='BrainX-skill checkout')
    parser.add_argument('--scope', type=Path, action='append', help='maintained file or directory, relative to root')
    parser.add_argument('--check', action='store_true', help='probe allowlisted links via HTTP GET')
    parser.add_argument('--anchors', action='store_true', help='check static HTML fragments for probed links')
    parser.add_argument('--host', action='append', help='exact hostname; replaces the default allowlist')
    parser.add_argument('--limit', type=int, help='maximum initial URL probes')
    parser.add_argument('--jobs', type=int, default=6, help='concurrent probes (1–16)')
    parser.add_argument('--timeout', type=float, default=6, help='socket timeout in seconds (0–30)')
    args = parser.parse_args()
    if args.anchors and not args.check:
        parser.error('--anchors requires --check')
    if not (1 <= args.jobs <= 16 and 0 < args.timeout <= 30) or (args.limit is not None and args.limit < 1):
        parser.error('expected jobs 1–16, timeout >0 and <=30, and limit >=1')
    root = args.root.resolve()
    if not (root / 'manifest.json').is_file() or not (root / 'skills').is_dir():
        parser.error('--root must be a BrainX-skill checkout')
    files = markdown_files(root)
    if args.scope:
        scopes = [(root / scope).resolve() for scope in args.scope]
        if any(not scope.is_relative_to(root) for scope in scopes):
            parser.error('--scope must stay within the checkout')
        if any(not any(p.resolve().is_relative_to(scope) for p in files) for scope in scopes):
            parser.error('each --scope must select maintained Markdown')
        files = [p for p in files if any(p.resolve().is_relative_to(scope) for scope in scopes)]
    refs, skipped = inventory(root, files)
    hosts = {host.lower() for host in args.host} if args.host else DEFAULT_HOSTS
    eligible = [url for url in sorted(refs) if urlsplit(url).hostname in hosts] if args.check else []
    selected = eligible[:args.limit]

    def probe_selected(url):
        fragments = {ref['fragment'] for ref in refs[url] if ref['fragment']} if args.anchors else ()
        return probe(url, hosts, args.timeout, fragments)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = dict(zip(selected, pool.map(probe_selected, selected)))
    entries = []
    for url in sorted(refs):
        reason = ('inventory_only' if not args.check else
                  'host_not_allowlisted' if urlsplit(url).hostname not in hosts else 'probe_budget')
        result = results.get(url, {'status': 'not_checked', 'reason': reason})
        entries.append({'url': url, 'references': refs[url], **result})
    print(json.dumps({'checked_at': datetime.now(timezone.utc).isoformat(),
                      'files_scanned': len(files), 'urls_found': len(entries),
                      'urls_probed': len(selected), 'skipped_url_occurrences': len(skipped),
                      'skipped_locations': skipped,
                      'missing_local_targets': missing_local_targets(root, files),
                      'missing_skill_routes': missing_skill_routes(root, files),
                      'entries': entries}, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
