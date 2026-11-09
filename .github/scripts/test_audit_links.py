from email.message import Message
from io import BytesIO
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[2]
AUDIT = runpy.run_path(str(ROOT / '.agents/skills/brainx-doc-sync/scripts/audit_links.py'))


class AuditLinksTest(unittest.TestCase):
    def test_table_links_and_skipped_query_locations(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            page = root / 'table.md'
            page.write_text(
                '|[A](https://example.com/a)|[B](https://example.com/b_(c)#section)|\n'
                '[Query](https://example.com/search?q=test)\n',
                encoding='utf-8',
            )
            refs, skipped = AUDIT['inventory'](root, [page])
            self.assertEqual(list(refs), ['https://example.com/a', 'https://example.com/b_(c)'])
            self.assertEqual(refs['https://example.com/b_(c)'][0]['fragment'], 'section')
            self.assertEqual(skipped, [{'file': 'table.md', 'line': 2}])

    def test_missing_relative_targets_ignore_examples(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            page = root / 'README.md'
            (root / 'present (1).md').touch()
            page.write_text(
                '[Present](<present (1).md#heading>)\n'
                '[Missing](missing(1).md)\n'
                '`[Inline example](example.md)`\n'
                '```markdown\n[Fenced example](example.md)\n```\n',
                encoding='utf-8',
            )
            self.assertEqual(AUDIT['missing_local_targets'](root, [page]), [
                {'file': 'README.md', 'line': 2, 'target': 'missing(1).md'},
            ])

    def test_static_fragment_lookup(self):
        response = BytesIO('<h2 id="café">Title</h2><a name="old"></a>'.encode())
        response.headers = Message()
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        self.assertEqual(AUDIT['check_anchors'](response, ['caf%C3%A9', 'old', 'absent']), {
            'caf%C3%A9': 'found',
            'old': 'found',
            'absent': 'missing_candidate',
        })

    def test_skill_routes_follow_published_roots(self):
        with TemporaryDirectory() as directory:
            nested = Path(directory).resolve() / 'nested'
            nested.mkdir()
            root = nested / '..'
            (root / 'manifest.json').write_text('{"skills":["workflow","model"]}', encoding='utf-8')
            workflow = root / 'skills/workflow'
            model = root / 'skills/package-skills/model'
            (workflow / 'references').mkdir(parents=True)
            (model / 'references').mkdir(parents=True)
            (workflow / 'SKILL.md').touch()
            (model / 'SKILL.md').touch()
            (model / 'references/input.md').touch()
            page = workflow / 'references/start.md'
            page.write_text(
                '`model` → `references/input.md`\n'
                '`model` → `references/absent.md`\n'
                '`references/start.md`\n'
                '`model` → `SKILL.md`\n'
                '`unknown` → `SKILL.md`\n'
                '`skills/package-skills/model/SKILL.md`\n'
                '`model` →\n'
                '  `references/input.md`\n'
                '`unknown`\n'
                '  -> `SKILL.md`\n'
                '\n'
                'Diagram label: `unknown` →\n'
                '\n'
                '`references/start.md` is a local route.\n'
                '```markdown\n'
                '`unknown` → `SKILL.md`\n'
                '`skills/package-skills/model/SKILL.md`\n'
                '```\n'
                '~~~text\n'
                '`unknown` -> `SKILL.md`\n'
                '~~~\n'
                '`model` → `references/after-fence.md`\n',
                encoding='utf-8',
            )
            self.assertEqual(AUDIT['missing_skill_routes'](root, [page]), [
                {'file': 'skills/workflow/references/start.md', 'line': 2,
                 'skill': 'model', 'target': 'references/absent.md'},
                {'file': 'skills/workflow/references/start.md', 'line': 5,
                 'skill': 'unknown', 'target': 'SKILL.md'},
                {'file': 'skills/workflow/references/start.md', 'line': 6,
                 'skill': 'workflow', 'target': 'skills/package-skills/model/SKILL.md'},
                {'file': 'skills/workflow/references/start.md', 'line': 10,
                 'skill': 'unknown', 'target': 'SKILL.md'},
                {'file': 'skills/workflow/references/start.md', 'line': 22,
                 'skill': 'model', 'target': 'references/after-fence.md'},
            ])


if __name__ == '__main__':
    unittest.main()
