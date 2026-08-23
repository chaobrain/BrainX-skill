import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const workflow = readFileSync(
  new URL('../workflows/agent-skills-validation.yml', import.meta.url),
  'utf8',
);

test('invokes the Agent Skills validator through its Python CLI', () => {
  assert.doesNotMatch(
    workflow,
    /^\s+skills-ref(?:\s|$)/m,
    'skills-ref 0.1.1 does not install a console command on GitHub runners',
  );
  assert.match(
    workflow,
    /python -c 'from skills_ref\.cli import main; main\(\)' validate "\$skill_dir"/,
  );
  assert.match(
    workflow,
    /find skills -type f -name SKILL\.md -printf '%h\\0'/,
  );
});
