import { execFileSync } from 'node:child_process';
import {
  appendFileSync,
  readFileSync,
  realpathSync,
  writeFileSync,
} from 'node:fs';
import { fileURLToPath } from 'node:url';

const FIELD_SEPARATOR = '\u001f';
const RECORD_SEPARATOR = '\u001e';

function git(args) {
  return execFileSync('git', args, {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  }).trim();
}

export function bumpPatch(version) {
  const match = /^(\d+)\.(\d+)\.(\d+)$/.exec(version);
  if (!match) {
    throw new Error(`Expected a semantic version, received "${version}"`);
  }

  return `${match[1]}.${match[2]}.${Number(match[3]) + 1}`;
}

function versionParts(version) {
  const match = /^(\d+)\.(\d+)\.(\d+)$/.exec(version);
  if (!match) {
    throw new Error(`Expected a semantic version, received "${version}"`);
  }
  return match.slice(1).map(Number);
}

export function compareVersions(left, right) {
  const leftParts = versionParts(left);
  const rightParts = versionParts(right);
  for (let index = 0; index < leftParts.length; index += 1) {
    if (leftParts[index] !== rightParts[index]) {
      return Math.sign(leftParts[index] - rightParts[index]);
    }
  }
  return 0;
}

export function selectPreviousRef({
  packageVersion,
  npmPublishedVersion,
  latestTag,
  npmVersionTagExists,
  catchUp,
}) {
  const packageTag = `v${packageVersion}`;
  if (latestTag !== packageTag) {
    throw new Error(
      `package.json version ${packageVersion} does not match ${latestTag}`,
    );
  }

  const versionOrder = compareVersions(npmPublishedVersion, packageVersion);
  if (versionOrder > 0) {
    throw new Error(
      `npm version ${npmPublishedVersion} is ahead of package.json ${packageVersion}`,
    );
  }
  if (versionOrder === 0) return latestTag;

  if (npmVersionTagExists) return `v${npmPublishedVersion}`;

  if (
    catchUp?.publishedVersion === npmPublishedVersion &&
    catchUp.previousRef
  ) {
    return catchUp.previousRef;
  }

  throw new Error(
    `No Git comparison baseline is configured for npm ${npmPublishedVersion}`,
  );
}

export function parseCommitLog(log) {
  return log
    .split(RECORD_SEPARATOR)
    .map((record) => record.trim())
    .filter(Boolean)
    .map((record) => {
      const [sha, subject, ...bodyParts] = record.split(FIELD_SEPARATOR);
      return {
        sha: sha.trim(),
        subject: subject.trim(),
        body: bodyParts.join(FIELD_SEPARATOR).trim(),
      };
    });
}

function sectionFor(subject) {
  if (/^[a-z]+(?:\([^)]*\))?!:/.test(subject)) return 'Breaking changes';
  if (/^feat(?:\([^)]*\))?:/.test(subject)) return 'New features';
  if (/^fix(?:\([^)]*\))?:/.test(subject)) return 'Fixes';
  if (/^perf(?:\([^)]*\))?:/.test(subject)) return 'Performance';
  if (/^docs(?:\([^)]*\))?:/.test(subject)) return 'Documentation';
  return 'Other changes';
}

function changedFilesSummary(files) {
  const visible = files.slice(0, 12).map((file) => `\`${file}\``);
  if (files.length > visible.length) {
    visible.push(`and ${files.length - visible.length} more`);
  }
  return visible.join(', ');
}

export function renderReleaseNotes({
  version,
  commits,
  repository,
  previousRef,
  headingLevel = 2,
}) {
  const groups = new Map();
  for (const commit of commits) {
    const section = sectionFor(commit.subject);
    const group = groups.get(section) ?? [];
    group.push(commit);
    groups.set(section, group);
  }

  const lines = [
    `BrainX Skill ${version} records ${commits.length} commit${commits.length === 1 ? '' : 's'} merged into \`main\`.`,
    '',
    `${'#'.repeat(headingLevel)} Release record`,
    '',
    `- **Version:** \`${version}\``,
    `- **Previous release:** \`${previousRef}\``,
    `- **Commits included:** ${commits.length}`,
    '',
  ];

  const order = [
    'Breaking changes',
    'New features',
    'Fixes',
    'Performance',
    'Documentation',
    'Other changes',
  ];

  for (const section of order) {
    const group = groups.get(section);
    if (!group) continue;

    lines.push(`${'#'.repeat(headingLevel)} ${section}`, '');
    for (const commit of group) {
      const shortSha = commit.sha.slice(0, 7);
      const commitUrl = `https://github.com/${repository}/commit/${commit.sha}`;
      lines.push(
        `${'#'.repeat(headingLevel + 1)} ${commit.subject} ([\`${shortSha}\`](${commitUrl}))`,
        '',
        commit.body || 'No additional commit description was provided.',
        '',
      );
      if (commit.files.length > 0) {
        lines.push(`**Changed files:** ${changedFilesSummary(commit.files)}`, '');
      }
    }
  }

  lines.push(
    `**Full Changelog:** https://github.com/${repository}/compare/${previousRef}...v${version}`,
    '',
  );
  return lines.join('\n');
}

export function updateChangelog(changelog, version, date, notes) {
  const firstRelease = changelog.indexOf('\n## ');
  if (firstRelease === -1) {
    throw new Error('CHANGELOG.md must contain an existing version heading');
  }

  const entry = `\n## ${version} - ${date}\n\n${notes.trim()}\n`;
  return `${changelog.slice(0, firstRelease)}${entry}${changelog.slice(firstRelease)}`;
}

function tryLatestTag() {
  try {
    return git(['describe', '--tags', '--abbrev=0', '--match', 'v[0-9]*']);
  } catch {
    return null;
  }
}

function refExists(ref) {
  try {
    git(['rev-parse', '--verify', '--quiet', `${ref}^{commit}`]);
    return true;
  } catch {
    return false;
  }
}

function collectCommits(range) {
  const log = git([
    'log',
    '--reverse',
    '--format=%H%x1f%s%x1f%b%x1e',
    range,
  ]);

  return parseCommitLog(log)
    .filter((commit) => !commit.subject.startsWith('chore(release):'))
    .map((commit) => ({
      ...commit,
      files: git([
        'diff-tree',
        '--root',
        '--no-commit-id',
        '--name-only',
        '-r',
        commit.sha,
      ])
        .split('\n')
        .filter(Boolean),
    }));
}

function writeOutput(name, value) {
  if (process.env.GITHUB_OUTPUT) {
    appendFileSync(process.env.GITHUB_OUTPUT, `${name}=${value}\n`);
  } else {
    console.log(`${name}=${value}`);
  }
}

export function main() {
  const packagePath = 'package.json';
  const changelogPath = 'CHANGELOG.md';
  const notesPath = 'RELEASE_NOTES.md';
  const config = JSON.parse(readFileSync('release-notes.config.json', 'utf8'));
  const packageJson = JSON.parse(readFileSync(packagePath, 'utf8'));
  const repository = process.env.GITHUB_REPOSITORY ?? config.repository;
  const npmPublishedVersion = process.env.NPM_PUBLISHED_VERSION;

  if (repository !== config.repository) {
    throw new Error(
      `Release repository must be ${config.repository}, received ${repository}`,
    );
  }

  const latestTag = tryLatestTag();
  if (!latestTag) {
    throw new Error('A version tag is required before preparing a release');
  }
  if (!npmPublishedVersion) {
    throw new Error(
      'NPM_PUBLISHED_VERSION is required before preparing a release',
    );
  }

  const previousRef = selectPreviousRef({
    packageVersion: packageJson.version,
    npmPublishedVersion,
    latestTag,
    npmVersionTagExists: refExists(`v${npmPublishedVersion}`),
    catchUp: config.npmCatchUp,
  });
  try {
    git(['merge-base', '--is-ancestor', previousRef, 'HEAD']);
  } catch {
    throw new Error(`Release baseline ${previousRef} is not an ancestor of HEAD`);
  }

  const commits = collectCommits(`${previousRef}..HEAD`);
  if (commits.length === 0) {
    writeOutput('skip', 'true');
    return;
  }

  const version = bumpPatch(packageJson.version);
  const releaseNotes = renderReleaseNotes({
    version,
    commits,
    repository,
    previousRef,
  });
  const changelogNotes = renderReleaseNotes({
    version,
    commits,
    repository,
    previousRef,
    headingLevel: 3,
  });
  const date = new Date().toISOString().slice(0, 10);

  packageJson.version = version;
  writeFileSync(packagePath, `${JSON.stringify(packageJson, null, 2)}\n`);
  writeFileSync(
    changelogPath,
    updateChangelog(
      readFileSync(changelogPath, 'utf8'),
      version,
      date,
      changelogNotes,
    ),
  );
  writeFileSync(notesPath, `${releaseNotes.trim()}\n`);

  writeOutput('skip', 'false');
  writeOutput('version', version);
  writeOutput('commit_count', String(commits.length));
}

const entryPoint = process.argv[1] ? realpathSync(process.argv[1]) : null;
if (entryPoint === fileURLToPath(import.meta.url)) {
  main();
}
