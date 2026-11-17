---
name: brainx-doc-sync
description: Audit and maintain BrainX skill Markdown, official documentation links, version-specific API claims, and bundled examples. Use to repair stale links, reconcile references with a declared BrainX release, or prepare a verified documentation refresh before freezing a refinement snapshot.
---

# BrainX documentation sync

## Purpose and boundary

Maintain the BrainX skill-authoring checkout. Use this development skill
from the repository. Scan the maintained corpus by
default; edit only within the requested scope and only where evidence supports
the change. For an audit request, report findings without modifying sources.

Keep three questions separate: can the link be reached, does its content
support the claim for the target release, and does the documented workflow run
in that environment? A pass on one does not establish the others.

## Inventory and verify

1. Read `AGENTS.md`, the relevant part of `plan.md`, and each target skill
   before editing. Record the requested scope and target BrainX release from
   the user's pin, project lock, or evaluation image receipt. Compare it with
   the checkout's advertised support in `README.md` and the install skill's
   compatibility matrix before changing runnable examples. Report conflicts
   and preserve supported versions unless the task changes the support policy.
2. Run the read-only inventory. Add `--check` for bounded HTTP probes; use
   `--scope` for focused work and `--anchors` with `--check` for static HTML
   fragments. Review missing local targets and skipped locations using
   `references/evidence.md`.

   ```bash
   python .agents/skills/brainx-doc-sync/scripts/audit_links.py --root . --check --limit 80
   ```

   Use the installed skill's script path when it lives outside the checkout.
   Treat the inventory as candidate locations, not proof of link identity or
   complete semantic coverage. Report links skipped by scope or budget.
3. Review version-sensitive claims and runnable examples by package; use the
   target release's change notes to prioritize. Track any unreviewed scope.
   Match each changed API claim to an official reference for the exact symbol,
   parameter, or behavior. Prefer sources tied to the target release. Use
   official examples for composition and release notes for version changes.
   Treat live unversioned pages as discovery evidence until reconciled with
   the target. Never inspect installed implementation internals for contracts.

## Repair and validate

1. Replace a dead link only after confirming the destination supports the same
   claim. Cite the exact API section beside the affected text and use
   `source_html_references/` for discovery. Keep unknown-version or unsupported
   claims explicit rather than rewriting an entire URL prefix.
   When editing an inventory, keep links that resolve to the intended content,
   including through redirects. Match the inventory's inclusion criteria and
   neighboring format before adding entries. Include deprecated module pages
   only when the user asks for them or the inventory explicitly tracks them.
2. Change signatures, imports, shapes, units, examples, or behavior only when
   the target release warrants it. Edit shared `braintools-references/` sources
   first. State each namespace rule once per reference; do not repeat it under
   every API family. Keep the full shared release and migration note in the
   authoring source.
3. Sync changed API facts and examples, not whole files. Preserve each
   installed skill's package-specific guidance and keep its copy usable on its
   own: name the canonical namespace and include a short version condition
   only where it changes the action. Do not copy release history or require a
   link to the repository's authoring source.
4. Review the changed source and its copies together for repeated instructions,
   duplicated migration notes, and missing package-specific guidance. Recheck
   changed destinations and local routes. Execute affected examples or
   focused behavior probes in the declared runtime when execution semantics
   change. Report the exact version, path tested, and limits. If the runtime or
   evidence is unavailable, leave the contract unresolved rather than claiming
   validation. Check the diff and preserve unrelated work.
5. Summarize inspected scope, evidence-backed edits, unresolved items, and
   checks. Separate link-only edits from API or runtime migrations. Do not
   report the corpus as verified merely because its links were inventoried.

## Reference routing

- Open `references/evidence.md` for HTTP result interpretation, claim evidence,
  and what to record when a reference is unversioned or unavailable.
- Open `references/refinement-handoff.md` when preparing a harness run or when
  a maintenance request changes the target release, for freeze and comparison
  rules. Run maintenance outside the evaluator's frozen workspace.
