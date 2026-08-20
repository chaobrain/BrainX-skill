Supported agents
================

The same skill bundle installs into seven coding agents. Each agent reads skills from its own directory, so the installer resolves one destination per agent and per scope.

Installation paths
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Agent
     - Global scope
     - Project scope
   * - Claude Code
     - ``~/.claude/skills/<skill-name>``
     - ``<cwd>/.claude/skills/<skill-name>``
   * - Codex
     - ``~/.agents/skills/<skill-name>``
     - ``<cwd>/.agents/skills/<skill-name>``
   * - Cursor
     - ``~/.cursor/skills/<skill-name>``
     - ``<cwd>/.cursor/skills/<skill-name>``
   * - Windsurf
     - ``~/.codeium/windsurf/skills/<skill-name>``
     - ``<cwd>/.codeium/windsurf/skills/<skill-name>``
   * - Gemini CLI
     - ``~/.gemini/skills/<skill-name>``
     - ``<cwd>/.gemini/skills/<skill-name>``
   * - Antigravity
     - ``~/.gemini/config/skills/<skill-name>``
     - ``<cwd>/.agents/skills/<skill-name>``
   * - OpenCode
     - ``~/.config/opencode/skills/<skill-name>``
     - ``<cwd>/.config/opencode/skills/<skill-name>``

Antigravity is the only agent whose project path differs from its global path. It reads global skills from ``~/.gemini/config/skills`` and workspace skills from ``<cwd>/.agents/skills``.

Shared destinations
-------------------

In project scope, Antigravity's workspace directory is the same one Codex uses. The installer groups adapters by resolved destination, so selecting both agents writes that directory once and records it for both.

Ownership is a property of the destination directory rather than of a single agent: a directory installed for Codex is still BrainX-owned when Antigravity is added later.

Scope selection
---------------

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - Scope
     - Base directory
     - Use when
   * - ``global``
     - your home directory
     - The skills should be available in every project on this machine.
   * - ``project``
     - the current working directory
     - The skills should be committed with, or scoped to, one repository.

Installation requires a terminal. In a non-interactive shell the installer exits with ``Interactive installation requires a terminal.`` rather than guessing a destination.

Ownership receipt
-----------------

Installation ownership is recorded in ``~/.brainx/receipt.json``. The receipt stores, per adapter, the label and the absolute destination that BrainX wrote, and it is validated on every later run: a receipt from a different package, an unknown adapter id, a relative destination, or a destination that does not end in that adapter's own skill path is rejected before anything is changed.

The receipt always lives under your home directory, even for a project-scope installation, because it tracks what BrainX wrote anywhere on the machine.

``npx brainx-skill update`` is receipt-driven. It selects only the adapters that already have a BrainX-managed record and rewrites those exact destinations. It never asks which agents to target and never introduces a new destination. With no managed installation it reports:

.. code-block:: text

   No BrainX-managed skills are currently installed.
   Run: npx brainx-skill install

Installing through the ``skills`` CLI
-------------------------------------

The bundle is also a plain Agent Skills repository, so the generic ``skills`` CLI can install it without going through ``brainx-skill``.

.. code-block:: bash

   # Claude Code, current project
   npx skills add chaobrain/BrainX-skill --agent claude-code

   # Codex, current project
   npx skills add chaobrain/BrainX-skill --agent codex

   # Claude Code, globally
   npx skills add chaobrain/BrainX-skill --agent claude-code --global

There is no BrainX receipt in that case, so ``npx brainx-skill update`` will not see the installation. Update it the same way you installed it.

Slow or blocked connections
---------------------------

.. code-block:: bash

   npx --registry=https://registry.npmmirror.com brainx-skill install
   npx --registry=https://registry.npmmirror.com brainx-skill update

Failure reporting
-----------------

When a destination cannot be written, the installer reports the agent label, the affected path, the action that was **not** performed, and how to resolve it, then leaves the other destinations untouched. An installation that fails for one agent does not partially change another.
