Installation
============

Install BrainX Skill with the guided BrainX installer or add it directly to a supported coding agent. Both methods install the same portable skills; choose the guided installer when you want to select several agents at once.

Requirements
------------

- Node.js 18 or newer
- macOS, Linux, or Windows

Guided installation
-------------------

Run the interactive installer, then select one or more coding agents and either the current project or your global user configuration.

.. code-block:: bash

   npx brainx-skill install

Direct installation
-------------------

Use the Agent Skills CLI when you already know the target agent and scope.

.. code-block:: bash

   # Claude Code, current project
   npx skills add chaobrain/BrainX-skill --agent claude-code

   # Codex, current project
   npx skills add chaobrain/BrainX-skill --agent codex

   # Claude Code, globally
   npx skills add chaobrain/BrainX-skill --agent claude-code --global

To install into a specific project, change to that project first. The selected agent then receives a project-scoped installation.

.. code-block:: bash

   cd /actual/path/to/project
   npx skills add chaobrain/BrainX-skill --agent codex

Update
------

Update every BrainX-managed skill recorded by the guided installer.

.. code-block:: bash

   npx brainx-skill update

Slower connections
------------------

Use the npm mirror when the default registry is slow or unreliable.

.. code-block:: bash

   # Install
   npx --registry=https://registry.npmmirror.com brainx-skill install

   # Update
   npx --registry=https://registry.npmmirror.com brainx-skill update

Installation locations
----------------------

Project installs live under the current project. Global installs live under your user directory.

.. list-table:: Default skill directories
   :header-rows: 1
   :widths: 24 38 38
   :class: installation-paths

   * - Agent
     - Global
     - Current project
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

The guided installer records BrainX-owned files in ``~/.brainx/receipt.json``. This receipt lets later updates replace only installations that BrainX can verify it owns.
