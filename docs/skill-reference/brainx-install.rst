.. rst-class:: skill-reference-page

brainx-install
==============

Complete skill: :skill-source:`source <skills/brainx-install/SKILL.md>`

Purpose and boundary
--------------------

Use ``brainx-install`` to inspect, diagnose, plan, install, upgrade, downgrade, pin, migrate, repair, verify, or remove BrainX in a confirmed Python environment. It owns environment compatibility, not the modeling task that follows installation.

Major contents
--------------

- Identify the active interpreter, environment manager, manifests, lockfiles, and installed BrainX/JAX packages.
- Match subpackage versions as an official BrainX release tuple rather than independent latest versions.
- Check Python compatibility and the intended CPU, CUDA, or TPU backend.
- Classify a working, absent, partial, drifted, or broken environment.
- Present exact changes and obtain explicit approval before modifying the environment.
- Apply the smallest approved change and verify dependencies, imports, versions, hardware, and project behavior.
- Remove BrainX deliberately while preserving the environment and unrelated dependencies.

Reference Markdown
------------------

- ``skills/brainx-install/references/compatibility-and-release-matching.md`` - :skill-source:`source <skills/brainx-install/references/compatibility-and-release-matching.md>`
- ``skills/brainx-install/references/uninstall-and-cleanup.md`` - :skill-source:`source <skills/brainx-install/references/uninstall-and-cleanup.md>`
