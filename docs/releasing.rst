Releasing and deploying
=======================

Two GitHub Actions workflows own everything that leaves the repository. Neither is triggered by an ordinary push to ``main``.

Release the npm package
-----------------------

**GitHub Actions → Release npm package and notes → Run workflow.**

Releases are manual and workflow-owned. Do not bump ``package.json`` ``version``, write ``CHANGELOG.md``, or create tags by hand; the release tooling does all three.

Running only on ``main`` of ``chaobrain/BrainX-skill``, the workflow:

#. runs ``npm test``;
#. recovers an incomplete tagged release if a previous run failed partway;
#. prepares the version, changelog, and release notes;
#. commits ``chore(release): v<version>``, tags ``v<version>``, and pushes atomically;
#. publishes to npm with provenance;
#. creates the GitHub Release from ``RELEASE_NOTES.md``.

Deploy the documentation
------------------------

**GitHub Actions → Deploy Docs.** The workflow runs on a published GitHub Release, and can also be dispatched manually.

.. warning::

   The release workflow creates its GitHub Release with the default ``GITHUB_TOKEN``. GitHub does not fire workflow-triggering events for actions taken with that token, so the ``release`` trigger will **not** fire on its own. Either give the release workflow a personal access token, or dispatch **Deploy Docs** manually after each release.

The deployment mirrors the pattern used by every other BrainX package site.

.. code-block:: text

   build            sphinx-build -b html docs docs/_build/html, assert index.html exists
   upload artifact  docs-html
   prepare          ssh: mkdir <base>/releases/<timestamp>-<sha7>
   upload           scp docs/_build/html/* into that release directory
   activate         ssh: symlink swap current -> the new release, then prune older releases
   reload           ssh: nginx -t && systemctl reload nginx

``DEPLOY_PATH`` is ``/var/www/brainx.chaobrain.com.pkgs/skills``, served at https://brainx.chaobrain.com/skills/.

Activation is atomic. The new release is linked only after ``index.html`` is confirmed present, and the swap is a single ``mv -Tf`` of a symlink, so a failed build never replaces a working site.

The build job sets ``BRAINX_DOCS_PRODUCTION=1``, which is what enables ``brainx_inject_base`` in ``docs/conf.py``. Local previews leave it unset and stay self-contained.

Required secrets
----------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Secret
     - Purpose
   * - ``DEPLOY_HOST``
     - Documentation server hostname.
   * - ``DEPLOY_PORT``
     - SSH port.
   * - ``DEPLOY_USER``
     - SSH user with write access to ``DEPLOY_PATH``.
   * - ``DEPLOY_SSH_KEY``
     - Private key for that user.

Server-side prerequisite
------------------------

The server needs an nginx location for the documentation path. Pair it with an exact-match block so the no-slash URL is served without a redirect, which is what ``brainx_inject_base`` is designed for.

.. code-block:: nginx

   location = /skills { alias /var/www/brainx.chaobrain.com.pkgs/skills/current/index.html; default_type text/html; }

   location /skills/ { alias /var/www/brainx.chaobrain.com.pkgs/skills/current/; try_files $uri $uri/ $uri/index.html =404; }

Both blocks go in the single ``brainx.chaobrain.com`` server block, alongside the equivalent pair for every other BrainX package site.

Build the documentation locally
-------------------------------

.. code-block:: bash

   python -m pip install -r requirements-doc.txt
   sphinx-build -b html docs docs/_build/html

Then open ``docs/_build/html/index.html``.

To preview an unpublished change to the shared BrainX header against a local landing-site build:

.. code-block:: bash

   BRAINX_HEADER_LOCAL=/path/to/brainx.chaobrain.com/dist/shared-header \
     sphinx-build -b html docs docs/_build/html

Set ``BRAINX_HEADER_TTL=0`` to force a fresh fetch of the brand header, as CI does.

Validate skills before opening a pull request
---------------------------------------------

A documentation change does not exempt a branch from the skill validators.

.. code-block:: bash

   python -m pip install --disable-pip-version-check skills-ref==0.1.1
   for d in skills/*/; do
     python -c 'from skills_ref.cli import main; main()' validate "$d"
   done
   node --test .github/scripts/*.test.mjs

Neither check reads routing tables, so neither catches a dangling reference target or an unrouted file. Verify those by hand against the rules in ``AGENTS.md``.
