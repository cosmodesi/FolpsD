This repository includes a GitHub Actions workflow (.github/workflows/docs.yml) that builds the Sphinx documentation and publishes the generated HTML to the `gh-pages` branch using `peaceiris/actions-gh-pages`.

Defaults and customization

- The workflow installs only Sphinx and the RTD theme for a fast build. If you need the documentation to render `autodoc` docstrings that import the package (SciPy, CLASS, JAX, ...), modify the "Install minimal docs dependencies" step to `pip install -r requirements.txt` or a pinned requirements file for docs.

- The action deploys to `gh-pages` by default and keeps history. The `GITHUB_TOKEN` provided by Actions is used to push the content.

How to trigger

- The workflow runs on pushes and PRs to `main`. You can also run it manually via the Actions tab (workflow_dispatch).
