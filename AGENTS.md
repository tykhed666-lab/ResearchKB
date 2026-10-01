# ResearchKB Agent Instructions

Before changing this repository, read `README.md` completely. It is the canonical project overview, architecture, setup, and verification document.

- Preserve existing PDF, multimodal, project-management, and SEC behavior.
- Do not introduce additional orchestration frameworks or data providers unless the user changes the scope.
- Keep Python comments and docstrings in Chinese where they explain business logic.
- Never commit `.env`, credentials, raw PDFs, local SQLite data, or generated images.
- Run `uv run python scripts/check_quality.py` before committing.
- Update `README.md` when the architecture, setup, or verification process materially changes.
