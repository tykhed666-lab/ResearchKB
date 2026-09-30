# ResearchKB Agent Instructions

Before changing this repository, read `docs/项目交接说明.md` completely. It is the canonical project history, architecture, runtime-state, constraints, and next-work document.

- Continue from Day 14; Day 13 is complete in the current main branch.
- Preserve existing PDF, multimodal, project-management, and SEC behavior.
- Do not introduce LangGraph, MCP, or additional data providers unless the user changes the scope.
- Keep Python comments and docstrings in Chinese where they explain business logic.
- Never commit `.env`, credentials, raw PDFs, local SQLite data, or generated images.
- Run relevant tests and `uv run pytest -q` before committing.
- Update the handoff document when the implementation state materially changes.
