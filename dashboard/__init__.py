"""The Streamlit dashboard: shell, connection layer, palette, and four views.

``app.py`` is the entrypoint Streamlit runs. Everything it needs is split by
what changes for what reason — :mod:`dashboard.theme` when the visual language
changes, :mod:`dashboard.database` when the warehouse or its hosting changes,
and :mod:`dashboard.views` when a chart changes.
"""

from __future__ import annotations

__all__: list[str] = []
