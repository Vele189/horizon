"""The four dashboard views, one module each.

The proposal sketched a single ``app.py``. Four views in one file was already
going to be unpleasant at the end of BI-04, and it also makes the navigation
impossible to test without executing the charts, so each view is its own module
and ``app.py`` is the shell that arranges them.
"""

from __future__ import annotations

from dashboard.views import anomaly_map, climate_matrix, risk_horizon, storm_dynamics

__all__ = [
    "ORDER",
    "anomaly_map",
    "climate_matrix",
    "risk_horizon",
    "storm_dynamics",
]

# The order the proposal lists them in, and the order they appear in the
# sidebar. Declared once so the navigation and its test read the same list.
ORDER = (anomaly_map, climate_matrix, storm_dynamics, risk_horizon)

