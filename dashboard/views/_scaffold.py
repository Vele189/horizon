"""What every view has in common, and what a view looks like before it has a chart.

Two dataclasses. :class:`ViewMeta` is the part the shell needs from every view
whether or not it is built — its title, its place in the navigation, the
question it answers and the one-line caption the acceptance criteria require.
:class:`PendingView` adds what a *stub* needs: the ticket that fills it in, and
a live probe of the mart it will read.

The stubs render a real query through the real connection layer rather than a
"coming soon". BI-02 is responsible for the app reaching Neon from every page
and waking it without erroring, and leaving three of the four pages inert would
leave that unexercised until the day their charts arrive.
"""

from __future__ import annotations

import math
import numbers
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dashboard import theme  # noqa: E402
from dashboard.database import run_query  # noqa: E402

__all__ = ["PendingView", "ViewMeta", "current_mode"]


def current_mode() -> theme.Mode:
    """The palette mode matching the viewer's Streamlit theme.

    Views painted on their own dark surface — the map — do not use this; the
    ramp follows the surface it sits on, not the theme of the page around it.
    """
    return theme.resolve_mode(getattr(st.context.theme, "type", None))


def _readable(value: object) -> str:
    """A count with thousands separators, a date as it is, a null as a dash.

    ``numbers.Integral`` rather than ``int``: pandas hands back ``numpy.int64``,
    which is not an ``int`` subclass, and the naive check silently falls through
    to ``str`` — turning 60,396 into 60396 on a page whose job is legibility.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if isinstance(value, numbers.Integral):
        return f"{int(value):,}"
    return str(value)


@dataclass(frozen=True)
class ViewMeta:
    """What the shell needs from a view, built or not.

    Attributes:
        title: Heading, and the label in the navigation.
        question: The question the view answers, from §Workstream 4 of the
            proposal. Kept verbatim so a built view can be checked against
            what was promised.
        caption: The one-line plain-English caption every view must carry.
        source_table: The gold mart it reads. Named here so a test can assert
            no view reaches for a layer that was never promoted.
    """

    title: str
    icon: str
    url_path: str
    question: str
    caption: str
    source_table: str


@dataclass(frozen=True)
class PendingView(ViewMeta):
    """A view whose chart has not landed yet.

    Attributes:
        encoding: ``"diverging"`` for the views that colour by signed anomaly —
            they render the shared key — or a sentence describing what the view
            uses instead.
        ticket: The ticket that fills this page in.
        probe_sql: A query returning one row of coverage facts about the source.
    """

    encoding: str
    ticket: str
    probe_sql: str
    probe_labels: Mapping[str, str]

    def render(self) -> None:
        mode = current_mode()

        st.title(self.title)
        st.caption(self.caption)
        st.markdown(f"**Answers:** {self.question}")

        if self.encoding == "diverging":
            st.markdown(theme.diverging_legend_html(mode), unsafe_allow_html=True)
        else:
            st.info(self.encoding, icon=":material/palette:")

        st.divider()
        st.subheader("Source", anchor=False)
        st.caption(
            f"`{self.source_table}` on the serving database, read live through "
            f"the cached connection layer. The chart lands in **{self.ticket}**."
        )

        frame = run_query(self.probe_sql)
        row = frame.iloc[0]
        columns = st.columns(len(self.probe_labels))
        for column, (field, label) in zip(columns, self.probe_labels.items()):
            column.metric(label, _readable(row[field]))
