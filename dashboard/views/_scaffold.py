"""What a view looks like before it has a chart in it.

BI-02 builds the shell; the four views land in BI-03 and BI-04. A stub that
said only "coming soon" would leave the thing this ticket is actually
responsible for — that the app reaches Neon from every page, wakes it without
erroring, and reuses one palette — untested until the day the charts arrive.

So each stub renders the parts of its view that already exist: the question it
answers, the plain-English caption it will carry, the colour key it will use,
and a live probe of the table it will read. The probe is a real query through
the real connection layer, which means opening any of the four pages exercises
the cold-start path end to end.
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

__all__ = ["PendingView", "current_mode"]


def current_mode() -> theme.Mode:
    """The palette mode matching the viewer's Streamlit theme."""
    return theme.resolve_mode(getattr(st.context.theme, "type", None))


def _readable(value: object) -> str:
    """A count with thousands separators, a date as it is, a null as a dash.

    ``numbers.Integral`` rather than ``int``: pandas hands back ``numpy.int64``,
    which is not an ``int`` subclass, and the naive check silently falls through
    to ``str`` — turning 60,396 into 60396 on a page whose job is legibility.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "\u2014"
    if isinstance(value, numbers.Integral):
        return f"{int(value):,}"
    return str(value)


@dataclass(frozen=True)
class PendingView:
    """One of the four views, before its chart exists.

    Attributes:
        title: Heading, and the label in the navigation.
        question: The question the finished view answers, from §Workstream 4
            of the proposal. Kept verbatim so the built view can be checked
            against what was promised.
        caption: The one-line plain-English caption the acceptance criteria
            require every view to carry.
        encoding: ``"diverging"`` for the views that colour by signed anomaly —
            they render the shared key — or a sentence describing what the view
            uses instead.
        ticket: The ticket that fills this page in.
        source: Which gold table it reads, and a query returning one row of
            coverage facts about it.
    """

    title: str
    icon: str
    url_path: str
    question: str
    caption: str
    encoding: str
    ticket: str
    source_table: str
    probe_sql: str
    probe_labels: Mapping[str, str]

    def render(self) -> None:
        mode = current_mode()

        st.title(self.title)
        st.caption(self.caption)

        st.markdown(f"**Answers:** {self.question}")

        if self.encoding == "diverging":
            st.markdown(
                theme.diverging_legend_html(mode),
                unsafe_allow_html=True,
            )
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
