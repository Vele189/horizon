"""What every view has in common.

:class:`ViewMeta` is the part the shell needs from each view — its title, its
place in the navigation, the question it answers, the one-line caption the
acceptance criteria require, and the gold mart it reads.

It carried a second dataclass until BI-06. ``PendingView`` rendered a stub
page — the question, the key it would use, and a live probe of its mart — so
that the connection layer BI-02 built was exercised from every page rather than
left unproven until the charts arrived. All four views are built now, so it is
gone: a scaffold kept after the building is finished is just something else to
maintain.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dashboard import theme  # noqa: E402

__all__ = ["ViewMeta", "current_mode"]


def current_mode() -> theme.Mode:
    """The palette mode matching the viewer's Streamlit theme.

    Views painted on their own dark surface — the map — do not use this; the
    ramp follows the surface it sits on, not the theme of the page around it.
    """
    return theme.resolve_mode(getattr(st.context.theme, "type", None))


@dataclass(frozen=True)
class ViewMeta:
    """What the shell needs from a view.

    Attributes:
        title: Heading, and the label in the navigation.
        question: The question the view answers, from §Workstream 4 of the
            proposal. Kept verbatim so a built view can be checked against what
            was promised.
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
