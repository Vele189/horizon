"""The dashboard shell: navigation, connection status, and failure states.

Run it with::

    streamlit run dashboard/app.py

There is nothing to configure first if ``.env`` already has
``SERVING_DATABASE_URL``; see :mod:`dashboard.database` for the full order in
which a connection string is looked for.

This file arranges. It holds no SQL, no colours, and no analysis. Those belong
to :mod:`dashboard.database`, :mod:`dashboard.theme` and the view modules
respectively, and keeping them out is what lets a view be rewritten in BI-03
without touching navigation.

**The failure states are the substance here.** A dashboard at a public URL is
read by people who cannot fix it and will not read a traceback, and the three
ways this one can fail are all ordinary rather than exceptional: the app may be
deployed before its secret is set, Neon may be asleep, and it may be deployed
before the marts have been promoted into it. Each gets a panel that says what
happened, what it means, and what to do, plus for the second a button to try
again, because "the database was waking up" is a condition that resolves on its
own.

The third is the one most easily left out, because it is not a *connection*
failure: the database answers, and then says the table does not exist. It fails
one page at a time rather than the app, so without a panel of its own it is the
one that reaches a visitor as a stack trace.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError
from streamlit.navigation.page import StreamlitPage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard import views  # noqa: E402
from dashboard.database import (  # noqa: E402
    CACHE_TTL_SECONDS,
    DashboardConfigError,
    Source,
    WarehouseUnreachable,
    clear_caches,
    get_engine,
    resolve_database_url,
    warehouse_status,
)

PAGE_TITLE = "Climate Volatility & Risk Engine"
PAGE_ICON = ":material/thermostat:"

# A public dashboard with no route back to its method is a number without a
# provenance, which is the thing this project exists not to produce.
REPOSITORY_URL = "https://github.com/Vele189/horizon"


def _navigation() -> StreamlitPage:
    """The four views, in the order the proposal lists them.

    Built from :data:`dashboard.views.ORDER` rather than a second list here.
    Pages are declared as callables rather than script paths so that importing
    this module does not execute a view, which is what makes the navigation
    testable without a database.
    """
    return st.navigation(
        [
            st.Page(
                module.render,
                title=module.VIEW.title,
                icon=module.VIEW.icon,
                url_path=module.VIEW.url_path,
                default=index == 0,
            )
            for index, module in enumerate(views.ORDER)
        ]
    )


def _sidebar_status(source: Source) -> None:
    """Where the data came from and how fresh it is.

    Degrades rather than raises. If the warehouse cannot be reached the main
    panel already says so at length; repeating it in the sidebar would push the
    navigation off the screen for the one thing a visitor can still usefully do,
    which is click something else.
    """
    st.sidebar.divider()
    st.sidebar.subheader("Warehouse", anchor=False)

    try:
        status = warehouse_status()
    except (WarehouseUnreachable, DashboardConfigError):
        st.sidebar.warning("Not reachable", icon=":material/cloud_off:")
        return
    except SQLAlchemyError:
        st.sidebar.warning("No marts yet", icon=":material/database_off:")
        return

    latest_observation = status.get("latest_observation")
    latest_forecast = status.get("latest_forecast")

    st.sidebar.caption(f"**Host** `{source.host}`")
    st.sidebar.caption(f"**Cities** {status.get('cities', '—')}")
    st.sidebar.caption(f"**Observed to** {latest_observation or '—'}")
    st.sidebar.caption(f"**Scored for** {latest_forecast or '—'}")


def _sidebar_controls() -> None:
    """The escape hatch the six-hour cache needs.

    A promotion is otherwise invisible for up to six hours. This is cheaper
    than a shorter TTL: it costs a wake-up only when someone asks for one,
    where a shorter TTL costs one on a timer whether anyone is looking or not.
    """
    st.sidebar.divider()
    hours = CACHE_TTL_SECONDS // 3600
    if st.sidebar.button(
        "Refresh data",
        icon=":material/refresh:",
        width="stretch",
        help=f"Results are cached for {hours} hours. This drops them and re-reads.",
    ):
        clear_caches()
        st.rerun()
    st.sidebar.caption(f"Cached for {hours} hours.")


def _render_config_error(exc: DashboardConfigError) -> None:
    st.title(PAGE_TITLE)
    st.error("This dashboard has no database to read.", icon=":material/key_off:")
    st.markdown(str(exc))
    st.caption(
        "No connection string is stored in this repository, and none ever has "
        "been, which is why one has to be supplied."
    )


def _render_unreachable(exc: WarehouseUnreachable, source: Source) -> None:
    st.error("The warehouse did not answer.", icon=":material/cloud_off:")
    st.markdown(
        f"""
{exc}

The serving database scales its compute to zero after five minutes idle and
resumes on the next query, so this is most often a resume that took longer
than expected rather than an outage. Trying again usually works.
"""
    )
    if st.button("Try again", icon=":material/refresh:", type="primary"):
        clear_caches()
        get_engine.clear()
        st.rerun()
    with st.expander("Connection details"):
        st.caption(f"Reading `{source.host}`, configured by **{source.origin}**.")
        st.code(source.masked, language="text")


def _render_query_failure(exc: SQLAlchemyError, source: Source) -> None:
    """The warehouse answered, and refused.

    Nearly always one thing: the app is pointed at a database the gold marts
    have not been promoted into yet. That is a deployment step, not a fault a
    visitor can wait out, so there is no retry button here. A message and the
    host it is talking to is all that can honestly be offered.
    """
    st.error("The warehouse answered, but the query did not.", icon=":material/database_off:")
    st.markdown(
        "This usually means the gold marts have not been promoted into this "
        "database yet. `python serving/promote.py` is what puts them there."
    )
    with st.expander("Details"):
        st.caption(f"Reading `{source.host}`, configured by **{source.origin}**.")
        st.code(str(exc.orig or exc), language="text")


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    try:
        source = resolve_database_url()
    except DashboardConfigError as exc:
        _render_config_error(exc)
        return

    # Built before the sidebar extras so the page links sit at the top of it,
    # and before run() because run() is what renders the selected page.
    navigation = _navigation()

    st.sidebar.title(PAGE_TITLE)
    st.sidebar.caption(
        "Thirty years of observations for fifteen cities, modelled into a "
        "governed warehouse and scored a week ahead."
    )
    _sidebar_status(source)
    _sidebar_controls()
    st.sidebar.divider()
    st.sidebar.caption(f"[Source and method]({REPOSITORY_URL})")

    try:
        navigation.run()
    except WarehouseUnreachable as exc:
        _render_unreachable(exc, source)
    except SQLAlchemyError as exc:
        _render_query_failure(exc, source)


if __name__ == "__main__":
    main()
