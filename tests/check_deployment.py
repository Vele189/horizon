"""Verifies a deployed dashboard, and times how long it takes to wake.

The counterpart to ``tests/check_connection.py``: that one proves the warehouse
is reachable from here, this one proves the app is reachable from anywhere.

    python tests/check_deployment.py https://your-app.streamlit.app
    python tests/check_deployment.py <url> --cold          # after a week idle
    python tests/check_deployment.py <url> --screenshots docs/images

**Two different cold starts, and they are not the same number.** Neon suspends
compute after five minutes and resumes in about a second (§Promotion to Neon).
Streamlit Community Cloud puts an app to sleep after roughly a week of no
visitors, and waking it rebuilds a container — tens of seconds, not one. A
visitor arriving at a long-idle portfolio link pays the second one and then the
first. ``--cold`` labels a run as that case; without it the timing is a warm
measurement and is recorded as such.

**What HTTP can and cannot prove.** A Streamlit page is a shell that fills
itself over a websocket, so a 200 from a view's URL proves the app is up and
routing, not that the chart drew. ``--screenshots`` closes that gap with
headless Chrome when it is installed: it captures each view so the four can be
checked by eye, which is the only honest way to verify a render.

Nothing here needs a credential. It reads a public URL.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from dashboard import views  # noqa: E402

HEALTH = "/_stcore/health"
WAKE_TIMEOUT_SECONDS = 180
CHROME_CANDIDATES = ("google-chrome", "chromium", "chromium-browser")


def _view_paths() -> list[tuple[str, str]]:
    """(title, url_path) for every view, from the navigation itself."""
    return [(module.VIEW.title, module.VIEW.url_path) for module in views.ORDER]


def wake(url: str, *, cold: bool) -> float | None:
    """Poll the health endpoint until the app answers. Returns seconds."""
    started = time.perf_counter()
    deadline = started + WAKE_TIMEOUT_SECONDS
    attempt = 0

    while time.perf_counter() < deadline:
        attempt += 1
        try:
            response = requests.get(url.rstrip("/") + HEALTH, timeout=20)
            if response.ok and response.text.strip().lower().startswith("ok"):
                elapsed = time.perf_counter() - started
                label = "cold start (app was asleep)" if cold else "warm"
                print(f"  health         ok after {elapsed:6.2f} s  <-- {label}")
                print(f"  attempts       {attempt}")
                return elapsed
        except requests.RequestException as exc:
            if attempt == 1:
                print(f"  waiting        {type(exc).__name__} — retrying")
        time.sleep(2)

    print(f"  FAIL           no health response within {WAKE_TIMEOUT_SECONDS} s")
    return None


def check_views(url: str) -> bool:
    """Each view's own URL must serve the app shell.

    Proves routing, not rendering — see the module docstring.
    """
    base = url.rstrip("/")
    ok = True
    for title, path in _view_paths():
        target = f"{base}/{path}"
        try:
            response = requests.get(target, timeout=30)
        except requests.RequestException as exc:
            print(f"  {title:20} FAIL  {type(exc).__name__}")
            ok = False
            continue
        good = response.status_code == 200
        ok &= good
        print(f"  {title:20} {'ok  ' if good else 'FAIL'}  {response.status_code}  {target}")
    return ok


def _chrome() -> str | None:
    for candidate in CHROME_CANDIDATES:
        found = shutil.which(candidate)
        if found:
            return found
    return None


def screenshots(url: str, destination: Path, *, settle: int = 25) -> bool:
    """Capture each view with headless Chrome, for eyes to check.

    The settle delay is not decoration: the page arrives empty and fills over a
    websocket, and a screenshot taken on load would faithfully record a blank
    app. It also has to cover the warehouse read behind the first render.
    """
    binary = _chrome()
    if binary is None:
        print(f"  SKIP           no browser found (tried {', '.join(CHROME_CANDIDATES)})")
        return False

    destination.mkdir(parents=True, exist_ok=True)
    base = url.rstrip("/")
    ok = True
    for title, path in _view_paths():
        out = destination / f"dashboard-{path}.png"
        command = [
            binary, "--headless", "--disable-gpu", "--no-sandbox",
            "--hide-scrollbars", "--window-size=1600,1200",
            f"--virtual-time-budget={settle * 1000}",
            f"--screenshot={out}", f"{base}/{path}",
        ]
        result = subprocess.run(command, capture_output=True, timeout=settle + 90)
        wrote = out.exists() and out.stat().st_size > 0
        ok &= wrote
        size = f"{out.stat().st_size / 1024:.0f} KB" if wrote else "nothing written"
        print(f"  {title:20} {'ok  ' if wrote else 'FAIL'}  {size}  {out}")
        if not wrote and result.stderr:
            print(f"      {result.stderr.decode('utf-8', 'ignore').strip().splitlines()[-1][:120]}")
    return ok


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="https://your-app.streamlit.app")
    parser.add_argument(
        "--cold",
        action="store_true",
        help="label the timing as a wake from sleep (run after ~a week idle)",
    )
    parser.add_argument(
        "--screenshots",
        type=Path,
        metavar="DIR",
        help="capture each view to DIR with headless Chrome",
    )
    args = parser.parse_args(argv)

    print(f"\n=== {args.url} ===")
    elapsed = wake(args.url, cold=args.cold)
    if elapsed is None:
        return 1

    print("\n=== views (routing) ===")
    routed = check_views(args.url)

    captured = True
    if args.screenshots:
        print("\n=== views (rendered) ===")
        captured = screenshots(args.url, args.screenshots)

    print("\n=== summary ===")
    print(f"  reachable      yes, in {elapsed:.2f} s"
          + ("   (record this as the cold start)" if args.cold else ""))
    print(f"  all four views {'route' if routed else 'DO NOT all route'}")
    if args.screenshots:
        print(f"  screenshots    {'captured — check them by eye' if captured else 'FAILED'}")
    else:
        print("  screenshots    not requested; rendering is unverified by this run")
    return 0 if routed and captured else 1


if __name__ == "__main__":
    raise SystemExit(main())
