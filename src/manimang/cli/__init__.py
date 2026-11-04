"""The ManimANG command line (`manimang`), 100% compatible with `manimgx`."""

import manimang  # noqa: F401
from manimgx.cli import (
    DRAWING,
    agent,
    app,
    export,
    inspect,
    present,
    preview,
    render,
)

__all__ = [
    "DRAWING",
    "agent",
    "app",
    "export",
    "inspect",
    "main",
    "present",
    "preview",
    "render",
]


def main() -> None:
    """Run the ManimANG CLI."""
    app(prog_name="manimang")
