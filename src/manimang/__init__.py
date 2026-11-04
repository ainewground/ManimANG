"""ManimANG (AI New Ground / Agent-Native Graphics): a next-generation superset of ManimGX.

Provides 100% backward compatibility with `import manimgx as m` while enabling
`import manimang as m` and submodule imports (`from manimang.scene import ThreeDScene`, etc.).
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from collections.abc import Sequence
from types import ModuleType

import manimgx as _manimgx
from manimgx import *  # noqa: F403

__all__ = _manimgx.__all__


class _ManimANGLoader(importlib.abc.Loader):
    """Loader that aliases `manimang.<subpath>` to `manimgx.<subpath>`."""

    def __init__(self, target_name: str) -> None:
        self.target_name = target_name

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return importlib.import_module(self.target_name)

    def exec_module(self, module: ModuleType) -> None:
        sys.modules[module.__name__] = module


class _ManimANGFinder(importlib.abc.MetaPathFinder):
    """MetaPathFinder mapping `manimang.*` submodules to `manimgx.*`."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith("manimang.") or fullname in (
            "manimang.cli",
            "manimang.__main__",
        ):
            return None
        target_name = "manimgx." + fullname.removeprefix("manimang.")
        target_spec = importlib.util.find_spec(target_name)
        if target_spec is None:
            return None
        is_pkg = target_spec.submodule_search_locations is not None
        spec = importlib.machinery.ModuleSpec(
            fullname, _ManimANGLoader(target_name), is_package=is_pkg
        )
        if is_pkg and target_spec.submodule_search_locations is not None:
            spec.submodule_search_locations = list(
                target_spec.submodule_search_locations
            )
        return spec


if not any(isinstance(f, _ManimANGFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _ManimANGFinder())
