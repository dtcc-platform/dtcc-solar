"""Public interface and runtime compatibility helpers for dtcc-solar."""

import sys
from types import ModuleType

# Older dtcc-viewer releases expect ``dtcc_core.get_logger`` to exist, but the
# current dtcc-core exposes it from ``dtcc_core.common.dtcc_logging`` instead of
# re-exporting it at package level. Patch the attribute when it is missing so
# downstream imports keep working after pip install.
try:  # pragma: no cover - defensive guard for optional dependency issues
    import dtcc_core  # type: ignore
except ImportError:  # dtcc-core not available; leave patching to caller
    dtcc_core = None
else:
    if not hasattr(dtcc_core, "get_logger"):
        try:
            from dtcc_core.common.dtcc_logging import get_logger as _dtcc_get_logger
        except ImportError:
            pass
        else:
            dtcc_core.get_logger = _dtcc_get_logger  # type: ignore[attr-defined]

# Older dtcc-model users import from a stand-alone package that no longer ships.
# Mirror the dtcc_core.model module under the legacy name when needed.
if "dtcc_model" not in sys.modules:
    try:
        import dtcc_model  # type: ignore
    except ImportError:
        try:
            from dtcc_core import model as _dtcc_model  # type: ignore
        except ImportError:
            _dtcc_model = None
        else:
            legacy_model = ModuleType("dtcc_model")
            legacy_model.__dict__.update(_dtcc_model.__dict__)
            sys.modules["dtcc_model"] = legacy_model

from .solar_engine import SolarEngine
from .sunpath import Sunpath

from .viewer import Viewer as SolarViewer
from .utils import SolarParameters, OutputCollection

# Classes and methods visible on the Docs page
__all__ = [
    "SolarEngine",
    "Sunpath",
    "OutputCollection",
    "SolarParameters",
    "SolarViewer",
]
