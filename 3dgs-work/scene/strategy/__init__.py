"""Densification strategies (P1-MultiStrategy).

See OPTIMIZATIONS.md §13 for the full spec.

Exports:
    Strategy     — abstract base class
    get_strategy — factory lookup by name (e.g. "default", "mcmc", "igs_plus")
    register     — decorator to register a Strategy subclass into the factory
    OptInTrainModule — abstract base class for cross-cutting train-time modules
    OPTIONAL_MODULES — global registry of @register-decorated wrappers
                        (scale_reg / admm / bilateral_grid / ppisp)

Importing this package triggers registration of all built-in strategies
(default, and later mcmc / igs_plus in P1-3 / P1-4 commits) AND all
opt-in train-time modules (P0-Bilateral-v2 / P2-PPISP / P2-ADMM / P1-5.7).
"""
from .base import Strategy
from .factory import get_strategy, register
from . import defaults  # noqa: F401  — registers @register("default")
from . import mcmc     # noqa: F401  — registers @register("mcmc") (P1-3)
from . import eq9      # noqa: F401  — Eq.(9) + histogram helper (P1-MCMCParity-A)
from . import igs_plus  # noqa: F401  — registers @register("igs_plus") (P1-4-IGS+phase1)
# OptInTrainModule wrappers — register themselves into OPTIONAL_MODULES on import.
from .optional_module import OptInTrainModule, OPTIONAL_MODULES, register as register_optional  # noqa: F401
from . import scale_reg_module  # noqa: F401  — registers @register (P1-5.7)
from . import admm_module        # noqa: F401  — registers @register (P2-ADMM)
from . import bilateral_module   # noqa: F401  — registers @register (P0-Bilateral-v2)
from . import ppisp_module       # noqa: F401  — registers @register (P2-PPISP)

__all__ = ["Strategy", "get_strategy", "register", "OptInTrainModule", "OPTIONAL_MODULES", "register_optional"]
