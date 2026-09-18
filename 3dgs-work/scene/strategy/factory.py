"""Strategy factory + registry decorator.

Concrete :class:`Strategy` subclasses register themselves by name via the
:func:`register` decorator. :func:`get_strategy` looks them up at runtime.

Adding a new strategy is two lines::

    # scene/strategy/foo.py
    from .factory import register
    from .base import Strategy

    @register("foo")
    class FooStrategy(Strategy):
        ...
"""

from typing import Dict, Type

_REGISTRY: Dict[str, Type] = {}


def register(name: str):
    """Class decorator. Registers the decorated ``Strategy`` subclass under
    ``name`` so :func:`get_strategy` can find it.

    The name must be unique within the registry. ``scene/strategy/__init__.py``
    imports every built-in strategy module so registration happens at package
    import time.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"strategy name must be a non-empty str, got {name!r}")

    def deco(cls):
        if name in _REGISTRY:
            raise RuntimeError(
                f"densification strategy {name!r} already registered "
                f"by {_REGISTRY[name].__module__}.{_REGISTRY[name].__name__}"
            )
        _REGISTRY[name] = cls
        return cls

    return deco


def get_strategy(name: str) -> Type:
    """Look up a registered strategy class by name.

    Raises ``ValueError`` listing all available names if ``name`` is unknown.
    """
    if name not in _REGISTRY:
        avail = ", ".join(sorted(_REGISTRY)) or "<none registered>"
        raise ValueError(
            f"unknown densification strategy {name!r}; available: {avail}"
        )
    return _REGISTRY[name]