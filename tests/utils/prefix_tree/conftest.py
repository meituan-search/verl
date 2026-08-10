# conftest.py - stub unavailable C/GPU packages so prefix_tree tests run on CPU.
import importlib.abc
import importlib.util
import sys
import types


class _StubModule(types.ModuleType):
    """Stub module: attribute access returns a callable child stub (also
    registered in sys.modules so ``from pkg.sub import name`` works)."""

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        child_name = f"{self.__name__}.{item}"
        child = _make_stub(child_name)
        setattr(self, item, child)
        sys.modules[child_name] = child
        return child


def _make_stub(name: str) -> _StubModule:
    mod = _StubModule(name)
    mod.__path__ = []
    mod.__package__ = name
    mod.__file__ = f"<stub:{name}>"
    mod.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    return mod


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Meta-path finder that auto-stubs any submodule of configured top-level
    packages. Catches ``megatron.core.enums``, ``magi_attention.api``, etc.
    without requiring each to be enumerated."""

    _prefixes: tuple[str, ...] = ()

    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        for prefix in cls._prefixes:
            if fullname == prefix or fullname.startswith(prefix + "."):
                return importlib.util.spec_from_loader(fullname, loader=cls)
        return None

    @classmethod
    def create_module(cls, spec):
        return _make_stub(spec.name)

    @classmethod
    def exec_module(cls, module):
        pass  # stubs have no body


def _install_stub_finder(prefixes: list[str]) -> None:
    _StubFinder._prefixes = tuple(prefixes)
    # Remove any existing instance and prepend so we run before the default finders.
    sys.meta_path = [f for f in sys.meta_path if not isinstance(f, _StubFinder)]
    sys.meta_path.insert(0, _StubFinder)


# Install the finder for packages whose real install isn't available on CPU.
# Each top-level package is also registered as a stub so `import megatron`
# succeeds even when the meta-path finder hasn't been queried for it yet.
_STUB_PACKAGES = ["megatron", "magi_attention", "apex", "transformer_engine"]
for _pkg in _STUB_PACKAGES:
    try:
        importlib.util.find_spec(_pkg)
    except (ModuleNotFoundError, ImportError):
        sys.modules[_pkg] = _make_stub(_pkg)

_install_stub_finder(_STUB_PACKAGES)
