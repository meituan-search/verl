# conftest.py — stub unavailable C/GPU packages so prefix_tree tests can run locally.
import importlib.util
import sys
import types


class _StubModule(types.ModuleType):
    """A module that acts as a real package and accepts arbitrary attribute access."""

    def __getattr__(self, item):
        child_name = f"{self.__name__}.{item}"
        child = _make_stub(child_name)
        setattr(self, item, child)
        # Also register in sys.modules so sub-imports find it.
        sys.modules[child_name] = child
        return child


def _make_stub(name: str) -> _StubModule:
    mod = _StubModule(name)
    mod.__path__ = []
    mod.__package__ = name
    mod.__file__ = f"<stub:{name}>"
    mod.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    return mod


def _stub_if_missing(pkg: str) -> None:
    try:
        importlib.import_module(pkg)
    except ModuleNotFoundError:
        sys.modules[pkg] = _make_stub(pkg)


for _pkg in ["magi_attention", "megatron", "apex", "transformer_engine"]:
    _stub_if_missing(_pkg)
