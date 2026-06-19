"""Lazy package init (PEP 562). `import tracer` must stay cheap and not import
the heavy ML stack until a real symbol is accessed."""
import importlib
import sys


def test_import_tracer_is_cheap():
    # Importing the package must not pull numpy/sklearn eagerly.
    for mod in list(sys.modules):
        if mod == "tracer" or mod.startswith("tracer."):
            del sys.modules[mod]
    import tracer  # noqa: F401

    assert "tracer" in sys.modules
    # The heavy submodules must NOT have been imported just by `import tracer`.
    assert "tracer.api" not in sys.modules
    assert "tracer.scanner" not in sys.modules


def test_version_present():
    import tracer

    assert isinstance(tracer.__version__, str)
    assert tracer.__version__.count(".") >= 1


def test_lazy_symbols_resolve():
    import tracer

    # These are exported lazily via __getattr__; accessing them loads the module.
    assert callable(tracer.fit)
    assert callable(tracer.scan)
    assert callable(tracer.watch)
    assert tracer.TraceRecord.__name__ == "TraceRecord"


def test_unknown_attribute_raises():
    import tracer

    try:
        tracer.does_not_exist  # noqa: B018
    except AttributeError as e:
        assert "does_not_exist" in str(e)
    else:
        raise AssertionError("expected AttributeError for unknown attribute")


def test_lazy_access_caches():
    import tracer

    a = tracer.TraceRecord
    b = tracer.TraceRecord
    assert a is b  # cached into module globals after first access
