"""Tests for similarity metric registry (Step 1).

Tests the registry mechanism only — individual metric implementations
are tested in Step 2.  The registry must:

  - ship with built-in metrics pre-registered at import time
  - allow users to add custom metrics
  - protect built-in metrics from accidental removal
  - reject invalid registrations eagerly
  - remain isolated across tests (no cross-test pollution)
"""

import pytest
from jsonld_ex.similarity import (
    register_similarity_metric,
    get_similarity_metric,
    list_similarity_metrics,
    unregister_similarity_metric,
    reset_similarity_registry,
    BUILTIN_METRIC_NAMES,
)


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the registry before *and* after every test.

    This guarantees test isolation — custom metrics registered in one
    test never leak into another.
    """
    reset_similarity_registry()
    yield
    reset_similarity_registry()


def _dummy_metric(a, b):
    """Minimal valid metric for registration tests."""
    return 0.0


def _another_dummy(a, b):
    """Second distinct callable for override tests."""
    return 1.0


# ── Built-in metrics are pre-registered ─────────────────────────────────

class TestBuiltinRegistration:
    """Built-in metrics must be available immediately after import."""

    def test_builtins_are_registered(self):
        names = list_similarity_metrics()
        for name in BUILTIN_METRIC_NAMES:
            assert name in names, f"Built-in metric '{name}' not registered"

    def test_builtin_names_constant_is_frozen(self):
        """BUILTIN_METRIC_NAMES should be immutable (frozenset)."""
        assert isinstance(BUILTIN_METRIC_NAMES, frozenset)

    def test_builtins_include_expected_set(self):
        """The exact set of built-ins we ship with v0.6."""
        expected = {
            "cosine", "euclidean", "dot_product", "manhattan",
            "chebyshev", "hamming", "jaccard",
        }
        assert BUILTIN_METRIC_NAMES == expected

    def test_get_builtin_returns_callable(self):
        for name in BUILTIN_METRIC_NAMES:
            fn = get_similarity_metric(name)
            assert callable(fn), f"'{name}' did not return a callable"


# ── Listing ─────────────────────────────────────────────────────────────

class TestListMetrics:
    def test_list_returns_sorted(self):
        names = list_similarity_metrics()
        assert names == sorted(names)

    def test_list_is_a_snapshot(self):
        """Mutating the returned list must not affect the registry."""
        names = list_similarity_metrics()
        names.append("bogus")
        assert "bogus" not in list_similarity_metrics()


# ── Custom registration ─────────────────────────────────────────────────

class TestRegisterCustomMetric:
    def test_register_new_metric(self):
        register_similarity_metric("my_metric", _dummy_metric)
        assert "my_metric" in list_similarity_metrics()

    def test_get_returns_registered_function(self):
        register_similarity_metric("my_metric", _dummy_metric)
        assert get_similarity_metric("my_metric") is _dummy_metric

    def test_register_lambda(self):
        register_similarity_metric("lam", lambda a, b: 0.5)
        fn = get_similarity_metric("lam")
        assert fn([1.0], [2.0]) == 0.5

    def test_register_callable_object(self):
        class MyMetric:
            def __call__(self, a, b):
                return 0.42

        obj = MyMetric()
        register_similarity_metric("obj_metric", obj)
        assert get_similarity_metric("obj_metric") is obj


# ── Duplicate / override semantics ──────────────────────────────────────

class TestDuplicateRegistration:
    def test_duplicate_name_raises_by_default(self):
        register_similarity_metric("dup", _dummy_metric)
        with pytest.raises(ValueError, match="already registered"):
            register_similarity_metric("dup", _another_dummy)

    def test_duplicate_allowed_with_force(self):
        register_similarity_metric("dup", _dummy_metric)
        register_similarity_metric("dup", _another_dummy, force=True)
        assert get_similarity_metric("dup") is _another_dummy

    def test_cannot_override_builtin_without_force(self):
        with pytest.raises(ValueError, match="already registered"):
            register_similarity_metric("cosine", _dummy_metric)

    def test_can_override_builtin_with_force(self):
        """Power-user escape hatch — intentional override of a built-in."""
        register_similarity_metric("cosine", _dummy_metric, force=True)
        assert get_similarity_metric("cosine") is _dummy_metric


# ── Invalid registrations ───────────────────────────────────────────────

class TestInvalidRegistration:
    def test_non_callable_raises_type_error(self):
        with pytest.raises(TypeError, match="callable"):
            register_similarity_metric("bad", 42)

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError, match="callable"):
            register_similarity_metric("bad", None)

    def test_string_raises_type_error(self):
        with pytest.raises(TypeError, match="callable"):
            register_similarity_metric("bad", "not a function")

    def test_empty_name_raises_value_error(self):
        with pytest.raises(ValueError, match="name"):
            register_similarity_metric("", _dummy_metric)

    def test_whitespace_name_raises_value_error(self):
        with pytest.raises(ValueError, match="name"):
            register_similarity_metric("  ", _dummy_metric)


# ── Retrieval of non-existent metrics ───────────────────────────────────

class TestGetNonExistent:
    def test_unknown_name_raises_key_error(self):
        with pytest.raises(KeyError):
            get_similarity_metric("no_such_metric")

    def test_error_message_includes_name(self):
        with pytest.raises(KeyError, match="no_such_metric"):
            get_similarity_metric("no_such_metric")


# ── Unregistration ──────────────────────────────────────────────────────

class TestUnregister:
    def test_unregister_custom_metric(self):
        register_similarity_metric("tmp", _dummy_metric)
        unregister_similarity_metric("tmp")
        assert "tmp" not in list_similarity_metrics()

    def test_unregister_nonexistent_raises(self):
        with pytest.raises(KeyError, match="not registered"):
            unregister_similarity_metric("ghost")

    def test_cannot_unregister_builtin(self):
        for name in BUILTIN_METRIC_NAMES:
            with pytest.raises(ValueError, match="[Bb]uilt-in"):
                unregister_similarity_metric(name)

    def test_unregister_then_re_register(self):
        register_similarity_metric("cycle", _dummy_metric)
        unregister_similarity_metric("cycle")
        register_similarity_metric("cycle", _another_dummy)
        assert get_similarity_metric("cycle") is _another_dummy


# ── Reset (test utility) ───────────────────────────────────────────────

class TestResetRegistry:
    def test_reset_removes_custom_metrics(self):
        register_similarity_metric("custom1", _dummy_metric)
        register_similarity_metric("custom2", _another_dummy)
        reset_similarity_registry()
        names = list_similarity_metrics()
        assert "custom1" not in names
        assert "custom2" not in names

    def test_reset_restores_builtins(self):
        """Even if a built-in was force-overridden, reset restores it."""
        original = get_similarity_metric("cosine")
        register_similarity_metric("cosine", _dummy_metric, force=True)
        reset_similarity_registry()
        restored = get_similarity_metric("cosine")
        # Must be the *original* built-in, not the dummy
        assert restored is not _dummy_metric
        assert restored is original

    def test_reset_is_idempotent(self):
        reset_similarity_registry()
        reset_similarity_registry()
        assert list_similarity_metrics() == sorted(BUILTIN_METRIC_NAMES)
