import builtins
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.conftest import _collected_tests_require_ray, manage_ray


def _collected_item(*, requires_ray: bool) -> Mock:
    item = Mock()
    item.get_closest_marker.return_value = object() if requires_ray else None
    return item


def _request(*items: Mock) -> SimpleNamespace:
    return SimpleNamespace(session=SimpleNamespace(items=list(items)))


def test_ray_is_not_required_for_an_unmarked_collection():
    items = [_collected_item(requires_ray=False), _collected_item(requires_ray=False)]

    assert not _collected_tests_require_ray(items)


def test_ray_is_required_when_any_collected_test_is_marked():
    items = [_collected_item(requires_ray=False), _collected_item(requires_ray=True)]

    assert _collected_tests_require_ray(items)


def test_manage_ray_does_not_import_ray_for_unmarked_collection(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "ray":
            raise AssertionError("Ray must not be imported for an unmarked collection")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    fixture = manage_ray.__wrapped__(_request(_collected_item(requires_ray=False)))

    next(fixture)
    with pytest.raises(StopIteration):
        next(fixture)


def test_manage_ray_propagates_initialization_failures(monkeypatch):
    fake_ray = SimpleNamespace(
        is_initialized=Mock(return_value=False),
        init=Mock(side_effect=RuntimeError("ray init failed")),
        shutdown=Mock(),
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    fixture = manage_ray.__wrapped__(_request(_collected_item(requires_ray=True)))

    with pytest.raises(RuntimeError, match="ray init failed"):
        next(fixture)
