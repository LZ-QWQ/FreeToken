from freetoken.engine import engine


def _clear_allocator_env(monkeypatch):
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)


def test_expandable_segments_are_not_enabled_on_rocm(monkeypatch):
    _clear_allocator_env(monkeypatch)
    calls = []
    monkeypatch.setattr(engine.torch.version, "hip", "7.14.1")
    monkeypatch.setattr(engine.torch.cuda.memory, "_set_allocator_settings", calls.append)

    engine._ensure_expandable_segments()

    assert calls == []


def test_expandable_segments_remain_enabled_on_cuda(monkeypatch):
    _clear_allocator_env(monkeypatch)
    calls = []
    monkeypatch.setattr(engine.torch.version, "hip", None)
    monkeypatch.setattr(engine.torch.cuda.memory, "_set_allocator_settings", calls.append)

    engine._ensure_expandable_segments()

    assert calls == ["expandable_segments:True"]
