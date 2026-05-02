"""If write_state crashes mid-write, the previous state on disk must survive."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from paper_trader import trade_log


def test_state_survives_mid_write_crash(tmp_path: Path, monkeypatch):
    target = tmp_path / "s.json"
    trade_log.write_state(target, {"v": 1})

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr(trade_log.os, "replace", boom)

    with pytest.raises(RuntimeError):
        trade_log.write_state(target, {"v": 2})

    # Original state intact, no half-written tmp adopted as primary.
    assert json.loads(target.read_text()) == {"v": 1}
