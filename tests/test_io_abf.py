"""Integration tests against a real .abf — skipped when the (gitignored) data is absent.

Locks in the validated numbers for cell 2026_06_02_0044.
"""
import os
import pytest

from neitz.io.abf import Recording
from neitz.spikes import detect_spikes
from neitz.analysis import flicker

# the .abf lives in the data store (gitignored)
_STORE = os.path.expanduser(os.environ.get("EPHYSDATAIO_ROOT", "~/Documents/ephysdataio"))
DATA = os.path.join(_STORE, "2026-06-02", "c01", "raw", "2026_06_02_0044.abf")

pytestmark = pytest.mark.skipif(not os.path.exists(DATA),
                                reason="abf data not present (gitignored)")


def test_recording_loads_and_channels_by_name():
    rec = Recording.load(DATA)
    assert rec.fs == 20000
    assert rec.channel_names[:1] == ["Im_prime"]
    assert "TTL" in rec.channel_names
    assert rec.channel("Im_prime").size > 0          # addressed by name
    assert rec.resolve_channel("TTL") == 2


def test_known_spike_count():
    rec = Recording.load(DATA)
    st = detect_spikes(rec.channel("Im_prime"), rec.fs, polarity="neg", method="mad", k=6)
    assert len(st) == 1157


def test_known_flicker():
    rec = Recording.load(DATA)
    fl = flicker.detect_flicker(rec.channel("TTL"), rec.fs)
    assert abs(fl.freq - 2.0) < 0.05
    assert 15.0 < fl.t0 < 16.0
    assert 73.0 < fl.t1 < 75.0
