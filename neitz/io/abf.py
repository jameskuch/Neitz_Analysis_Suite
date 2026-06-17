"""
neitz.io.abf — load Axon .abf recordings by EXPLICIT path.

Replaces the old Neitz.abfread, which baked in channel indices, a datapath, and
plotting side effects. Here a Recording just gives you clean access to channels
(by name OR index), the time vector, the sample rate, and metadata — nothing more.

    from neitz.io.abf import Recording
    rec = Recording.load("~/Documents/ephysdataio/2026-06-02/c01/raw/2026_06_02_0044.abf")
    im  = rec.channel("Im_prime")     # by name (case-insensitive, partial ok)
    ttl = rec.channel("TTL")
    t   = rec.time()
"""

from __future__ import annotations
import os
import threading
import numpy as np
import pyabf


class Recording:
    """Thin, path-explicit wrapper around a pyabf.ABF."""

    def __init__(self, abf: "pyabf.ABF", path: str):
        self._abf = abf
        # pyabf.setSweep mutates shared state on the ABF object; serialize setSweep+read so two
        # threads reading different channels of the same file can't interleave and return (and then
        # cache) the wrong channel's data. One lock per Recording — different files don't contend.
        self._lock = threading.Lock()
        self.path = path
        self.fs = float(abf.dataRate)
        self.n_sweeps = int(abf.sweepCount)
        self.n_channels = int(abf.channelCount)
        self.channel_names = list(abf.adcNames)
        self.channel_units = list(abf.adcUnits)
        self.protocol = abf.protocol
        self.duration = float(abf.sweepLengthSec)

    @classmethod
    def load(cls, path) -> "Recording":
        return cls(pyabf.ABF(str(path)), str(path))

    # -- channel access -----------------------------------------------------
    def resolve_channel(self, channel) -> int:
        """Map a channel name (or index) to its integer index."""
        if isinstance(channel, (int, np.integer)):
            return int(channel)
        name = str(channel).lower()
        names = [n.lower() for n in self.channel_names]
        if name in names:
            return names.index(name)
        hits = [i for i, n in enumerate(names) if name in n]
        if len(hits) == 1:
            return hits[0]
        raise KeyError(f"channel {channel!r} not found / ambiguous in {self.channel_names}")

    def channel(self, channel, sweep: int = 0) -> np.ndarray:
        """Return one channel's data (a copy) for a sweep."""
        idx = self.resolve_channel(channel)
        with self._lock:                       # setSweep + read must be atomic (see __init__)
            self._abf.setSweep(sweep, channel=idx)
            return self._abf.sweepY.copy()

    def time(self, sweep: int = 0) -> np.ndarray:
        with self._lock:
            self._abf.setSweep(sweep, channel=0)
            return self._abf.sweepX.copy()

    def units(self, channel) -> str:
        return self.channel_units[self.resolve_channel(channel)]

    def summary(self) -> dict:
        return dict(path=self.path, fs=self.fs, duration=self.duration,
                    n_sweeps=self.n_sweeps, protocol=self.protocol,
                    channels=dict(zip(self.channel_names, self.channel_units)))

    def metadata(self) -> dict:
        """Human-readable acquisition metadata for display."""
        a = self._abf
        md = {
            "file": os.path.basename(self.path),
            "protocol": self.protocol,
            "sample rate": f"{self.fs:.0f} Hz",
            "duration": f"{self.duration:.2f} s",
            "samples/sweep": int(round(self.duration * self.fs)),
            "sweeps": self.n_sweeps,
            "channels": ", ".join(f"{n} [{u}]" for n, u in
                                  zip(self.channel_names, self.channel_units)),
        }
        for attr, key in [("abfDateTime", "recorded"),
                          ("abfVersionString", "abf version"),
                          ("abfID", "abf id"),
                          ("creator", "creator")]:
            try:
                val = getattr(a, attr)
                if val not in (None, ""):
                    md[key] = str(val)
            except Exception:
                pass
        return md

    def __repr__(self):
        return (f"<Recording {self.path!r} fs={self.fs:.0f}Hz "
                f"dur={self.duration:.1f}s chans={self.channel_names} "
                f"protocol={self.protocol!r}>")
