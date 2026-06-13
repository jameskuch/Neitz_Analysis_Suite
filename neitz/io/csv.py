"""
neitz.io.csv — CSV loaders for the siso / Gaussian-noise pipeline.

Ported from the Neitz.py monolith as plain functions that RETURN data (no hidden
instance state). Used by the extract_siso* scripts.

    from neitz.io import csv as ncsv
    time, spikes = ncsv.load_spikes_csv("siso-spikes.csv")   # col0=time, cols1..=channels
    spike_times  = ncsv.spike_times_from_matrix(spikes, time)
    epochs, phases = ncsv.load_stimulus_epochs_csv("siso-stdev.csv")
    stim = ncsv.stim_phase_only(epochs, phases)              # STIM-phase rows only
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd


def load_spikes_csv(path):
    """
    Load a spike CSV. Column 0 = time vector, columns 1.. = spike channels.
    A non-numeric first row is treated as a header and dropped. Spike columns may
    be binary (0/1) or hold the spike amplitude (only `> 0` is used downstream).

    Returns
    -------
    time : (N,) float
    spikes : (N, M) float
    """
    raw = pd.read_csv(path, header=None)
    first = pd.to_numeric(raw.iloc[0], errors="coerce")
    if first.isna().any():                       # header row present
        raw = raw.iloc[1:].reset_index(drop=True)
    arr = raw.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if arr.shape[1] < 2:
        raise ValueError(f"spike CSV needs >= 2 columns, got shape {arr.shape}")
    return arr[:, 0], arr[:, 1:]


def spike_times_from_matrix(spike_matrix, time_vec):
    """Per-channel spike times: time_vec where each column is > 0."""
    spike_matrix = np.asarray(spike_matrix)
    time_vec = np.asarray(time_vec)
    return [time_vec[spike_matrix[:, i] > 0] for i in range(spike_matrix.shape[1])]


def load_stimulus_epochs_csv(path):
    """
    Tolerant stimulus-epoch loader. Handles header/no-header and a leading Phase
    column (PRE/STIM labels) or a leading frame-number column.

    Returns
    -------
    stim_epochs : (num_rows, num_epochs) float
    phases : (num_rows,) str array, or None if the CSV has no phase column
    """
    peek = pd.read_csv(path, header=None, nrows=5)
    if peek.shape[1] < 2:
        raise ValueError(f"stimulus CSV needs >= 2 columns, got {peek.shape}")

    first_row = [str(x).strip().lower() for x in peek.iloc[0].tolist()]
    has_header = ("phase" in first_row
                  or any(x.startswith("stim") for x in first_row)
                  or any("epoch" in x for x in first_row)
                  or any("frame" in x for x in first_row))

    if has_header:
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        cols_lower = [c.lower() for c in df.columns]
        if "phase" in cols_lower:
            phase_col = df.columns[cols_lower.index("phase")]
            stim_cols = [c for c in df.columns
                         if c != phase_col and (c.lower().startswith("stim")
                                                or "epoch" in c.lower())]
            if not stim_cols:
                stim_cols = list(df.columns[1:])
            stim_df = df[stim_cols].apply(pd.to_numeric, errors="coerce")
            valid = stim_df.notna().any(axis=1)
            phases = df.loc[valid, phase_col].astype(str).to_numpy()
            stim_epochs = stim_df.loc[valid].to_numpy(dtype=float)
        else:
            stim_df = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
            valid = stim_df.notna().any(axis=1)
            phases = None
            stim_epochs = stim_df.loc[valid].to_numpy(dtype=float)
    else:
        df = pd.read_csv(path, header=None)
        first_col_num = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        if first_col_num.notna().all():          # leading frame-number column
            phases = None
        else:                                    # leading phase-label column
            phases = df.iloc[:, 0].astype(str).to_numpy()
        stim_epochs = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    return stim_epochs, phases


def stim_phase_only(stim_epochs, phases):
    """Rows whose phase == STIM; if no phase info, returns all rows unchanged."""
    if phases is None:
        return stim_epochs
    p = np.char.upper(np.char.strip(np.asarray(phases).astype(str)))
    mask = p == "STIM"
    if not mask.any():
        raise ValueError("phase column present but no rows labeled 'STIM'")
    return stim_epochs[mask, :]


class CsvSpikeRecording:
    """
    Adapter presenting a spike CSV (col 0 = time, cols 1.. = spike channels) with
    the same interface as neitz.io.abf.Recording, so the viewer and tools can open a
    spike CSV like an .abf. Each spike column becomes a channel ("ch1", "ch2", ...);
    values may be binary (0/1) or spike amplitudes (only `> 0` marks a spike).
    """

    def __init__(self, path):
        time, spikes = load_spikes_csv(path)
        self.path = str(path)
        self._time = np.asarray(time, dtype=float)
        self._spikes = np.asarray(spikes, dtype=float)
        t = self._time
        if t.size < 2 or not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
            raise ValueError(
                f"{os.path.basename(str(path))}: column 0 is not a finite, increasing time "
                "vector — this looks like a stimulus / header CSV, not a spike recording")
        dt = float(np.median(np.diff(t)))
        self.fs = (1.0 / dt) if dt > 0 else 1.0
        self.duration = float(t[-1])
        m = self._spikes.shape[1]
        self.channel_names = [f"ch{i + 1}" for i in range(m)]
        self.channel_units = ["spikes"] * m
        self.n_channels = m
        self.n_sweeps = 1
        self.protocol = "csv-spikes"

    @classmethod
    def load(cls, path):
        return cls(path)

    def resolve_channel(self, channel) -> int:
        if isinstance(channel, (int, np.integer)):
            return int(channel)
        name = str(channel).lower()
        names = [n.lower() for n in self.channel_names]
        if name in names:
            return names.index(name)
        raise KeyError(f"channel {channel!r} not in {self.channel_names}")

    def channel(self, channel, sweep: int = 0) -> np.ndarray:
        return self._spikes[:, self.resolve_channel(channel)].copy()

    def time(self, sweep: int = 0) -> np.ndarray:
        return self._time.copy()

    def units(self, channel) -> str:
        return "spikes"

    def metadata(self) -> dict:
        return {
            "file": os.path.basename(self.path),
            "protocol": self.protocol,
            "sample rate": f"{self.fs:.0f} Hz",
            "duration": f"{self.duration:.2f} s",
            "samples": int(self._time.size),
            "channels": ", ".join(self.channel_names),
        }

    def __repr__(self):
        return (f"<CsvSpikeRecording {self.path!r} fs={self.fs:.0f}Hz "
                f"dur={self.duration:.1f}s channels={self.channel_names}>")
