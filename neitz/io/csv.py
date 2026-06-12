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
