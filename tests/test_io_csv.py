"""CSV loaders (ported from the Neitz.py monolith) — synthetic tests."""
import numpy as np
from neitz.io import csv as ncsv


def test_load_spikes_no_header(tmp_path):
    p = tmp_path / "spk.csv"
    p.write_text("0,0,0\n0.1,1,0\n0.2,0,1\n0.3,1,1\n")
    time, spikes = ncsv.load_spikes_csv(str(p))
    assert time.shape == (4,) and spikes.shape == (4, 2)
    assert np.allclose(time, [0, 0.1, 0.2, 0.3])
    st = ncsv.spike_times_from_matrix(spikes, time)
    assert np.allclose(st[0], [0.1, 0.3])
    assert np.allclose(st[1], [0.2, 0.3])


def test_load_spikes_with_header_and_amplitudes(tmp_path):
    p = tmp_path / "spk.csv"
    p.write_text("time,ch1,ch2\n0,0,0\n0.1,29.0,0\n0.2,0,40.0\n")
    time, spikes = ncsv.load_spikes_csv(str(p))
    assert time.shape == (3,) and spikes.shape == (3, 2)
    st = ncsv.spike_times_from_matrix(spikes, time)   # only > 0 matters
    assert np.allclose(st[0], [0.1])
    assert np.allclose(st[1], [0.2])


def test_stimulus_epochs_frame_number(tmp_path):
    p = tmp_path / "stim.csv"
    p.write_text("1,0.5,-0.5\n2,0.1,0.2\n3,-0.3,0.4\n")
    ep, ph = ncsv.load_stimulus_epochs_csv(str(p))
    assert ph is None and ep.shape == (3, 2)
    assert np.allclose(ncsv.stim_phase_only(ep, ph), ep)   # no phase -> all rows


def test_stimulus_epochs_phase_column(tmp_path):
    p = tmp_path / "stim.csv"
    p.write_text("Phase,Stim1,Stim2\nPRE,0,0\nSTIM,0.5,-0.5\nSTIM,0.1,0.2\n")
    ep, ph = ncsv.load_stimulus_epochs_csv(str(p))
    assert ph is not None
    stim = ncsv.stim_phase_only(ep, ph)
    assert stim.shape == (2, 2)                            # only the STIM rows
    assert np.allclose(stim, [[0.5, -0.5], [0.1, 0.2]])
