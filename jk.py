"""jk.py — quick look at one .abf: current (spike) channel + stimulus/frame-sync.

Thin client of neitz.io.Recording (no plotting side-effects, unlike the old monolith).
"""
import numpy as np
from matplotlib import pyplot as pl

from neitz.io.abf import Recording

# pick a file (channel 0 = current/spikes, channel 2 = TTL/stimulus)
ABF = "data/ipRGC barak/ipRGC/2026_06_02_0040.abf"
SPIKE_CH = 0
STIM_CH = 2

rec = Recording.load(ABF)
time_vec = rec.time()
spike_ch = rec.channel(SPIKE_CH)
stim_raw = rec.channel(STIM_CH)
stim_ch = (stim_raw - float(np.max(stim_raw))) * -1.0     # offset + invert (as before)

pl.subplot(2, 1, 1)                 # top panel
pl.plot(time_vec, spike_ch)         # spike train
pl.ylabel(f"{rec.channel_names[SPIKE_CH]} ({rec.channel_units[SPIKE_CH]})")

pl.subplot(2, 1, 2)                 # bottom panel
pl.plot(time_vec, stim_ch)          # stimulus / frame sync
pl.ylabel("stimulus")
pl.xlabel("time (s)")
pl.show()
