"""jk2.py — quick look at one spike channel from a siso-spikes.csv.

Thin client of neitz.io.csv.
"""
import numpy as np
from matplotlib import pyplot as pl

from neitz.io import csv as ncsv

fn = "/Users/j/Neitz_Analysis_Suite/SaraipRGC/siso-spikes.csv"
trim_fr_s = 0.500
trim_en_s = 0.500
channel = 3

time_t, spikes_t = ncsv.load_spikes_csv(fn)

# trim a window and re-zero time
mask = (time_t >= time_t[0] + trim_fr_s) & (time_t <= time_t[-1] - trim_en_s)
time_s = time_t[mask] - time_t[mask][0]
spikes = spikes_t[mask, :]

spike_times = ncsv.spike_times_from_matrix(spikes, time_s)
sp = spike_times[channel]

pl.figure(figsize=(12, 6))
pl.plot(time_s, spikes[:, channel])
pl.plot(sp, np.zeros_like(sp), "k|", markersize=25)
pl.xlabel("Time (s)")
pl.title(f"Spike times, channel {channel}  ({len(sp)} spikes)")
pl.show()
