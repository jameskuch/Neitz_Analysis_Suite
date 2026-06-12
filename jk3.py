"""jk3.py — bin one channel's spike train into a 360 Hz response.

Thin client of neitz.io.csv (was a broken fragment under the old monolith).
"""
import numpy as np

from neitz.io import csv as ncsv

fn = "/Users/j/Neitz_Analysis_Suite/SaraipRGC/siso-spikes.csv"
stim_le_s = 10.0
bin_rate = 6 * 60                       # 360 Hz
bins_total = int(stim_le_s * bin_rate)  # 3600 bins
ch = 0

time_vec, spikes = ncsv.load_spikes_csv(fn)
spike_times = ncsv.spike_times_from_matrix(spikes, time_vec)

edges = np.linspace(0, stim_le_s, bins_total + 1)
response_counts, _ = np.histogram(spike_times[ch], bins=edges)
response = response_counts.astype(float) * bin_rate

print(f"channel {ch}: {len(spike_times[ch])} spikes; response shape {response.shape}, "
      f"max {response.max():.0f} spikes/s")
