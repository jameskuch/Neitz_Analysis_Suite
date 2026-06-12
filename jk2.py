import Neitz as ne
import numpy as np

import pandas as pd

from matplotlib import pyplot as pl

#fn = '/Users/j/Library/CloudStorage/GoogleDrive-j@jkuchen.com/My Drive/Tuo/SaraipRGC/siso-spikes.csv'
fn = '/Users/j/Neitz_Analysis_Suite/SaraipRGC/siso-spikes.csv'
#fn = 'SaraipRGC/siso-spikes.csv'

bins_per_frame  = 6
fps             = 60

fs              = 10000

stim_le_s       = 10.0
trim_fr_s       = 0.500
trim_en_s       = 0.500

bins_total      = int(stim_le_s * bins_per_frame * fps)






n = ne.Neitz(
    peak_height=50,
    filepath=".")

time_s_t, spikes_t = n.load_csv_spikes(fn)

# Define time window
t_beg   = time_s_t[0]  + trim_fr_s
t_end   = time_s_t[-1] - trim_en_s

# Boolean mask
mask = (time_s_t >= t_beg) & (time_s_t <= t_end)

# Apply mask
time_s = time_s_t[mask] - time_s_t[mask][0]   # re-zero time
spikes = spikes_t[mask, :]


spike_times = [
    n.t_rel[n.spikes[:, i] > 0]
    for i in range(n.spikes.shape[1])
]



t = np.histogram(spikes, bins_total)




channel = 3
spike_mask = spikes[:, channel] > 0
spike_times_pl = time_s[spike_mask]

pl.figure(figsize=(12, 6))
pl.plot(time_s, np.zeros_like(time_s), alpha=0)  # dummy axis
pl.plot(time_s, spikes[:,channel])
pl.plot(spike_times_pl, np.zeros_like(spike_times_pl), 'k|', markersize=25)
pl.xlabel('Time (s)')
pl.title(f'Spike times, channel {channel}')
pl.show()


