import Neitz as ne
import numpy as np

#fn = '/Users/j/Library/CloudStorage/GoogleDrive-j@jkuchen.com/My Drive/Tuo/SaraipRGC/siso-spikes.csv'
fn = '/Users/j/Neitz_Analysis_Suite/SaraipRGC/siso-spikes.csv'
#fn = 'SaraipRGC/siso-spikes.csv'

bins_per_frame = 6
fps = 60
fs = 10000

stim_le_s = 10.0
trim_fr_s = 0.500
trim_en_s = 0.500

bin_rate = bins_per_frame * fps          # 360 Hz
bin_width_s = 1.0 / bin_rate
bins_total = int(stim_le_s * bin_rate)   # 3600 bins

n = ne.Neitz(
    peak_height=50,
    filepath="."
)


n.t_rel = time_s
n.spikes = spikes

spike_times = n.extract_spike_times_from_matrix()



edges = np.linspace(0, stim_le_s, bins_total + 1)
response_counts, _ = np.histogram(spike_times[ch], bins=edges)
response = response_counts.astype(float) * bin_rate