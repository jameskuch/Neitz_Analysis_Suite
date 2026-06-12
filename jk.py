import Neitz as ne
import numpy as np

import pandas as pd

from matplotlib import pyplot as pl

start = 40
end = 40
#abf_names = [f"ep{i:01d}.abf" for i in range(start, end + 1)]
abf_names = [f"2026_06_02_{i:04d}.abf" for i in range(start, end + 1)]
duration_s = 6.0  # seconds to show (starting from t=0)




n = ne.Neitz(
    peak_height=50,
    filepath=".")


for i in np.arange(0,len(abf_names)):
    
    #n.abfread(abf_names[1])
    n.abfread(abf_names[i])


    pl.subplot(2, 1, 1)                 # top panel
    pl.plot(n.time_vec, n.spike_ch)     # spike train
    pl.ylabel("spikes (pA)")

    pl.subplot(2, 1, 2)                 # bottom panel
    pl.plot(n.time_vec, n.stim_ch)      # stimulus / frame sync
    pl.ylabel("stimulus")
    pl.xlabel("time (s)")

    pl.show()
    #print(n.stim_ch_m)


#time = n.time_vec
#voltage = n.spike_ch

#pl.plot(n.time_vec, n.spike_ch)
#pl.show()


#peaks = n.find_spikes(spike_polarity="neg")
    #peaks = n.find_spikes()


#print(peaks)              # indices of spikes
#print(n.time_vec[peaks])  # spike times
#print(n.spike_ch[peaks])  # spike voltages

#pl.plot(n.time_vec, n.spike_ch)
#pl.plot(n.time_vec[peaks], [1]*len(peaks), "k|", markersize=12)
#pl.xlabel("Time (s)")
#pl.yticks([])
#pl.show()

#pl.plot(n.time_vec[peaks], n.spike_ch[peaks], "ro", label="spikes")
#pl.xlabel("Time (s)")
#pl.ylabel("Voltage")
#pl.legend()
#pl.show()