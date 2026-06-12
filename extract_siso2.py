import numpy as np
import matplotlib.pyplot as pl
from neitz.io import csv as ncsv


# ============================================================
# User settings
# ============================================================

#spike_csv = '/Users/j/Library/CloudStorage/GoogleDrive-j@jkuchen.com/My Drive/Tuo/SaraipRGC/siso-spikes.csv'
#stim_csv  = '/Users/j/Library/CloudStorage/GoogleDrive-j@jkuchen.com/My Drive/Tuo/SaraipRGC/siso-stdev.csv'

#spike_csv = 'SaraipRGC/siso-spikes.csv'
#stim_csv  = 'SaraipRGC/siso-stdev.csv'

spike_csv = '/Users/j/Neitz_Analysis_Suite/SaraipRGC/siso-spikes.csv'
stim_csv  = '/Users/j/Neitz_Analysis_Suite/SaraipRGC/siso-stdev.csv'




bins_per_frame = 6
fps = 60
fs = 10000

stim_le_s = 10.0
trim_fr_s = 0.500
trim_en_s = 0.500

blank_bins = 0
filter_len = bins_per_frame * fps   # 360 bins = 1 second
bin_rate = bins_per_frame * fps
bins_total = int(stim_le_s * bin_rate)


# ============================================================
# Helper functions
# ============================================================

def normalize_dlp_to_centered(stim):
    """
    Convert DLP 0..255 intensities to centered range roughly [-1, +1].
    128 becomes near 0.
    """
    return (stim.astype(float) - 127.5) / 127.5


def bin_spike_times(spike_times, stim_le_s, bins_total, bin_rate):
    edges = np.linspace(0, stim_le_s, bins_total + 1)
    counts, _ = np.histogram(spike_times, bins=edges)
    response = counts.astype(float) * bin_rate
    return counts, response, edges


def compute_filter_fft(stimulus, response, filter_len, zero_pad=None):
    n = len(stimulus)

    if len(response) != n:
        raise RuntimeError(
            f"Stimulus and response length mismatch: {len(stimulus)} vs {len(response)}"
        )

    if zero_pad is None:
        zero_pad = n

    s_pad = np.concatenate([stimulus, np.zeros(zero_pad)])
    r_pad = np.concatenate([response, np.zeros(zero_pad)])

    Rsr = np.fft.fft(r_pad) * np.conj(np.fft.fft(s_pad))
    filt = np.real(np.fft.ifft(Rsr))

    return filt[:filter_len]


def compute_temporal_tuning(filt, bin_rate):
    F = np.fft.rfft(filt)
    freqs = np.fft.rfftfreq(len(filt), d=1.0 / bin_rate)
    amp = np.abs(F)
    return freqs, amp


# ============================================================
# Main
# ============================================================

# ------------------------------------------------------------
# Load and trim spikes
# ------------------------------------------------------------
time_vec, spike_ch = ncsv.load_spikes_csv(spike_csv)

t_beg = time_vec[0]  + trim_fr_s
t_end = time_vec[-1] - trim_en_s

mask = (time_vec >= t_beg) & (time_vec < t_end)

time_s = time_vec[mask] - time_vec[mask][0]
spikes = spike_ch[mask, :]

spike_times_all = ncsv.spike_times_from_matrix(spikes, time_s)


# ------------------------------------------------------------
# Load stimulus epochs from CSV
# ------------------------------------------------------------
stim_epochs_all, stim_phases = ncsv.load_stimulus_epochs_csv(stim_csv)
stim_epochs = ncsv.stim_phase_only(stim_epochs_all, stim_phases)

# stim_epochs should now be (600, 15)
print("stim_epochs shape:", stim_epochs.shape)
print("spikes shape:", spikes.shape)

num_stim_rows, num_epochs = stim_epochs.shape

if num_epochs != spikes.shape[1]:
    raise RuntimeError(
        f"Mismatch: stimulus has {num_epochs} epochs but spikes has {spikes.shape[1]} columns"
    )

# Upsample from 600 frame values -> 3600 bins
stim_upsampled_all = np.repeat(stim_epochs, bins_per_frame, axis=0)


if stim_upsampled_all.shape[0] != bins_total:
    raise RuntimeError(
        f"Stimulus upsampled length mismatch: {stim_upsampled_all.shape[0]} vs expected {bins_total}"
    )






#n.compute_sta()



#trials = []

#sta_norm, lags_ms, fs, n_spikes = n.compute_sta_from_trials(trials)


#trials = []

    # def load_trial(
    #     self,
    #     abf_name: str,
    #     csv_filename: str | None = None,
    #     *,
    #     spike_polarity: str = "auto",
    #     stim_thr: float | None = None,
    #     active_high: bool | None = None,
    #     long_pause_s: float = 0.5,
    #     search_from_s: float = 0.0,
    # ) -> "Neitz":
    #     """
    #     Run full pipeline: abfread → find_spikes → find_stim_on_off → load_csv → align_contrast.
    #     Returns self so you can chain e.g. n.load_trial("file.abf").plot_trial().
    #     """
    #     self.abfread(abf_name)
    #     self.find_spikes(spike_polarity=spike_polarity)
    #     thr = self.stim_threshold if stim_thr is None else stim_thr
    #     self.find_stim_on_off_by_first_rise_and_pause(
    #         thr=thr,
    #         active_high=active_high,
    #         long_pause_s=long_pause_s,
    #         search_from_s=search_from_s,
    #     )
    #     self.load_csv(csv_filename)
    #     self.align_contrast()
    #     return self



stim_upsampled_all = stim_upsampled_all.astype(float)


if blank_bins > 0:
    stim_upsampled_all[:blank_bins, :] = 0

# ------------------------------------------------------------
# Per-epoch filter computation
# ------------------------------------------------------------
per_epoch_filters = np.zeros((num_epochs, filter_len), dtype=float)
per_epoch_tuning = []
per_epoch_responses = []

for ep in range(num_epochs):
    # Spike times for epoch ep
    st = spike_times_all[ep]

    counts, response, edges = bin_spike_times(
        spike_times=st,
        stim_le_s=stim_le_s,
        bins_total=bins_total,
        bin_rate=bin_rate
    )


    if blank_bins > 0:
        response[:blank_bins] = 0

    stim_ep = stim_upsampled_all[:, ep].astype(float)
    #stim_ep = stim_ep - np.mean(stim_ep)
   
    
    
    filt = compute_filter_fft(
        stimulus=stim_ep,
        response=response,
        filter_len=filter_len,
        zero_pad=bins_total
    )

    
    
    per_epoch_filters[ep, :] = filt
    per_epoch_responses.append(response)

    freqs, amp = compute_temporal_tuning(filt, bin_rate)
    per_epoch_tuning.append(amp)

    print(f"Processed epoch {ep + 1:02d} / {num_epochs}")

# Average raw filter across epochs
avg_filter = np.mean(per_epoch_filters, axis=0)

# MATLAB step 2: normalize average filter by its std
avg_filter = avg_filter / np.std(avg_filter, ddof=1)

# Frequency axis for rfft
freqs = np.fft.rfftfreq(filter_len, d=1.0 / bin_rate)

# MATLAB per-epoch normalization before FFT
per_epoch_filters_norm = np.zeros_like(per_epoch_filters)
per_epoch_tuning = np.zeros((num_epochs, len(freqs)), dtype=float)

for ep in range(num_epochs):
    norm_ep = per_epoch_filters[ep, :] / np.std(per_epoch_filters[ep, :], ddof=1)
    per_epoch_filters_norm[ep, :] = norm_ep
    per_epoch_tuning[ep, :] = np.abs(np.fft.rfft(norm_ep))

avg_tuning = np.abs(np.fft.rfft(avg_filter))
avg_response = np.mean(np.vstack(per_epoch_responses), axis=0)


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
filter_time_ms = 1000.0 * np.arange(filter_len) / bin_rate
bin_centers = 0.5 * (edges[:-1] + edges[1:])

pl.figure(figsize=(12, 12))

# Top: filters
ax1 = pl.subplot(2, 1, 1)
for ep in range(num_epochs):
    ax1.plot(
        filter_time_ms,
        per_epoch_filters_norm[ep, :],
        linewidth=0.8,
        alpha=0.25
    )

ax1.plot(
    filter_time_ms,
    avg_filter,
    linewidth=3.0
)

ax1.axhline(0, color='k', linewidth=0.8, alpha=0.4)
ax1.set_xlabel('time (msec)')
ax1.set_ylabel('normalized amplitude')
ax1.set_title('S-iso gaussian noise: all 15 epochs + average')
ax1.grid(True, alpha=0.25)

# Bottom: tuning
nyquist_idx = len(freqs)
ax2 = pl.subplot(2, 1, 2)
for ep in range(num_epochs):
    ax2.plot(freqs[:nyquist_idx], per_epoch_tuning[ep, :nyquist_idx], linewidth=0.8, alpha=0.25)

ax2.plot(freqs[:nyquist_idx], avg_tuning[:nyquist_idx], linewidth=3.0)
ax2.set_xlim(0, 200)

ax2.set_xlabel('frequency (hz)')
ax2.set_ylabel('amplitude')
ax2.set_title('S-iso temporal tuning: all 15 epochs + average')
ax2.grid(True, alpha=0.25)

pl.tight_layout()
pl.show()