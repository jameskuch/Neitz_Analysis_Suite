import numpy as np
import matplotlib.pyplot as pl
from neitz.io import csv as ncsv


# ============================================================
# User settings
# ============================================================

spike_csv = '/Users/j/Neitz_Analysis_Suite/SaraipRGC/siso-spikes.csv'
stim_csv  = '/Users/j/Neitz_Analysis_Suite/SaraipRGC/siso-stdev.csv'


# ------------------------------------------------------------
# Normalization toggles  (the whole point of v4 — flip these to compare)
# ------------------------------------------------------------
# NORMALIZE_STA:    master on/off for normalizing the AVERAGE filter (the
#                   spike-triggered average). False -> raw reverse-correlation
#                   units (spikes/s).
#
# STA_NORM_METHOD:  how to normalize when NORMALIZE_STA is True --
#                   'max' -> divide by max(abs(filter)). This reproduces Sara's
#                            graphDataOnline display EXACTLY (her plot does
#                            linearFilter / max(max(abs(linearFilter))), so the
#                            largest deflection sits at +/-1.0). Use this to
#                            overlay directly on the MATLAB blue trace.
#                   'std' -> divide by std(filter). This is Sara's INTERNAL
#                            analysis.linearFilter value (peak ends up ~5).
#
# NORMALIZE_EPOCHS: divide EACH epoch's filter by its own std before drawing
#                   the faint background traces. This is a DISPLAY choice that
#                   is NOT in Sara's pipeline (she only ever shows the average).
#                   Set False to see the raw per-epoch filters at true scale.
#
# Note: the FFT temporal-tuning curves are computed from whatever filter the
# flags produce, so toggling NORMALIZE_STA also rescales the tuning plot.
NORMALIZE_STA    = True
STA_NORM_METHOD  = 'max'    # 'max' (matches MATLAB plot) or 'std' (Sara internal)
NORMALIZE_EPOCHS = True

# Sample-std (ddof=1, i.e. divide by N-1) matches MATLAB's default std().
STD_DDOF = 1


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
    # delegates to the shared engine (numerically identical to the old inline FFT)
    from neitz.analysis import revcorr
    if zero_pad is None:
        zero_pad = len(stimulus)
    return revcorr.reverse_correlation(stimulus, response, filter_len, zero_pad=zero_pad)


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

# ------------------------------------------------------------
# Average filter (the spike-triggered average)
# ------------------------------------------------------------
# Raw mean over epochs (same as Sara's linearFilter/numEpochs).
avg_raw = np.mean(per_epoch_filters, axis=0)

# Normalization FACTOR — computed from the average and applied CONSISTENTLY to both
# the average AND the per-epoch display traces, so the overlay is on a common scale.
# (Previously each epoch was divided by its OWN std while the average was divided by
# max/std of the average -> different scales -> the average looked misleadingly small.
# Averaging a coherent signal + incoherent noise makes the average ~1.7x the typical
# single epoch here, which is now visible.)
if NORMALIZE_STA:
    if STA_NORM_METHOD == 'max':
        norm_factor = np.max(np.abs(avg_raw))
        sta_ylabel = 'normalized amplitude'
        sta_state = 'STA normalized (/max|.|, matches MATLAB)'
    elif STA_NORM_METHOD == 'std':
        norm_factor = np.std(avg_raw, ddof=STD_DDOF)
        sta_ylabel = 'normalized amplitude'
        sta_state = 'STA normalized (/std, Sara internal)'
    else:
        raise ValueError(f"STA_NORM_METHOD must be 'max' or 'std', got {STA_NORM_METHOD!r}")
else:
    norm_factor = 1.0
    sta_ylabel = 'amplitude (spikes/s)'
    sta_state = 'STA raw (no normalization)'

avg_filter = avg_raw / norm_factor

print(f"\n{sta_state}; "
      f"peak |amp| = {np.max(np.abs(avg_filter)):.4g} at "
      f"{1000.0 * np.argmax(np.abs(avg_filter)) / bin_rate:.1f} ms")

# Frequency axis for rfft
freqs = np.fft.rfftfreq(filter_len, d=1.0 / bin_rate)

# ------------------------------------------------------------
# Per-epoch traces for display — scaled by the SAME factor as the average so the
# overlay shares one scale (NORMALIZE_EPOCHS=False shows the raw per-epoch filters).
# ------------------------------------------------------------
per_epoch_display = np.zeros_like(per_epoch_filters)
per_epoch_tuning = np.zeros((num_epochs, len(freqs)), dtype=float)

for ep in range(num_epochs):
    trace = per_epoch_filters[ep, :] / (norm_factor if NORMALIZE_EPOCHS else 1.0)
    per_epoch_display[ep, :] = trace
    per_epoch_tuning[ep, :] = np.abs(np.fft.rfft(trace))

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
        per_epoch_display[ep, :],
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
ax1.set_ylabel(sta_ylabel)
ax1.set_title(f'S-iso gaussian noise: all 15 epochs + average  [{sta_state}]')
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
