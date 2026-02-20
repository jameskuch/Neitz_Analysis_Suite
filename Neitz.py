from pathlib import Path

import numpy as np
import pandas as pd
import pyabf
from scipy.interpolate import interp1d
from scipy.signal import find_peaks


class Neitz:
    """
    - abfread() offsets+inverts stim channel but does NOT normalize to 0–100.
    - find_spikes() supports spike_polarity="auto" for negative-going spikes.
    """

    def __init__(
        self,
        filepath=Path.cwd(),
        fs=1e4,
        sweep=0,
        spike_ch_num=0,
        stim_ch_num=2,
        peak_height=20.0,
        stim_threshold=1.0,
        t_omit_on=1.0,
        sta_win_s=0.2,
        csv_path=None,
        csv_header=None,
    ):
        self.filepath = Path(filepath)
        self.datapath = self.filepath / "data"

        self.fs = float(fs)
        self.sweep = int(sweep)
        self.spike_ch_num = int(spike_ch_num)
        self.stim_ch_num = int(stim_ch_num)

        self.peak_height = float(peak_height)
        self.stim_threshold = float(stim_threshold)
        self.t_omit_on = float(t_omit_on)
        self.sta_win_s = float(sta_win_s)

        self.csv_path = Path(csv_path) if csv_path is not None else None
        self.csv_header = csv_header  # if you *want* to force a specific header row index

    def _require(self, name: str):
        if not hasattr(self, name):
            raise RuntimeError(f"Missing '{name}'. Call the required earlier step first.")

    # ----------------------------
    # IO
    # ----------------------------
    def abfread(self, filename: str) -> "Neitz":
        abf = pyabf.ABF(str(self.datapath / filename))

        abf.setSweep(self.sweep, channel=self.spike_ch_num)
        self.spike_ch = abf.sweepY.copy()
        self.time_vec = abf.sweepX.copy()

        abf.setSweep(self.sweep, channel=self.stim_ch_num)
        stim_raw = abf.sweepY.copy()

        # offset + invert (not normalized)
        stim_m = float(np.max(stim_raw))
        self.stim_ch = (stim_raw - stim_m) * -1.0
        self.stim_ch_m = stim_m

        self.dt = float(np.mean(np.diff(self.time_vec)))
        self.fs = 1.0 / self.dt

        return self

    def load_csv(self, csv_filename: str | None = None):
        """
        Robust CSV loader.

        Preferred (header present): columns include "time_s" and "contrast".
        Fallback (no header): assume columns are [frame, time_s, intensity, contrast]
                              i.e. time_s = col 1, contrast = col 3.
        """
        if csv_filename is not None:
            self.csv_path = self.datapath / csv_filename
        if self.csv_path is None:
            raise RuntimeError("No csv_path set (pass csv_filename or set csv_path in constructor).")

        required = {"time_s", "contrast"}

        # 1) Try reading with a header (normal case)
        if self.csv_header is None:
            df = pd.read_csv(self.csv_path)  # header='infer'
        else:
            df = pd.read_csv(self.csv_path, header=self.csv_header)

        # clean column names if they are strings
        if len(df.columns) and isinstance(df.columns[0], str):
            df.columns = [c.strip() for c in df.columns]

        if required.issubset(set(df.columns)):
            self.t_rel_60 = df["time_s"].to_numpy(dtype=float)
            self.C_60 = df["contrast"].to_numpy(dtype=float)

            if "frame" in df.columns:
                self.frame = df["frame"].to_numpy()
            if "intensity" in df.columns:
                self.intensity = df["intensity"].to_numpy(dtype=float)

            return self.t_rel_60, self.C_60

        # 2) Fallback: read with no header and use fixed column indices
        df2 = pd.read_csv(self.csv_path, header=None)

        if df2.shape[1] < 4:
            raise RuntimeError(
                f"CSV does not have named columns {required} and also has <4 columns. "
                f"Shape={df2.shape}, columns={list(df.columns)}"
            )

        # assume [frame, time_s, intensity, contrast]
        self.t_rel_60 = df2.iloc[:, 1].to_numpy(dtype=float)
        self.C_60 = df2.iloc[:, 3].to_numpy(dtype=float)

        # optional extras
        self.frame = df2.iloc[:, 0].to_numpy()
        self.intensity = df2.iloc[:, 2].to_numpy(dtype=float)

        return self.t_rel_60, self.C_60

    # ----------------------------
    # Spike detection (supports negative spikes)
    # ----------------------------
    def find_spikes(self, *, spike_polarity: str = "auto"):
        """
        spike_polarity:
            "pos"  -> find positive peaks on spike_ch
            "neg"  -> find positive peaks on -spike_ch (negative deflections)
            "auto" -> try both and pick more peaks; tie-break by larger median height
        """
        self._require("spike_ch")
        y = np.asarray(self.spike_ch, dtype=float)

        if spike_polarity not in {"pos", "neg", "auto"}:
            raise ValueError("spike_polarity must be one of: 'pos', 'neg', 'auto'")

        def _peaks(sig):
            p, props = find_peaks(sig, height=self.peak_height)
            h = props.get("peak_heights", np.array([], dtype=float))
            return p, h

        if spike_polarity == "pos":
            peaks, _ = _peaks(y)
            self.spike_polarity = "pos"
        elif spike_polarity == "neg":
            peaks, _ = _peaks(-y)
            self.spike_polarity = "neg"
        else:
            p_pos, h_pos = _peaks(y)
            p_neg, h_neg = _peaks(-y)

            if len(p_pos) > len(p_neg):
                peaks = p_pos
                self.spike_polarity = "pos"
            elif len(p_neg) > len(p_pos):
                peaks = p_neg
                self.spike_polarity = "neg"
            else:
                med_pos = float(np.median(h_pos)) if len(h_pos) else -np.inf
                med_neg = float(np.median(h_neg)) if len(h_neg) else -np.inf
                if med_neg > med_pos:
                    peaks = p_neg
                    self.spike_polarity = "neg"
                else:
                    peaks = p_pos
                    self.spike_polarity = "pos"

        self.peaks = np.asarray(peaks, dtype=int)
        return self.peaks

    # ----------------------------
    # Stim epoch detection
    # ----------------------------
    def find_stim_on_off_by_first_rise_and_pause(
        self,
        *,
        thr: float | None = None,
        long_pause_s: float = 0.25,
        search_from_s: float = 0.0,
        active_high: bool | None = None,
    ):
        """
        Onset = first transition into "pulse state".
        Offset = end of last pulse before a long pause (gap between pulse starts > long_pause_s).

        If pulses are dips from a high baseline, use active_high=False.
        """
        self._require("stim_ch")
        self._require("time_vec")
        self._require("fs")

        t = self.time_vec
        y = np.asarray(self.stim_ch, dtype=float)
        thr = self.stim_threshold if thr is None else float(thr)

        i0 = int(np.searchsorted(t, search_from_s)) if search_from_s > 0 else 0

        # auto polarity: if mostly above thr, pulses are likely dips
        if active_high is None:
            frac_above = float(np.mean(y[i0:] > thr))
            active_high = frac_above < 0.5

        stim_on = (y > thr) if active_high else (y < thr)

        edges = np.diff(stim_on.astype(int))
        rise = np.where(edges == 1)[0] + 1
        fall = np.where(edges == -1)[0] + 1

        rise = rise[rise >= i0]
        fall = fall[fall >= i0]

        if len(rise) == 0:
            raise RuntimeError("No rising edges found. Tune thr or active_high.")
        if len(fall) == 0:
            raise RuntimeError("No falling edges found. Tune thr or active_high.")

        i_on = int(rise[0])
        self.t_on = float(t[i_on])

        rise_t = t[rise]
        gaps = np.diff(rise_t)
        idx = np.where(gaps > long_pause_s)[0]
        last_rise = int(rise[int(idx[0])]) if len(idx) > 0 else int(rise[-1])

        j = np.searchsorted(fall, last_rise, side="right")
        if j >= len(fall):
            raise RuntimeError("Found last rise but no falling edge after it. Lower thr or check polarity.")

        i_off = int(fall[j])
        self.t_off = float(t[i_off])

        self.stim_on_idx = i_on
        self.stim_off_idx = i_off
        return self.t_on, self.t_off

    # ----------------------------
    # Contrast alignment
    # ----------------------------
    def align_contrast(self):
        self._require("time_vec")

        if not hasattr(self, "t_on"):
            self.find_stim_on_off_by_first_rise_and_pause(
                thr=self.stim_threshold, active_high=None, long_pause_s=0.5, search_from_s=0.0
            )

        if not hasattr(self, "t_rel_60") or not hasattr(self, "C_60"):
            self.load_csv()

        frame_dt = float(np.median(np.diff(self.t_rel_60)))
        shift_frames = 0  # <- change this for testing

        t_abs_60 = self.t_on + np.asarray(self.t_rel_60, dtype=float) + shift_frames * frame_dt
        c = np.asarray(self.C_60, dtype=float)

        order = np.argsort(t_abs_60)
        t_abs_60 = t_abs_60[order]
        c = c[order]

        f = interp1d(
            t_abs_60,
            c,
            kind="previous",
            bounds_error=False,
            fill_value=(0.0, 0.0),
            assume_sorted=True,
        )
        self.C_20k = f(self.time_vec)
        return self.C_20k

    def align_contrast_to_stim_edges(
        self,
        *,
        thr: float | None = None,
        active_high: bool | None = None,
    ) -> np.ndarray:
        """
        Align contrast to the actual frame times from the stim signal (high-low transitions).
        Contrast changes only at frame boundaries, so skips or jitter in recorded frames
        are corrected. Use this for STA; use align_contrast() for simple flicker viewing.
        """
        self._require("time_vec")
        self._require("stim_ch")
        if not hasattr(self, "t_on") or not hasattr(self, "t_off"):
            self.find_stim_on_off_by_first_rise_and_pause(
                thr=thr or self.stim_threshold,
                active_high=active_high,
                long_pause_s=0.5,
                search_from_s=0.0,
            )
        if not hasattr(self, "t_rel_60") or not hasattr(self, "C_60"):
            self.load_csv()

        t = self.time_vec
        y = np.asarray(self.stim_ch, dtype=float)
        thr = thr if thr is not None else self.stim_threshold
        if active_high is None:
            frac_above = float(np.mean(y > thr))
            active_high = frac_above < 0.5

        # Detect frame boundaries: use only the transition INTO the active state
        # (one edge per frame, not two). Each CSV row = one frame.
        stim_on = (y > thr) if active_high else (y < thr)
        edges = np.diff(stim_on.astype(int))
        # Rising edge of stim_on = transition into active state = start of new frame
        frame_starts_idx = np.where(edges == 1)[0] + 1
        frame_starts_t = t[frame_starts_idx]
        frame_starts_t = frame_starts_t[
            (frame_starts_t >= self.t_on) & (frame_starts_t <= self.t_off)
        ]
        if len(frame_starts_t) < 2:
            self.C_20k = self.align_contrast()
            return self.C_20k

        # Each interval [frame_starts_t[i], frame_starts_t[i+1]) gets C_60[i]
        n_seg = len(frame_starts_t) - 1
        C_60 = np.asarray(self.C_60, dtype=float)
        n_csv = len(C_60)
        fill = float(C_60[-1]) if n_csv else 0.0
        seg_contrast = np.full(n_seg, fill)
        if n_csv > 0:
            seg_contrast[: min(n_seg, n_csv)] = C_60[: min(n_seg, n_csv)]

        idx = np.searchsorted(frame_starts_t, t, side="right") - 1
        idx = np.clip(idx, 0, n_seg - 1)
        self.C_20k = seg_contrast[idx].astype(float)
        return self.C_20k

    # ----------------------------
    # STA
    # ----------------------------
    def compute_sta(self):
        """Compute spike-triggered average of contrast using align_contrast() (CSV-time based)."""
        self._require("time_vec")
        self._require("fs")

        if not hasattr(self, "peaks"):
            self.find_spikes()

        if not hasattr(self, "t_on") or not hasattr(self, "t_off"):
            self.find_stim_on_off_by_first_rise_and_pause(
                thr=self.stim_threshold, active_high=None, long_pause_s=0.5, search_from_s=0.0
            )

        if not hasattr(self, "C_20k"):
            self.align_contrast()

        sta_win_n = int(self.sta_win_s * self.fs)
        spike_times = self.time_vec[self.peaks]

        keep = (spike_times >= self.t_on + self.t_omit_on) & (spike_times <= self.t_off)
        peaks_keep = self.peaks[keep]

        segs = []
        for p in peaks_keep:
            if p < sta_win_n:
                continue
            seg = self.C_20k[p - sta_win_n : p]
            if len(seg) == sta_win_n:
                segs.append(seg)

        if len(segs) == 0:
            self.sta = np.full(sta_win_n, np.nan)
            self.sta_norm = self.sta.copy()
            return self.sta_norm

        segs = np.asarray(segs)
        sta = segs.mean(axis=0)[::-1]
        self.sta = sta

        mx = float(np.max(np.abs(sta)))
        self.sta_norm = sta / mx if mx > 0 else sta
        return self.sta_norm

    # ----------------------------
    # Plotting
    # ----------------------------
    def _gaussian_smooth(self, x, sigma_ms: float = 5.0):
        x = np.asarray(x, dtype=float)
        sigma = (sigma_ms / 1000.0) * float(self.fs)
        if sigma <= 0:
            return x
        radius = int(np.ceil(4 * sigma))
        k = np.arange(-radius, radius + 1)
        g = np.exp(-0.5 * (k / sigma) ** 2)
        g /= g.sum()
        return np.convolve(x, g, mode="same")

    def plot_sta(self, smooth_ms: float = 1.0):
        import matplotlib.pyplot as plt

        self._require("fs")
        if not hasattr(self, "sta_norm"):
            self.compute_sta()

        sta = self.sta_norm
        sta_s = self._gaussian_smooth(sta, sigma_ms=smooth_ms)

        n = len(sta_s)
        dt = 1.0 / float(self.fs)
        lags_ms = np.arange(n) * dt * 1000.0

        plt.figure()
        plt.plot(lags_ms, sta_s)
        plt.xlabel("Time before spike (ms)")
        plt.ylabel("Normalized contrast")
        plt.title("Average stimulus history preceding spikes")
        plt.xlim(0, self.sta_win_s * 1000.0)
        plt.ylim(-1, 1)
        plt.tight_layout()
        plt.show()

    # ----------------------------
    # High-level: load trial + plot (single call)
    # ----------------------------
    def load_trial(
        self,
        abf_name: str,
        csv_filename: str | None = None,
        *,
        spike_polarity: str = "auto",
        stim_thr: float | None = None,
        active_high: bool | None = None,
        long_pause_s: float = 0.5,
        search_from_s: float = 0.0,
    ) -> "Neitz":
        """
        Run full pipeline: abfread → find_spikes → find_stim_on_off → load_csv → align_contrast.
        Returns self so you can chain e.g. n.load_trial("file.abf").plot_trial().
        """
        self.abfread(abf_name)
        self.find_spikes(spike_polarity=spike_polarity)
        thr = self.stim_threshold if stim_thr is None else stim_thr
        self.find_stim_on_off_by_first_rise_and_pause(
            thr=thr,
            active_high=active_high,
            long_pause_s=long_pause_s,
            search_from_s=search_from_s,
        )
        self.load_csv(csv_filename)
        self.align_contrast()
        return self

    def load_trial_flicker(
        self,
        abf_name: str,
        csv_filename: str | None = None,
        *,
        stim_thr: float | None = None,
        active_high: bool | None = None,
        long_pause_s: float = 0.5,
        search_from_s: float = 0.0,
    ) -> "Neitz":
        """
        Load one trial for flicker viewing (raw spike_ch + stim ON blocks).
        No spike detection. CSV/contrast alignment is optional (skipped if no csv provided).
        Use with plot_trial_flicker().
        """
        self.abfread(abf_name)
        thr = self.stim_threshold if stim_thr is None else stim_thr
        self.find_stim_on_off_by_first_rise_and_pause(
            thr=thr,
            active_high=active_high,
            long_pause_s=long_pause_s,
            search_from_s=search_from_s,
        )
        if csv_filename is not None or self.csv_path is not None:
            self.load_csv(csv_filename)
            self.align_contrast()
        return self

    def _stim_on_spans(self, i0: int, i1: int, thr: float | None = None, merge_gap_s: float = 0.03):
        """
        Return list of (t_start, t_end) for ON blocks in stim_ch[i0:i1].
        Adjacent spans separated by less than merge_gap_s are merged into one
        continuous block (avoids tiny gaps between individual 60 Hz pulses).
        """
        y = np.asarray(self.stim_ch[i0:i1], dtype=float)
        t = self.time_vec[i0:i1]
        thr = thr if thr is not None else self.stim_threshold
        frac_above = float(np.mean(y > thr))
        on = (y > thr) if frac_above < 0.5 else (y < thr)
        edges = np.diff(on.astype(int))
        rises = np.where(edges == 1)[0] + 1
        falls = np.where(edges == -1)[0] + 1
        if on[0]:
            rises = np.concatenate(([0], rises))
        if on[-1]:
            falls = np.concatenate((falls, [len(on) - 1]))
        raw = [(float(t[r]), float(t[min(f, len(t) - 1)])) for r, f in zip(rises, falls)]
        if not raw:
            return []
        merged = [raw[0]]
        for s0, s1 in raw[1:]:
            if s0 - merged[-1][1] <= merge_gap_s:
                merged[-1] = (merged[-1][0], s1)
            else:
                merged.append((s0, s1))
        return merged

    def plot_trial_flicker(
        self,
        t_start_s: float | None = None,
        duration_s: float = 6.0,
        show: bool = True,
    ):
        """
        Plot raw spike channel with grey-shaded ON blocks from stim signal.
        Returns the matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        self._require("time_vec")
        self._require("spike_ch")
        self._require("stim_ch")
        if t_start_s is None:
            t_start_s = 0.0
        t_end_s = t_start_s + duration_s

        i0 = int(np.searchsorted(self.time_vec, t_start_s))
        i1 = int(np.searchsorted(self.time_vec, t_end_s))
        t = self.time_vec[i0:i1]
        spike = self.spike_ch[i0:i1]
        spans = self._stim_on_spans(i0, i1)

        fig, ax = plt.subplots(1, 1, figsize=(12, 3))
        for s0, s1 in spans:
            ax.axvspan(s0, s1, color="0.85", zorder=0)
        ax.plot(t, spike, color="C0", linewidth=0.4, label="spike_ch")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (mV)")
        ax.set_title("4 Hz Black/White Flicker (ON blocks shaded)")
        ax.legend(loc="upper left")
        ax.set_xlim(t_start_s, t_end_s)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def load_trial_and_plot_flicker(
        self,
        abf_name: str,
        csv_filename: str | None = None,
        duration_s: float = 2.0,
        show: bool = True,
        **load_trial_kw,
    ):
        """Load one trial and plot stim + contrast only (flicker view). Returns (self, figure)."""
        self.load_trial_flicker(abf_name, csv_filename=csv_filename, **load_trial_kw)
        fig = self.plot_trial_flicker(duration_s=duration_s, show=show)
        return self, fig

    def plot_trial(
        self,
        t_start_s: float | None = None,
        duration_s: float = 2.0,
        show: bool = True,
    ):
        """
        Plot one figure: stim sync, aligned contrast, and spike raster.
        Returns the matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        self._require("time_vec")
        self._require("stim_ch")
        self._require("C_20k")
        self._require("peaks")
        if t_start_s is None:
            t_start_s = getattr(self, "t_on", 0.0)
        t_end_s = t_start_s + duration_s

        i0 = int(np.searchsorted(self.time_vec, t_start_s))
        i1 = int(np.searchsorted(self.time_vec, t_end_s))
        t = self.time_vec[i0:i1]
        stim = self.stim_ch[i0:i1]
        contrast = self.C_20k[i0:i1]
        spike_times = self.time_vec[self.peaks]
        in_win = (spike_times >= t_start_s) & (spike_times <= t_end_s)
        spikes_in_win = spike_times[in_win]

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 5))
        axes[0].plot(t, stim, color="C0", label="stim sync")
        axes[0].set_ylabel("Stim sync")
        axes[0].legend(loc="upper right")
        axes[0].set_title("Single trial")
        axes[1].plot(t, contrast, color="C1", label="contrast (C_20k)")
        axes[1].set_ylabel("Contrast")
        axes[1].legend(loc="upper right")
        axes[2].scatter(spikes_in_win, np.ones_like(spikes_in_win), marker="|", color="C2", s=80)
        axes[2].set_ylabel("Spikes")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_yticks([])
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def load_trial_and_plot(
        self,
        abf_name: str,
        csv_filename: str | None = None,
        duration_s: float = 2.0,
        show: bool = True,
        **load_trial_kw,
    ):
        """Load one trial and plot it in one figure. Returns (self, figure)."""
        self.load_trial(abf_name, csv_filename=csv_filename, **load_trial_kw)
        fig = self.plot_trial(duration_s=duration_s, show=show)
        return self, fig

    # ----------------------------
    # Multi-trial: load + plot, or load + STA + plot (class methods)
    # ----------------------------
    @staticmethod
    def _resolve_trial_kwargs(filepath, csv_path, csv_name, neitz_kw):
        """Pop filepath/csv_path/csv_name from neitz_kw so callers can safely unpack NEITZ_KW."""
        neitz_kw = dict(neitz_kw)
        filepath = filepath if filepath is not None else neitz_kw.pop("filepath", None)
        neitz_kw.pop("filepath", None)
        if csv_path is None:
            csv_path = neitz_kw.pop("csv_path", None)
        else:
            neitz_kw.pop("csv_path", None)
        if csv_name is None:
            csv_name = neitz_kw.pop("csv_name", None)
        else:
            neitz_kw.pop("csv_name", None)
        filepath = Path(filepath) if filepath is not None else Path.cwd()
        if csv_path is None and csv_name is not None:
            csv_path = filepath / "data" / csv_name
        elif csv_path is not None:
            csv_path = Path(csv_path)
        return filepath, csv_path, csv_name, neitz_kw

    @classmethod
    def load_trials(
        cls,
        abf_names: list[str],
        filepath: Path | str | None = None,
        csv_path: Path | str | None = None,
        csv_name: str | None = None,
        **neitz_kw,
    ) -> list["Neitz"]:
        """Load multiple trials (with spike detection). neitz_kw passed to constructor."""
        filepath, csv_path, csv_name, neitz_kw = cls._resolve_trial_kwargs(
            filepath, csv_path, csv_name, neitz_kw
        )
        kwargs = {**neitz_kw, "filepath": filepath, "csv_path": csv_path}
        trials = []
        for abf_name in abf_names:
            n = cls(**kwargs)
            n.load_trial(abf_name, csv_filename=csv_name)
            trials.append(n)
        return trials

    @classmethod
    def load_trials_flicker(
        cls,
        abf_names: list[str],
        filepath: Path | str | None = None,
        csv_path: Path | str | None = None,
        csv_name: str | None = None,
        **neitz_kw,
    ) -> list["Neitz"]:
        """Load multiple trials for flicker viewing only (no spike detection)."""
        filepath, csv_path, csv_name, neitz_kw = cls._resolve_trial_kwargs(
            filepath, csv_path, csv_name, neitz_kw
        )
        kwargs = {**neitz_kw, "filepath": filepath, "csv_path": csv_path}
        trials = []
        for abf_name in abf_names:
            n = cls(**kwargs)
            n.load_trial_flicker(abf_name, csv_filename=csv_name)
            trials.append(n)
        return trials

    @staticmethod
    def compute_sta_from_trials(
        trials: list["Neitz"],
        sta_win_s: float | None = None,
        t_omit_on: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        """
        Collect spike-triggered contrast segments from multiple trials and average.
        Returns (sta_norm, lags_ms, fs, n_spikes).
        Uses sta_win_s and t_omit_on from the first trial if not provided.
        """
        if not trials:
            raise ValueError("Need at least one trial.")
        n0 = trials[0]
        sta_win_s = sta_win_s if sta_win_s is not None else n0.sta_win_s
        t_omit_on = t_omit_on if t_omit_on is not None else n0.t_omit_on

        fs_ref = float(n0.fs)
        sta_win_n = int(round(sta_win_s * fs_ref))
        all_segs = []

        for n in trials:
            if abs(float(n.fs) - fs_ref) / fs_ref > 1e-3:
                raise RuntimeError(f"Sampling rate mismatch: expected {fs_ref}, got {n.fs}")
            spike_times = n.time_vec[n.peaks]
            keep = (spike_times >= n.t_on + t_omit_on) & (spike_times <= n.t_off)
            peaks_keep = n.peaks[keep]
            for p in peaks_keep:
                if p < sta_win_n:
                    continue
                seg = n.C_20k[p - sta_win_n : p]
                if len(seg) == sta_win_n:
                    all_segs.append(seg)

        if not all_segs:
            raise RuntimeError("No spike segments collected across trials.")
        all_segs = np.asarray(all_segs)
        n_spikes = all_segs.shape[0]
        sta = all_segs.mean(axis=0)[::-1]
        mx = float(np.max(np.abs(sta)))
        sta_norm = sta / mx if mx > 0 else sta
        dt = 1.0 / fs_ref
        lags_ms = np.arange(len(sta_norm)) * dt * 1000.0
        return sta_norm, lags_ms, fs_ref, n_spikes

    @staticmethod
    def plot_sta_from_arrays(
        sta_norm: np.ndarray,
        lags_ms: np.ndarray,
        sta_win_s: float,
        n_spikes: int,
        smooth_ms: float = 1.0,
        fs: float | None = None,
    ):
        """Plot STA from precomputed arrays. If fs is None, inferred from lags_ms."""
        import matplotlib.pyplot as plt

        if fs is None and len(lags_ms) > 1:
            fs = 1000.0 / (lags_ms[1] - lags_ms[0])
        elif fs is None:
            fs = 10000.0
        # reuse instance smoother via a minimal dummy
        dummy = Neitz(sta_win_s=sta_win_s)
        dummy.fs = fs
        sta_plot = dummy._gaussian_smooth(sta_norm, sigma_ms=smooth_ms)

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(lags_ms, sta_plot)
        ax.set_xlabel("Time before spike (ms)")
        ax.set_ylabel("Normalized contrast")
        ax.set_title(f"Average stimulus history preceding spikes (n={n_spikes} spikes)")
        ax.set_xlim(0, sta_win_s * 1000.0)
        ax.set_ylim(-1, 1)
        plt.tight_layout()
        return fig

    @classmethod
    def load_trials_and_plot(
        cls,
        abf_names: list[str],
        filepath: Path | str | None = None,
        csv_name: str | None = None,
        duration_s: float = 1.0,
        overlay: bool = True,
        show: bool = True,
        labels: list[str] | None = None,
        **neitz_kw,
    ):
        """Load multiple trials and plot contrast in one figure (overlay or stacked)."""
        import matplotlib.pyplot as plt

        filepath_r, _, csv_name_r, nkw = cls._resolve_trial_kwargs(
            filepath, None, csv_name, neitz_kw
        )
        trials = cls.load_trials(abf_names, filepath=filepath_r, csv_name=csv_name_r, **nkw)
        n0 = trials[0]
        t_start_s = getattr(n0, "t_on", 0.0)
        t_end_s = t_start_s + duration_s
        if labels is None:
            labels = [Path(name).stem for name in abf_names]

        if overlay:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            for n, label in zip(trials, labels):
                i0 = int(np.searchsorted(n.time_vec, t_start_s))
                i1 = int(np.searchsorted(n.time_vec, t_end_s))
                ax.plot(n.time_vec[i0:i1], n.C_20k[i0:i1], alpha=0.7, label=label)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Contrast (C_20k)")
            ax.set_title("Multiple trials (overlay)")
            ax.legend(loc="upper right")
        else:
            fig, axes = plt.subplots(len(trials), 1, sharex=True, figsize=(10, 2 * len(trials)))
            axes = np.atleast_1d(axes)
            for ax, n, label in zip(axes, trials, labels):
                i0 = int(np.searchsorted(n.time_vec, t_start_s))
                i1 = int(np.searchsorted(n.time_vec, t_end_s))
                ax.plot(n.time_vec[i0:i1], n.C_20k[i0:i1], label=label)
                ax.set_ylabel("Contrast")
                ax.legend(loc="upper right")
            axes[-1].set_xlabel("Time (s)")
            fig.suptitle("Multiple trials (stacked)")
        plt.tight_layout()
        if show:
            plt.show()
        return trials, fig

    @classmethod
    def load_trials_and_plot_flicker(
        cls,
        abf_names: list[str],
        filepath: Path | str | None = None,
        csv_name: str | None = None,
        duration_s: float = 1.2,
        t_start_s: float | None = None,
        show: bool = True,
        labels: list[str] | None = None,
        **neitz_kw,
    ):
        """
        Load multiple trials (flicker, no spike detection).
        Plot vertically offset raw spike traces with grey ON boxes.
        Returns (list of Neitz instances, figure).
        """
        import matplotlib.pyplot as plt

        filepath_r, _, csv_name_r, nkw = cls._resolve_trial_kwargs(
            filepath, None, csv_name, neitz_kw
        )
        trials = cls.load_trials_flicker(
            abf_names, filepath=filepath_r, csv_name=csv_name_r, **nkw
        )
        if labels is None:
            labels = [str(i + 1) for i in range(len(trials))]

        n0 = trials[0]
        if t_start_s is None:
            t_start_s = 0.0
        t_end_s = t_start_s + duration_s
        i0_ref = int(np.searchsorted(n0.time_vec, t_start_s))
        i1_ref = int(np.searchsorted(n0.time_vec, t_end_s))

        fig, ax = plt.subplots(1, 1, figsize=(12, 1.5 * len(trials)))

        # Grey stim overlay from trial 1 (same method as single-trial figure)
        spans = n0._stim_on_spans(i0_ref, i1_ref)
        for s0, s1 in spans:
            ax.axvspan(s0, s1, color="0.85", zorder=0)

        for k, (n, label) in enumerate(zip(trials, labels)):
            i0 = int(np.searchsorted(n.time_vec, t_start_s))
            i1 = int(np.searchsorted(n.time_vec, t_end_s))
            t = n.time_vec[i0:i1]
            sig = n.spike_ch[i0:i1].copy()
            offset = (k + 1)
            sig_range = float(np.ptp(sig)) or 1.0
            sig_norm = sig / sig_range
            ax.plot(t, sig_norm + offset, linewidth=0.4, color=f"C{k}")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Trial #")
        ax.set_yticks(range(1, len(trials) + 1))
        ax.set_yticklabels(labels)
        ax.set_xlim(t_start_s, t_end_s)
        ax.set_ylim(0.3, len(trials) + 0.7)
        ax.set_title("4 Hz black/white flicker")
        plt.tight_layout()
        if show:
            plt.show()
        return trials, fig

    @classmethod
    def load_trials_sta_and_plot(
        cls,
        abf_names: list[str],
        filepath: Path | str | None = None,
        csv_name: str | None = None,
        smooth_ms: float = 1.0,
        show: bool = True,
        **neitz_kw,
    ):
        """
        Load multiple trials, compute STA across all spikes, and plot the STA figure.
        Uses align_contrast() (CSV-time based) for contrast alignment.
        Returns (trials, sta_norm, lags_ms, figure).
        """
        filepath_r, _, csv_name_r, nkw = cls._resolve_trial_kwargs(
            filepath, None, csv_name, neitz_kw
        )
        trials = cls.load_trials(abf_names, filepath=filepath_r, csv_name=csv_name_r, **nkw)
        sta_norm, lags_ms, fs, n_spikes = cls.compute_sta_from_trials(trials)
        sta_win_s = trials[0].sta_win_s
        fig = cls.plot_sta_from_arrays(
            sta_norm, lags_ms, sta_win_s, n_spikes, smooth_ms=smooth_ms, fs=fs
        )
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        return trials, sta_norm, lags_ms, fig


# ----------------------------
# Example usage (single main block)
# ----------------------------
if __name__ == "__main__":
    n = Neitz(
        filepath=Path.cwd(),
        sweep=0,
        spike_ch_num=0,
        stim_ch_num=2,
        peak_height=20.0,
        stim_threshold=0.02,
        sta_win_s=0.2,
        t_omit_on=1.0,
        csv_path=Path.cwd() / "data" / "achromatic_gaussian_120s_60Hz_seed1234_20260204_160729.csv",
    )
    n.load_trial_and_plot(
        "2026_02_04_0005.abf",
        csv_filename="achromatic_gaussian_120s_60Hz_seed1234_20260204_160729.csv",
        active_high=False,
        search_from_s=2.0,
    )
