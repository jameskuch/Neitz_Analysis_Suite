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


    # ----------------------------
    # STA
    # ----------------------------
    def compute_sta(self):
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
        plt.title("Spike-triggered average (STA)")
        plt.xlim(0, self.sta_win_s * 1000.0)
        plt.ylim(-1, 1)
        plt.tight_layout()
        plt.show()


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

    n.abfread("2026_02_04_0005.abf")
    n.find_spikes(spike_polarity="auto")

    n.find_stim_on_off_by_first_rise_and_pause(
        thr=0.02,
        active_high=False,
        long_pause_s=0.5,
        search_from_s=2.0,
    )

    n.load_csv("achromatic_gaussian_120s_60Hz_seed1234_20260204_160729.csv")
    n.align_contrast()
    n.compute_sta()
    n.plot_sta(smooth_ms=1.0)
