from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyabf
from scipy.interpolate import interp1d
from scipy.signal import find_peaks


@dataclass
class Neitz:
    filepath: Path = Path.cwd()
    fs: float = 1e4

    sweep: int = 0
    spike_ch_num: int = 0
    stim_ch_num: int = 2

    peak_height: float = 20.0 # needs to be adjustible based on recording. Maybe add a quick GUI
    stim_threshold: float = 1.0
    t_omit_on: float = 1.0
    sta_win_s: float = 0.2


    csv_path = None
    csv_header = None
    csv_time_col: int = 1 # starts at 0, is frame number
    csv_val_col: int = 2 # R value, but RGB are all same

    def __init__(self, filepath=Path.cwd(), fs=1e4,
                 sweep=0, spike_ch_num=0, stim_ch_num=2,
                 peak_height=20.0, stim_threshold=1.0,
                 t_omit_on=1.0, sta_win_s=0.2,
                 csv_path=None, csv_header=None,
                 csv_time_col=1, csv_val_col=2):

        self.filepath = Path(filepath)
        self.datapath = self.filepath / "data"

        self.fs = fs
        self.sweep = sweep
        self.spike_ch_num = spike_ch_num
        self.stim_ch_num = stim_ch_num

        self.peak_height = peak_height
        self.stim_threshold = stim_threshold
        self.t_omit_on = t_omit_on
        self.sta_win_s = sta_win_s

        self.csv_path = Path(csv_path) if csv_path is not None else None
        self.csv_header = csv_header
        self.csv_time_col = csv_time_col
        self.csv_val_col = csv_val_col

    def _require(self, name: str):
        if not hasattr(self, name):
            raise RuntimeError(f"Missing '{name}'. Call the required earlier step first.")

    def abfread(self, filename: str) -> "Neitz":
        abf = pyabf.ABF(str(self.datapath / filename))

        abf.setSweep(self.sweep, channel=self.spike_ch_num)
        self.spike_ch = abf.sweepY.copy()
        self.time_vec = abf.sweepX.copy()

        abf.setSweep(self.sweep, channel=self.stim_ch_num)
        self.stim_ch = abf.sweepY.copy()
        self.stim_ch_m = float(np.max(self.stim_ch))
        self.stim_ch = (self.stim_ch - self.stim_ch_m) * -1

        self.dt = float(np.mean(np.diff(self.time_vec)))
        self.fs = 1.0 / self.dt

        return self

    def find_spikes(self):
        self._require("spike_ch")
        self.peaks, _ = find_peaks(self.spike_ch, height=self.peak_height)
        return self.peaks

    def find_stim_on_off(self):
        self._require("stim_ch")
        self._require("time_vec")

        stim_on = self.stim_ch > self.stim_threshold
        edges = np.diff(stim_on.astype(int))

        rise = np.where(edges == 1)[0] + 1
        fall = np.where(edges == -1)[0] + 1

        if len(rise) == 0 or len(fall) == 0:
            raise RuntimeError("Could not find stim on/off edges. Check stim_threshold or stim channel.")

        i_on = int(rise[0])
        fall_after = fall[fall > i_on]
        if len(fall_after) == 0:
            raise RuntimeError("Found stim rise but no falling edge after it.")

        i_off = int(fall_after[0])

        self.t_on = float(self.time_vec[i_on])
        self.t_off = float(self.time_vec[i_off])
        return self.t_on, self.t_off

    def load_csv(self, csv_filename=None):
        if csv_filename is not None:
            self.csv_path = self.datapath / csv_filename
        if self.csv_path is None:#shouldn't be none, it will be the same as the data. So should be different error.
            raise RuntimeError("No csv_path set.")

        df = pd.read_csv(self.csv_path, header=self.csv_header)

        self.t_60 = df.iloc[:, 1].to_numpy()
        self.v_60 = df.iloc[:, 2].to_numpy()

        self.v0 = self.v_60.mean()
        self.C_60 = (self.v_60 - self.v0) / self.v0


        return self.t_60, self.C_60

    def align_contrast(self):
        self._require("time_vec")
        if not hasattr(self, "t_60") or not hasattr(self, "C_60"):
            self.load_csv()

        f = interp1d(
            self.t_60,
            self.C_60,
            kind="previous",
            bounds_error=False,
            fill_value=(0.0, 0.0),
        )
        self.C_20k = f(self.time_vec)
        return self.C_20k

    def compute_sta(self):
        self._require("time_vec")
        self._require("fs")

        if not hasattr(self, "peaks"):
            self.find_spikes()
        if not hasattr(self, "t_on") or not hasattr(self, "t_off"):
            self.find_stim_on_off()
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
