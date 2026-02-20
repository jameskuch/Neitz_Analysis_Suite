"""
STA (Spike-Triggered Average) tutorial: load data and plot in one figure.

  Easiest way to run (no coding): open sta_tutorial.ipynb in Jupyter and run the cells.

  Command line:  python sta_tutorial.py  (runs all three examples)

================================================================================
How to install
================================================================================

  From the project root (Neitz_Analysis_Suite):

    pip install -r requirements.txt

  Or:  pip install numpy pandas pyabf scipy matplotlib

================================================================================
Three examples (single function calls from the Neitz package)
================================================================================

  1. Single trial (4 Hz flicker):  Neitz(...).load_trial_and_plot_flicker(abf_name, ...)
  2. Multiple trials (flicker):    Neitz.load_trials_and_plot_flicker(abf_names, ...)
  3. STA (contrast aligned to stim frame edges):  Neitz.load_trials_sta_and_plot(abf_names, ...)
"""

from pathlib import Path

from Neitz import Neitz


# -----------------------------------------------------------------------------
# Config (change for your data paths and stim/spike settings)
# -----------------------------------------------------------------------------
FILEPATH = Path.cwd()
CSV_NAME = "achromatic_gaussian_120s_60Hz_seed1234_20260204_160729.csv"
# Optional: override constructor defaults for Neitz
NEITZ_KW = dict(
    filepath=FILEPATH,
    sweep=0,
    spike_ch_num=0,
    stim_ch_num=2,
    peak_height=20.0,
    stim_threshold=0.02,
    sta_win_s=0.2,
    t_omit_on=1.0,
    csv_path=FILEPATH / "data" / CSV_NAME,
)
STIM_KW = dict(active_high=False, long_pause_s=0.5, search_from_s=2.0)


# -----------------------------------------------------------------------------
# Example 1: Single trial (simple 4 Hz flicker – stim + contrast only)
# -----------------------------------------------------------------------------
def example_1_single_trial_one_fig(
    abf_name: str = "2026_02_04_0005.abf",
    csv_name: str | None = None,
    duration_s: float = 2.0,
    show: bool = True,
):
    """Load a single trial and show stim + contrast only (4 Hz flicker, no spikes)."""
    n = Neitz(**NEITZ_KW)
    return n.load_trial_and_plot_flicker(
        abf_name,
        csv_filename=csv_name or CSV_NAME,
        duration_s=duration_s,
        show=show,
        **STIM_KW,
    )


# -----------------------------------------------------------------------------
# Example 2: Multiple trials (4 Hz flicker – overlay or stacked)
# -----------------------------------------------------------------------------
def example_2_multiple_trials_one_fig(
    abf_names: list[str] | None = None,
    csv_name: str | None = None,
    duration_s: float = 1.0,
    overlay: bool = True,
    show: bool = True,
):
    """Load multiple trials and show contrast in one figure (flicker view, no spikes)."""
    if abf_names is None:
        abf_names = [f"2026_02_04_{i:04d}.abf" for i in range(5, 10)]
    return Neitz.load_trials_and_plot_flicker(
        abf_names,
        filepath=FILEPATH,
        csv_name=csv_name or CSV_NAME,
        duration_s=duration_s,
        overlay=overlay,
        show=show,
        **NEITZ_KW,
    )


# -----------------------------------------------------------------------------
# Example 3: Multiple trials → compute STA → plot
# -----------------------------------------------------------------------------
def example_3_multiple_trials_sta(
    abf_names: list[str] | None = None,
    csv_name: str | None = None,
    smooth_ms: float = 1.0,
    show: bool = True,
):
    """Load multiple trials, compute STA across all spikes, then show the STA figure."""
    if abf_names is None:
        abf_names = [f"2026_02_04_{i:04d}.abf" for i in range(5, 10)]
    return Neitz.load_trials_sta_and_plot(
        abf_names,
        filepath=FILEPATH,
        csv_name=csv_name or CSV_NAME,
        smooth_ms=smooth_ms,
        show=show,
        **NEITZ_KW,
    )


# -----------------------------------------------------------------------------
# Main: run all three examples
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Example 1: Single trial in one figure")
    example_1_single_trial_one_fig()

    print("Example 2: Multiple trials in one figure")
    example_2_multiple_trials_one_fig(overlay=True)

    print("Example 3: Multiple trials → STA → figure")
    example_3_multiple_trials_sta(smooth_ms=1.0)
