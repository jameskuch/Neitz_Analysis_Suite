"""
One-time migration of existing repo data into the ephysdataio store.

NON-DESTRUCTIVE: copies files into ~/Documents/ephysdataio and writes manifests.
Nothing in the repo is deleted here — verify the store, then do the cleanup separately.

    /Users/j/miniconda3/bin/python migrate_to_ephysdataio.py
"""
import glob
import os

from neitz.dataio import DataStore
from neitz.io.abf import Recording
from neitz.analysis import flicker as flk

ds = DataStore()
print(f"data root: {ds.root}\n")


def import_flicker_cell(folder, label):
    cm = ds.new_cell("2026-06-02", label=label, cell_type=label)
    for f in sorted(glob.glob(f"{folder}/*.abf")):
        rec = Recording.load(f)
        fl = flk.detect_flicker(rec.channel("TTL"), rec.fs)
        params = {"flicker_hz": round(fl.freq, 3) if fl else None,
                  "carrier_hz": None, "cone_isolation": None, "frame_rate": 60}
        cm.add_recording(f, label=os.path.basename(f),
                         stimulus={"type": "flicker", "params": params, "source": "ttl"},
                         channels={"signal": "Im_prime", "ttl": "TTL"},
                         fs=rec.fs, duration_s=round(rec.duration, 2))
    cm.save()
    print(f"  2026-06-02/{cm.data['cell']} ({label}): {len(cm.data['recordings'])} recordings")


print("=== Barak 2026-06-02 ===")
import_flicker_cell("data/ipRGC barak/ipRGC", "ipRGC")
import_flicker_cell("data/ipRGC barak/notipRGC", "notipRGC")

print("\n=== SaraipRGC -> 2017-01-18 / 20170118Bc4 ===")
cm = ds.new_cell("2017-01-18", label="20170118Bc4", cell_type="ipRGC")
DATA_EXT = (".csv", ".mat")
keep = sorted(glob.glob("SaraipRGC/*.csv") + glob.glob("SaraipRGC/*.mat")
              + glob.glob("SaraipRGC/*.m") + glob.glob("SaraipRGC/*.pdf"))
for f in keep:
    ext = os.path.splitext(f)[1].lower()
    base = os.path.basename(f)
    if ext in DATA_EXT:
        stim = None
        if "siso-spikes" in base or "siso-stdev" in base:
            stim = {"type": "gaussian_noise",
                    "params": {"cone_isolation": "S", "stdev": 0.3,
                               "frame_rate": 60, "bins_per_frame": 6}, "source": "file"}
        cm.add_recording(f, label=base, kind="recording", stimulus=stim)
    else:                                           # .m loader / .pdf docs -> references
        cm.add_recording(f, label=base, kind="reference")
cm.save()
print(f"  2017-01-18/{cm.data['cell']} (20170118Bc4): {len(cm.data['recordings'])} files "
      f"({sum(r['kind']=='recording' for r in cm.data['recordings'])} data, "
      f"{sum(r['kind']=='reference' for r in cm.data['recordings'])} reference)")

idx = ds.update_index()
print("\nindex:")
for c in idx:
    print(f"  {c['date']}/{c['cell']}  label={c['label']!r}  recordings={c['n_recordings']}")
