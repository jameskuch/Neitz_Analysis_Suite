from .abf import Recording
from .csv import CsvSpikeRecording
from .stim import (find_manifest, load_manifest, iter_epochs, stimulus_metadata,
                   align_abfs_to_epochs, build_import_plan, epoch_groups,
                   noise_from_record, sent_codes_from_record)


def load_recording(path):
    """Load an .abf as a Recording, or a spike .csv as a CsvSpikeRecording."""
    if str(path).lower().endswith(".csv"):
        return CsvSpikeRecording.load(path)
    return Recording.load(path)


__all__ = ["Recording", "CsvSpikeRecording", "load_recording",
           "find_manifest", "load_manifest", "iter_epochs", "stimulus_metadata",
           "align_abfs_to_epochs", "build_import_plan", "epoch_groups",
           "noise_from_record", "sent_codes_from_record"]
