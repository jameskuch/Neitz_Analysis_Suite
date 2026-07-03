from .abf import Recording
from .csv import CsvSpikeRecording
from .stim import (load_session_manifest, find_session_manifest, stimulus_metadata,
                   pair_by_order, noise_from_record, sent_codes_from_record,
                   apply_session_manifest, epoch_groups)


def load_recording(path):
    """Load an .abf as a Recording, or a spike .csv as a CsvSpikeRecording."""
    if str(path).lower().endswith(".csv"):
        return CsvSpikeRecording.load(path)
    return Recording.load(path)


__all__ = ["Recording", "CsvSpikeRecording", "load_recording",
           "load_session_manifest", "find_session_manifest", "stimulus_metadata",
           "pair_by_order", "noise_from_record", "sent_codes_from_record",
           "apply_session_manifest", "epoch_groups"]
