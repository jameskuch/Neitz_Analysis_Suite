from .abf import Recording
from .csv import CsvSpikeRecording


def load_recording(path):
    """Load an .abf as a Recording, or a spike .csv as a CsvSpikeRecording."""
    if str(path).lower().endswith(".csv"):
        return CsvSpikeRecording.load(path)
    return Recording.load(path)


__all__ = ["Recording", "CsvSpikeRecording", "load_recording"]
