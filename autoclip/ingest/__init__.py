"""Media ingest: probe, scene detection, transcription, tagging and dedupe."""

from __future__ import annotations

from .pipeline import IngestOptions, ingest_folder
from .probe import ProbeError, probe, sha1_file
from .scenes import detect_scenes, merge_short_scenes
from .transcribe import Transcriber, TranscriptSegment

__all__ = [
    "IngestOptions",
    "ProbeError",
    "Transcriber",
    "TranscriptSegment",
    "detect_scenes",
    "ingest_folder",
    "merge_short_scenes",
    "probe",
    "sha1_file",
]
