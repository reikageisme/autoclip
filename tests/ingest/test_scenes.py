from __future__ import annotations

from pathlib import Path

import pytest

from autoclip.ingest.scenes import detect_scenes, merge_short_scenes


def test_merge_keeps_long_scenes() -> None:
    scenes = [(0.0, 1.5), (1.5, 3.0)]
    assert merge_short_scenes(scenes, min_scene_sec=0.8) == [(0.0, 1.5), (1.5, 3.0)]


def test_merge_folds_short_scene_into_previous() -> None:
    scenes = [(0.0, 1.0), (1.0, 1.4), (1.4, 3.0)]
    assert merge_short_scenes(scenes, min_scene_sec=0.8) == [(0.0, 1.4), (1.4, 3.0)]


def test_merge_folds_short_first_scene_into_next() -> None:
    scenes = [(0.0, 0.3), (0.3, 2.0)]
    assert merge_short_scenes(scenes, min_scene_sec=0.8) == [(0.0, 2.0)]


def test_merge_single_short_scene_is_kept() -> None:
    assert merge_short_scenes([(0.0, 0.4)], min_scene_sec=0.8) == [(0.0, 0.4)]


def test_merge_empty_falls_back_to_duration() -> None:
    assert merge_short_scenes([], min_scene_sec=0.8, duration=4.0) == [(0.0, 4.0)]
    assert merge_short_scenes([], min_scene_sec=0.8) == []


def test_merge_sorts_and_clamps_to_duration() -> None:
    scenes = [(1.0, 2.5), (0.0, 1.0)]
    assert merge_short_scenes(scenes, min_scene_sec=0.8, duration=2.0) == [
        (0.0, 1.0),
        (1.0, 2.0),
    ]


def test_merge_coalesces_overlaps() -> None:
    scenes = [(0.0, 1.0), (0.9, 2.0)]
    assert merge_short_scenes(scenes, min_scene_sec=0.8) == [(0.0, 2.0)]


def test_detect_scenes_on_color_cuts(media_files: dict[str, Path]) -> None:
    pytest.importorskip("scenedetect")
    scenes = detect_scenes(media_files["scenes"], min_scene_sec=0.8, threshold=20.0)
    assert len(scenes) >= 2
    assert scenes[0][0] == pytest.approx(0.0, abs=0.2)
    assert scenes[-1][1] == pytest.approx(3.0, abs=0.3)
    for start, end in scenes:
        assert end > start


def test_split_long_scenes() -> None:
    from autoclip.ingest.scenes import split_long_scenes

    assert split_long_scenes([(0.0, 4.0), (4.0, 17.0)], 6.0) == [
        (0.0, 4.0),
        (4.0, 8.333),
        (8.333, 12.667),
        (12.667, 17.0),
    ]
    assert split_long_scenes([(0.0, 12.0)], 6.0) == [(0.0, 6.0), (6.0, 12.0)]
    assert split_long_scenes([(0.0, 20.0)], 0) == [(0.0, 20.0)]
