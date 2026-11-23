"""Tests for celldform.demo.source (stages 1-2 of the demo)."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from celldform.demo.source import Recording, read_frame_log, read_notes, remove_overlay, to_gray

REAL = Path(__file__).resolve().parent.parent / "data/test/Test_1_2026_11_12_12_32_PM"


def _video(path: Path, n=6, size=(160, 120), colour=True):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 22.0, size, colour)
    for i in range(n):
        f = np.full((size[1], size[0], 3), (150, 190, 215), np.uint8)
        cv2.circle(f, (80, 60), 15, (60, 60, 60), -1)
        w.write(f if colour else f[..., 0])
    w.release()


def test_overlay_text_is_removed_background_kept():
    f = np.full((120, 160, 3), (150, 190, 215), np.uint8)        # warm background (BGR)
    cv2.putText(f, "Current: 322 mA", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    clean = remove_overlay(f)
    assert (clean[:30, :100].astype(int) - np.array([150, 190, 215])).__abs__().max() < 40
    assert np.array_equal(clean[60:, :], f[60:, :])              # untouched away from the text


def test_gray_frame_passes_through_overlay_removal():
    g = np.random.default_rng(0).integers(0, 255, (50, 60), np.uint8)
    assert remove_overlay(g) is g


def test_to_gray_size_and_dtype():
    out = to_gray(np.zeros((120, 160, 3), np.uint8), (80, 60))
    assert out.shape == (60, 80) and out.dtype == np.uint8


def test_notes_and_empty_log(tmp_path):
    (tmp_path / "SKBR3.txt").write_text("Cell label: SKBR3\nStarted: 2026-11-12 12:32:52 PM\n")
    (tmp_path / "frames.csv").write_text("")
    assert read_notes(tmp_path / "SKBR3.txt")["Cell label"] == "SKBR3"
    assert read_notes(tmp_path / "missing.txt") == {}
    assert read_frame_log(tmp_path / "frames.csv") is None       # empty = failed copy, not a log
    assert read_frame_log(tmp_path / "nope.csv") is None


def test_recording_folder_streams_gray_frames(tmp_path):
    _video(tmp_path / "video.avi")
    (tmp_path / "SKBR3.txt").write_text("Cell label: SKBR3\n")
    rec = Recording(tmp_path, size=(80, 60))
    assert rec.cell_label == "SKBR3"
    got = list(rec.frames())
    assert len(got) == 6 and got[0][2].shape == (60, 80) and got[0][2].ndim == 2
    assert got[3][0] == 3 and got[3][1] == pytest.approx(3 / 22.0)
    assert [i for i, _, _ in rec.frames(every_n=2)] == [0, 2, 4]
    assert rec.complete()


def test_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Recording(tmp_path)
    with pytest.raises(FileNotFoundError):
        Recording(tmp_path / "x.avi")


def test_truncated_video_stops_cleanly(tmp_path):
    _video(tmp_path / "video.avi", n=40)
    p = tmp_path / "video.avi"
    p.write_bytes(p.read_bytes()[: p.stat().st_size // 2])
    rec = Recording(p, size=(80, 60))
    n = rec.usable_frames()
    assert 0 < n < 40


@pytest.mark.skipif(not (REAL / "video.avi").exists(), reason="new test recording not present")
def test_real_recording_whatever_has_arrived():
    """The recording may still be copying; only properties that hold at any copy stage are asserted."""
    rec = Recording(REAL)
    assert rec.cell_label == "SKBR3" and rec.declared_frames > 10000
    i, t, f = next(rec.frames())
    assert f.shape == (922, 1228) and f.dtype == np.uint8
    cap = cv2.VideoCapture(str(rec.video)); _, raw = cap.read(); cap.release()
    clean = remove_overlay(raw)
    def yellow(im):
        b, g, r = (im[:200, :600, k].astype(int) for k in range(3))
        return ((r > 200) & (g > 200) & (b < 90)).sum()
    assert yellow(clean) < 0.1 * (yellow(raw) + 1)


def test_select_predicate_picks_indices(tmp_path):
    _video(tmp_path / "video.avi", n=10)
    rec = Recording(tmp_path, size=(80, 60))
    assert [i for i, _, _ in rec.frames(select=lambda i: i in (1, 4, 8))] == [1, 4, 8]
    assert [i for i, _, _ in rec.frames(select=lambda i: True, stop=3)] == [0, 1, 2]


# ── still images as the source ───────────────────────────────────────────────────────────────────────────────────

def _stills(folder: Path, frames, size=(160, 120), table=True):
    """A recording folder with a video and a ``stills_1GB`` folder holding the stills of the given 1-based video frames."""
    import pandas as pd
    folder.mkdir(exist_ok=True)
    _video(folder / "video.avi", n=6, size=size)
    d = folder / "stills_1GB"
    d.mkdir()
    names = []
    for k, vf in enumerate(frames, 1):
        n = f"still_{k:06d}_350mA.png"
        cv2.imwrite(str(d / n), np.full((size[1], size[0], 3), 10 * vf, np.uint8))
        names.append(n)
    if table:
        pd.DataFrame({"still_file": names, "video_frame": list(frames)}).to_csv(folder / "stills.csv", index=False)
    return folder


def test_images_use_the_video_frame_index_from_the_table(tmp_path):
    from celldform.demo.source import find_images
    folder = _stills(tmp_path / "r", frames=[1, 2, 4, 5])               # video frame 3 has no still
    _, idx = find_images(folder)
    assert sorted(idx) == [0, 1, 3, 4]                                  # 1-based video_frame -> 0-based index
    rec = Recording(folder, size=(80, 60), source="images")
    got = [(i, int(g.mean())) for i, _, g in rec.frames()]
    assert [i for i, _ in got] == [0, 1, 3, 4] and got[0][1] == 10 and got[2][1] == 40
    assert rec.declared_frames == 5 and len(rec.images) == 4


def test_images_only_read_the_selected_frames_and_respect_stop(tmp_path):
    rec = Recording(_stills(tmp_path / "r", frames=list(range(1, 7))), size=(80, 60), source="images")
    assert [i for i, _, _ in rec.frames(select=lambda i: i in (1, 4))] == [1, 4]
    assert [i for i, _, _ in rec.frames(stop=3)] == [0, 1, 2]
    assert [i for i, _, _ in rec.frames(every_n=2)] == [0, 2, 4]


def test_a_plain_folder_of_images_is_indexed_by_sorted_order(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    for k in (3, 1, 2):
        cv2.imwrite(str(d / f"a{k}.png"), np.full((60, 80, 3), 50 * k, np.uint8))
    rec = Recording(d, size=(80, 60), source="images")
    assert [(i, int(g.mean())) for i, _, g in rec.frames()] == [(0, 50), (1, 100), (2, 150)]


def test_images_source_errors(tmp_path):
    with pytest.raises(ValueError):
        Recording(tmp_path, source="gif")
    empty = tmp_path / "e"; empty.mkdir()
    _video(empty / "video.avi")
    (empty / "stills_0GB").mkdir()
    with pytest.raises(FileNotFoundError, match="no images"):
        Recording(empty, source="images")
    with pytest.raises(FileNotFoundError, match="no images"):                # a video file: its sibling stills are used
        Recording(empty / "video.avi", source="images")


def test_video_source_is_unchanged_and_still_the_default(tmp_path):
    folder = _stills(tmp_path / "r", frames=[1, 2])
    rec = Recording(folder, size=(80, 60))
    assert rec.source == "video" and rec.declared_frames == 6 and len(list(rec.frames())) == 6


# ── choosing the video file or the images yourself ──────────────────────────────────────────────────────────────

def _folder_with_two_videos(tmp_path, notes=True, logs=True):
    import pandas as pd
    d = tmp_path / "many"; d.mkdir()
    _video(d / "a_cell.avi", n=6); _video(d / "b_cell.avi", n=9)
    if notes:
        (d / "b_cell.txt").write_text("Cell label: B cell\n")
    if logs:
        pd.DataFrame({"time_s": np.arange(9) / 22.0, "set_current_mA": np.arange(9) + 400}).to_csv(d / "b_cell.csv", index=False)
    return d


def test_a_chosen_video_file_is_used_not_the_first_one_in_the_folder(tmp_path):
    d = _folder_with_two_videos(tmp_path)
    assert Recording(d, size=(80, 60)).video.name == "a_cell.avi"                     # a folder still takes the first
    rec = Recording(d / "b_cell.avi", size=(80, 60))
    assert rec.video.name == "b_cell.avi" and rec.declared_frames == 9 and rec.name == "b_cell.avi"
    assert rec.cell_label == "B cell" and list(rec.frame_log.set_current_mA)[:2] == [400, 401]     # its own sidecars
    other = Recording(d / "a_cell.avi", size=(80, 60))
    assert other.declared_frames == 6 and other.cell_label == "" and other.frame_log is None       # not b's


def test_generic_sidecars_apply_to_a_video_that_is_alone_in_its_folder(tmp_path):
    import pandas as pd
    d = tmp_path / "one"; d.mkdir()
    _video(d / "movie.avi")
    (d / "SKBR3.txt").write_text("Cell label: SKBR3\n")
    pd.DataFrame({"time_s": [0.0, 0.1], "set_current_mA": [500, 510]}).to_csv(d / "frames.csv", index=False)
    rec = Recording(d / "movie.avi", size=(80, 60))
    assert rec.cell_label == "SKBR3" and len(rec.frame_log) == 2
    other = tmp_path / "x"; other.mkdir()
    d2 = _folder_with_two_videos(other, notes=False, logs=False)
    (d2 / "frames.csv").write_text("time_s,set_current_mA\n0,500\n")             # ambiguous with two videos: not used
    assert Recording(d2 / "a_cell.avi", size=(80, 60)).frame_log is None


def test_a_video_file_outside_any_recording_folder_works_bare(tmp_path):
    _video(tmp_path / "lone.avi")
    rec = Recording(tmp_path / "lone.avi", size=(80, 60))
    assert rec.frame_log is None and rec.cell_label == "" and rec.fps > 0 and len(list(rec.frames())) == 6


def test_a_non_video_file_is_refused(tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    with pytest.raises(ValueError, match="not a video file"):
        Recording(tmp_path / "notes.txt")


def test_hand_picked_images_are_indexed_in_natural_order(tmp_path):
    d = tmp_path / "pics"; d.mkdir()
    files = []
    for name, v in (("img10.png", 100), ("img2.png", 20), ("img1.png", 10)):
        cv2.imwrite(str(d / name), np.full((60, 80, 3), v, np.uint8))
        files.append(d / name)
    rec = Recording(None, image_files=files, size=(80, 60))
    assert rec.source == "images" and rec.name == "3 images from pics" and rec.declared_frames == 3
    assert [(i, int(g.mean())) for i, _, g in rec.frames()] == [(0, 10), (1, 20), (2, 100)]     # img2 before img10
    assert rec.fps == 22.0 and rec.frame_log is None
    from celldform.demo.source import FrameReader
    assert int(FrameReader(rec).read(2).mean()) == 100


def test_hand_picked_stills_keep_their_video_frame_numbers(tmp_path):
    folder = _stills(tmp_path / "r", frames=list(range(1, 7)))
    stills = sorted((folder / "stills_1GB").glob("*.png"))
    pick = [stills[1], stills[4], stills[2]]                                       # frames 2, 5, 3 (1-based)
    rec = Recording(None, image_files=pick, size=(80, 60))
    assert sorted(rec.images) == [1, 2, 4] and rec.declared_frames == 5            # their video frames - 1
    assert [i for i, _, _ in rec.frames()] == [1, 2, 4]


def test_hand_picked_images_errors(tmp_path):
    d = tmp_path / "p"; d.mkdir()
    cv2.imwrite(str(d / "a.png"), np.zeros((6, 8, 3), np.uint8))
    (d / "b.txt").write_text("x")
    with pytest.raises(ValueError, match="not an image file: b.txt"):
        Recording(None, image_files=[d / "a.png", d / "b.txt"])
    with pytest.raises(FileNotFoundError):
        Recording(None, image_files=[d / "nope.png"])
    with pytest.raises(ValueError):
        Recording(None)


def test_from_spec_takes_paths_and_lists(tmp_path):
    d = tmp_path / "p"; d.mkdir()
    cv2.imwrite(str(d / "a.png"), np.zeros((6, 8, 3), np.uint8)); cv2.imwrite(str(d / "b.png"), np.zeros((6, 8, 3), np.uint8))
    assert Recording.from_spec([str(d / "a.png"), str(d / "b.png")], size=(8, 6)).declared_frames == 2
    v = _folder_with_two_videos(tmp_path)
    assert Recording.from_spec(str(v / "b_cell.avi"), size=(80, 60)).declared_frames == 9
    assert Recording.from_spec(v, size=(80, 60)).video.name == "a_cell.avi"
