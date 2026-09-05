"""The two-stage tracker, driven by a fake detector."""

from badminton_coach.roi import Box, RoiTracker, RoiTrackerOptions, bounds, crop_window, unproject


def pose_at(cx, cy, h=0.2, w=0.08):
    return {
        "image": [
            [cx - w / 2, cy - h / 2, 0, 1],
            [cx + w / 2, cy - h / 2, 0, 1],
            [cx + w / 2, cy + h / 2, 0, 1],
            [cx - w / 2, cy + h / 2, 0, 1],
        ],
        "world": [[0, 0, 0]],
    }


def make(full=(), crop=None, **kwargs):
    calls = {"full": 0, "crop": 0}

    def detect_full(_frame):
        calls["full"] += 1
        return list(full() if callable(full) else full)

    def detect_crop(_frame, _window):
        calls["crop"] += 1
        return crop() if callable(crop) else crop

    return RoiTracker(detect_full=detect_full, detect_crop=detect_crop, **kwargs), calls


def test_bounds_and_crop_window():
    b = bounds(pose_at(0.5, 0.5, 0.2, 0.1)["image"])
    assert abs(b.cx - 0.5) < 1e-9
    assert abs(b.height - 0.2) < 1e-9
    # In a 16:9 frame the window is wider than tall in normalised units, so that
    # it is square in pixels.
    w = crop_window(b, 0.85, 16 / 9)
    assert w.width * (16 / 9) > w.height * 0.99


def test_crop_window_stays_inside_the_frame():
    w = crop_window(Box(0.0, 0.0, 0.1, 0.1), 2.0, 1.0)
    assert w.x0 >= 0 and w.y0 >= 0 and w.x1 <= 1 and w.y1 <= 1


def test_unproject_maps_back_to_the_full_frame():
    p = unproject([[0.5, 0.5, 0, 1]], Box(0.2, 0.4, 0.6, 0.8))[0]
    assert abs(p[0] - 0.4) < 1e-9
    assert abs(p[1] - 0.6) < 1e-9
    assert p[3] == 1


def test_first_frame_searches_then_crops():
    t, calls = make(full=[pose_at(0.3, 0.5)], crop=pose_at(0.5, 0.5, 0.5, 0.4))
    assert t.step(None)["source"] == "full"
    assert t.step(None)["source"] == "crop"
    assert calls == {"full": 1, "crop": 1}


def test_with_no_history_the_biggest_person_is_chosen():
    t, _ = make(full=[pose_at(0.3, 0.5, 0.15), pose_at(0.8, 0.5, 0.6)])
    assert abs(bounds(t.step(None)["image"]).cx - 0.8) < 1e-6


def test_a_tap_overrides_the_size_preference():
    t, _ = make(full=[pose_at(0.3, 0.5, 0.15), pose_at(0.8, 0.5, 0.6)])
    t.target_hint = (0.3, 0.5)
    assert abs(bounds(t.step(None)["image"]).cx - 0.3) < 1e-6


def test_after_a_loss_the_nearest_person_beats_the_biggest():
    # The bystander case: someone walks between the phone and the court.
    t, _ = make(
        full=[pose_at(0.30, 0.5, 0.20), pose_at(0.34, 0.5, 0.90)],
        crop=None,
        options=RoiTrackerOptions(max_misses=0),
    )
    t.target_hint = (0.30, 0.5)
    t.step(None)
    t.step(None)      # crop fails, the lock is dropped
    assert abs(bounds(t.step(None)["image"]).cx - 0.30) < 1e-6


def test_a_crop_result_that_jumps_too_far_is_rejected():
    t, _ = make(
        full=[pose_at(0.3, 0.5, 0.2)],
        crop=pose_at(0.95, 0.95, 0.08, 0.04),
        options=RoiTrackerOptions(max_jump=0.1),
    )
    t.step(None)
    box = t.box
    assert t.step(None) is None
    assert t.box is box


def test_a_crop_result_of_a_different_size_is_rejected():
    t, _ = make(full=[pose_at(0.5, 0.5, 0.2)], crop=pose_at(0.5, 0.5, 0.99))
    t.step(None)
    t.options.max_size_ratio = 1.2
    assert t.step(None) is None


def test_the_lock_is_dropped_only_after_several_misses():
    t, calls = make(
        full=[pose_at(0.5, 0.5, 0.2)], crop=None,
        options=RoiTrackerOptions(max_misses=3),
    )
    t.step(None)
    assert calls["full"] == 1
    for _ in range(3):
        assert t.step(None) is None
        assert calls["full"] == 1, "gave up too early"
    t.step(None)
    assert calls["full"] == 2, "never went back to a full-frame search"


def test_an_empty_frame_yields_none():
    t, _ = make(full=[])
    assert t.step(None) is None


def test_reset_clears_the_lock():
    t, calls = make(full=[pose_at(0.5, 0.5)], crop=pose_at(0.5, 0.5, 0.5))
    t.step(None)
    assert t.tracking
    t.reset()
    assert not t.tracking
    t.step(None)
    assert calls["full"] == 2
