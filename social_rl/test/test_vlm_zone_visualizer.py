"""Regression tests for the visualization-only VLM zone hold."""

from social_rl.vlm_zone_visualizer import ZoneMarkerHold


def test_confirmed_zone_is_kept_for_three_seconds_after_labels_disappear():
    hold = ZoneMarkerHold(3.0)
    source_message = object()
    source_zones = ['talking-zone']

    hold.update(source_message, source_zones, 1_000_000_000)
    hold.update(object(), [], 2_000_000_000)

    assert hold.active(3_999_999_999) == (source_message, ('talking-zone',))
    assert hold.active(4_000_000_001) == (None, ())


def test_new_confirmed_zone_restarts_the_hold_window():
    hold = ZoneMarkerHold(3.0)
    first_message = object()
    latest_message = object()

    hold.update(first_message, ['first-zone'], 0)
    hold.update(latest_message, ['latest-zone'], 2_000_000_000)

    assert hold.active(4_999_999_999) == (latest_message, ('latest-zone',))
    assert hold.active(5_000_000_001) == (None, ())
