"""Tests for business costs and threshold selection."""

from latedeliveries.evaluation import (
    THRESHOLDS,
    choose_threshold,
    order_cost,
)


def test_cost_of_hand_calculated_example() -> None:
    y_true = [1, 0, 1, 1, 0, 0]
    flagged = [True, True, False, False, False, True]

    # Three vouchers cost $12 and two missed late orders cost $30.
    assert order_cost(y_true, flagged) == 42


def test_threshold_can_choose_flag_nobody() -> None:
    y_true = [0, 0, 0]
    probabilities = [0.2, 0.4, 0.6]

    threshold, _ = choose_threshold(
        y_true,
        probabilities,
        thresholds=[0.05, 0.95],
    )

    assert threshold == 0.95


def test_threshold_can_choose_flag_everybody() -> None:
    y_true = [1, 1, 1]
    probabilities = [0.2, 0.4, 0.6]

    threshold, _ = choose_threshold(
        y_true,
        probabilities,
        thresholds=[0.05, 0.95],
    )

    assert threshold == 0.05


def test_default_threshold_range() -> None:
    assert len(THRESHOLDS) == 19
    assert THRESHOLDS[0] == 0.05
    assert THRESHOLDS[-1] == 0.95
