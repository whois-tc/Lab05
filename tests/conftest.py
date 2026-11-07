"""Shared pytest fixtures."""

import pandas as pd
import pytest

from latedeliveries.data import make_orders


@pytest.fixture
def orders() -> pd.DataFrame:
    """Return a small repeatable order table for tests."""
    return make_orders(n=400, seed=7)
