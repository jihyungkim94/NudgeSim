import random

import pytest

from nudgesim.agents.persona import DEFAULT_SOCIETY
from nudgesim.data import ClaimPool, MotifLibrary


@pytest.fixture(scope="session")
def pool() -> ClaimPool:
    return ClaimPool.synthetic(seed=11)


@pytest.fixture(scope="session")
def library() -> MotifLibrary:
    return MotifLibrary.synthetic(seed=11).with_min_reach(1, DEFAULT_SOCIETY.all_ids)


@pytest.fixture
def rng() -> random.Random:
    return random.Random(0)
