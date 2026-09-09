import numpy as np
import pytest

from foxes.base import Observation
from foxes.domain import Domain, Issue, Utility, target_utility


@pytest.fixture
def domain3() -> Domain:
    return Domain("test3", (
        Issue("price", "continuous", low=0.0, high=1.0, steps=11),
        Issue("closing", "discrete", values=("30d", "60d", "90d")),
        Issue("employment", "discrete", values=("none", "1yr", "2yr")),
    ))


@pytest.fixture
def utility3(domain3) -> Utility:
    return Utility(domain3, {"price": 0.6, "closing": 0.3, "employment": 0.1},
                   {"price": (0.0, 1.0),
                    "closing": {"30d": 0.0, "60d": 0.5, "90d": 1.0},
                    "employment": {"none": 0.0, "1yr": 0.5, "2yr": 1.0}},
                   reservation_value=0.3)


def conceding_observations(utility: Utility, rv: float, n: int = 10,
                           max_rounds: int = 20, e: float = 1.0) -> list[Observation]:
    """A series of offers by a counterpart that concedes from 1.0 towards `rv`."""
    space = utility.domain.outcome_space()
    values = np.array([utility(o) for o in space])
    obs = []
    for k in range(1, n + 1):
        target = target_utility(k / max_rounds, e, p_min=rv, p_max=1.0)
        idx = int(np.argmin(np.abs(values - target)))
        obs.append(Observation(round=k, party="cp", offer=space[idx], action="propose",
                               est_utility_to_proposer=float(values[idx]),
                               max_rounds=max_rounds))
    return obs
