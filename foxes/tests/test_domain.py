import numpy as np

from foxes.domain import (kalai_smorodinsky, nash_point, pareto_frontier, target_utility, zopa)


def test_utility_normalises_weights(utility3):
    assert abs(sum(utility3.weights.values()) - 1.0) < 1e-12


def test_utility_extremes(domain3, utility3):
    best = {"price": 1.0, "closing": "90d", "employment": "2yr"}
    worst = {"price": 0.0, "closing": "30d", "employment": "none"}
    assert abs(utility3(best) - 1.0) < 1e-9
    assert abs(utility3(worst)) < 1e-9


def test_boulware_concedes_late_conceder_early():
    # e < 1 (Boulware) withholds utility halfway; e > 1 (Conceder) has already given it up.
    assert target_utility(0.5, 0.2) > 0.9
    assert target_utility(0.5, 3.0) < 0.5
    for e in (0.2, 1.0, 3.0):
        assert abs(target_utility(1.0, e) - 0.0) < 1e-9
        assert abs(target_utility(0.0, e) - 1.0) < 1e-9


def test_pareto_frontier_is_undominated(domain3, utility3):
    from foxes.domain import Utility
    other = Utility(domain3, {"price": 0.2, "closing": 0.3, "employment": 0.5},
                    {"price": (1.0, 0.0),
                     "closing": {"30d": 1.0, "60d": 0.5, "90d": 0.0},
                     "employment": {"none": 1.0, "1yr": 0.5, "2yr": 0.0}})
    space = domain3.outcome_space()
    frontier = pareto_frontier(space, utility3, other)
    pts = np.array([[utility3(o), other(o)] for o in frontier])
    for a, b in pts:
        dominated = np.any((pts[:, 0] >= a) & (pts[:, 1] >= b)
                           & ((pts[:, 0] > a) | (pts[:, 1] > b)))
        assert not dominated


def test_nash_and_ks_lie_on_the_frontier(domain3, utility3):
    from foxes.domain import Utility
    other = Utility(domain3, {"price": 0.5, "closing": 0.25, "employment": 0.25},
                    {"price": (1.0, 0.0),
                     "closing": {"30d": 1.0, "60d": 0.5, "90d": 0.0},
                     "employment": {"none": 1.0, "1yr": 0.5, "2yr": 0.0}},
                    reservation_value=0.2)
    space = domain3.outcome_space()
    frontier = {tuple(sorted(o.items(), key=str)) for o in pareto_frontier(space, utility3, other)}
    nash, _ = nash_point(space, utility3, other)
    ks = kalai_smorodinsky(space, utility3, other)
    assert tuple(sorted(nash.items(), key=str)) in frontier
    assert tuple(sorted(ks.items(), key=str)) in frontier
    assert len(zopa(space, utility3, other)) > 0
