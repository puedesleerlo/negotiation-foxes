"""Decision tools: properties that must always hold (task 2.8)."""
import numpy as np
import pytest

from foxes.domain import Domain, Issue, Utility
from tools.decision import (counterpart_target, evpi, expected_information_gain,
                            p_accept_from_belief)


@pytest.fixture
def theta() -> dict:
    rng = np.random.default_rng(0)
    n = 3000
    return {"rv": rng.uniform(0.05, 0.75, n), "beta": rng.uniform(0.2, 3.0, n),
            "T": rng.uniform(10, 30, n)}


@pytest.fixture
def price_domain() -> Domain:
    return Domain("price_only", (Issue("price", "continuous", low=5000, high=60000, steps=111),))


def test_p_accept_grows_with_what_is_offered(theta):
    probs = [p_accept_from_belief(u, 5, theta).value for u in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert all(b >= a - 1e-9 for a, b in zip(probs, probs[1:])), probs


def test_p_accept_carries_an_interval_containing_the_value(theta):
    est = p_accept_from_belief(0.5, 5, theta)
    assert est.low <= est.value <= est.high
    assert est.n_effective > 100


def test_the_counterpart_concedes_with_time(theta):
    """The counterpart's target descends towards its rv as the negotiation advances."""
    early = counterpart_target(theta["rv"], theta["beta"], theta["T"], 1).mean()
    late = counterpart_target(theta["rv"], theta["beta"], theta["T"], 25).mean()
    assert late < early
    assert late >= theta["rv"].mean() - 1e-9


def test_eig_is_zero_when_the_answer_is_certain(theta):
    """An offer that is surely accepted (or surely rejected) teaches nothing."""
    assert expected_information_gain(1.0, 5, theta)["eig"] == pytest.approx(0.0, abs=1e-9)
    assert expected_information_gain(0.0, 5, theta)["eig"] == pytest.approx(0.0, abs=1e-9)


def test_eig_peaks_where_the_answer_is_uncertain(theta):
    """The most information lies where p(accept) is intermediate, not at the extremes."""
    grid = np.linspace(0.05, 0.95, 19)
    rows = [(u, expected_information_gain(u, 5, theta)) for u in grid]
    best_u, best = max(rows, key=lambda r: r[1]["eig"])
    assert best["eig"] > 0.05
    assert 0.05 < best["p_accept"] < 0.95, best["p_accept"]


def test_eig_is_never_negative(theta):
    for u in np.linspace(0.0, 1.0, 21):
        assert expected_information_gain(float(u), 5, theta)["eig"] >= 0.0


def test_evpi_is_non_negative_and_zero_without_uncertainty(price_domain):
    own = Utility(price_domain, {"price": 1.0}, {"price": (5000.0, 60000.0)})
    own.reservation_value = 0.18
    offers = price_domain.outcome_space()
    rng = np.random.default_rng(1)
    uncertain = {"rv": rng.uniform(0.05, 0.75, 2000), "beta": rng.uniform(0.2, 3.0, 2000),
                 "T": rng.uniform(10, 30, 2000)}
    assert evpi(offers, own, 5, uncertain)["evpi"] > 0.0

    certain = {"rv": np.full(500, 0.4), "beta": np.full(500, 1.0), "T": np.full(500, 20.0)}
    assert evpi(offers, own, 5, certain)["evpi"] == pytest.approx(0.0, abs=1e-9)
