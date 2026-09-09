"""The `econ` personality: the decision rule and the effect of tau and lambda (§7.2)."""
import numpy as np
import pytest

from agents.personality import Personality
from foxes.domain import Domain, Issue, Utility


@pytest.fixture
def setup():
    domain = Domain("price_only", (Issue("price", "continuous", low=5000, high=60000, steps=56),))
    own = Utility(domain, {"price": 1.0}, {"price": (5000.0, 60000.0)})
    own.reservation_value = 0.18
    rng = np.random.default_rng(0)
    n = 2500
    theta = {"rv": rng.uniform(0.05, 0.75, n), "beta": rng.uniform(0.2, 3.0, n),
             "T": rng.uniform(10, 30, n)}
    return domain, own, theta


def test_econ_loads_its_parameters():
    econ = Personality.load("econ")
    assert econ.personality_id == "econ"
    assert 0.0 <= econ.tau <= 1.0 and econ.lambda_info >= 0.0


def test_optimism_makes_the_agent_ask_for_more(setup):
    """RQ2's central lever: without optimism the agent drifts towards its own reserve."""
    domain, own, theta = setup
    econ = Personality.load("econ")
    neutral = econ.with_overrides(optimism_quantile_tau=0.5, info_gain_weight_lambda=0.0)
    optimistic = econ.with_overrides(optimism_quantile_tau=0.75, info_gain_weight_lambda=0.0)
    offers = domain.outcome_space()
    best_neutral = neutral.score_offers(offers, own, 5, theta)[0]
    best_optimistic = optimistic.score_offers(offers, own, 5, theta)[0]
    assert best_optimistic.u_own > best_neutral.u_own


def test_lambda_shifts_the_choice_towards_informative_offers(setup):
    domain, own, theta = setup
    econ = Personality.load("econ")
    no_info = econ.with_overrides(info_gain_weight_lambda=0.0)
    with_info = econ.with_overrides(info_gain_weight_lambda=2.0)
    offers = domain.outcome_space()
    a = no_info.score_offers(offers, own, 5, theta)[0]
    b = with_info.score_offers(offers, own, 5, theta)[0]
    assert b.eig >= a.eig


def test_optimistic_theta_leaves_the_belief_alone_with_neutral_tau(setup):
    _, _, theta = setup
    neutral = Personality.load("econ").with_overrides(optimism_quantile_tau=0.5)
    assert neutral.optimistic_theta(theta) is theta


def test_never_accepts_below_its_own_reservation_value(setup):
    domain, own, theta = setup
    econ = Personality.load("econ")
    cands = econ.score_offers(domain.outcome_space(), own, 5, theta)
    accept, detail = econ.should_accept(own.reservation_value - 0.05, cands, own)
    assert not accept and detail["rv"] == own.reservation_value


def test_walks_away_if_nothing_reachable_beats_its_reserve(setup):
    domain, own, theta = setup
    econ = Personality.load("econ")
    demanding = Utility(domain, {"price": 1.0}, {"price": (5000.0, 60000.0)})
    demanding.reservation_value = 0.98          # would only accept almost the maximum
    cands = econ.score_offers(domain.outcome_space(), demanding, 5, theta)
    walk, _ = econ.should_walk_away(cands, demanding)
    assert walk


def test_block_concession_runs_from_aspiration_to_reserve(setup):
    econ = Personality.load("econ")
    targets = [econ.concession_target(r, 20, aspiration=0.9, reservation=0.2)
               for r in (1, 8, 15, 20)]
    assert targets == sorted(targets, reverse=True)
    assert targets[0] <= 0.9 and targets[-1] == pytest.approx(0.2)
