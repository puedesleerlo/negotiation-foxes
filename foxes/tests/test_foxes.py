"""Minimal tests of every fox. These are the ones section 8 of each SKILL.md cites."""
import numpy as np
import pytest

from foxes.base import Observation, Posterior, counterpart_offers
from foxes.domain import Utility
from foxes.tests.conftest import conceding_observations


# ------------------------------------------------------------------ helpers
def _state_with(fox, domain, observations):
    state = fox.init_state(domain)
    state.data["counterpart_party"] = "cp"
    for obs in observations:
        state = fox.update(state, obs)
    return state


# ------------------------------------------------------------------ base
def test_counterpart_offers_excludes_responses(domain3, utility3):
    """A reject/accept carries *our* offer with the counterpart as `party`; it is not a proposal."""
    from foxes.f06_uninformed_prior import UninformedPrior
    fox = UninformedPrior()
    proposals = conceding_observations(utility3, 0.4, n=3)
    response = Observation(round=4, party="cp", offer=proposals[0].offer, action="reject",
                           est_utility_to_proposer=0.9, max_rounds=20)
    state = _state_with(fox, domain3, proposals + [response])
    assert len(counterpart_offers(state)) == 3


# ------------------------------------------------------------------ f01
def test_f01_recovers_a_known_rv(domain3, utility3):
    from foxes.f01_bayes_rv_concession import BayesRvConcession
    fox = BayesRvConcession()
    rv_true = 0.35
    obs = conceding_observations(utility3, rv_true, n=18, max_rounds=20, e=1.0)
    post = fox.posterior(_state_with(fox, domain3, obs))
    lo, hi = post.interval("rv", 0.9)
    assert lo <= rv_true <= hi, f"true rv {rv_true} outside [{lo:.3f}, {hi:.3f}]"
    assert post.scope_ok


def test_f01_declares_out_of_scope_without_concession(domain3, utility3):
    from foxes.f01_bayes_rv_concession import BayesRvConcession
    fox = BayesRvConcession()
    space = domain3.outcome_space()
    best = max(space, key=utility3)
    obs = [Observation(round=k, party="cp", offer=best, action="propose",
                       est_utility_to_proposer=float(utility3(best)), max_rounds=20)
           for k in range(1, 9)]
    post = fox.posterior(_state_with(fox, domain3, obs))
    assert not post.scope_ok
    assert any("concession" in w for w in post.warnings)


def test_f01_puts_no_mass_above_the_lowest_offer(domain3, utility3):
    from foxes.f01_bayes_rv_concession import BayesRvConcession
    fox = BayesRvConcession()
    obs = conceding_observations(utility3, 0.4, n=12, max_rounds=20)
    post = fox.posterior(_state_with(fox, domain3, obs))
    lowest = min(o.est_utility_to_proposer for o in obs)
    mass_above = float(post.weights[post.column("rv") > lowest + 1e-9].sum())
    assert mass_above < 1e-9


# ------------------------------------------------------------------ f02 and f09
@pytest.mark.parametrize("fox_name", ["f02", "f09"])
def test_weights_rank_the_most_conceded_issue_last(domain3, fox_name):
    """The counterpart gives everything on `employment` and nothing on `price`: price must weigh more."""
    if fox_name == "f02":
        from foxes.f02_concession_issue_weights import ConcessionIssueWeights as Fox
    else:
        from foxes.f09_hypothesis_issue_weights import HypothesisIssueWeights as Fox
    fox = Fox()
    offers = [{"price": 1.0, "closing": "90d", "employment": "2yr"},
              {"price": 1.0, "closing": "90d", "employment": "1yr"},
              {"price": 1.0, "closing": "60d", "employment": "none"},
              {"price": 0.9, "closing": "60d", "employment": "none"}]
    obs = [Observation(round=k + 1, party="cp", offer=o, action="propose", max_rounds=20)
           for k, o in enumerate(offers)]
    post = fox.posterior(_state_with(fox, domain3, obs))
    w_price = post.mean("w.price")
    w_emp = post.mean("w.employment")
    assert w_price > w_emp, f"price {w_price:.3f} should weigh more than employment {w_emp:.3f}"


def test_f02_is_out_of_scope_with_a_single_issue():
    from foxes.domain import Domain, Issue
    from foxes.f02_concession_issue_weights import ConcessionIssueWeights
    fox = ConcessionIssueWeights()
    domain = Domain("one", (Issue("price", "continuous", low=0, high=1, steps=5),))
    post = fox.posterior(_state_with(fox, domain, []))
    assert not post.scope_ok


# ------------------------------------------------------------------ f03
def test_f03_recovers_parameters_of_a_known_curve(domain3, utility3):
    from foxes.f03_time_concession_regression import TimeConcessionRegression
    fox = TimeConcessionRegression(n_bootstrap=40)
    rv_true, e_true, deadline = 0.30, 1.0, 20
    obs = conceding_observations(utility3, rv_true, n=16, max_rounds=deadline, e=e_true)
    post = fox.posterior(_state_with(fox, domain3, obs))
    lo, hi = post.interval("rv", 0.9)
    assert lo - 0.15 <= rv_true <= hi + 0.15
    assert post.scope_ok


# ------------------------------------------------------------------ f04
def test_f04_p_accept_is_monotone_in_utility(domain3):
    from foxes.f04_accept_boundary_kde import AcceptBoundaryKde
    fox = AcceptBoundaryKde()
    state = fox.init_state(domain3)
    for u, acc in [(0.2, False), (0.35, False), (0.6, True), (0.75, True), (0.5, False)]:
        fox.observe_response(state, u, acc, t=0.4)
    probs = [fox.p_accept(state, {}, utility_to_counterpart=u, t=0.4).value
             for u in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert all(b >= a - 1e-9 for a, b in zip(probs, probs[1:])), probs
    est = fox.p_accept(state, {}, utility_to_counterpart=0.5, t=0.4)
    assert est.low <= est.value <= est.high


def test_f04_receives_responses_through_update(domain3):
    from foxes.f04_accept_boundary_kde import AcceptBoundaryKde
    fox = AcceptBoundaryKde()
    state = fox.init_state(domain3)
    for k, (u, act) in enumerate([(0.3, "reject"), (0.5, "reject"), (0.8, "accept")], 1):
        state = fox.update(state, Observation(round=k, party="cp", offer={"price": 0.5},
                                              action=act, est_utility_to_proposer=u,
                                              max_rounds=20))
    assert len(state.data["responses"]) == 3
    assert state.data["responses"][-1]["accepted"] is True


# ------------------------------------------------------------------ f06
def test_f06_nominal_coverage(domain3):
    from foxes.f06_uninformed_prior import UninformedPrior
    fox = UninformedPrior(params=["rv"], n=4000)
    post = fox.posterior(fox.init_state(domain3))
    rng = np.random.default_rng(0)
    truths = rng.uniform(0, 1, 400)
    lo, hi = post.interval("rv", 0.9)
    coverage = float(np.mean((truths >= lo) & (truths <= hi)))
    assert 0.82 <= coverage <= 0.98, coverage


def test_f06_is_idempotent_on_the_same_state(domain3):
    """Two calls on the same state return the same samples: the entropy recorded in a fox_call
    is the entropy of what entered the pool."""
    from foxes.f06_uninformed_prior import UninformedPrior
    fox = UninformedPrior(params=["rv"], n=500)
    state = fox.init_state(domain3)
    a, b = fox.posterior(state), fox.posterior(state)
    assert np.array_equal(a.samples, b.samples)
    state.round = 3
    c = fox.posterior(state)
    assert not np.array_equal(a.samples, c.samples), "a different round draws afresh"


# ------------------------------------------------------------------ f07
def test_f07_with_exact_theta_reproduces_the_zopa(domain3, utility3):
    from foxes.f07_zopa_pareto_estimator import ZopaParetoEstimator
    from foxes.f07_zopa_pareto_estimator.fox import utility_from_theta
    fox = ZopaParetoEstimator(n_draws=5)
    weights = np.array([0.2, 0.3, 0.5])
    directions = np.array([-1.0, -1.0, -1.0])
    other = utility_from_theta(domain3, weights, directions, 0.25)
    state = fox.init_state(domain3, utility3)
    state.data["theta_samples"] = {"weights": np.tile(weights, (5, 1)),
                                   "directions": np.tile(directions, (5, 1)),
                                   "rv": np.full(5, 0.25)}
    post = fox.posterior(state)
    space = domain3.outcome_space()
    true_fraction = float(np.mean([(utility3(o) >= utility3.reservation_value)
                                   and (other(o) >= other.reservation_value) for o in space]))
    assert abs(post.mean("zopa_fraction") - true_fraction) < 0.02


# ------------------------------------------------------------------ f08
def test_f08_returns_an_outcome_distribution(domain3, utility3):
    from foxes.registry import FoxRegistry
    reg = FoxRegistry()
    fox = reg.create("f08_outcome_simulator", allow_experimental=True, n_draws=12)
    rng = np.random.default_rng(0)
    state = fox.init_state(domain3, utility3)
    state.data.update(
        theta_samples={"weights": rng.dirichlet(np.ones(3), 12),
                       "directions": rng.choice([-1, 1], (12, 3)),
                       "rv": rng.uniform(0.2, 0.5, 12)},
        strategy={"e": 0.5, "deadline": 20})
    post = fox.posterior(state)
    assert set(post.params) == {"u_own", "u_other", "joint_utility", "agreement", "rounds"}
    assert len(post.samples) == 12
    assert 0.0 <= fox.p_impasse(state) <= 1.0
    assert 0.0 <= post.mean("u_own") <= 1.0


def test_f08_without_a_strategy_is_out_of_scope(domain3, utility3):
    from foxes.registry import FoxRegistry
    fox = FoxRegistry().create("f08_outcome_simulator", allow_experimental=True)
    state = fox.init_state(domain3, utility3)
    post = fox.posterior(state)
    assert not post.scope_ok


# ------------------------------------------------------------------ registry and pool
def test_registry_blocks_candidates_and_unvalidated_foxes():
    from foxes.registry import FoxRegistry
    reg = FoxRegistry()
    with pytest.raises(ValueError, match="candidate"):
        reg.create("f05_case_reader_prior")
    with pytest.raises(ValueError, match="implemented"):
        reg.create("f08_outcome_simulator")
    assert reg.create("f08_outcome_simulator", allow_experimental=True) is not None


def test_registry_records_every_call(domain3, utility3):
    from foxes.registry import FoxRegistry
    reg = FoxRegistry()
    fox = reg.create("f06_uninformed_prior", params=["rv"])
    state = fox.init_state(domain3, utility3)
    state.program_id, state.party = "prog-test", "A"
    reg.call(fox, state, step="preparation")
    assert len(reg.calls) == 1
    call = reg.calls[0]
    assert call.fox_id == "f06_uninformed_prior" and call.program_id == "prog-test"
    assert call.step == "preparation" and call.seconds >= 0.0
    assert "rv" in call.summary and "n_samples" in call.summary


def test_equal_weight_pool_lies_between_its_components():
    from foxes.pooling import disagreement, leave_one_out, linear_pool
    rng = np.random.default_rng(0)
    a = Posterior("f01", "0.1", ["rv"], rng.normal(0.3, 0.05, 2000))
    b = Posterior("f03", "0.1", ["rv"], rng.normal(0.6, 0.05, 2000))
    pooled = linear_pool([a, b], "rv")
    assert 0.3 < pooled.quantile("rv", 0.5) < 0.6
    assert pooled.entropy("rv") > a.entropy("rv")      # the pool is wider than each part
    assert disagreement([a, b], "rv") > 1.0
    assert set(leave_one_out([a, b], "rv")) == {"f01", "f03"}


def test_pool_ignores_out_of_scope_posteriors():
    from foxes.pooling import linear_pool
    rng = np.random.default_rng(0)
    good = Posterior("f01", "0.1", ["rv"], rng.normal(0.3, 0.05, 500))
    bad = Posterior("f03", "0.1", ["rv"], rng.normal(0.9, 0.01, 500), scope_ok=False)
    pooled = linear_pool([good, bad], "rv")
    assert pooled.quantile("rv", 0.95) < 0.5
