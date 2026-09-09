"""Integrative tools (t06-t08): declared applicability and properties."""
import numpy as np
import pytest

from foxes.domain import Domain, Issue, Utility
from gym.case import CaseBundle
from tools.integrative import build_contingent, find_logrolling, generate_mesos


@pytest.fixture
def multi() -> tuple[Domain, Utility]:
    domain = Domain("m", (
        Issue("price", "continuous", low=0.0, high=1.0, steps=11),
        Issue("closing", "discrete", values=("30d", "60d", "90d")),
        Issue("emp", "discrete", values=("none", "1yr", "2yr"))))
    util = Utility(domain, {"price": 0.6, "closing": 0.3, "emp": 0.1},
                   {"price": (0.0, 1.0),
                    "closing": {"30d": 0.0, "60d": 0.5, "90d": 1.0},
                    "emp": {"none": 0.0, "1yr": 0.5, "2yr": 1.0}})
    return domain, util


def test_meso_does_not_apply_to_a_single_issue():
    bundle = CaseBundle.load("parker_gibson")
    result = generate_mesos(bundle.domain(), bundle.true_utility("parkers"), 0.4)
    assert not result
    assert "two issues" in result.reason


def test_mesos_are_equivalent_for_oneself(multi):
    domain, util = multi
    meso = generate_mesos(domain, util, target_utility=0.5, k=3, tolerance=0.02)
    assert len(meso.offers) == 3
    assert meso.detail["max_utility_gap"] <= 0.05, "a MESO must be equivalent to whoever makes it"


def test_mesos_differ_from_each_other(multi):
    domain, util = multi
    meso = generate_mesos(domain, util, target_utility=0.5, k=3)
    signatures = {tuple(sorted(o.items(), key=str)) for o in meso.offers}
    assert len(signatures) == 3 and meso.spread > 0.2


def test_logrolling_does_not_apply_to_a_single_issue():
    bundle = CaseBundle.load("parker_gibson")
    result = find_logrolling(bundle.domain(), bundle.true_utility("parkers"),
                             {"price": 30000.0}, np.ones((3, 1)), np.ones((3, 1)))
    assert not result and "logrolling" in result.reason


def test_logrolling_without_a_belief_over_weights_does_not_apply(multi):
    domain, util = multi
    result = find_logrolling(domain, util, {"price": 0.5, "closing": "60d", "emp": "1yr"},
                             np.array([]), np.array([]))
    assert not result and "f02" in result.reason


def test_logrolling_only_proposes_joint_improvements(multi):
    domain, util = multi
    rng = np.random.default_rng(0)
    trades = find_logrolling(domain, util, {"price": 0.5, "closing": "60d", "emp": "1yr"},
                             rng.dirichlet(np.ones(3), 40), np.tile([-1, -1, -1], (40, 1)))
    assert trades, "it should find trades with opposed preferences"
    assert all(t.expected_joint_delta > 0 for t in trades)
    assert trades == sorted(trades, key=lambda t: t.expected_joint_delta, reverse=True)


def test_the_contingent_clause_gives_both_sides_positive_value():
    clause = build_contingent("rezoning", 0.7, 0.3, stake=0.2)
    assert clause
    assert clause.own_expected_value > 0 and clause.counterpart_expected_value > 0
    assert clause.own_expected_value == pytest.approx(0.2 * 0.4 / 2)


def test_no_clause_without_a_difference_in_beliefs():
    result = build_contingent("x", 0.50, 0.48, stake=0.2)
    assert not result and "difference" in result.reason


def test_the_clause_is_symmetric_when_beliefs_swap():
    a = build_contingent("x", 0.75, 0.35, stake=0.2)
    b = build_contingent("x", 0.35, 0.75, stake=0.2)
    assert a.own_expected_value == pytest.approx(b.own_expected_value)
    assert a.payment_if_event == pytest.approx(-b.payment_if_event)
