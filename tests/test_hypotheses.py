"""Hypothesis platform (§9): registration, updating, priority, resolution and isolation."""
import pytest

from hypotheses.schema import HypothesisTest
from hypotheses.store import HypothesisStore, disagreement_hypothesis


@pytest.fixture
def store(tmp_path) -> HypothesisStore:
    return HypothesisStore("prog-test", "parkers", runs_dir=tmp_path)


def _test_obj() -> HypothesisTest:
    return HypothesisTest(kind="observe_rounds", resolves_if="if they concede twice in a row")


def test_register_and_persist(store, tmp_path):
    h = store.register(party="parkers", statement="The counterpart is in a hurry",
                       type="behavior", test=_test_obj(), prior_belief=0.6)
    store.save()
    reloaded = HypothesisStore("prog-test", "parkers", runs_dir=tmp_path)
    assert [x.hypothesis_id for x in reloaded.all()] == [h.hypothesis_id]
    assert reloaded.all()[0].prior_belief == 0.6


def test_a_party_cannot_register_the_other_partys_hypotheses(store):
    with pytest.raises(PermissionError):
        store.register(party="gibsons", statement="…", test=_test_obj())


def test_updating_moves_the_belief_and_leaves_a_trail(store):
    h = store.register(party="parkers", statement="rv < 0.5", variable="theta.rv",
                       test=_test_obj())
    store.update(h.hypothesis_id, observation="accepted the probe", confirmed=True, at="round:6")
    updated = store.all()[0]
    assert updated.current_belief == h.test.p_true_if_confirmed
    assert updated.updates[0]["at"] == "round:6"


def test_resolution_and_brier(store):
    h = store.register(party="parkers", statement="rv < 0.5", variable="theta.rv",
                       test=_test_obj(), prior_belief=0.5)
    store.update(h.hypothesis_id, observation="…", confirmed=True, at="round:6")
    store.resolve(h.hypothesis_id, truth_value=True, evidence="true rv 0.41")
    resolved = store.all()[0]
    assert resolved.status == "supported"
    assert resolved.brier == pytest.approx((0.9 - 1.0) ** 2)
    assert store.scores()["resolved_fraction"] == 1.0


def test_priority_by_value_of_information(store):
    uncertain = store.register(party="parkers", statement="a", variable="theta.rv",
                               test=_test_obj(), prior_belief=0.5)
    almost_sure = store.register(party="parkers", statement="b", variable="theta.rv",
                                 test=_test_obj(), prior_belief=0.97)
    order = [h.hypothesis_id for h in store.by_priority()]
    assert order.index(uncertain.hypothesis_id) < order.index(almost_sure.hypothesis_id)


def test_evpi_weighs_in_the_priority(store):
    cheap = store.register(party="parkers", statement="a", variable="theta.beta",
                           test=_test_obj(), prior_belief=0.5)
    valuable = store.register(party="parkers", statement="b", variable="theta.rv",
                              test=_test_obj(), prior_belief=0.5)
    order = [h.hypothesis_id for h in store.by_priority(evpi={"theta.rv": 10.0,
                                                              "theta.beta": 0.01})]
    assert order[0] == valuable.hypothesis_id and order[-1] == cheap.hypothesis_id


def test_fox_disagreement_generates_a_hypothesis_with_a_probe(store):
    h = disagreement_hypothesis(store, party="parkers", variable="theta.rv",
                                fox_a="f01_bayes_rv_concession", median_a=0.30,
                                fox_b="f03_time_concession_regression", median_b=0.55,
                                sd_gap=1.6, at="round:4", probe_utility=0.42)
    assert h.test.kind == "probe_offer"
    assert h.interval == (0.30, 0.55)
    assert "f01_bayes_rv_concession" in h.evidence[0]
    assert "0.420" in h.test.resolves_if


# ------------------------------------------------------------------ resolution against the truth
def test_a_structured_threshold_resolves_the_hypothesis(tmp_path):
    from feedback.metrics import resolve_hypotheses
    from gym.case import CaseBundle
    from hypotheses.schema import Threshold

    bundle = CaseBundle.load("parker_gibson")
    store = HypothesisStore("prog-res", "parkers", runs_dir=tmp_path)
    store.register(party="parkers", variable="theta.rv",
                   statement="The Gibsons' reservation value exceeds 30,000",
                   threshold=Threshold(comparison="gte", values=[30000], unit="USD"),
                   test=_test_obj(), prior_belief=0.5)
    store.save()
    summary = resolve_hypotheses(bundle, "prog-res", runs_dir=tmp_path)
    reloaded = HypothesisStore("prog-res", "parkers", runs_dir=tmp_path).all()[0]
    assert reloaded.status == "supported", "the gibsons' true rv exceeds 30,000"
    assert reloaded.brier == pytest.approx(0.25)
    assert summary["parkers"]["n_resolved"] == 1


def test_the_threshold_is_extracted_from_prose_in_both_orders():
    from hypotheses.thresholds import threshold_from_prose

    assert threshold_from_prose("is at least 15,500 USD") == ("gte", 15500.0)
    assert threshold_from_prose("is 20,000 or less") == ("lte", 20000.0)
    assert threshold_from_prose("above 30 000") == ("gte", 30000.0)
    assert threshold_from_prose("does not exceed 25,000") == ("lte", 25000.0)
    # the parser also reads the Spanish wording of older recorded programs
    assert threshold_from_prose("es al menos 15 500 USD") == ("gte", 15500.0)
    assert threshold_from_prose("es 20 000 o menos") == ("lte", 20000.0)


def test_without_a_bound_no_criterion_is_invented():
    from hypotheses.thresholds import threshold_from_prose
    assert threshold_from_prose("they are in a hurry to close") is None
    assert threshold_from_prose("they concede by reciprocity") is None


def test_an_unresolved_hypothesis_can_be_retried(tmp_path):
    """`unresolved` is not a verdict: if the feedback improves it must be able to score it."""
    from feedback.metrics import resolve_hypotheses
    from gym.case import CaseBundle

    bundle = CaseBundle.load("parker_gibson")
    store = HypothesisStore("prog-retry", "gibsons", runs_dir=tmp_path)
    h = store.register(party="gibsons", variable="theta.rv",
                       statement="The Parkers' reservation value is 20,000 or less",
                       test=_test_obj())
    store.mark_unresolved(h.hypothesis_id, "first pass without a criterion")
    store.save()
    resolve_hypotheses(bundle, "prog-retry", runs_dir=tmp_path)
    reloaded = HypothesisStore("prog-retry", "gibsons", runs_dir=tmp_path).all()[0]
    assert reloaded.status == "supported", "the parkers' true rv is <= 20,000"


def test_a_behavioural_hypothesis_stays_unresolved_and_says_why(tmp_path):
    from feedback.metrics import resolve_hypotheses
    from gym.case import CaseBundle

    bundle = CaseBundle.load("parker_gibson")
    store = HypothesisStore("prog-beh", "parkers", runs_dir=tmp_path)
    store.register(party="parkers", type="behavior", variable="behavior",
                   statement="They are in a hurry to close", test=_test_obj())
    store.save()
    resolve_hypotheses(bundle, "prog-beh", runs_dir=tmp_path)
    reloaded = HypothesisStore("prog-beh", "parkers", runs_dir=tmp_path).all()[0]
    assert reloaded.status == "unresolved"
    assert "does not fix" in reloaded.resolution.evidence
