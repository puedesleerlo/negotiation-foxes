"""Canary test: one party's corpus never reaches the other (§15)."""
import json

import pytest

from gym.case import CaseBundle
from gym.isolation import CorpusLeak, LeakDetector, assert_no_leak, canaries_of, view_contains_only_own

CASE = "parker_gibson"


@pytest.fixture(scope="module")
def bundle() -> CaseBundle:
    return CaseBundle.load(CASE)


def test_every_role_has_a_unique_canary(bundle):
    canaries = canaries_of(bundle)
    assert set(canaries) == set(bundle.role_ids())
    assert len(set(canaries.values())) == len(canaries), "canaries must differ"


def test_each_party_view_contains_only_its_own_corpus(bundle):
    canaries = canaries_of(bundle)
    for role_id in bundle.role_ids():
        view = bundle.party_view(role_id)
        assert view_contains_only_own(view, canaries), f"leak in the view of {role_id}"


def test_the_view_does_not_expose_the_sealed_truth(bundle):
    view = bundle.party_view("parkers")
    text = json.dumps(view.__dict__, default=str)
    # The true reservation values of BOTH parties cannot be in the view (read from the bundle,
    # never written here).
    for role_id in bundle.role_ids():
        rv = bundle.spec["sealed_truth"][role_id]["reservation_value"]["value"]
        assert str(int(rv)) not in text, f"the view exposes the sealed rv of {role_id}"
    assert not hasattr(view, "sealed_truth")


def test_a_contaminated_artifact_fails_hard(bundle):
    canaries = canaries_of(bundle)
    memo = {"round": 3, "rationale": f"leaked this: {canaries['gibsons']}"}
    with pytest.raises(CorpusLeak):
        assert_no_leak(memo, owner_role="parkers", canaries=canaries, context="decision_memo")


def test_a_clean_artifact_passes(bundle):
    canaries = canaries_of(bundle)
    memo = {"round": 3, "rationale": "the counterpart conceded 2,000 dollars",
            "offer": {"price": 32000}}
    assert_no_leak(memo, owner_role="parkers", canaries=canaries)


def test_the_detector_records_every_check(bundle):
    detector = LeakDetector.for_case(bundle)
    detector.check({"text": "no canaries"}, owner_role="parkers", context="prep")
    detector.check({"text": "none here either"}, owner_role="gibsons", context="prep")
    assert detector.summary() == {"checks": 2, "leaks": 0, "roles": ["gibsons", "parkers"]}


def test_the_detector_catches_a_leak_through_a_file_path(bundle):
    """An indirect leak: the artifact carries not the text but the path of the foreign file."""
    detector = LeakDetector.for_case(bundle)
    path = bundle.root / "roles" / "gibsons" / "confidential.md"
    with pytest.raises(CorpusLeak):
        detector.check({"sources": [path]}, owner_role="parkers", context="preparation_memo")
