"""Researchers (§8): typed output, explicit failure and respect for scope."""
from pathlib import Path

import pytest

from agents.declared_utility import derive_from_view
from agents.researcher import Researcher, integrate, run_plan
from gym.case import CaseBundle


@pytest.fixture(scope="module")
def bundle() -> CaseBundle:
    return CaseBundle.load("parker_gibson")


@pytest.fixture
def researcher(bundle, tmp_path) -> Researcher:
    view = bundle.party_view("parkers")
    declared = derive_from_view(view)
    return Researcher(view, declared.to_utility(view), "prog-test", runs_dir=tmp_path)


def test_a_candidate_fox_returns_an_explicit_failure(researcher):
    out = researcher.run_task({"task_id": "t01", "fox_or_tool": "f05_case_reader_prior"})
    assert not out.ok
    assert "candidate" in out.error
    assert out.result == {}, "no result at all is returned when the task cannot be done"


def test_an_unknown_fox_also_fails(researcher):
    out = researcher.run_task({"task_id": "t01", "fox_or_tool": "f99_made_up"})
    assert not out.ok and "catalog" in out.error


def test_the_output_has_the_shape_of_section_8(researcher):
    out = researcher.run_task({"task_id": "t02", "fox_or_tool": "f06_uninformed_prior"})
    assert out.ok
    for field in ("task_id", "fox_id", "version", "result", "diagnostics", "evidence", "cost",
                  "notes"):
        assert field in out.to_dict(), field
    assert "posterior_ref" in out.result and "summary" in out.result
    assert Path(out.result["posterior_ref"]).exists()
    assert out.cost["seconds"] >= 0.0


def test_the_evidence_cites_papers_corpus_and_skill(researcher):
    out = researcher.run_task({"task_id": "t03", "fox_or_tool": "f01_bayes_rv_concession"})
    assert any(e.startswith("paper:") for e in out.evidence)
    assert any(e.startswith("corpus:") for e in out.evidence)
    assert any(e.startswith("skill:") for e in out.evidence)


def test_online_foxes_declare_themselves_out_of_scope_in_preparation(researcher):
    """Without observed offers f01 can say nothing: it must say so, not make it up."""
    out = researcher.run_task({"task_id": "t04", "fox_or_tool": "f01_bayes_rv_concession"})
    assert out.ok
    assert out.diagnostics["scope_ok"] is False
    assert "pool" in out.notes


def test_the_pool_excludes_out_of_scope_and_keeps_a_record(researcher):
    plan = [{"task_id": "t01", "fox_or_tool": "f01_bayes_rv_concession"},
            {"task_id": "t02", "fox_or_tool": "f06_uninformed_prior"},
            {"task_id": "t03", "fox_or_tool": "f05_case_reader_prior"}]
    outputs = run_plan(plan, researcher)
    result = integrate(outputs)
    assert result["foxes"] == ["f06_uninformed_prior"]
    reasons = {d["fox_id"]: d["why"] for d in result["discarded"]}
    assert "f01_bayes_rv_concession" in reasons and "scope" in reasons["f01_bayes_rv_concession"]
    assert "f05_case_reader_prior" in reasons


def test_the_researcher_never_sees_the_counterpart_corpus(researcher, bundle):
    from gym.isolation import canaries_of
    out = researcher.run_task({"task_id": "t05", "fox_or_tool": "f06_uninformed_prior"})
    foreign_canary = canaries_of(bundle)["gibsons"]
    assert foreign_canary not in str(out.to_dict())
    assert all("gibsons" not in e for e in out.evidence)
