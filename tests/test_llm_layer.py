"""Model layer: client, planner validation and turn writing.

No test calls the API: the client is replaced by a fake one. What is tested is the contract —
what is accepted, what is rejected and what happens when the model fails — not the model.
"""
import functools
import json

import pytest

from agents.llm import EmptyCompletion, LLMClient, LLMResponse, LLMUsage, load_env
from agents.negotiator.llm_writer import TurnWriter
from agents.planner import PartyPlanner
from gym.case import CaseBundle
from gym.protocol import Action


class FakeClient:
    """A client that returns whatever it is told, with no network."""

    def __init__(self, payload, fail: Exception | None = None) -> None:
        self.payload = payload
        self.fail = fail
        self.model = "fake"
        self.endpoint = "https://fake/v1"
        self.calls = 0
        self.configured = True

    def complete(self, messages, **kwargs) -> LLMResponse:
        self.calls += 1
        if self.fail:
            raise self.fail
        text = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return LLMResponse(text=text, usage=LLMUsage(10, 20, 5, 0, 0.1, 1), model="fake",
                           finish_reason="stop", reasoning="simulated reasoning")


@pytest.fixture(scope="module")
def bundle() -> CaseBundle:
    return CaseBundle.load("parker_gibson")


@functools.lru_cache(maxsize=1)
def _sealed_parkers_rv() -> float:
    """The parkers' reservation value, read from the local bundle rather than written here."""
    from agents.declared_utility import derive_from_view
    view = CaseBundle.load("parker_gibson").party_view("parkers")
    return derive_from_view(view).reservation_value_raw


def _valid_strategy(quote: str) -> dict:
    return {
        "case_view": {"issues": [{"id": "price"}], "my_interests": ["sell high"],
                      "inferred_counterpart_interests": [], "uncertainty_sources": ["their cap"]},
        "declared_utility": {"reservation_value": {"value": _sealed_parkers_rv(), "quote": quote},
                             "aspiration": {"value": 35000, "reasoning": "…"},
                             "batna": {"description": "sell to the buyer of the house"}},
        "strategy": {"features": ["unique parcel"],
                     "negotiation_parameters": {"first_offer": {"value": 45000, "reasoning": "…"},
                                                "concession_plan": "…", "walk_away_rule": "…"},
                     "approach": ["prepare", "probe", "close"],
                     "uncertainty_levers": [{"lever": "optimism by quantile", "use": "yes",
                                             "why": "…"}]},
        "plan": [{"task_id": "t01", "objective": "estimate their cap",
                  "fox_or_tool": "f01_bayes_rv_concession", "phase": "online",
                  "why_in_scope": "there will be more than 2 offers", "expected_output": "rv posterior"}],
        "foxes_rejected": [{"id": "f02_concession_issue_weights", "why": "single issue"}],
        "hypotheses": [{"statement": "they are in a hurry", "variable": "theta.T", "prior_belief": 0.5,
                        "test": {"kind": "observe_rounds", "resolves_if": "if they concede fast"}}],
    }


def _real_quote(bundle) -> str:
    from agents.declared_utility import derive_from_view
    return derive_from_view(bundle.party_view("parkers")).provenance[0].split("«")[1].rstrip("»")


def test_a_correct_strategy_validates(bundle):
    view = bundle.party_view("parkers")
    planner = PartyPlanner(view, client=FakeClient(_valid_strategy(_real_quote(bundle))))
    result = planner.plan()
    assert result.ok, result.validation["errors"]
    assert result.validation["n_plan_tasks"] == 1


def test_rejects_an_invented_quote(bundle):
    strategy = _valid_strategy("The Parkers would accept anything above a thousand dollars.")
    planner = PartyPlanner(bundle.party_view("parkers"), client=FakeClient(strategy))
    result = planner.plan()
    assert not result.ok
    assert any("quote" in e for e in result.validation["errors"])


def test_rejects_a_reservation_value_outside_the_space(bundle):
    strategy = _valid_strategy(_real_quote(bundle))
    strategy["declared_utility"]["reservation_value"]["value"] = 900000
    planner = PartyPlanner(bundle.party_view("parkers"), client=FakeClient(strategy))
    result = planner.plan()
    assert not result.ok
    assert any("outside the case range" in e for e in result.validation["errors"])


def test_rejects_a_candidate_fox(bundle):
    strategy = _valid_strategy(_real_quote(bundle))
    strategy["plan"][0]["fox_or_tool"] = "f05_case_reader_prior"
    planner = PartyPlanner(bundle.party_view("parkers"), client=FakeClient(strategy))
    result = planner.plan()
    assert not result.ok
    assert any("candidate" in e for e in result.validation["errors"])


def test_rejects_an_unknown_fox(bundle):
    strategy = _valid_strategy(_real_quote(bundle))
    strategy["plan"][0]["fox_or_tool"] = "f99_imaginary_fox"
    planner = PartyPlanner(bundle.party_view("parkers"), client=FakeClient(strategy))
    result = planner.plan()
    assert not result.ok and any("catalog" in e for e in result.validation["errors"])


def test_rejects_a_weight_fox_in_a_single_issue_case(bundle):
    strategy = _valid_strategy(_real_quote(bundle))
    strategy["plan"][0]["fox_or_tool"] = "f09_hypothesis_issue_weights"
    planner = PartyPlanner(bundle.party_view("parkers"), client=FakeClient(strategy))
    result = planner.plan()
    assert not result.ok
    assert any("single issue" in e for e in result.validation["errors"])


def test_the_control_f06_is_never_rejected_on_scope(bundle):
    """Regression: f06 can produce a prior over weights, but its scope is 'always applicable'.
    Rejecting it in a single-issue case was a validation error, not a plan error: the protocol
    (§6.5, rule 2) requires the control in every program."""
    strategy = _valid_strategy(_real_quote(bundle))
    strategy["plan"].append({"task_id": "t02", "objective": "baseline",
                             "fox_or_tool": "f06_uninformed_prior", "phase": "preparation",
                             "why_in_scope": "always applicable", "expected_output": "prior"})
    planner = PartyPlanner(bundle.party_view("parkers"), client=FakeClient(strategy))
    result = planner.plan()
    assert result.ok, result.validation["errors"]


def test_rejects_a_hypothesis_without_a_test(bundle):
    strategy = _valid_strategy(_real_quote(bundle))
    strategy["hypotheses"][0]["test"] = {"kind": "observe_rounds"}
    planner = PartyPlanner(bundle.party_view("parkers"), client=FakeClient(strategy))
    result = planner.plan()
    assert not result.ok and any("test" in e for e in result.validation["errors"])


def test_a_non_json_output_fails_with_a_clear_message(bundle):
    planner = PartyPlanner(bundle.party_view("parkers"), client=FakeClient("not json {"))
    result = planner.plan()
    assert not result.ok and "JSON" in result.validation["errors"][0]


def test_the_catalog_the_planner_sees_hides_candidates(bundle):
    digest = PartyPlanner(bundle.party_view("parkers"), client=FakeClient({})).catalog_digest()
    assert "f01_bayes_rv_concession" in digest
    assert "f05_case_reader_prior" in digest.split("Not available")[1]
    assert "f05_case_reader_prior" not in digest.split("Not available")[0]


# ------------------------------------------------------------------ turn writing
def _memo() -> dict:
    return {"round": 4, "chosen": {"action": "propose", "offers": [{"price": 32000.0}]},
            "belief_summary": {"rv": {"median": 0.4}, "foxes_in_scope": ["f01"],
                               "foxes_out_of_scope": [], "disagreement_sd": 0.3},
            "candidates": [], "rationale": "deterministic reasoning", "concession_target": 0.5}


def test_writing_cannot_change_the_action():
    writer = TurnWriter("parkers", "seller", "gibsons", "price",
                        client=FakeClient({"rationale": "because", "message": "Here is my offer",
                                           "expects_to_learn": "…"}))
    action = Action(4, "parkers", "propose", [{"price": 32000.0}])
    written = writer.write(action, _memo())
    assert written.message == "Here is my offer"
    assert action.offers == [{"price": 32000.0}], "the action is immutable for the writer"


def test_if_the_model_fails_the_turn_continues_with_the_deterministic_memo():
    writer = TurnWriter("parkers", "seller", "gibsons", "price",
                        client=FakeClient(None, fail=RuntimeError("500 from the provider")))
    written = writer.write(Action(4, "parkers", "propose", [{"price": 32000.0}]), _memo())
    assert written.failed and "500" in written.failed
    assert written.rationale == "deterministic reasoning"
    assert written.message is None


def test_without_credentials_the_writer_calls_nobody():
    client = LLMClient(env={})
    assert not client.configured
    writer = TurnWriter("parkers", "seller", "gibsons", "price", client=client)
    written = writer.write(Action(1, "parkers", "propose", [{"price": 1.0}]), _memo())
    assert "credentials" in (written.failed or "")


def test_the_client_requires_credentials_before_calling():
    from agents.llm import CredentialsMissing
    with pytest.raises(CredentialsMissing):
        LLMClient(env={}).complete([{"role": "user", "content": "hi"}])


def test_load_env_ignores_comments_and_quotes(tmp_path):
    path = tmp_path / ".env"
    path.write_text('# comment\nLLM_API_KEY="abc123"\nLLM_ENDPOINT=https://x/v1\n')
    values = load_env(path)
    assert values["LLM_API_KEY"] == "abc123" and values["LLM_ENDPOINT"] == "https://x/v1"


# ------------------------------------------------------------------ preparation orchestration
def test_the_full_preparation_produces_the_artifacts_of_section_5_1(bundle, tmp_path):
    """With a simulated planner, the step must leave every artifact and seal the utility."""
    from gym.prepare import run_preparation

    quote = _real_quote(bundle)
    client = FakeClient(_valid_strategy(quote))
    results = run_preparation("parker_gibson", "prog-prep-test", roles=["parkers"],
                              runs_dir=tmp_path, client=client)
    res = results["parkers"]
    assert res["ok"], res["validation"]["errors"]

    prep = tmp_path / "prog-prep-test" / "parkers" / "prep"
    for name in ("strategy.json", "strategy.md", "plan.json", "declared_utility.json",
                 "preparation_memo.md", "research.json", "outcomes.json",
                 "integration.json", "distributions.parquet"):
        assert (prep / name).exists(), f"missing {name}"

    # The sealed utility matches the deterministic extraction.
    declared = json.loads((prep / "declared_utility.json").read_text())
    assert declared["reservation_value_raw"] == _sealed_parkers_rv()

    # The memo contrasts the two independent readings of the same material.
    memo = (prep / "preparation_memo.md").read_text()
    assert "deterministic extraction" in memo and "agree" in memo


def test_the_preparation_runs_the_plan_with_researchers(bundle, tmp_path):
    from gym.prepare import run_preparation

    client = FakeClient(_valid_strategy(_real_quote(bundle)))
    results = run_preparation("parker_gibson", "prog-prep-2", roles=["parkers"],
                              runs_dir=tmp_path, client=client)
    research = results["parkers"]["research"]
    assert len(research) == 1 and research[0].fox_id == "f01_bayes_rv_concession"
    # In preparation f01 has no observations: it must declare itself out of scope and stay out
    # of the pool, on record.
    integration = results["parkers"]["integration"]
    assert any(d["fox_id"] == "f01_bayes_rv_concession" for d in integration["discarded"])


def test_an_invalid_strategy_seals_nothing(bundle, tmp_path):
    from gym.prepare import run_preparation

    strategy = _valid_strategy("a quote that exists in no corpus of the case")
    results = run_preparation("parker_gibson", "prog-prep-3", roles=["parkers"],
                              runs_dir=tmp_path, client=FakeClient(strategy))
    assert not results["parkers"]["ok"]
    prep = tmp_path / "prog-prep-3" / "parkers" / "prep"
    assert not (prep / "strategy.json").exists(), "no artifacts are written for an invalid strategy"
    assert (prep / "llm_calls.jsonl").exists(), "but the raw call stays, for diagnosis"
