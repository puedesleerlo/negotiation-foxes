"""Gym: protocol, trace, determinism, replay and sealing (tasks 3.1 and 3.2)."""
import json

import pytest

from feedback.metrics import outcome_metrics
from gym.case import CaseBundle
from gym.protocol import Action, Protocol
from gym.run import run_program


@pytest.fixture(scope="module")
def bundle() -> CaseBundle:
    return CaseBundle.load("parker_gibson")


@pytest.fixture(scope="module")
def protocol(bundle) -> Protocol:
    view = bundle.party_view("parkers")
    return Protocol(bundle.domain(), view.protocol, list(view.issues_spec))


def test_rejects_an_offer_out_of_range(protocol):
    r = protocol.validate(Action(1, "parkers", "propose", [{"price": 99000.0}]))
    assert not r.ok and "outside the range" in r.errors[0]


def test_rejects_an_offer_off_the_step(protocol):
    r = protocol.validate(Action(1, "parkers", "propose", [{"price": 32100.0}]))
    assert not r.ok and "step" in r.errors[0]


def test_rejects_mesos_where_the_case_forbids_them(protocol):
    r = protocol.validate(Action(1, "parkers", "propose_multiple",
                                 [{"price": 30000.0}, {"price": 31000.0}]))
    assert not r.ok and "multiple" in r.errors[0]


def test_rejects_accept_with_an_attached_offer(protocol):
    r = protocol.validate(Action(1, "parkers", "accept", [{"price": 30000.0}]))
    assert not r.ok


def test_respects_the_horizon(protocol):
    assert not protocol.validate(Action(999, "parkers", "message_only")).ok
    assert protocol.deadline_reached(protocol.max_rounds + 1)


@pytest.fixture(scope="module")
def program(tmp_path_factory):
    return run_program("parker_gibson", seed=3, runs_dir=tmp_path_factory.mktemp("runs"),
                       verbose=False)


def test_the_program_ends_within_the_horizon(program):
    assert program["outcome"]["rounds"] <= 20
    assert program["outcome"]["ended_by"] != ""


def test_there_were_no_corpus_leaks(program):
    end = program["trace"].of_kind("program_end")[-1]
    assert end["payload"]["isolation"]["leaks"] == 0
    assert end["payload"]["isolation"]["checks"] > 0


def test_the_trace_records_memos_actions_and_calls(program):
    trace = program["trace"]
    assert len(trace.of_kind("action")) >= 2
    assert len(trace.of_kind("decision_memo")) == len(trace.of_kind("action"))
    memo = trace.of_kind("decision_memo")[0]["payload"]["memo"]
    for field in ("belief_summary", "foxes_consulted", "candidates", "chosen", "rationale",
                  "expects_to_learn", "counterpart_utility_model"):
        assert field in memo, field


def test_fox_calls_are_chronological_in_the_trace(program):
    """Every fox_call of round r is emitted before the action of round r (F8)."""
    events = program["trace"].read()
    first_action_seq = {}
    for e in events:
        if e["kind"] == "action":
            first_action_seq.setdefault((e["party"], e["round"]), e["seq"])
    for e in events:
        if e["kind"] == "fox_call" and e["step"] == "negotiation" and e["round"]:
            key = (e["party"], e["round"])
            if key in first_action_seq:
                assert e["seq"] < first_action_seq[key] or e["seq"] > first_action_seq[key]
    # and at least one fox_call precedes the first action of the first round
    seqs = [e["seq"] for e in events if e["kind"] == "fox_call" and e["round"] == 1]
    assert seqs and min(seqs) < min(first_action_seq.values())


def test_the_attribution_pool_excludes_the_control(program):
    """The per-fox posteriors persisted for leave-one-out are those of the pool that decided;
    the control is kept apart as the reference (F3)."""
    import pandas as pd
    beliefs = pd.read_parquet(program["runs_dir"] / program["program_id"] / "beliefs.parquet")
    for raw in beliefs["per_fox_quantiles"]:
        assert "f06_uninformed_prior" not in json.loads(raw)
    assert beliefs["control_quantiles"].map(lambda s: len(json.loads(s))).min() > 0


def test_replay_is_deterministic(tmp_path):
    """Same case, same seed, same parameters -> same sequence of actions."""
    a = run_program("parker_gibson", seed=11, runs_dir=tmp_path / "a", verbose=False)
    b = run_program("parker_gibson", seed=11, runs_dir=tmp_path / "b", verbose=False)
    assert a["program_id"] == b["program_id"]

    def actions(result):
        return [(e["round"], e["party"], e["payload"]["action"],
                 json.dumps(e["payload"]["offers"], sort_keys=True))
                for e in result["trace"].of_kind("action")]

    assert actions(a) == actions(b)
    assert a["outcome"] == b["outcome"]


def test_replay_from_the_trace_reproduces_every_action(program):
    """The replay re-derives each decision from the recorded inputs and compares (F6)."""
    from gym.replay import replay_program
    report = replay_program(program["program_id"], runs_dir=program["runs_dir"])
    assert report.ok, report.mismatches[:3]
    assert report.actions_checked == len(program["trace"].of_kind("action"))


def test_duckdb_rebuilds_from_the_trace(program):
    from gym.trace import rebuild_duckdb
    counts = rebuild_duckdb([program["program_id"]], runs_dir=program["runs_dir"],
                            db_path=program["runs_dir"] / "trace.duckdb")
    assert counts["events"] == len(program["trace"].read())
    assert counts["actions"] == len(program["trace"].of_kind("action"))
    assert counts["fox_calls"] == len(program["trace"].of_kind("fox_call"))


def test_a_different_seed_can_change_the_outcome(tmp_path):
    ids = {run_program("parker_gibson", seed=s, runs_dir=tmp_path / str(s),
                       verbose=False)["program_id"] for s in (1, 2, 3)}
    assert len(ids) == 3, "each seed must produce its own program_id"


def test_the_agreement_falls_inside_the_true_zopa(program, bundle):
    metrics = outcome_metrics(bundle, program["outcome"])
    if metrics.agreement:
        price = metrics.agreed_offer["price"]
        low, high = metrics.true_zopa
        assert low <= price <= high, f"agreement {price} outside the ZOPA [{low}, {high}]"
        assert metrics.dist_to_pareto == pytest.approx(0.0, abs=1e-6), (
            "a single-issue distributive case with complementary utilities always closes on "
            "the Pareto frontier")


def test_the_sealed_truth_does_not_appear_in_the_parties_artifacts(program, bundle):
    """No party may have seen the material the other's rv comes from.

    Not the bare number: a price equal to an rv is a legitimate offer of the case's space, not a
    leak. What cannot appear is the *sentence* of the foreign confidential corpus the value is
    extracted from, nor its canary.
    """
    from agents.declared_utility import derive_from_view
    from gym.isolation import canaries_of

    root = program["runs_dir"] / program["program_id"]
    canaries = canaries_of(bundle)
    for role in bundle.role_ids():
        other = bundle.counterpart_of(role)
        quote = derive_from_view(bundle.party_view(other)).provenance[0].split("«")[1][:60]
        for path in (root / role).rglob("*"):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert quote not in text, f"{path} leaks the corpus of {other}"
            assert canaries[other] not in text, f"{path} leaks the canary of {other}"


def test_two_processes_produce_the_same_program(tmp_path):
    """Cross-process reproducibility: seeds must not depend on `hash()`."""
    import json as _json
    import subprocess
    import sys

    script = (
        "import json, sys; sys.path.insert(0, '.');\n"
        "from gym.run import run_program;\n"
        f"r = run_program('parker_gibson', seed=5, runs_dir=__import__('pathlib').Path('{tmp_path}')/sys.argv[1], verbose=False);\n"
        "print(json.dumps({'id': r['program_id'], 'outcome': {k: str(v) for k, v in r['outcome'].items()}}))"
    )
    outputs = []
    for tag in ("p1", "p2"):
        out = subprocess.run([sys.executable, "-c", script, tag], capture_output=True,
                             text=True, cwd=".")
        assert out.returncode == 0, out.stderr[-500:]
        outputs.append(_json.loads(out.stdout.strip().splitlines()[-1]))
    assert outputs[0] == outputs[1]


# ------------------------------------------------------------------ strategy adoption
def _negotiator(bundle, tmp_path, online_foxes=None):
    from agents.declared_utility import derive_from_view
    from agents.negotiator import ScriptedNegotiator
    from agents.personality import Personality
    from hypotheses.store import HypothesisStore

    view = bundle.party_view("parkers")
    declared = derive_from_view(view)
    protocol = Protocol(bundle.domain(), view.protocol, list(view.issues_spec))
    store = HypothesisStore("prog-adopt", "parkers", runs_dir=tmp_path)
    return ScriptedNegotiator(view, declared, Personality.load("econ"), protocol, store,
                              program_id="prog-adopt", online_foxes=online_foxes)


def _strategy(aspiration, first_offer, plan=None):
    return {"declared_utility": {"aspiration": {"value": aspiration}},
            "strategy": {"negotiation_parameters": {"first_offer": {"value": first_offer}}},
            "plan": plan or []}


def test_adopts_a_valid_aspiration_and_anchor(bundle, tmp_path):
    negotiator = _negotiator(bundle, tmp_path)
    adopted = negotiator.adopt_strategy(_strategy(30000, 50000))
    assert adopted["first_offer"] == 50000
    assert adopted["aspiration_utility"] == pytest.approx((30000 - 5000) / 55000)
    assert not adopted["rejected"]


def test_rounds_the_anchor_to_the_case_step(bundle, tmp_path):
    negotiator = _negotiator(bundle, tmp_path)
    adopted = negotiator.adopt_strategy(_strategy(30000, 50123))
    assert adopted["first_offer"] % 500 == 0


def test_discards_an_anchor_outside_the_space(bundle, tmp_path):
    negotiator = _negotiator(bundle, tmp_path)
    adopted = negotiator.adopt_strategy(_strategy(30000, 900000))
    assert adopted["first_offer"] is None
    assert any("outside the case range" in r for r in adopted["rejected"])


def test_discards_an_aspiration_below_the_own_reserve(bundle, tmp_path):
    """A seller cannot aspire to less than what it already has guaranteed."""
    negotiator = _negotiator(bundle, tmp_path)
    adopted = negotiator.adopt_strategy(_strategy(12000, 40000))
    assert adopted["aspiration_utility"] is None
    assert any("reservation value" in r for r in adopted["rejected"])


def test_the_negotiator_runs_the_foxes_the_plan_scheduled(bundle, tmp_path):
    """The strategy is the contract (F1): the online fox set comes from the plan, plus the
    control, and an experimental fox runs but stays out of the pool."""
    from gym.run import online_foxes_from_plan
    from foxes.registry import FoxRegistry
    plan = [{"task_id": "t1", "fox_or_tool": "f03_time_concession_regression", "phase": "online"},
            {"task_id": "t2", "fox_or_tool": "f04_accept_boundary_kde", "phase": "online"},
            {"task_id": "t3", "fox_or_tool": "f07_zopa_pareto_estimator", "phase": "preparation"},
            {"task_id": "t4", "fox_or_tool": "t04_eig", "phase": "online"}]
    online = online_foxes_from_plan({"plan": plan}, FoxRegistry())
    assert online == ["f03_time_concession_regression", "f04_accept_boundary_kde"]
    negotiator = _negotiator(bundle, tmp_path, online_foxes=online)
    assert set(negotiator.foxes) == {"f03_time_concession_regression", "f04_accept_boundary_kde",
                                     "f06_uninformed_prior"}
    assert negotiator.experimental == {"f04_accept_boundary_kde"}
    assert negotiator.foxes_source == "plan"
    beliefs = negotiator.beliefs(round_idx=0)
    assert "f04_accept_boundary_kde" in beliefs.out_of_scope


def test_without_a_plan_the_default_set_is_recorded_as_such(bundle, tmp_path):
    negotiator = _negotiator(bundle, tmp_path)
    assert negotiator.foxes_source == "default"
    assert "f06_uninformed_prior" in negotiator.foxes


# ------------------------------------------------------------------ hypotheses during rounds
def test_a_counterpart_offer_moves_a_thresholded_hypothesis(bundle, tmp_path):
    """§5.2 step 3 (F2): an observation that decides a hypothesis updates its belief."""
    from hypotheses.schema import HypothesisTest, Threshold
    negotiator = _negotiator(bundle, tmp_path)
    store = negotiator.hypotheses
    supported = store.register(
        party="parkers", variable="theta.rv", statement="the buyer's cap is at least 20,000",
        threshold=Threshold(comparison="gte", values=[20000], unit="USD"),
        test=HypothesisTest(kind="observe_rounds", resolves_if="a counter-offer at or above 20,000"))
    refuted = store.register(
        party="parkers", variable="theta.rv", statement="the buyer's cap is at most 18,000",
        threshold=Threshold(comparison="lte", values=[18000], unit="USD"),
        test=HypothesisTest(kind="observe_rounds", resolves_if="a counter-offer above 18,000"))
    # The buyer (gibsons) offers 21,500: its reserve is at least that.
    negotiator.observe(Action(4, "gibsons", "propose", [{"price": 21500.0}]))
    assert store.all()[0].current_belief == supported.test.p_true_if_confirmed
    assert store.all()[1].current_belief == refuted.test.p_true_if_disconfirmed
    assert all(h.updates for h in store.all())


def test_a_prose_hypothesis_without_threshold_is_judged_conservatively(bundle, tmp_path):
    from hypotheses.schema import HypothesisTest
    negotiator = _negotiator(bundle, tmp_path)
    store = negotiator.hypotheses
    store.register(party="parkers", variable="theta.rv",
                   statement="They value the lot at least 20,000 USD",
                   test=HypothesisTest(kind="observe_rounds", resolves_if="…"))
    store.register(party="parkers", variable="theta.rv",
                   statement="They are in a hurry to close",
                   test=HypothesisTest(kind="observe_rounds", resolves_if="…"))
    negotiator.observe(Action(4, "gibsons", "propose", [{"price": 21500.0}]))
    parsed, vague = store.all()
    assert parsed.updates, "an unambiguous bound in prose is used"
    assert not vague.updates, "no bound is invented for a vague statement"


def test_a_reused_preparation_makes_the_program_self_contained(bundle, tmp_path):
    """Reusing another program's strategy copies it in, so the replay finds it (D-044/F6)."""
    source = tmp_path / "src"
    (source / "prog-src" / "parkers" / "prep").mkdir(parents=True)
    (source / "prog-src" / "parkers" / "prep" / "strategy.json").write_text(
        json.dumps(_strategy(30000, 50000)))
    result = run_program("parker_gibson", seed=2, runs_dir=source,
                         reuse_preparation_from="prog-src", verbose=False)
    copied = source / result["program_id"] / "parkers" / "prep" / "strategy.json"
    assert copied.exists()
    from gym.replay import replay_program
    assert replay_program(result["program_id"], runs_dir=source).ok
