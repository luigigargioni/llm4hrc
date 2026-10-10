"""End-to-end replay of user-study conversations through the full graph.

The scenario suite in ``scenario_runner.py`` exercises only
``world_state_updater -> plan_validator``. This runner instead drives the
*whole* LangGraph pipeline, so the three components the suite never touches
are measured in situ:

  * the Action Canonicalizer as an LLM call (real mapping error rate, real
    latency, and how often it re-canonicalises the same activity),
  * the hard-enforcement override in Next Steps (how often it fires, and
    whether it fires on an actual proposal or merely on a refusal that
    names the blocked activity),
  * the Goal Planner, reachable only through the perception router.

Replay works because ``robot_perception_node`` asks for exactly the three
fields that ``conversation_message`` writes to each transcript: the
simulated time, the patient request, and the situation perception. Feeding
them back in order reproduces the session.

Usage::

    python -m evaluation.replay_runner --dry-run          # no LLM calls
    python -m evaluation.replay_runner --limit 3
    python -m evaluation.replay_runner --min-turns 3
"""

import argparse
import builtins
from copy import deepcopy
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.transcript_parser import Transcript, load_corpus  # noqa: E402

# The user-study sessions live outside the code repository, one folder per
# participant. Folder names carry real names, so only the participant
# number reaches the results (see transcript_parser.Transcript.name).
DEFAULT_CORPUS = (
    r"C:\Users\Gargioni\Il mio Drive\Academic\_Unibs\PhD"
    r"\Hybrid Architecture\User Study\Users"
)


class ReplayFinished(Exception):
    """Raised when the scripted input feeder is exhausted, ending the run."""


class ReplayBudgetExceeded(Exception):
    """Raised when a transcript consumes more graph iterations than allowed."""


class ReplayTokenBudgetExceeded(Exception):
    """Raised when a transcript spends more tokens than allowed."""


class ReplayTaskCompleted(Exception):
    """Raised when the graph reaches the Knowledge Manager with every
    activity done or skipped.

    The base architecture routes there once the day's task is over, but the
    Knowledge Manager never sets the end flag, so the graph restarts from
    Situation Assessment without returning to perception and cycles with no
    input until the stall guard stops it. Those cycles would add decisions
    on an unchanging state to every statistic. The architecture is left as
    it is; the runner ends the session at that point instead, recording how
    many transcript turns were still unread.
    """


class ReplayStalled(Exception):
    """Raised when the graph cycles without reading a new conversation turn.

    A healthy session runs roughly one cycle per turn. The runaway observed
    in the user-study replay spins tens of cycles without ever reaching
    ``robot_perception_node``, so no transcript input is consumed while
    tokens keep being spent. Watching the feeder's input counter catches
    that within a few cycles instead of hundreds.
    """


class InputFeeder:
    """Answers the graph's ``input()`` calls from a transcript.

    The graph asks for three setup values (patient id, doctor specs, start
    time) and then three values per perception cycle. ``robot_perception_node``
    refuses a time earlier than the current one and re-prompts, so emitted
    times are clamped to be non-decreasing.
    """

    def __init__(self, transcript: Transcript, meter: "TokenMeter | None" = None):
        self.transcript = transcript
        self.meter = meter
        self.script: list[str] = []
        self.consumed = 0
        self.last_time = transcript.start_time

        # Setup: task_synthesizer_node asks these three, in this order.
        self.script.append(transcript.patient_id)
        self.script.append(transcript.doctor_specs)
        self.script.append(transcript.start_time)

        # One perception cycle per recorded turn.
        for turn in transcript.turns:
            emitted = max(turn.time, self.last_time)
            self.last_time = emitted
            self.script.append(emitted)
            self.script.append(turn.patient_request or "")
            self.script.append(turn.situation_perception)

    def __call__(self, prompt: str = "") -> str:
        if self.consumed >= len(self.script):
            raise ReplayFinished(
                f"transcript {self.transcript.name} exhausted after "
                f"{self.consumed} inputs"
            )
        value = self.script[self.consumed]
        self.consumed += 1
        if self.meter is not None:
            # The first three inputs are setup; each later group of three
            # opens the next perception cycle.
            self.meter.current_turn = max(0, self.consumed - 3 + 2) // 3
        print(f"[replay] input #{self.consumed} -> {value!r}")
        return value


class TokenMeter:
    """Attributes OpenAI token usage to the node that spent it.

    Every node obtains its model from ``utils.models.define_model`` and then
    calls ``.invoke`` on it, so wrapping the returned runnable captures all
    remote usage. ``progress_checking_node`` builds its ``ChatOllama``
    directly and is therefore correctly excluded: it runs locally and costs
    nothing.
    """

    def __init__(self, token_budget: int | None = None):
        self.token_budget = token_budget
        self.current_node = "graph"
        self.current_turn = 0
        self.by_node: dict[str, dict[str, int]] = {}
        # One entry per remote call, so prompt growth over a session can be
        # measured rather than estimated.
        self.calls: list[dict] = []

    def record(self, response) -> None:
        usage = getattr(response, "usage_metadata", None) or {}
        if not usage:
            meta = getattr(response, "response_metadata", None) or {}
            usage = meta.get("token_usage", {}) or {}
        prompt = int(
            usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        )
        completion = int(
            usage.get("output_tokens") or usage.get("completion_tokens") or 0
        )

        bucket = self.by_node.setdefault(
            self.current_node, {"calls": 0, "input": 0, "output": 0}
        )
        bucket["calls"] += 1
        bucket["input"] += prompt
        bucket["output"] += completion

        # Reasoning models bill hidden chain-of-thought as output, so
        # separating it tells whether a node's output cost is the answer
        # itself or the thinking that produced it.
        details = usage.get("output_token_details") or {}
        self.calls.append({
            "seq": len(self.calls) + 1,
            "turn": self.current_turn,
            "node": self.current_node,
            "input": prompt,
            "output": completion,
            "reasoning": int(details.get("reasoning") or 0),
        })

        if self.token_budget is not None:
            spent = self.totals()
            used = spent["input_tokens"] + spent["output_tokens"]
            if used > self.token_budget:
                raise ReplayTokenBudgetExceeded(
                    f"spent {used} tokens, budget {self.token_budget}"
                )

    def wrap_model(self, model):
        meter = self

        class MeteredModel:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, item):
                return getattr(self._inner, item)

            def invoke(self, *args, **kwargs):
                response = self._inner.invoke(*args, **kwargs)
                meter.record(response)
                return response

            def bind_tools(self, *args, **kwargs):
                return MeteredModel(self._inner.bind_tools(*args, **kwargs))

        return MeteredModel(model)

    def totals(self) -> dict:
        return {
            "calls": sum(b["calls"] for b in self.by_node.values()),
            "input_tokens": sum(b["input"] for b in self.by_node.values()),
            "output_tokens": sum(b["output"] for b in self.by_node.values()),
        }


class Recorder:
    """Wraps graph nodes to record decisions, latency, and call counts."""

    def __init__(self, max_iterations: int, meter: TokenMeter,
                 max_idle_cycles: int | None = None):
        self.max_iterations = max_iterations
        self.meter = meter
        self.max_idle_cycles = max_idle_cycles
        # Set after construction: the feeder needs the meter, which needs
        # to exist first.
        self.feeder: "InputFeeder | None" = None
        self.idle_cycles = 0
        self.inputs_at_last_cycle = 0
        self.iterations = 0
        self.node_calls: dict[str, int] = {}
        self.node_seconds: dict[str, float] = {}
        self.canonicalizations: list[dict] = []
        self.validations: list[dict] = []
        self.enforcements: list[dict] = []
        self.goal_planner_calls = 0

    def wrap(self, name: str, fn):
        def wrapped(state):
            self.iterations += 1
            if self.iterations > self.max_iterations:
                raise ReplayBudgetExceeded(
                    f"exceeded {self.max_iterations} node invocations"
                )

            # One cycle is one pass through the canonicalizer. A healthy
            # session advances the transcript every cycle; if several
            # cycles go by with no new input read, the graph is spinning.
            if name == "action_canonicalizer" and self.max_idle_cycles:
                consumed = self.feeder.consumed if self.feeder else 0
                if consumed > self.inputs_at_last_cycle:
                    self.inputs_at_last_cycle = consumed
                    self.idle_cycles = 0
                else:
                    self.idle_cycles += 1
                    if self.idle_cycles > self.max_idle_cycles:
                        raise ReplayStalled(
                            f"{self.idle_cycles} cycles without reading a "
                            f"new turn (stuck after "
                            f"{self.meter.current_turn} turns)"
                        )

            if name == "knowledge_manager":
                task = _get(state, "task") or {}
                activities = task.get("activities") or []
                if activities and all(
                    a.get("done") is True or a.get("skipped") is True
                    for a in activities
                ):
                    raise ReplayTaskCompleted(
                        f"task completed after {self.meter.current_turn} "
                        f"turns"
                    )

            before = _snapshot_activities(state)
            previous_node = self.meter.current_node
            self.meter.current_node = name
            started = time.perf_counter()
            try:
                result = fn(state)
            finally:
                elapsed = time.perf_counter() - started
                self.meter.current_node = previous_node

            self.node_calls[name] = self.node_calls.get(name, 0) + 1
            self.node_seconds[name] = self.node_seconds.get(name, 0.0) + elapsed

            self._record(name, before, result, elapsed)
            return result

        return wrapped

    def _record(self, name: str, before: dict, result, elapsed: float) -> None:
        activities = _activities_of(result)

        if name == "action_canonicalizer":
            for activity in activities:
                activity_name = activity.get("activity_name", "")
                self.canonicalizations.append({
                    "activity_name": activity_name,
                    "canonical_action": activity.get("canonical_action"),
                    "is_known": activity.get("canonical_action") != "unknown",
                    "previous_mapping": before.get(activity_name, {}).get(
                        "canonical_action"
                    ),
                    "node_seconds": round(elapsed, 3),
                    "done": activity.get("done", False),
                })

        elif name == "plan_validator":
            for activity in activities:
                self.validations.append({
                    "activity_name": activity.get("activity_name", ""),
                    "canonical_action": activity.get("canonical_action"),
                    "arbitrated_status": activity.get("arbitrated_status"),
                    "reasons": activity.get("not_executable_reasons", []),
                    "recovery_actions": [
                        r.get("canonical_action")
                        for r in activity.get("recovery_actions", []) or []
                    ],
                })

        elif name == "goal_planner":
            self.goal_planner_calls += 1

        elif name == "next_steps":
            action = _get(result, "robot_action")
            answer = _get(result, "vocal_answer") or ""
            # These two actions are the shape an override leaves behind,
            # but the model also chooses them on its own, so counting them
            # overstates enforcement. The authoritative count is the
            # hard_enforcement entries in the structured log; what is
            # collected here is only a candidate for that cross-check.
            if action in ("SUGGEST_RECOVERY_ACTION", "WAIT_FOR_PATIENT"):
                blocked = [
                    a for a in activities if a.get("executable") is False
                ]
                self.enforcements.append({
                    "robot_action_after": action,
                    "vocal_answer_after": answer[:400],
                    "blocked_activities": [
                        a.get("activity_name", "") for a in blocked
                    ],
                    "statuses": [a.get("arbitrated_status") for a in blocked],
                    # An override whose pre-override answer was itself a
                    # refusal is a false positive of the substring check;
                    # that text comes from the structured log.
                    "classification": "candidate_only",
                })


def _get(state, key):
    try:
        return state[key]
    except (KeyError, TypeError):
        return None


def _activities_of(state) -> list[dict]:
    task = _get(state, "updated_task")
    if not task:
        return []
    return task.get("activities", []) or []


def _snapshot_activities(state) -> dict:
    return {
        a.get("activity_name", ""): {"canonical_action": a.get("canonical_action")}
        for a in _activities_of(state)
    }


_PRISTINE_INITIAL_STATE: dict | None = None


def _pristine_initial_state(interfaces_module) -> dict:
    """Return the untouched INITIAL_GRAPH_STATE, captured on first call."""
    global _PRISTINE_INITIAL_STATE
    if _PRISTINE_INITIAL_STATE is None:
        _PRISTINE_INITIAL_STATE = deepcopy(
            dict(interfaces_module.INITIAL_GRAPH_STATE)
        )
    return _PRISTINE_INITIAL_STATE


# The patient files in mock_data are the care plans every session is built
# from. The Goal Planner used to append its activities to them, so a
# replayed session inherited the activities of the sessions before it. The
# node no longer writes there; as a safeguard the runner also snapshots the
# directory before each session and restores it afterwards, recording in the
# result whether anything had changed.
MOCK_DATA_DIR = Path(__file__).resolve().parent.parent / "mock_data"
# The patient file the replayed sessions are built from.
CARE_PLAN_FILE = "789.txt"


def _snapshot_mock_data() -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in MOCK_DATA_DIR.iterdir()
            if p.is_file()}


def _restore_mock_data(snapshot: dict[str, bytes]) -> list[str]:
    """Put mock_data back as it was; return the names that had changed."""
    changed = []
    for path in MOCK_DATA_DIR.iterdir():
        if path.is_file() and path.name not in snapshot:
            path.unlink()
            changed.append(path.name)
    for name, content in snapshot.items():
        path = MOCK_DATA_DIR / name
        if not path.exists() or path.read_bytes() != content:
            path.write_bytes(content)
            changed.append(name)
    return sorted(changed)


def replay_transcript(
    transcript: Transcript,
    max_iterations: int,
    token_budget: int | None = None,
    max_idle_cycles: int | None = None,
) -> dict:
    """Replay one transcript through the full graph and return its metrics."""
    import graph.graph_init as graph_init_module
    import utils.interfaces as interfaces_module
    import utils.models as models_module
    import utils.structured_logs as structured_logs
    from utils.logs import create_log_file

    # graph_init() runs `inputs = INITIAL_GRAPH_STATE` with no copy, and
    # world_state_updater_node mutates that dict in place. Running one
    # session per process (as main.py does) hides this, but a replay runs
    # many, so each session would inherit the previous one's world state.
    # Restore the pristine contents in place, since the nodes hold their
    # own bindings to the same object.
    pristine = _pristine_initial_state(interfaces_module)
    interfaces_module.INITIAL_GRAPH_STATE.clear()
    interfaces_module.INITIAL_GRAPH_STATE.update(deepcopy(pristine))

    create_log_file()
    # Force a fresh structured log file per transcript so entries are
    # attributable without offset arithmetic.
    structured_logs._structured_log_filename = None  # noqa: SLF001

    # The canonicalizer remembers each activity name it has mapped for the
    # length of a session; a replay runs many sessions in one process, so
    # the memory has to be cleared between them or a name would carry its
    # mapping into the next transcript.
    from nodes.action_canonicalizer_node import reset_canonicalization_cache
    reset_canonicalization_cache()

    meter = TokenMeter(token_budget=token_budget)
    original_define_model = models_module.define_model

    def metered_define_model(*args, **kwargs):
        return meter.wrap_model(original_define_model(*args, **kwargs))

    models_module.define_model = metered_define_model
    # Nodes import define_model by value, so each module's own binding has
    # to be replaced too.
    model_importers = [
        module
        for name, module in list(sys.modules.items())
        if (name.startswith("nodes.") or name.startswith("edges."))
        and getattr(module, "define_model", None) is original_define_model
    ]
    for module in model_importers:
        module.define_model = metered_define_model

    recorder = Recorder(
        max_iterations=max_iterations,
        meter=meter,
        max_idle_cycles=max_idle_cycles,
    )
    instrumented = {
        "action_canonicalizer": "action_canonicalizer_node",
        "world_state_updater": "world_state_updater_node",
        "plan_validator": "plan_validator_node",
        "next_steps": "next_steps_node",
        "goal_planner": "goal_planner_node",
        "task_progress": "task_progress_node",
        "progress_checking": "progress_checking_node",
        "knowledge_manager": "knowledge_manager_node",
    }
    originals = {}
    for label, attr in instrumented.items():
        original = getattr(graph_init_module, attr)
        originals[attr] = original
        setattr(graph_init_module, attr, recorder.wrap(label, original))

    feeder = InputFeeder(transcript, meter=meter)
    recorder.feeder = feeder
    original_input = builtins.input
    builtins.input = feeder

    outcome = "completed"
    error = None
    started = time.perf_counter()
    try:
        graph_init_module.graph_init()
    except ReplayFinished:
        outcome = "transcript_exhausted"
    except ReplayBudgetExceeded as exc:
        outcome = "budget_exceeded"
        error = str(exc)
    except ReplayTokenBudgetExceeded as exc:
        outcome = "token_budget_exceeded"
        error = str(exc)
    except ReplayTaskCompleted as exc:
        outcome = "task_completed"
        error = str(exc)
    except ReplayStalled as exc:
        outcome = "stalled"
        error = str(exc)
    except Exception as exc:  # noqa: BLE001 - a crash is itself a finding
        outcome = "error"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        builtins.input = original_input
        for attr, original in originals.items():
            setattr(graph_init_module, attr, original)
        models_module.define_model = original_define_model
        for module in model_importers:
            module.define_model = original_define_model
    wall_seconds = time.perf_counter() - started

    canon = recorder.canonicalizations
    remapped = [
        c for c in canon
        if c["previous_mapping"] and c["previous_mapping"] != c["canonical_action"]
    ]

    return {
        "transcript": transcript.name,
        "patient_id": transcript.patient_id,
        "turns_in_transcript": len(transcript.turns),
        "inputs_consumed": feeder.consumed,
        "turns_replayed": max(0, feeder.consumed - 3) // 3,
        "outcome": outcome,
        "error": error,
        "wall_seconds": round(wall_seconds, 2),
        "structured_log": structured_logs._ensure_log_file(),  # noqa: SLF001
        "node_calls": recorder.node_calls,
        "node_seconds": {k: round(v, 2) for k, v in recorder.node_seconds.items()},
        "tokens_by_node": meter.by_node,
        "tokens_total": meter.totals(),
        "llm_calls": meter.calls,
        "goal_planner_calls": recorder.goal_planner_calls,
        "canonicalization_count": len(canon),
        "canonicalization_unknown": sum(1 for c in canon if not c["is_known"]),
        "canonicalization_remapped": len(remapped),
        "canonicalizations": canon,
        "validations": recorder.validations,
        "hard_enforcements": recorder.enforcements,
    }


def summarise(results: list[dict]) -> dict:
    total_canon = sum(r["canonicalization_count"] for r in results)
    statuses: dict[str, int] = {}
    tokens_by_node: dict[str, dict[str, int]] = {}
    for r in results:
        for v in r["validations"]:
            key = v["arbitrated_status"] or "NONE"
            statuses[key] = statuses.get(key, 0) + 1
        for node, usage in r["tokens_by_node"].items():
            bucket = tokens_by_node.setdefault(
                node, {"calls": 0, "input": 0, "output": 0}
            )
            for field_name in ("calls", "input", "output"):
                bucket[field_name] += usage[field_name]

    return {
        "tokens_by_node": tokens_by_node,
        "tokens_total": {
            "calls": sum(b["calls"] for b in tokens_by_node.values()),
            "input_tokens": sum(b["input"] for b in tokens_by_node.values()),
            "output_tokens": sum(b["output"] for b in tokens_by_node.values()),
        },
        "transcripts": len(results),
        "completed": sum(1 for r in results if r["outcome"] == "transcript_exhausted"),
        "errors": sum(1 for r in results if r["outcome"] == "error"),
        "budget_exceeded": sum(
            1 for r in results if r["outcome"] == "budget_exceeded"
        ),
        "token_budget_exceeded": sum(
            1 for r in results if r["outcome"] == "token_budget_exceeded"
        ),
        "stalled": sum(1 for r in results if r["outcome"] == "stalled"),
        "task_completed": sum(
            1 for r in results if r["outcome"] == "task_completed"
        ),
        "turns_replayed": sum(r["turns_replayed"] for r in results),
        "wall_seconds": round(sum(r["wall_seconds"] for r in results), 2),
        "canonicalizer_calls": total_canon,
        "canonicalizer_unknown": sum(
            r["canonicalization_unknown"] for r in results
        ),
        "canonicalizer_remapped_same_activity": sum(
            r["canonicalization_remapped"] for r in results
        ),
        "canonicalizer_seconds": round(
            sum(r["node_seconds"].get("action_canonicalizer", 0.0) for r in results), 2
        ),
        "validator_seconds": round(
            sum(r["node_seconds"].get("plan_validator", 0.0) for r in results), 2
        ),
        "goal_planner_calls": sum(r["goal_planner_calls"] for r in results),
        # Candidates, not confirmed overrides: see Recorder._record.
        "hard_enforcement_candidates": sum(
            len(r["hard_enforcements"]) for r in results
        ),
        "arbitrated_status_counts": statuses,
    }


def write_review_csv(results: list[dict], path: str) -> None:
    """Emit the rows that still need a human judgement."""
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "transcript", "kind", "activity_name", "canonical_action",
            "detail", "verdict_correct_yes_no",
        ])
        for r in results:
            for c in r["canonicalizations"]:
                writer.writerow([
                    r["transcript"], "canonicalization", c["activity_name"],
                    c["canonical_action"],
                    f"previous={c['previous_mapping']}", "",
                ])
            for e in r["hard_enforcements"]:
                writer.writerow([
                    r["transcript"], "hard_enforcement",
                    "; ".join(e["blocked_activities"]),
                    e["robot_action_after"], e["vocal_answer_after"], "",
                ])


def _write_results(results: list[dict], out_path: str, csv_path: str) -> dict:
    """Write the results file and review CSV, returning the payload."""
    output = {
        "generated_at": datetime.now().isoformat(),
        "provider": os.getenv("LLM_PROVIDER"),
        "model": (
            os.getenv("OPENROUTER_MODEL")
            if os.getenv("LLM_PROVIDER") == "openrouter"
            else os.getenv("OPENAI_MODEL")
        ),
        "summary": summarise(results),
        "transcripts": results,
    }
    # Write to a temporary file first, so an interrupt during the dump
    # cannot leave a truncated JSON behind.
    tmp_path = f"{out_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, out_path)
    write_review_csv(results, csv_path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end conversation replay")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS,
                        help="directory holding the transcripts to replay")
    parser.add_argument("--flat", action="store_true",
                        help="do not recurse into per-participant subfolders")
    parser.add_argument("--min-turns", type=int, default=3,
                        help="skip transcripts with fewer recorded turns")
    parser.add_argument("--session", default=None,
                        help="replay one session by name fragment, e.g. "
                             "'16072025_120235'; use when a participant "
                             "has more than one recorded session")
    parser.add_argument("--only", default=None,
                        help="replay just these participants, e.g. 'P08' or "
                             "'P08,P14' (matched against the participant id)")
    parser.add_argument("--limit", type=int, default=None,
                        help="replay at most N sessions after filtering")
    parser.add_argument("--max-iterations", type=int, default=400,
                        help="abort a transcript after N instrumented node calls")
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="absolute token ceiling per session; 0 means "
                             "scale it with --max-tokens-per-turn")
    parser.add_argument("--max-tokens-per-turn", type=int, default=60_000,
                        help="token ceiling per recorded turn, used when "
                             "--max-tokens is 0; a reasoning model spends "
                             "around 30k per turn, so this leaves headroom")
    parser.add_argument("--max-idle-cycles", type=int, default=4,
                        help="abort after N cycles that read no new turn "
                             "(0 disables); a healthy session reads one "
                             "turn per cycle")
    parser.add_argument("--provider", default=None,
                        help="override LLM_PROVIDER for this run")
    parser.add_argument("--model", default=None,
                        help="override the model id for this run")
    parser.add_argument("--out", default="evaluation/replay_results.json")
    parser.add_argument("--review-csv", default="evaluation/replay_review.csv")
    parser.add_argument("--repeat", type=int, default=1,
                        help="replay each session N times, to measure how "
                             "much run-to-run behaviour varies on "
                             "identical input")
    parser.add_argument("--care-plan-variant", default=None,
                        help="file that replaces mock_data/789.txt for each "
                             "session; the original is restored afterwards")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the input script without calling any model")
    args = parser.parse_args()

    corpus = load_corpus(
        args.corpus,
        min_turns=args.min_turns,
        recursive=not args.flat,
    )
    if args.only:
        available = {t.participant.upper() for t in corpus}
        wanted = {p.strip().upper() for p in args.only.split(",") if p.strip()}
        missing = wanted - available
        if missing:
            parser.error(
                f"nessuna sessione per {sorted(missing)}; "
                f"disponibili: {sorted(available)}"
            )
        corpus = [t for t in corpus if t.participant.upper() in wanted]
    if args.session:
        corpus = [t for t in corpus if args.session in t.name]
        if not corpus:
            parser.error(f"nessuna sessione contiene {args.session!r}")
    if args.limit:
        corpus = corpus[: args.limit]

    print(f"{len(corpus)} sessioni selezionate da {args.corpus} "
          f"(>= {args.min_turns} turni, "
          f"{sum(len(t.turns) for t in corpus)} turni totali)")

    if args.dry_run:
        for transcript in corpus:
            feeder = InputFeeder(transcript)
            print(f"\n--- {transcript.name} ({len(transcript.turns)} turni, "
                  f"{len(feeder.script)} input) ---")
            for i, value in enumerate(feeder.script):
                tag = "setup" if i < 3 else f"turn {(i - 3) // 3 + 1}"
                print(f"  [{tag}] {value!r}")
        return

    # The environment must be loaded before importing the graph:
    # robot_effector constructs an OpenAI client at import time.
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

    # graph_init() renders the workflow with pygraphviz when this is "true".
    # A replay re-enters graph_init() once per transcript, so the rendering
    # is both useless here and a hard failure without graphviz installed.
    os.environ["SAVE_GRAPH_WORKFLOW_IMAGES"] = "false"

    # load_dotenv overrides the environment, so command-line choices are
    # applied after it to make them final.
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        provider = os.environ.get("LLM_PROVIDER", "")
        key = ("OPENROUTER_MODEL" if provider == "openrouter"
               else "GROQ_MODEL" if provider == "groq"
               else "OLLAMA_MODEL" if provider == "ollama"
               else "OPENAI_MODEL")
        os.environ[key] = args.model

    # Each session is replayed --repeat times; repetitions of the same
    # session on identical input show how much run-to-run variation the
    # LLM calls introduce.
    variant_bytes = (
        Path(args.care_plan_variant).read_bytes()
        if args.care_plan_variant else None
    )
    schedule = [(t, rep) for t in corpus for rep in range(1, args.repeat + 1)]
    results = []
    for index, (transcript, repetition) in enumerate(schedule, start=1):
        label = transcript.name
        if args.repeat > 1:
            label = f"{label}#{repetition}"
        print(f"\n===== [{index}/{len(schedule)}] {label} "
              f"({len(transcript.turns)} turni) =====")
        mock_data = _snapshot_mock_data()
        if variant_bytes is not None:
            # The variant replaces the care plan for this session only; the
            # snapshot taken after placing it detects writes by the session,
            # and the original is put back afterwards.
            (MOCK_DATA_DIR / CARE_PLAN_FILE).write_bytes(variant_bytes)
        during = _snapshot_mock_data()
        try:
            result = replay_transcript(
                transcript,
                max_iterations=args.max_iterations,
                token_budget=(
                    args.max_tokens
                    or (args.max_tokens_per_turn * len(transcript.turns))
                    or None
                ),
                max_idle_cycles=args.max_idle_cycles or None,
            )
        finally:
            restored = _restore_mock_data(during)
            _restore_mock_data(mock_data)
        result["mock_data_restored"] = restored
        result["care_plan_variant"] = args.care_plan_variant
        if restored:
            print(f"  !! mock_data changed during the session, restored: "
                  f"{restored}")
        result["repetition"] = repetition
        result["transcript"] = label
        results.append(result)
        print(f"  -> {result['outcome']} in {result['wall_seconds']}s, "
              f"{result['canonicalization_count']} canonicalizzazioni, "
              f"{len(result['hard_enforcements'])} override")

        # Persist after every session: a session can run for many minutes,
        # and an interrupted run used to lose everything held in memory.
        _write_results(results, args.out, args.review_csv)

    output = _write_results(results, args.out, args.review_csv)

    print("\n=== SUMMARY ===")
    print(json.dumps(output["summary"], indent=2))
    print(f"\nrisultati: {args.out}\nrighe da rivedere a mano: {args.review_csv}")


if __name__ == "__main__":
    main()
