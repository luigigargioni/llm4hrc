"""Regression tests for the fixes prompted by the conversation replay.

All of them run offline: no OpenAI call, no Ollama call, no tokens spent.

    python tests/test_replay_fixes.py
"""

import os
import sys

os.environ.setdefault("OPENAI_API_KEY", "offline-test")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import graph.graph_init  # noqa: F401,E402  - breaks the node import cycle
from datetime import datetime  # noqa: E402

from nodes.action_canonicalizer_node import (  # noqa: E402
    _catalogue,
    _parse_batch,
    cached_action,
    remember_action,
    reset_canonicalization_cache,
)
from nodes.plan_validator_node import (  # noqa: E402
    _check_activity_canonical_consistency,
    _parse_activity_time,
    plan_validator_node,
)
from nodes.next_steps_node import (  # noqa: E402
    SPEECH_ONLY_ACTIONS,
    evaluate_enforcement,
    resolve_declaration,
)
from nodes.world_state_updater_node import world_state_updater_node  # noqa: E402
from planning.canonical_actions import CANONICAL_ACTIONS  # noqa: E402

failures: list[str] = []


def check(label: str, got, expected) -> None:
    ok = got == expected
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: {got!r}")
    if not ok:
        failures.append(f"{label}: atteso {expected!r}, ottenuto {got!r}")


def _state(activities, world_state=None, perception="", now="10:00"):
    """Build the subset of GraphState the two nodes and the loggers read."""
    return {
        "task": None,
        "updated_task": {"activities": activities},
        "world_state": dict(world_state or {}),
        "situation_perception": perception,
        "progress_checking_output": None,
        "conversation_history": None,
        "robot_action": None,
        "vocal_answer": None,
        "patient_request": None,
        "robot_behaviour": "proactive",
        "doctor_specs": None,
        "doctor_specifications": None,
        "patient_id": None,
        "end": False,
        "fake_datetime": datetime.combine(
            datetime.today(), datetime.strptime(now, "%H:%M").time()
        ),
    }


print("\n--- catalogo del canonicalizer ---")
catalogue = _catalogue()
check("una riga per azione", len(catalogue.splitlines()), len(CANONICAL_ACTIONS))
check(
    "le descrizioni sono incluse",
    "Read a book or magazine" in catalogue,
    True,
)

print("\n--- parsing della risposta batch ---")
check("array valido", _parse_batch('["rest", "exercise"]', 2), ["rest", "exercise"])
check("fence markdown", _parse_batch('```json\n["rest"]\n```', 1), ["rest"])
check("lunghezza errata respinta", _parse_batch('["a"]', 2), None)
check("testo non JSON respinto", _parse_batch("nope", 1), None)

print("\n--- parsing degli orari ---")
for raw, expected in [
    ("09:00", "09:00"), ("9:00", "09:00"), ("9.00", "09:00"),
    ("0900", "09:00"), ("9AM", "09:00"), ("9PM", "21:00"),
    ("12AM", "00:00"), ("7h30", "07:30"),
]:
    parsed = _parse_activity_time(raw)
    check(f"{raw!r}", parsed.strftime("%H:%M") if parsed else None, expected)
for raw in ("25:99", "", "abc"):
    check(f"{raw!r} resta non parsabile", _parse_activity_time(raw), None)

print("\n--- consistency check con alias ---")
check(
    "mappatura corretta non segnalata",
    _check_activity_canonical_consistency("Take Vitamin D", "take_medication"),
    True,
)
check(
    "mappatura sbagliata segnalata",
    _check_activity_canonical_consistency("Give journal", "social_interaction"),
    False,
)
check(
    "alternativa corretta accettata",
    _check_activity_canonical_consistency("Give journal", "reading_activity"),
    True,
)

print("\n--- attivita' duplicate ---")
state = _state([
    {"activity_name": "Read a book", "canonical_action": "reading_activity",
     "activity_time": "10:00", "done": False, "critical": False},
    {"activity_name": "Read a book", "canonical_action": "reading_activity",
     "activity_time": "10:00", "done": False, "critical": False},
], world_state={"patient_awake": True, "book_available": True})
plan_validator_node(state)
first, second = state["updated_task"]["activities"]
check("la prima resta eseguibile", first["executable"], True)
check("la seconda e' bloccata", second["executable"], False)
check(
    "motivo registrato",
    "duplicate_activity" in second["not_executable_reasons"],
    True,
)

state = _state([
    {"activity_name": "Take Vitamin D", "canonical_action": "take_medication",
     "activity_time": "09:00", "done": False, "critical": False},
    {"activity_name": "Take Vitamin D", "canonical_action": "take_medication",
     "activity_time": "21:00", "done": False, "critical": False},
], world_state={"patient_awake": True, "medication_available": True,
                "correct_time": True}, now="09:00")
plan_validator_node(state)
check(
    "stesso nome a orari diversi non e' duplicato",
    "duplicate_activity" in state["updated_task"]["activities"][1]
    ["not_executable_reasons"],
    False,
)

print("\n--- la percezione prevale sui fatti dedotti ---")
state = _state(
    [{"activity_name": "Prepare patient for sleep",
      "canonical_action": "prepare_for_sleep", "done": True}],
    world_state={"patient_awake": True, "patient_tired": True},
    perception="Patient is speaking",
)
world_state_updater_node(state)
check("prepare_for_sleep ha dedotto il sonno",
      CANONICAL_ACTIONS["prepare_for_sleep"]["effects"], ["patient_sleeping"])
check("ma l'osservazione vince: sveglia", state["world_state"]["patient_awake"], True)
check("e non dorme", state["world_state"]["patient_sleeping"], False)

state = _state(
    [{"activity_name": "Prepare patient for sleep",
      "canonical_action": "prepare_for_sleep", "done": True}],
    world_state={"patient_awake": True, "patient_tired": True},
    perception="Anna is sleeping",
)
world_state_updater_node(state)
check(
    "percezione concorde: resta addormentata",
    state["world_state"]["patient_sleeping"],
    True,
)

print("\n--- memoria delle mappature per sessione ---")
reset_canonicalization_cache()
check("nome mai visto", cached_action("Take Vitamin D"), None)

remember_action("Take Vitamin D", "take_medication")
check("mappatura ricordata", cached_action("Take Vitamin D"),
      "take_medication")
check("chiave insensibile a maiuscole e spazi",
      cached_action("  take   vitamin d "), "take_medication")

# La prima risposta nota vince: lo sfarfallio fra cicli e' cio' che si
# vuole togliere, quindi un secondo parere diverso non sovrascrive.
remember_action("Take Vitamin D", "take_antibiotic")
check("la prima mappatura resta", cached_action("Take Vitamin D"),
      "take_medication")

# "unknown" non si ricorda: e' l'esito fail-closed, e congelarlo
# escluderebbe l'attivita' per tutta la sessione su una sola risposta.
remember_action("Give journal", "unknown")
check("unknown non viene ricordato", cached_action("Give journal"), None)

remember_action("Garden therapy", "not_an_action")
check("azione inesistente non ricordata",
      cached_action("Garden therapy"), None)

reset_canonicalization_cache()
check("il reset dimentica la sessione",
      cached_action("Take Vitamin D"), None)

os.environ["CANONICALIZATION_CACHE"] = "false"
remember_action("Take Vitamin D", "take_medication")
check("con la cache spenta non si ricorda nulla",
      cached_action("Take Vitamin D"), None)
os.environ.pop("CANONICALIZATION_CACHE")

print("\n--- enforcement sull'intento dichiarato ---")

BLOCKED = ["walk in the park"]
EXECUTABLE = ["have lunch", "read a book"]
REFUSAL = (
    "I'm sorry, we cannot go for the walk in the park now: "
    "you have not had lunch yet."
)
PROPOSAL = (
    "The weather is lovely, all conditions are good, "
    "let's go for our walk in the park now."
)


def decide(action, answer, declared, present=True, mode="declared",
           blocked=None, executable=None):
    return evaluate_enforcement(
        robot_action=action,
        vocal_answer=answer,
        proposed_activity=declared,
        declaration_present=present,
        blocked_names=blocked if blocked is not None else BLOCKED,
        executable_names=executable if executable is not None else EXECUTABLE,
        mode=mode,
    )


# The three corpus false positives: a sound refusal under an invented
# action label, overridden because the label was not on the exempt list.
legacy_fp = decide("EXECUTE_DECLINE", REFUSAL, None, mode="legacy")
check("legacy: rifiuto corretto sovrascritto", legacy_fp["override"], True)

declared_fp = decide("EXECUTE_DECLINE", REFUSAL, None)
check("dichiarato: rifiuto corretto rispettato",
      declared_fp["override"], False)
check("  motivo", declared_fp["reason"], "declared_no_proposal")
check("  l'attivita' resta registrata",
      declared_fp["mentioned_activities"], ["walk in the park"])

# The intervention that must survive: the model proposes the blocked walk.
tp = decide("START_WALK", PROPOSAL, "Walk in the park")
check("dichiarato: proposta bloccata intercettata", tp["override"], True)
check("  regola", tp["match_mode"], "declared")
check("  attivita'", tp["violated_activities"], ["walk in the park"])

# A declaration naming an allowed activity is trusted, even when the text
# also names the blocked one to explain the swap.
swap = decide("SPEAKING", "Instead of the walk in the park, let's have lunch.",
              "Have lunch")
check("dichiarato: alternativa eseguibile", swap["override"], False)
check("  motivo", swap["reason"], "declared_executable")

# No declaration at all: the name-matching backstop decides, exactly as
# the legacy rule would, so dropping the field cannot buy a bypass.
missing = decide("START_WALK", PROPOSAL, None, present=False)
check("campo assente: interviene il backstop", missing["override"], True)
check("  regola", missing["match_mode"], "fallback_name_match")
missing_speech = decide("SPEAKING", REFUSAL, None, present=False)
check("campo assente ma azione di sola parola",
      missing_speech["override"], False)

# Ambiguous names must not be guessed.
check("il nome esatto vince sull'omonimia parziale",
      resolve_declaration("stretching", ["stretching", "stretching exercises"]),
      "stretching")
check("nome esatto risolto",
      resolve_declaration("stretching exercises",
                          ["stretching", "stretching exercises"]),
      "stretching exercises")
check("qualificatore in piu' risolto",
      resolve_declaration("walk in the park (afternoon)",
                          ["walk in the park", "have lunch"]),
      "walk in the park")
check("parafrasi non risolta, non indovinata",
      resolve_declaration("the walk", ["walk in the park", "have lunch"]),
      None)
# "stretching" fits an activity on each list: the declaration says
# nothing, so the backstop decides instead of a coin toss.
ambiguous = decide("START_ACTIVITY", "Let's do the stretching exercises now.",
                   "Stretching",
                   blocked=["stretching exercises"],
                   executable=["stretching routine"])
check("dichiarazione ambigua: decide il backstop",
      ambiguous["match_mode"], "fallback_name_match")

print("\n--- il backstop conserva l'esenzione di sola parola ---")
check("SPEAKING esente", "SPEAKING" in SPEECH_ONLY_ACTIONS, True)
check("NO_ACTION esente", "NO_ACTION" in SPEECH_ONLY_ACTIONS, True)
check(
    "un'azione esecutiva resta soggetta",
    "PREPARE_PARACETAMOL" in SPEECH_ONLY_ACTIONS,
    False,
)

print("\n" + "=" * 56)
if failures:
    print(f"FALLITI: {len(failures)}")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)
print("Tutti i controlli superati.")
