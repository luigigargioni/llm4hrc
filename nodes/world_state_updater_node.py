# Iterates over all activities, checks those that are executable and simultaneously DONE,
#
# Each completed activity has its effects applied once, the first cycle it is
# seen done with a known action. The task is regenerated every cycle, and
# re-applying every done activity's effects on every cycle meant a fact about
# the patient's condition could never lapse: a morning walk kept the patient
# tired for the rest of the day. Facts in LAPSING_FACTS now become false again
# a fixed time after they were last set.

import re
from datetime import datetime, timedelta
from utils.logs import log_graph_state, log_message
from utils.structured_logs import log_world_state_update, log_structured
from typing import TYPE_CHECKING
from colorama import Fore, Style
from planning.canonical_actions import CANONICAL_ACTIONS

if TYPE_CHECKING:
    from graph.graph_init import GraphState

# Mutually exclusive fact pairs: when one is set True, the other is set False
MUTUALLY_EXCLUSIVE = {
    "patient_awake": "patient_sleeping",
    "patient_tired": "patient_not_tired",
    "patient_sad": "patient_emotionally_stable",
}

# Facts the robot can observe directly. The world state is otherwise built
# by *inferring* effects from completed activities, which can contradict
# what is actually happening: a completed "prepare for sleep" sets
# patient_sleeping, and the mutual-exclusivity rule then clears
# patient_awake even while the patient is talking to the robot. An observed
# fact is stronger evidence than an inferred one, so perception is applied
# last and wins.
PERCEPTION_FACTS = {
    "patient_awake": (
        "is speaking", "is awake", "awakening", "woke up", "is talking",
        "just woke", "is answering",
    ),
    "patient_sleeping": (
        "is sleeping", "is asleep", "fell asleep", "falls asleep",
        "sta dormendo", "dorme",
    ),
    "patient_tired": ("is tired", "looks tired", "e stanco", "e stanca"),
}


# Facts about the patient's condition, and how long each stays true after it
# was last set (in simulated time). Completion facts are not listed: once an
# activity is done it stays done.
LAPSING_FACTS = {
    "patient_tired": timedelta(hours=2),
    "patient_fed": timedelta(hours=4),
    "after_meal": timedelta(hours=2),
}


# What the patient says about their own tiredness is an observation, and the
# only one the recorded sessions contain: no perception there describes it,
# while patients say "I'm a little bit tired" or "un po' stanca". "Tired of"
# is about something else, and a negation within a few words ("not tired",
# "non sono stanca", "don't feel tired") reports the opposite.
_TIRED = re.compile(r"\b(tired|stanc[ao])\b(?!\s+(of|di)\b)")
_NEGATED_TIRED = re.compile(
    r"(\bnot\b|n't\b|\bnon\b|\bnever\b|\bno longer\b)(\W+\w+){0,3}?\W+"
    r"(tired|stanc[ao])\b"
)


def _patient_report(text: str) -> str | None:
    """The tiredness fact the patient's own words report, if any."""
    text = (text or "").lower()
    if not _TIRED.search(text):
        return None
    return "patient_not_tired" if _NEGATED_TIRED.search(text) else (
        "patient_tired")


def _current_time(state) -> datetime:
    return state.get("fake_datetime") or datetime.now()


def _applied_key(activity: dict, canonical_action: str) -> str:
    """Identifies one completed activity across regenerated tasks."""
    name = str(activity.get("activity_name", "")).strip().lower()
    time = str(activity.get("activity_time") or "").strip()
    return f"{name}|{time}|{canonical_action}"


def _mark_true(world_state: dict, true_since: dict, fact: str, now: datetime) -> list:
    """Set a fact true, record when, and enforce mutual exclusivity."""
    world_state[fact] = True
    if fact in LAPSING_FACTS:
        true_since[fact] = now.isoformat()
    return _apply_mutual_exclusivity(world_state, fact)


def _lapse_facts(world_state: dict, true_since: dict, now: datetime) -> list:
    """Make lapsing facts false once their duration has passed."""
    lapsed = []
    for fact, duration in LAPSING_FACTS.items():
        since = true_since.get(fact)
        if not world_state.get(fact, False) or since is None:
            continue
        set_at = datetime.fromisoformat(since)
        if now - set_at < duration:
            continue
        world_state[fact] = False
        del true_since[fact]
        # A lapsed tiredness makes the patient not tired again; the other
        # lapsing facts have no counterpart.
        restored = MUTUALLY_EXCLUSIVE.get(fact)
        if restored:
            world_state[restored] = True
        lapsed.append(fact)
        log_structured({
            "node": "world_state_updater",
            "decision_type": "fact_lapsed",
            "fact": fact,
            "true_since": since,
            "current_time": now.isoformat(),
            "duration_minutes": int(duration.total_seconds() // 60),
            "restored_fact": restored,
            "message": f"{fact} set at {since} lapsed after {duration}",
        })
    return lapsed


def _apply_mutual_exclusivity(world_state: dict, fact: str) -> list:
    """When a fact is set True, set its opposite False. Returns changed facts."""
    changed = []
    opposite = MUTUALLY_EXCLUSIVE.get(fact)
    if opposite and world_state.get(opposite, False):
        world_state[opposite] = False
        changed.append(opposite)
    # Also check reverse mapping
    for key, val in MUTUALLY_EXCLUSIVE.items():
        if val == fact and world_state.get(key, False):
            world_state[key] = False
            changed.append(key)
    return changed



def world_state_updater_node(state: "GraphState"):
    print(Fore.GREEN + "[position: WORLD STATE UPDATER NODE]" + Style.RESET_ALL)

    log_graph_state(state, "world_state_updater_node")

    if state["updated_task"] is None:
        return state

    world_state = state["world_state"]
    meta = state.get("world_state_meta") or {}
    applied = set(meta.get("applied", []))
    true_since = dict(meta.get("true_since", {}))
    now = _current_time(state)

    activities = state["updated_task"]["activities"]

    modifications = []  # for debug, list of world state changes to print

    for activity in activities:

        # the action must be completed
        if not activity.get("done", False):
            continue

        canonical_action = activity.get(
            "canonical_action"
        )

        if not canonical_action:
            continue

        action_def = CANONICAL_ACTIONS.get(
            canonical_action
        )

        if not action_def:
            continue

        # Effects are applied once per completed activity; re-applying them
        # every cycle would renew facts that are meant to lapse.
        key = _applied_key(activity, canonical_action)
        if key in applied:
            continue
        applied.add(key)

        effects = action_def.get(
            "effects",
            []
        )

        # Detect idempotent re-application: all effects already True
        already_true = [e for e in effects if world_state.get(e, False)]
        if effects and len(already_true) == len(effects):
            log_structured({
                "node": "world_state_updater",
                "decision_type": "idempotent_reapplication",
                "activity_name": activity.get("activity_name", ""),
                "canonical_action": canonical_action,
                "effects": effects,
                "message": "All effects already True; re-applying has no observable effect",
            })

        world_state_before = dict(world_state)

        for effect in effects:
            modifications.append(effect)
            # Enforce mutual exclusivity: set opposite facts to False
            exclusivity_changes = _mark_true(
                world_state, true_since, effect, now
            )
            modifications.extend(exclusivity_changes)
            # Log each mutual-exclusivity correction so it is auditable
            for changed_fact in exclusivity_changes:
                log_structured({
                    "node": "world_state_updater",
                    "decision_type": "mutual_exclusivity_correction",
                    "activity_name": activity.get("activity_name", ""),
                    "canonical_action": canonical_action,
                    "trigger_effect": effect,
                    "corrected_fact": changed_fact,
                    "message": f"Setting {effect}=True auto-corrected {changed_fact}=False (mutual exclusivity)",
                })

        log_world_state_update(
            activity_name=activity.get("activity_name", ""),
            canonical_action=canonical_action,
            effects_applied=effects,
            world_state_before=world_state_before,
            world_state_after=world_state,
        )

    # Facts about the patient's condition lapse after a fixed time.
    modifications.extend(_lapse_facts(world_state, true_since, now))

    # The patient's own report overrides the inferred facts, and is itself
    # overridden by a perception below.
    reported = _patient_report(state.get("patient_request"))
    if reported:
        was_true = world_state.get(reported, False)
        corrections = _mark_true(world_state, true_since, reported, now)
        if not was_true:
            modifications.append(reported)
            modifications.extend(corrections)
            log_structured({
                "node": "world_state_updater",
                "decision_type": "patient_report",
                "fact": reported,
                "patient_request": state.get("patient_request"),
                "corrected_facts": corrections,
                "message": (
                    f"The patient's own words set {reported}=True, "
                    f"overriding the fact inferred from completed activities"
                ),
            })

    # Apply perception last so an observation overrides an inferred fact.
    perception = (state.get("situation_perception") or "").lower()
    for fact, markers in PERCEPTION_FACTS.items():
        if not any(marker in perception for marker in markers):
            continue
        if world_state.get(fact, False):
            # Observed again: the lapse counts from this observation.
            if fact in LAPSING_FACTS:
                true_since[fact] = now.isoformat()
            continue
        modifications.append(fact)
        corrections = _mark_true(world_state, true_since, fact, now)
        modifications.extend(corrections)
        log_structured({
            "node": "world_state_updater",
            "decision_type": "perception_override",
            "fact": fact,
            "situation_perception": state.get("situation_perception"),
            "corrected_facts": corrections,
            "message": (
                f"Observed situation set {fact}=True, overriding the fact "
                f"inferred from completed activities"
            ),
        })

    state["world_state"] = world_state
    state["world_state_meta"] = {
        "applied": sorted(applied),
        "true_since": true_since,
    }

    print(Fore.LIGHTBLUE_EX + f"WS set true: {modifications}" + Style.RESET_ALL)

    return state