# nodes/action_canonicalizer_node.py

import json
import os

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from planning.canonical_actions import CANONICAL_ACTIONS

from utils.models import define_model
from utils.logs import log_graph_state, log_message
from utils.structured_logs import log_canonicalizer_decision
from colorama import Fore, Style

from graph.graph_init import GraphState


# Mapping a name onto a fixed vocabulary has one right answer, so sampling
# only adds variance: replaying the same session repeatedly showed the same
# activity mapped to different canonical actions from run to run.
CANONICALIZER_TEMPERATURE = 0.0

# Canonicalization is closed-vocabulary classification, which a small
# open-weight model does at least as well as a large one and far more
# cheaply, while the generative nodes need a stronger model. CANONICALIZER_MODEL
# (and optionally CANONICALIZER_PROVIDER) select it; unset, this node runs on
# whatever the rest of the graph uses.
CANONICALIZER_MODEL_ENV = "CANONICALIZER_MODEL"
CANONICALIZER_PROVIDER_ENV = "CANONICALIZER_PROVIDER"

# Mapping the same activity name twice in one session and getting two
# different answers is the layer's most frequent defect on real traffic:
# across the replayed corpus 48 of 220 activity names were mapped
# inconsistently within a single session, and in 26 of those the
# arbitrated status changed between blocked and executable as a result.
# The task is regenerated on every cycle, so the same name comes back over
# and over; remembering the first answer removes the flapping by
# construction, and removes the calls that produced it.
#
# Only known actions are remembered. An "unknown" is the fail-closed
# outcome, and caching it would freeze an activity out of the plan for the
# rest of the session on the strength of one bad answer; leaving it
# uncached lets the next cycle try again.
_MAPPING_CACHE: dict[str, str] = {}


def reset_canonicalization_cache() -> None:
    """Forget the session's mappings. Call it when a session starts."""
    _MAPPING_CACHE.clear()


def cache_enabled() -> bool:
    """Read the switch at call time, so evaluation can flip it per run."""
    return os.getenv("CANONICALIZATION_CACHE", "true").strip().lower() != "false"


def _cache_key(activity_name: str) -> str:
    return " ".join((activity_name or "").split()).lower()


def cached_action(activity_name: str) -> str | None:
    """The action this name was mapped to earlier in the session, if any."""
    if not cache_enabled():
        return None
    return _MAPPING_CACHE.get(_cache_key(activity_name))


def remember_action(activity_name: str, canonical_action: str) -> None:
    """Remember a known mapping; unknown is deliberately not remembered."""
    if not cache_enabled():
        return
    if canonical_action and canonical_action in CANONICAL_ACTIONS:
        _MAPPING_CACHE.setdefault(_cache_key(activity_name), canonical_action)


INITIAL_PROMPT = """
You are an action canonicalization module.

You receive a numbered list of patient activities. Map EACH one to exactly
one canonical action from the catalogue below.

Canonical actions (name: description):

{canonical_actions}

Rules:
- Choose the action whose description best matches the activity.
- If no action fits, use "unknown". Never invent a name.
- Answer with ONLY a JSON array of strings, one per activity, in the same
  order as the input. No explanations, no extra keys.

Example for three activities:
["take_medication", "reading_activity", "unknown"]
"""


def _catalogue() -> str:
    """Render the action catalogue with descriptions, not just names.

    Passing bare names made the model guess from the identifier alone; the
    description is what separates, say, reading_activity from
    social_interaction.

    The per-action alias lists are deliberately *not* included here. Adding
    them does map "Give journal" onto reading_activity, but on the 50-item
    stress test it cost more than it gained: the presence of an alias list
    makes the model match out-of-scope requests loosely, turning safe
    "unknown" answers into valid-but-wrong actions, which is the error
    class that bypasses the fail-closed policy. Aliases are therefore used
    only by the validator's consistency check, where a wider vocabulary
    suppresses false warnings without licensing a mapping.
    """
    return "\n".join(
        f"- {name}: {definition.get('description', name)}"
        for name, definition in CANONICAL_ACTIONS.items()
    )


def _parse_batch(raw_output, expected: int) -> list[str] | None:
    """Parse the model's JSON array, or return None if it is unusable."""
    if isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], dict):
        # Some providers wrap the answer in content blocks.
        raw_output = raw_output[0].get("text") or raw_output[0].get("content")
    if isinstance(raw_output, dict):
        raw_output = raw_output.get("text") or raw_output.get("content")
    if raw_output is None:
        return None

    text = str(raw_output).strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, list) or len(parsed) != expected:
        return None
    return [str(item).strip() for item in parsed]


def _canonicalize_one(model, activity_name: str, catalogue: str) -> str:
    """Fallback path: ask for a single activity when the batch is unusable."""
    prompt = PromptTemplate.from_template(INITIAL_PROMPT).invoke(
        {"canonical_actions": catalogue}
    )
    messages = [
        SystemMessage(content=prompt.to_string()),
        HumanMessage(content=f"1. {activity_name}"),
    ]
    parsed = _parse_batch(model.invoke(messages).content, expected=1)
    return parsed[0] if parsed else "unknown"


def action_canonicalizer_node(state: GraphState):
    print(Fore.GREEN + "[position: ACTION CANONICALIZER NODE]" + Style.RESET_ALL)

    log_graph_state(state, "action_canonicalizer_node")

    if state["updated_task"] is None:
        raise ValueError("Task is None in action_canonicalizer_node")

    activities = state["updated_task"]["activities"]
    if not activities:
        return {**state, "updated_task": state["updated_task"]}

    catalogue = _catalogue()

    # Activities whose name was already mapped in this session are not sent
    # to the model again: reusing the answer keeps the mapping stable
    # across cycles, and shortens the batch.
    pending = [
        activity for activity in activities
        if cached_action(activity["activity_name"]) is None
    ]

    answers: dict[int, str] = {}
    if pending:
        model = define_model(
            temperature=CANONICALIZER_TEMPERATURE,
            model=os.getenv(CANONICALIZER_MODEL_ENV) or None,
            provider=os.getenv(CANONICALIZER_PROVIDER_ENV) or None,
        )

        # One call for the whole batch instead of one per activity: the node
        # runs on every graph cycle, so per-activity calls dominated the
        # layer's cost.
        numbered = "\n".join(
            f"{index}. {activity['activity_name']}"
            for index, activity in enumerate(pending, start=1)
        )
        prompt = PromptTemplate.from_template(INITIAL_PROMPT).invoke(
            {"canonical_actions": catalogue}
        )
        messages = [
            SystemMessage(content=prompt.to_string()),
            HumanMessage(content=numbered),
        ]

        result = model.invoke(messages)
        batch = _parse_batch(result.content, expected=len(pending))

        if batch is None:
            log_message(
                "ACTION_CANONICALIZER - batch answer unusable, "
                "falling back to one call per activity"
            )
            print(
                Fore.YELLOW
                + "[WARNING] Canonicalizer batch answer unusable; "
                "falling back to per-activity calls."
                + Style.RESET_ALL
            )
            batch = [
                _canonicalize_one(model, activity["activity_name"], catalogue)
                for activity in pending
            ]
        answers = {id(activity): answer
                   for activity, answer in zip(pending, batch)}

    for activity in activities:

        activity_name = activity["activity_name"]

        remembered = cached_action(activity_name)
        if remembered is not None:
            activity["canonical_action"] = remembered
            log_canonicalizer_decision(
                activity_name=activity_name,
                raw_llm_output="",
                canonical_action=remembered,
                is_known=True,
                source="session_cache",
            )
            print(f"canonicalization: '{activity_name}' -> "
                  f"'{remembered}' (cached)")
            continue

        raw_answer = answers.get(id(activity), "unknown")
        canonical_action = str(raw_answer).strip()

        # --------------------------------------------------
        # VALIDAZIONE: fail-closed se l'LLM restituisce un
        # nome non presente in CANONICAL_ACTIONS.
        # L'attività viene marcata come "unknown" e il
        # Plan Validator la tratterà come non eseguibile.
        # --------------------------------------------------
        if canonical_action not in CANONICAL_ACTIONS:
            log_message(
                f"ACTION_CANONICALIZER - unknown canonical action "
                f"'{canonical_action}' for activity '{activity_name}', "
                f"marking as 'unknown'"
            )
            print(
                Fore.YELLOW
                + f"[WARNING] Canonicalizer returned unknown action "
                f"'{canonical_action}' for '{activity_name}'. "
                f"Marking as 'unknown' (fail-closed)."
                + Style.RESET_ALL
            )
            log_canonicalizer_decision(
                activity_name=activity_name,
                raw_llm_output=canonical_action,
                canonical_action="unknown",
                is_known=False,
            )
            canonical_action = "unknown"
        else:
            log_canonicalizer_decision(
                activity_name=activity_name,
                raw_llm_output=canonical_action,
                canonical_action=canonical_action,
                is_known=True,
            )

        print(f"canonicalization: '{activity_name}' -> '{canonical_action}'")

        activity["canonical_action"] = canonical_action
        remember_action(activity_name, canonical_action)

    return {
        **state,
        "updated_task": state["updated_task"]
    }
