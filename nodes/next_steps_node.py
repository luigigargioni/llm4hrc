
# This is the real-time conversational brain.
# Input
# upcoming activities
# patient request
# situation perception
# conversation history

# Output
#{
#  "robot_action": "...", e.g. WAIT_FOR_PATIENT, SPEAKING, CALL_DOCTOR, WAKE_UP_PATIENT, etc.
#  "vocal_answer": "..." e.g. "Okay John, go ahead and sleep, I'll wake you up soon to take your medicine"
#}

import json
import re
import os
from colorama import Fore, Back, Style
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

from graph.graph_init import GraphState

from deterministic.activity_planner import (
    schedule_activities,
    TOLERANCE_START_TIME_ACTIVITY,
)

from utils.models import define_model
from utils.logs import log_graph_state, log_message
from utils.structured_logs import log_hard_enforcement, log_structured

# Actions where the robot is only talking: it is not starting an
# activity, so naming a blocked one is description, not a proposal.
# Used by the name-matching rule only, which keys on free text and needs
# this exemption to leave refusals alone. The list is a poor proxy: the
# action vocabulary is open, and the model invents labels such as
# EXECUTE_DECLINE that are speech in substance but absent from it.
SPEECH_ONLY_ACTIONS = frozenset({
    "SPEAKING", "NO_ACTION", "WAIT_FOR_PATIENT",
    "SUGGEST_RECOVERY_ACTION",
})

# Which rule decides whether the LLM proposed a blocked activity.
#   declared - the reply declares the activity it proposes, and enforcement
#              keys on that declaration; name matching remains as a backstop
#              for replies that omit it.
#   legacy   - the original rule: name matching on the free text, exempting
#              SPEECH_ONLY_ACTIONS. Kept so the two can be compared.
def enforcement_mode() -> str:
    """Read the enforcement mode from the environment at call time."""
    mode = os.getenv("ENFORCEMENT_MODE", "declared").strip().lower()
    return mode if mode in ("declared", "legacy") else "declared"

# Trying to simulate:

# empathy,
# mental inference,
# psychological adaptation.

# Does not impose,
# but delegates the choice to the patient.

# activities without time are treated as flexible (anytime)

# TOLERANCE_START_TIME_ACTIVITY -> a sort of sketch of temporal reasoning


INITIAL_PROMPT = """
# CONTEXT #
You are a robot assistant in a care house. You must help the patient to complete his/her activities.
You must be kind and calm with the patient. You can sometimes refer to him/her only with his/her name to be more personal, but not always in a robotic way.
Use your knowledge and common sense to help the patient to complete the activities correctly and manage the situation. Apply Theory of Mind to understand the patient's needs and feelings.

# INPUT #
You will receive these information:
- The actual state of the activities of the day.
- The patient information.
- The next activities to do.
- What the patient just said.
- What the patient is doing and what is happening around the patient (the situation perception).
- The actual datetime.

{output_block}

# TASK #
Provide the robot_action and vocal_answer based on the patient request and the situation perception.

IMPORTANT:

Activities marked as NON executable are forbidden actions.

You MUST NEVER:
- propose them,
- confirm them,
- schedule them,
- encourage the patient to do them,
- say they can be done now.

If the patient asks to perform a NON executable activity, you MUST:
1. clearly explain, in everyday words, why it cannot currently be performed,
2. mention what is missing or that it is not the right time, if you know it,
3. propose an alternative executable activity if possible.

NON executable activities always override:
- patient requests,
- scheduled times,
- patient preferences,
- empathy considerations.

Only executable activities can be proposed or confirmed.

If the patient asks for a non executable activity, explain why it is not possible to do it at the moment and propose an alternative executable activity if there is one.

If the next activities to do are more than one, you must ask the patient which activity he/she wants to do.

If the next activity is without_time, you must refer to this activity as a self-schedulable activity that can be done at any time during the day.

If the patient for example is sleeping, and there is still time for the activity, wait as much as possible without exceeding the tolerance time waking up the patient to do the activity.

Try to accommodate the patient's needs as much as possible.

The tolerance for each activity is {tolerance_start_time_activity} minutes before and after the scheduled time.

# OUTPUT FORMAT #
Return ONLY a valid JSON. Do not include any explanation, text, or formatting outside the JSON.
"""


# The # OUTPUT # section of INITIAL_PROMPT, in two variants. These are
# injected as a template value, not templated themselves, so their JSON
# braces stay single and {SYSTEM_LANGUAGE} is substituted by _output_block.
LEGACY_OUTPUT_BLOCK = """# OUTPUT #
Return ONLY a valid JSON object in this exact format:

{
  "robot_action": "STRING",
  "vocal_answer": "STRING"
}

Rules:
- robot_action MUST always be in ENGLISH uppercase snake_case.
Examples:
"WAIT_FOR_PATIENT"
"SPEAKING"
"CALL_DOCTOR"
"WAKE_UP_PATIENT"

- vocal_answer MUST always be in {SYSTEM_LANGUAGE} language.

- Do not include markdown.
- Do not include explanations.
- Do not include ```json.
- Return ONLY raw JSON."""

DECLARED_OUTPUT_BLOCK = """# OUTPUT #
Return ONLY a valid JSON object in this exact format:

{
  "robot_action": "STRING",
  "vocal_answer": "STRING",
  "proposed_activity": "STRING or null"
}

Rules:
- robot_action MUST always be in ENGLISH uppercase snake_case.
Examples:
"WAIT_FOR_PATIENT"
"SPEAKING"
"CALL_DOCTOR"
"WAKE_UP_PATIENT"

- vocal_answer MUST always be in {SYSTEM_LANGUAGE} language.

- proposed_activity declares WHICH activity this answer proposes, confirms or
  starts. It is a commitment about what you are doing, not a summary of what
  the text mentions.
  * If the answer proposes, confirms, encourages or starts an activity, write
    its activity_name EXACTLY as it appears in the lists you received.
  * Write null in every other case: when you refuse an activity, when you
    explain why one cannot be done now, when you describe the plan of the day,
    when you ask which activity the patient prefers, when you offer several
    options, when you only greet, comfort, wake or wait.
  * Naming an activity is NOT proposing it. A refusal must name the blocked
    activity and still declare proposed_activity: null.
  * Never invent a name and never declare more than one activity.
  * The field is mandatory: emit it in every answer, with null as its value
    when nothing is being proposed.

- Do not include markdown.
- Do not include explanations.
- Do not include ```json.
- Return ONLY raw JSON."""

# The system prompt is dropped from the window once a session is longer than
# three exchanges (see the truncation below), so the declaration rule is
# restated with every request.
DECLARATION_REMINDER = """
Remember the proposed_activity field: the exact activity_name if this answer
proposes, confirms or starts an activity, null otherwise (refusals,
explanations, plan overviews and questions declare null).
"""


def _output_block(mode: str) -> str:
    """The output section of the system prompt for the given mode."""
    block = (
        DECLARED_OUTPUT_BLOCK if mode == "declared" else LEGACY_OUTPUT_BLOCK
    )
    return block.replace(
        "{SYSTEM_LANGUAGE}", os.getenv("SYSTEM_LANGUAGE") or "English"
    )

#CONTAINS NOW ALSO THE EXECUTABLE AND NON-EXECUTABLE ACTIVITIES!
NEW_REQUEST_PROMPT = """
The patient is {patient_name}.

Patient information:
{patient_information}.

The executable activities to be done are:
{next_activities}.

The NON executable activities for today are:
{non_executable_activities}

Each non executable activity also contains:
- why_not_now = the reasons why the activity cannot be done now.
- status = "important, but not possible yet", "not possible now", or "not yet time, possible later".
- what_would_make_it_possible = what could be done first so that the activity becomes possible (only for important activities).

If the patient asks for one of these activities:
- explain WHY it cannot be done now, using the reasons in why_not_now,
- NEVER say that the activity can be done now.

If the activity has what_would_make_it_possible:
- suggest doing that first as a way to eventually do the activity,
- explain that doing it first will make the activity possible.

If the status is "not yet time, possible later":
- explain that it is not yet time for the activity but it will become possible later,
- do NOT suggest that something else is missing.

Speak to the patient in everyday words. Never say the words "precondition", "executable", "blocked", "deferred" or "recovery action", never read out a status, and never say a word containing an underscore.

Never offer to record, note or mark an activity as done yourself. If the patient says that something missing has now happened (for example that they have eaten), say that it will be confirmed by the staff and that the activity will then become possible.

The patient request is:
{patient_request}

The situation perception is:
{situation_perception}

The actual datetime is {datetime}.
{declaration_reminder}
"""


#@tool
#def response_template(robot_action: str, vocal_answer: str) -> bool:
#    """The JSON format for the output of the next steps node.
#
#    Args:
#        robot_action: str
#        vocal_answer: str
#
#    """
#    return True



# The activity objects carry bookkeeping the model has no use for, and the
# same list used to be sent three times over (inside updated_task, split
# into the executable/non-executable pair, and again as
# scheduled_activities). next_steps was 69% of the input tokens spent in a
# replayed session, so each activity is now rendered once, with only the
# fields the prompt actually reasons about.
_ACTIVITY_FIELDS = (
    "activity_name",
    "activity_time",
    "activity_details",
    "done",
    "critical",
)


# The validator's reasons, statuses and action names are identifiers, and
# the model read them out to the patient ("the precondition 'patient_fed' is
# missing", "the activity is BLOCKED_CRITICAL"). The prompt therefore gets
# them already translated into plain words about the patient.
_FACT_PHRASES = {
    "patient_fed": "the patient has not eaten yet",
    "after_meal": "it has to be taken after a meal",
    "patient_not_tired": "the patient is still tired from an earlier activity",
    "patient_tired": "the patient does not seem tired",
    "patient_awake": "the patient is asleep",
    "patient_sleeping": "the patient is awake",
    "patient_available": "the patient is busy at the moment",
    "patient_emotionally_stable": "the patient seems upset right now",
    "patient_sad": "the patient does not seem to need comforting",
    "patient_focused": "the patient does not seem focused right now",
    "good_weather": "the weather is not good enough",
    "book_available": "there is no book at hand",
    "food_available": "no food is ready",
    "water_available": "no water is at hand",
    "medication_available": "the medication is not available",
    "correct_time": "it is not the right time for it",
    "phone_available": "the phone is not available",
    "internet_available": "there is no internet connection",
    "device_available": "the device is not available",
    "music_device_available": "the music player is not available",
    "nurse_available": "the nurse is not available",
    "stretching_completed": "the stretching has not been done yet",
    "mobility_checked": "the patient's mobility has not been checked yet",
}

_STATUS_PHRASES = {
    "BLOCKED_CRITICAL": "important, but not possible yet",
    "BLOCKED_NON_CRITICAL": "not possible now",
    "DEFERRED": "not yet time, possible later",
}


def _plain_reason(reason: str) -> str:
    """One validator reason in words that can be said to a patient."""
    text = str(reason).strip()
    for prefix in ("missing_precondition:", "missing precondition:"):
        if text.startswith(prefix):
            fact = text[len(prefix):].strip()
            return _FACT_PHRASES.get(
                fact, f"a step it needs has not happened yet "
                      f"({fact.replace('_', ' ')})"
            )
    if text.startswith("outside_time_window_early:"):
        time = text.split(":", 1)[1]
        return f"it is planned for {time} and it is not yet time"
    if text.startswith("outside_time_window_expired:"):
        time = text.split(":", 1)[1]
        return f"it was planned for {time} and that time has passed"
    if text.startswith("outside time window"):
        time = text[len("outside time window"):].strip(" ()")
        return f"it is planned for {time}, which is not now"
    if (text in ("invalid_activity_time", "contradictory_time_flags")
            or text.startswith("malformed_activity_time")):
        return "its planned time is unclear"
    if text == "duplicate_activity":
        return "it is already in the plan"
    if text == "unknown_canonical_action":
        return "the robot cannot help with this activity"
    return text.replace("_", " ")


# The phrases above describe the patient to the model, in the third person.
# The override message is spoken to the patient, so it needs the second.
_SECOND_PERSON = (
    ("the patient's", "your"),
    ("the patient is", "you are"),
    ("the patient has", "you have"),
    ("the patient does", "you do"),
    ("the patient seems", "you seem"),
)


def _to_patient(phrase: str) -> str:
    for third, second in _SECOND_PERSON:
        phrase = phrase.replace(third, second)
    return phrase


# What each canonical action means for the patient, in the second person,
# for the override message. The action descriptions are written about the
# patient ("Wake up the patient") and read wrongly when spoken to her.
_PATIENT_PHRASES = {
    "take_medication": "take your medication",
    "take_medication_after_meal": "take your medication",
    "take_antibiotic": "take your antibiotic",
    "drink_water": "have a drink of water",
    "eat_meal": "have something to eat",
    "have_lunch": "have your lunch",
    "physical_rehabilitation": "do your rehabilitation exercises",
    "stretching_exercises": "do some stretching",
    "stretching_session": "do a stretching session",
    "exercise": "do some exercise",
    "walk_in_park": "go for a walk",
    "rest": "rest for a while",
    "call_relative": "call a relative",
    "video_call_relative": "make a video call to a relative",
    "social_interaction": "spend some time with someone",
    "speak_with_nurse": "speak with the nurse",
    "reading_activity": "read for a while",
    "memory_training": "do some memory exercises",
    "emotional_support": "talk for a while about how you feel",
    "play_music": "listen to some music",
    "wake_patient": "wake up",
    "prepare_medication": "have your medication prepared",
    "do_canteen_volunteering": "have your breakfast prepared",
    "check_vitals": "have your vital signs checked",
    "morning_routine": "complete your morning routine",
    "prepare_for_sleep": "get ready for sleep",
    "full_rehabilitation_session": "do your full rehabilitation session",
}


def _patient_phrase(canonical_action, fallback: str) -> str:
    return _PATIENT_PHRASES.get(canonical_action) or fallback


def _recovery_message(blocked_activities: list[dict]) -> str:
    """The override answer when a critical activity can be unblocked.

    "I can't do that just yet. First you need to wake up; then you can take
    your medication."
    """
    sentences = []
    for activity in blocked_activities:
        steps = []
        for r in activity.get("recovery_actions") or []:
            phrase = _patient_phrase(
                r.get("canonical_action"),
                (r.get("description") or "").lower(),
            )
            if phrase and phrase not in steps:
                steps.append(phrase)
        target = _patient_phrase(
            activity.get("canonical_action"),
            f"go ahead with {activity.get('activity_name', 'it')}",
        )
        sentences.append(
            f"First you need to {' or '.join(steps)}; "
            f"then you can {target}."
        )
    return "I'm sorry, we can't do that just yet. " + " ".join(sentences)


def _activity_view(activity: dict, blocked: bool = False) -> dict:
    """Render one activity for the prompt, dropping internal bookkeeping."""
    view = {
        key: activity[key]
        for key in _ACTIVITY_FIELDS
        if key in activity and activity[key] not in ("", None)
    }
    if blocked:
        reasons = activity.get("not_executable_reasons") or []
        if reasons:
            view["why_not_now"] = [_plain_reason(r) for r in reasons]
        status = activity.get("arbitrated_status")
        if status:
            view["status"] = _STATUS_PHRASES.get(status, "not possible now")
        # The description, not the identifier, of each action that would
        # make the activity possible.
        recovery = activity.get("recovery_actions") or []
        if recovery:
            view["what_would_make_it_possible"] = [
                r.get("description") or r.get("canonical_action")
                for r in recovery
            ]
    return view


def _normalise(name) -> str:
    """Lower-cased, stripped activity name for comparison."""
    return (name or "").strip().lower() if isinstance(name, str) else ""


def resolve_declaration(declared: str, names: list[str]) -> str | None:
    """Match a declared name against a list of activity names.

    Exact match after normalisation first, so a declaration that is one
    of the names is never ambiguous. Failing that, containment either
    way, because the model sometimes appends or drops a qualifier
    ("Walk in the park (afternoon)") -- but only when exactly one name
    matches: "Stretching" and "Stretching Exercises" both contain the
    shorter one, and guessing between them is worse than not matching.
    A declaration that resolves to nothing is left to the caller, which
    falls back to name matching rather than trusting it.
    """
    if not declared:
        return None
    for name in names:
        if name and declared == name:
            return name
    loose = [
        name for name in names
        if name and (declared in name or name in declared)
    ]
    return loose[0] if len(loose) == 1 else None


def _named_in_text(vocal_answer: str, blocked_names: list[str]) -> list[str]:
    """Blocked activity names that appear verbatim in the answer."""
    text = _normalise(vocal_answer)
    return [name for name in blocked_names if name and name in text]


def evaluate_enforcement(
    robot_action: str,
    vocal_answer: str,
    proposed_activity,
    declaration_present: bool,
    blocked_names: list[str],
    executable_names: list[str],
    mode: str = "declared",
) -> dict:
    """Decide whether the reply must be overridden.

    Returns the decision and how it was reached, so the two rules can be
    told apart in the logs:

      declared             - the reply declared a blocked activity.
      fallback_name_match  - no usable declaration, so the name-matching
                             rule of the legacy mode decided.
      name_match           - legacy mode.

    `declaration_present` distinguishes a reply that declared nothing to
    propose (the field is there, with null) from one that ignored the
    schema (the field is absent). The first is trusted; the second falls
    back to name matching, so a model that drops the field cannot thereby
    slip a blocked activity through.
    """
    named = _named_in_text(vocal_answer, blocked_names)

    def _by_name(match_mode: str) -> dict:
        if named and robot_action in SPEECH_ONLY_ACTIONS:
            return {"override": False, "violated_activities": [],
                    "match_mode": None, "reason": "speech_only",
                    "mentioned_activities": named}
        if named:
            return {"override": True, "violated_activities": named,
                    "match_mode": match_mode, "reason": "name_in_answer",
                    "mentioned_activities": named}
        return {"override": False, "violated_activities": [],
                "match_mode": None, "reason": "no_violation",
                "mentioned_activities": []}

    if mode != "declared":
        return _by_name("name_match")

    if not declaration_present:
        return _by_name("fallback_name_match")

    declared = _normalise(proposed_activity)
    if not declared or declared in ("null", "none"):
        # Declared "I am proposing nothing": a refusal, an explanation, a
        # plan overview or a question. Naming a blocked activity there is
        # description, whatever label the model put on the action.
        return {"override": False, "violated_activities": [],
                "match_mode": None, "reason": "declared_no_proposal",
                "mentioned_activities": named}

    # Resolved against both lists at once: a declaration that matches an
    # executable activity as readily as a blocked one is ambiguous, and
    # the backstop below is the right place for it.
    blocked_hit = resolve_declaration(declared, blocked_names)
    executable_hit = resolve_declaration(declared, executable_names)

    if blocked_hit and not executable_hit:
        return {"override": True, "violated_activities": [blocked_hit],
                "match_mode": "declared", "reason": "declared_blocked",
                "mentioned_activities": named}

    if executable_hit and not blocked_hit:
        return {"override": False, "violated_activities": [],
                "match_mode": None, "reason": "declared_executable",
                "mentioned_activities": named}

    # A declaration matching nothing, or matching on both lists at once,
    # is not evidence of anything: the name-matching backstop decides.
    return _by_name("fallback_name_match")


def next_steps_node(state: GraphState) -> GraphState:
    print(Fore.GREEN + "[position: NEXT STEPS NODE]" + Style.RESET_ALL)

    log_graph_state(state, "next_steps_node")

    if state["updated_task"] is None:
        raise ValueError("next_steps_node - updated_task is None")

    # formatted_datetime = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    if state["fake_datetime"] is None:
        raise ValueError("next_steps_node - fake_datetime is None")
    formatted_datetime = state["fake_datetime"].strftime("%d-%m-%Y %H:%M:%S")

    mode = enforcement_mode()

    initial_prompt_template = PromptTemplate.from_template(INITIAL_PROMPT)
    initial_prompt = initial_prompt_template.invoke(
        {
            "tolerance_start_time_activity": TOLERANCE_START_TIME_ACTIVITY,
            "SYSTEM_LANGUAGE": os.getenv("SYSTEM_LANGUAGE"),
            "output_block": _output_block(mode),
        }
    )

    #scheduled_activities, next_activities = schedule_activities(
    #    state["updated_task"]["activities"], state["fake_datetime"]
    #)

    #skipping scheduling based on my definition of next activities.
    #-----------------------------------------------------
    # HERE I ONLY DISCERN THE EXECUTABLE ONES, UNLIKE BEFORE
    #------------------------------------------------------

    # all
    scheduled_activities = state["updated_task"]["activities"]

    # feasible
    executable_activities = [activity for activity in scheduled_activities if activity["executable"] == True]

    # not feasible
    non_executable_activities = [activity for activity in scheduled_activities if activity["executable"] == False]


    print("activity status:")
    for activity in scheduled_activities:
        status = activity.get("arbitrated_status", "N/A")
        print(
            f"- {activity['activity_name']} - executable: {activity.get('executable', 'N/A')} - status: {status}"
        )

    # Ablation test: Skip scheduler if the environment variable is set
    if os.getenv("ABLATION_MODE", "false").lower() == "true":
        print("ABLATION MODE ACTIVE: Skipping scheduler, using all activities as next activities.")
        log_message("Skipping scheduler as per environment variable ABLATION_MODE")
        scheduled_activities = state["updated_task"]["activities"]
        executable_activities = state["updated_task"]["activities"]

    log_message("next_steps_node - scheduled_activities")
    log_message(json.dumps(scheduled_activities))
    log_message("next_steps_node - executable")
    log_message(json.dumps(executable_activities))
    log_message("next_steps_node - non executable")
    log_message(json.dumps(non_executable_activities))

    executable_activities_prompt = (
        [_activity_view(a) for a in executable_activities]
        if len(executable_activities) > 0
        else "No activities to do at the moment."
    )
    non_executable_prompt = [
        _activity_view(a, blocked=True) for a in non_executable_activities
    ]

    new_request_prompt_template = PromptTemplate.from_template(NEW_REQUEST_PROMPT)
    new_request_prompt = new_request_prompt_template.invoke(
        {
            "datetime": formatted_datetime,
            "patient_name": state["updated_task"].get("patient_name", "the patient"),
            "next_activities": executable_activities_prompt,
            "non_executable_activities": non_executable_prompt,
            "patient_request": state["patient_request"] or "",
            "situation_perception": state["situation_perception"] or "",
            "patient_information": state["updated_task"]["patient_information"] or "",
            "declaration_reminder": (
                DECLARATION_REMINDER if mode == "declared" else ""
            ),
        }
    )

    #was causing issues with groq/Ollama, didn't handle tool calling well
    #model = define_model(response_template)
    model = define_model(None)

    messages = state["conversation_history"] or []
    if len(messages) == 0:
        messages.append(SystemMessage(content=initial_prompt.to_string()))

    messages.append(HumanMessage(content=new_request_prompt.to_string()))

    #removed to make the output more flexible so that responses are better formatted
    """ result = model.invoke(
        messages,
    ).to_json()

    log_message("\nNEXT_STEPS_NODE - result")
    if result["kwargs"]["tool_calls"] is None:
        raise ValueError("nNEXT_STEPS_NODE - tool_calls is None")

    log_message(
        "robot_action - " + result["kwargs"]["tool_calls"][0]["args"]["robot_action"]
    )
    log_message(
        "vocal_answer - " + result["kwargs"]["tool_calls"][0]["args"]["vocal_answer"]
    )

    task_progress_history = {
        "situation_perception": state["situation_perception"],
        "patient_request": state["patient_request"],
        "robot_action": result["kwargs"]["tool_calls"][0]["args"]["robot_action"],
        "vocal_answer": result["kwargs"]["tool_calls"][0]["args"]["vocal_answer"],
    } """

    #------------------------------------------------------------------------------------------------------------------------------
    # ------------- TRYING TO KEEP ONLY THE LAST 3 EXCHANGES = 6 MESSAGES TO AVOID TOKEN EXHAUSTION AND SEE HOW IT GOES ------------- #
    #------------------------------------------------------------------------------------------------------------------------------
    messages = messages[-6:]

    #reasoning
    raw_response = model.invoke(messages)

    raw_content = raw_response.content

    # LangChain may return either a string or a list
    if not isinstance(raw_content, str):
        raw_content = str(raw_content)
    
    # Remove markdown fences if present
    raw_content = raw_content.strip()

    if raw_content.startswith("```json"):
        raw_content = raw_content.removeprefix("```json").strip()

    if raw_content.startswith("```"):
        raw_content = raw_content.removeprefix("```").strip()

    if raw_content.endswith("```"):
        raw_content = raw_content.removesuffix("```").strip()

    #log_message("\nNEXT_STEPS_NODE - raw_content")
    #log_message(raw_content)

    

    #try:
    #    parsed_response = json.loads(raw_content)
    #except Exception as e:
    #    print("❌ JSON PARSING ERROR") 
    #    print(raw_content)
    #    raise e
    
    #having removed tool calling, it sometimes derails with groq or ollama. So here instead of that, I use a fallback format:
    try:
        parsed_response = json.loads(raw_content)

    except Exception:

        print("⚠️ JSON parsing failed, trying fallback parser...")
        print(raw_content)

        # FALLBACK FORMAT:
        # Robot action: xxx
        # Robot answer: YYY

        robot_action = None
        vocal_answer = None
        # Sentinel: the declaration is absent until the reply provides one,
        # and absent is not the same as declared-null.
        recovered_declaration: dict = {}

        lines = raw_content.splitlines()

        for line in lines:

            line = line.strip()

            if line.lower().startswith("robot action:"):
                robot_action = line.split(":", 1)[1].strip()

            elif line.lower().startswith("robot answer:"):
                vocal_answer = line.split(":", 1)[1].strip()

            elif line.lower().startswith("proposed activity:"):
                value = line.split(":", 1)[1].strip()
                recovered_declaration["proposed_activity"] = (
                    None if value.lower() in ("null", "none", "-", "")
                    else value
                )

        if robot_action is None or vocal_answer is None:
            # Last resort: the model sometimes wraps a perfectly good JSON
            # object in prose, or emits the two fields with different
            # labels. Recovering them is preferable to aborting the whole
            # session on the formatting of a single reply.
            embedded = re.search(r"\{.*\}", raw_content, re.S)
            if embedded:
                try:
                    recovered = json.loads(embedded.group(0))
                    robot_action = recovered.get("robot_action") or robot_action
                    vocal_answer = recovered.get("vocal_answer") or vocal_answer
                    if "proposed_activity" in recovered:
                        recovered_declaration["proposed_activity"] = (
                            recovered["proposed_activity"]
                        )
                except json.JSONDecodeError:
                    pass

        if robot_action is None or vocal_answer is None:
            # Never abort: a malformed reply is a communication failure, not
            # a safety one, and the deterministic checks downstream still
            # apply. Falling back to waiting keeps the session alive and
            # leaves the event in the log.
            log_message(
                "NEXT_STEPS_NODE - unparseable model response, "
                "falling back to WAIT_FOR_PATIENT"
            )
            log_structured({
                "node": "next_steps",
                "decision_type": "unparseable_response",
                "raw_content": str(raw_content)[:500],
            })
            robot_action = robot_action or "WAIT_FOR_PATIENT"
            vocal_answer = vocal_answer or (
                "Sorry, I did not catch that. Could you repeat?"
            )

        parsed_response = {
            "robot_action": robot_action,
            "vocal_answer": vocal_answer,
            **recovered_declaration,
        }

    # Protection against malformed output. `.get(key, default)` is not
    # enough: the model sometimes emits the key with an explicit null, and
    # a present-but-null value skips the default and yields None, which
    # then breaks on the first string concatenation downstream.
    robot_action = parsed_response.get("robot_action") or "NO_ACTION"
    vocal_answer = parsed_response.get("vocal_answer") or "..."

    # Absent and present-but-null mean different things to the rule below,
    # so the two are kept apart rather than collapsed by `.get`.
    declaration_present = "proposed_activity" in parsed_response
    proposed_activity = parsed_response.get("proposed_activity")

    # --------------------------------------------------
    # DETERMINISTIC POST-CHECK: hard enforcement of Plan Validator
    # If the reply proposes a non-executable activity, override it with a
    # safe fallback. This is a deterministic check downstream of the LLM:
    # in declared mode it keys on the activity the reply declares it is
    # proposing, and falls back to matching names in the free text when
    # that declaration is missing.
    # --------------------------------------------------
    non_executable_names_lower = [
        _normalise(activity.get("activity_name"))
        for activity in non_executable_activities
        if _normalise(activity.get("activity_name"))
    ]
    executable_names_lower = [
        _normalise(activity.get("activity_name"))
        for activity in executable_activities
        if _normalise(activity.get("activity_name"))
    ]

    decision = evaluate_enforcement(
        robot_action=robot_action,
        vocal_answer=vocal_answer,
        proposed_activity=proposed_activity,
        declaration_present=declaration_present,
        blocked_names=non_executable_names_lower,
        executable_names=executable_names_lower,
        mode=mode,
    )
    violated_activities = decision["violated_activities"]

    # Every reply is recorded before the override is applied: this is what
    # makes declaration compliance measurable (how often the field is
    # emitted at all) and lets the two rules be re-scored offline on the
    # same replies.
    log_structured({
        "node": "next_steps",
        "decision_type": "reply",
        "enforcement_mode": mode,
        "robot_action": robot_action,
        "vocal_answer": vocal_answer,
        "proposed_activity": proposed_activity,
        "declaration_present": declaration_present,
        "blocked_activities": non_executable_names_lower,
        "executable_activities": executable_names_lower,
        "override": decision["override"],
        "match_mode": decision["match_mode"],
        "reason": decision["reason"],
    })

    if not decision["override"] and decision["mentioned_activities"]:
        # The answer names a blocked activity without proposing it: a
        # refusal, an explanation or a plan overview. Recorded so these
        # can still be reviewed, but not overridden.
        log_message(
            "NEXT_STEPS_NODE - mention of non-executable activity/ies "
            + ", ".join(decision["mentioned_activities"])
            + f" while robot_action={robot_action}"
            + f" ({decision['reason']}); not overridden"
        )
        log_structured({
            "node": "next_steps",
            "decision_type": f"enforcement_skipped_{decision['reason']}",
            "mentioned_activities": decision["mentioned_activities"],
            "robot_action": robot_action,
            "vocal_answer": vocal_answer,
            "proposed_activity": proposed_activity,
            "enforcement_mode": mode,
        })

    if violated_activities:
        robot_action_before = robot_action
        vocal_answer_before = vocal_answer
        log_message(
            "NEXT_STEPS_NODE - DETERMINISTIC OVERRIDE: "
            "LLM proposed non-executable activity/ies: "
            + ", ".join(violated_activities)
        )
        print(
            Fore.RED
            + f"[HARD ENFORCEMENT] LLM proposed non-executable "
            f"activity/ies: {violated_activities} "
            f"(via {decision['match_mode']}). "
            f"Overriding response."
            + Style.RESET_ALL
        )

        # Build a context-aware fallback response based on
        # arbitrated_status and recovery_actions
        blocked_critical_with_recovery = [
            activity for activity in non_executable_activities
            if activity.get("activity_name", "").lower().strip() in violated_activities
            and activity.get("arbitrated_status") == "BLOCKED_CRITICAL"
            and activity.get("recovery_actions")
        ]

        if blocked_critical_with_recovery:
            # There are recovery actions available for critical activities
            robot_action = "SUGGEST_RECOVERY_ACTION"
            vocal_answer = _recovery_message(blocked_critical_with_recovery)
        else:
            robot_action = "WAIT_FOR_PATIENT"
            reasons = []
            for activity in non_executable_activities:
                if (activity.get("activity_name", "").lower().strip()
                        in violated_activities):
                    reasons.extend(
                        _to_patient(_plain_reason(r))
                        for r in activity.get("not_executable_reasons") or []
                    )
            because = (
                f" because {'; '.join(dict.fromkeys(reasons))}"
                if reasons else ""
            )
            vocal_answer = (
                "I'm sorry, but we cannot do that right now"
                + because
                + ". Let's wait until it becomes possible."
            )

        # Structured log of the hard enforcement override
        arbitrated_statuses = [
            activity.get("arbitrated_status", "N/A")
            for activity in non_executable_activities
            if activity.get("activity_name", "").lower().strip() in violated_activities
        ]
        log_hard_enforcement(
            violated_activities=violated_activities,
            robot_action_before=robot_action_before,
            robot_action_after=robot_action,
            arbitrated_statuses=arbitrated_statuses,
            recovery_suggested=robot_action == "SUGGEST_RECOVERY_ACTION",
            vocal_answer_before=vocal_answer_before,
            vocal_answer_after=vocal_answer,
            declared_activity=proposed_activity,
            match_mode=decision["match_mode"],
            enforcement_mode=mode,
        )

    log_message("robot_action - " + robot_action)
    log_message("vocal_answer - " + vocal_answer)

    task_progress_history = {
        "situation_perception": state["situation_perception"],
        "patient_request": state["patient_request"],
        "robot_action": robot_action,
        "vocal_answer": vocal_answer,
    }

    messages.append(AIMessage(content=json.dumps(task_progress_history)))
    
    #contextual modification of the above modification
    #messages.append(
    #AIMessage(
    #    content=f"""
    #    Robot action: {robot_action}
    #
    #    Robot answer: {vocal_answer}
    #    """
    #    )
    #)
    

    return {
        **state,
        #"robot_action": result["kwargs"]["tool_calls"][0]["args"]["robot_action"],
        #"vocal_answer": result["kwargs"]["tool_calls"][0]["args"]["vocal_answer"],
        "robot_action": robot_action,
        "vocal_answer": vocal_answer,
        "conversation_history": messages,
        "scheduled_activities": scheduled_activities,
        "next_activities": executable_activities
    }
