
# The NON-LLM deterministic logic. Here is what it does: first, it defines a knowledge base with rules and facts. Rules
#  are conditions that, when satisfied, lead to certain actions (for example, if an activity is critical, then it must be
#  kept). Facts represent the current state of the world (for example, "activity X is scheduled at 10:00").
#  The inference engine applies rules to facts to derive new conclusions or make decisions (for example,
#  if an activity is past and has not been marked as done, then it is automatically skipped). The expert scheduler
#  uses the inference engine to schedule activities optimally, taking into account priorities and temporal
#  constraints. This way, the system can handle complex and dynamic situations without relying exclusively
#  on the LLM.

from __future__ import annotations

from datetime import datetime, timedelta
import math
from typing import Dict, List, Callable, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from utils.interfaces import Activity, Task
from utils.logs import log_message

# Constants
TOLERANCE_START_TIME_ACTIVITY = timedelta(minutes=30)  # activity is current if +- 30 minutes from the current time
START_DAY_TIME_LIMIT = datetime.combine(
    datetime.today(), datetime.strptime("07:00", "%H:%M").time()
)
END_DAY_TIME_LIMIT = datetime.combine(
    datetime.today(), datetime.strptime("22:00", "%H:%M").time()
)

# actions that a rule can produce on an activity
class RuleAction(Enum):
    REPLACE = "replace"
    KEEP_EXISTING = "keep_existing"
    KEEP_BOTH = "keep_both"
    SKIP = "skip"

# rules governing activity scheduling, such as skipping some if they are expired, or resolving temporal
# conflicts between activities based on priority and criticality, ...
@dataclass
class Rule:
    """Represents a rule in the knowledge base"""

    name: str
    condition: Callable[[Activity, Activity | datetime], bool]  # takes an activity as input and decides whether the rule applies
    action: RuleAction
    explanation: str
    priority: int = 0  # Higher number = higher priority rule


@dataclass
class Fact:
    """Represents a fact in the working memory"""

    subject: str
    predicate: str
    object: Any
    timestamp: datetime
    confidence: float = 1.0


class KnowledgeBase:
    """Stores domain knowledge about activity scheduling"""

    def __init__(self):
        self.rules: List[Rule] = []
        self.domain_knowledge: Dict[str, Any] = {}
        self._initialize_rules()
        self._initialize_domain_knowledge()

    # Initialize rules and append them to self.rules
    def _initialize_rules(self):
        """Initialize the rule base with scheduling rules"""

        # Critical activity rules
        self.rules.append(
            # if new activity is critical, and existing activity is not critical, then replace the existing one with the new one
            Rule(
                name="critical_over_normal",
                condition=lambda new, existing: (
                    isinstance(existing, dict)
                    and new.get("critical")
                    and not existing.get("critical")
                ),
                action=RuleAction.REPLACE,
                explanation="Critical activities take priority over non-critical ones",
                priority=10,
            )
        )

        self.rules.append(
            # if new activity is not critical, and existing activity is critical, then keep the existing one and discard the new one
            Rule(
                name="normal_kept_over_critical",
                condition=lambda new, existing: (
                    isinstance(existing, dict)
                    and existing.get("critical")
                    and not new.get("critical")
                ),
                action=RuleAction.KEEP_EXISTING,
                explanation="Existing critical activities are kept over new non-critical ones",
                priority=10,
            )
        )

        # Priority rules
        # with equal criticality flag, pick the one with higher priority (lower number)
        self.rules.append(
            Rule(
                name="higher_priority_replaces",
                condition=lambda new, existing: (
                    isinstance(existing, dict)
                    and new.get("critical") == existing.get("critical")
                    and new.get("priority") < existing.get("priority")
                ),
                action=RuleAction.REPLACE,
                explanation="Higher priority activities replace lower priority ones",
                priority=5,
            )
        )

        self.rules.append(
            # with equal criticality flag, if new activity has lower priority (higher number), then keep
            # the existing one and discard the new one
            Rule(
                name="lower_priority_kept",
                condition=lambda new, existing: (
                    isinstance(existing, dict)
                    and new.get("critical") == existing.get("critical")
                    and new.get("priority") > existing.get("priority")
                ),
                action=RuleAction.KEEP_EXISTING,
                explanation="Existing higher priority activities are kept",
                priority=5,
            )
        )

        # Tolerance rules
        self.rules.append(
            # if activity is outside tolerance, then skip that activity (mark it as "skipped" and "done": False)
            Rule(
                name="skip_overdue_activity",
                condition=lambda activity, current_time: self._is_overdue(
                    activity,
                    (
                        current_time
                        if isinstance(current_time, datetime)
                        else datetime.now()
                    ),
                ),
                action=RuleAction.SKIP,
                explanation="Activities past tolerance window are automatically skipped",
                priority=15,
            )
        )
    # Initialize domain knowledge, e.g., time limits, tolerances, time format, etc.
    def _initialize_domain_knowledge(self):
        """Initialize domain-specific knowledge"""
        self.domain_knowledge = {
            "day_limits": {"start": START_DAY_TIME_LIMIT, "end": END_DAY_TIME_LIMIT},
            "tolerance": TOLERANCE_START_TIME_ACTIVITY,
            "conflict_tolerance_percentage": 0.1,  # Used for temporal conflicts. Adds 10% of time to the activity duration to consider it conflicting with other activities that overlap even partially (pessimistic approach)
            "time_format": "%H:%M",
        }
    # check if activity is overdue
    def _is_overdue(self, activity: Activity, current_time: datetime) -> bool:
        """Check if activity is overdue based on tolerance"""
        if (
            not activity.get("activity_time")
            or activity["activity_time"] is None
            or activity.get("without_time")
        ):
            return False

        activity_time = self._parse_time(activity["activity_time"], current_time)
        if not activity_time:
            return False

        return current_time - activity_time >= self.domain_knowledge["tolerance"]

    # parse a time string with a reference date
    def _parse_time(
        self, time_str: str, reference_date: datetime
    ) -> Optional[datetime]:
        """Parse time string with reference date"""
        try:
            return datetime.strptime(
                time_str, self.domain_knowledge["time_format"]
            ).replace(
                year=reference_date.year,
                month=reference_date.month,
                day=reference_date.day,
            )
        except (ValueError, TypeError):
            return None

    # Return rules applicable to a specific context (e.g., conflict resolution or activity status)
    def get_applicable_rules(self, context: str) -> List[Rule]:
        """Get rules applicable to a specific context"""
        context_rules = {
            "conflict_resolution": [
                r
                for r in self.rules
                if r.name
                in [
                    "critical_over_normal",
                    "normal_kept_over_critical",
                    "higher_priority_replaces",
                    "lower_priority_kept",
                ]
            ],
            "activity_status": [
                r for r in self.rules if r.name in ["skip_overdue_activity"]
            ],
        }
        return context_rules.get(context, self.rules)


class WorkingMemory:
    """Manages current facts and inferences"""
    # Defines facts, inferred facts, and current conflicts. Facts represent the current state of the world,
    # inferred facts are those derived from applying rules,
    # and conflicts are situations where rules have produced conflicting conclusions requiring a decision.
    def __init__(self):
        self.facts: List[Fact] = []
        self.inferred_facts: List[Fact] = []
        self.conflict_set: List[Dict[str, Any]] = []

    def add_fact(
        self, subject: str, predicate: str, object: Any, confidence: float = 1.0
    ):
        """Add a fact to working memory"""
        fact = Fact(subject, predicate, object, datetime.now(), confidence)
        self.facts.append(fact)

    def get_facts(
        self, subject: str | None = None, predicate: str | None = None
    ) -> List[Fact]:
        """Retrieve facts matching criteria"""
        results = self.facts + self.inferred_facts
        if subject:
            results = [f for f in results if f.subject == subject]
        if predicate:
            results = [f for f in results if f.predicate == predicate]
        return results

    def add_conflict(
        self, new_activity: Activity, existing_activity: Activity, conflict_type: str
    ):
        """Record a conflict for resolution"""
        self.conflict_set.append(
            {
                "new_activity": new_activity,
                "existing_activity": existing_activity,
                "type": conflict_type,
                "timestamp": datetime.now(),
            }
        )

# The actual inference engine that applies rules to facts to derive new conclusions or make decisions
class InferenceEngine:
    """Handles reasoning and rule application"""

    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.working_memory = WorkingMemory()
        self.explanation_trace: List[Dict[str, Any]] = []

    # For each rule applicable to the specific context, apply the rule to the activity list
    def forward_chaining(
        self, activities: List[Activity], context: str | None = None
    ) -> List[Activity]:
        """Apply rules to derive new facts and resolve conflicts"""
        applicable_rules = self.kb.get_applicable_rules(
            context or "conflict_resolution"
        )
        applicable_rules.sort(key=lambda r: r.priority, reverse=True)

        for rule in applicable_rules:
            activities = self._apply_rule(rule, activities)

        return activities

    # Apply a single rule to the activity list. If it's a skip rule, iterate over all activities and mark as
    # "skipped" those that satisfy the rule's condition (e.g., overdue activities). If it's a conflict resolution
    # rule, apply apply_conflict_rule
    def _apply_rule(self, rule: Rule, activities: List[Activity]) -> List[Activity]:
        """Apply a single rule to the activity list"""
        if rule.name == "skip_overdue_activity":
            return self._apply_skip_rule(rule, activities)
        else:
            return self._apply_conflict_rule(rule, activities)

    # The activity must not be done, must not be skipped, and must satisfy the rule's condition (e.g., being overdue), then mark that activity as "skipped" and "done": False
    def _apply_skip_rule(
        self, rule: Rule, activities: List[Activity]
    ) -> List[Activity]:
        """Apply rules for skipping activities"""
        current_time = datetime.now()

        for activity in activities:
            if (
                not activity.get("done")
                and not activity.get("skipped")
                and rule.condition(activity, current_time)
            ):

                activity["skipped"] = True
                activity["done"] = False

                self._log_decision(rule, activity, None, "skipped")

        return activities

    # Same as above, apply a conflict resolution rule to the activity list -> not implemented yet, returns the list as is
    def _apply_conflict_rule(
        self, rule: Rule, activities: List[Activity]
    ) -> List[Activity]:
        """Apply rules for conflict resolution"""
        # This will be called from resolve_conflicts method
        return activities

    # Set up conflicts for all pairs of activities where conflicts are detected, and apply conflict resolution rules
    # to decide which activities to keep, which to replace, and which to leave both for user decision
    def resolve_conflicts(
        self, new_activity: Activity, activities: List[Activity]
    ) -> List[Activity]:
        """Resolve conflicts using rule-based approach"""
        conflicting_activities = self._find_conflicts(new_activity, activities)

        if not conflicting_activities:
            return activities + [new_activity]

        # Apply conflict resolution rules
        applicable_rules = self.kb.get_applicable_rules("conflict_resolution")
        applicable_rules.sort(key=lambda r: r.priority, reverse=True)

        for existing_activity in conflicting_activities:
            self.working_memory.add_conflict(
                new_activity, existing_activity, "time_overlap"
            )

            action_taken = False
            for rule in applicable_rules:
                if rule.condition(new_activity, existing_activity):
                    if rule.action == RuleAction.REPLACE:
                        activities.remove(existing_activity)
                        self._log_decision(
                            rule, new_activity, existing_activity, "replaced"
                        )
                        action_taken = True
                        break
                    elif rule.action == RuleAction.KEEP_EXISTING:
                        self._log_decision(
                            rule, new_activity, existing_activity, "kept_existing"
                        )
                        return activities  # Don't add new activity

            if not action_taken:
                # Default case: keep both for user decision
                self._log_decision(None, new_activity, existing_activity, "kept_both")

        return activities + [new_activity]

    # Return the list of activities that conflict with the new_activity
    def _find_conflicts(
        self, new_activity: Activity, activities: List[Activity]
    ) -> List[Activity]:
        """Find activities that conflict with the new one"""
        conflicts = []

        for existing in activities:
            if self._activities_conflict(new_activity, existing):
                conflicts.append(existing)

        return conflicts

    # Used by the method above to determine if two activities conflict
    # If either one has no start time, they are not necessarily in conflict
    # Otherwise, takes their duration, computes when they end, and returns true (conflict) if their time intervals overlap
    # even partially, taking into account the tolerance that extends each activity's duration by 10% (pessimistic approach)
    def _activities_conflict(self, activity1: Activity, activity2: Activity) -> bool:
        """Check if two activities have overlapping time slots"""
        start1 = self._parse_time(activity1.get("activity_time"))
        start2 = self._parse_time(activity2.get("activity_time"))

        if not start1 or not start2:
            return False

        duration1 = timedelta(minutes=activity1["activity_duration"])
        duration2 = timedelta(minutes=activity2["activity_duration"])

        # Apply tolerance (pessimistic approach)
        tolerance_pct = self.kb.domain_knowledge["conflict_tolerance_percentage"]
        tolerance1 = timedelta(
            minutes=math.ceil(duration1.total_seconds() / 60 * tolerance_pct)
        )
        tolerance2 = timedelta(
            minutes=math.ceil(duration2.total_seconds() / 60 * tolerance_pct)
        )

        end1 = start1 + duration1 + tolerance1
        end2 = start2 + duration2 + tolerance2

        return start1 < end2 and start2 < end1

    def _parse_time(self, time_str: str | None) -> Optional[datetime]:
        """Parse time string to datetime object"""
        if not time_str:
            return None

        try:
            return datetime.combine(
                datetime.today(),
                datetime.strptime(
                    time_str, self.kb.domain_knowledge["time_format"]
                ).time(),
            )
        except (ValueError, TypeError):
            return None

    # Debug log -> appends everything to the explanation trace and prints a log message explaining the decision made
    # (e.g., "Replacing activity X with activity Y because X is critical and Y is not")
    def _log_decision(
        self,
        rule: Rule | None,
        new_activity: Activity,
        existing_activity: Activity | None,
        action: str,
    ):
        """Log reasoning decisions with explanations"""
        explanation_entry = {
            "timestamp": datetime.now(),
            "rule": rule.name if rule else "default",
            "explanation": rule.explanation if rule else "No applicable rule found",
            "new_activity": new_activity["activity_name"],
            "existing_activity": (
                existing_activity["activity_name"] if existing_activity else None
            ),
            "action": action,
        }

        self.explanation_trace.append(explanation_entry)

        # Generate appropriate log message
        if action == "replaced":
            if existing_activity is not None:
                log_message(
                    f"Replacing {existing_activity['activity_name']} with {new_activity['activity_name']} - {(rule.explanation if rule else 'No applicable rule found')}"
                )
            else:
                log_message(
                    f"Replacing activity with {new_activity['activity_name']} - {(rule.explanation if rule else 'No applicable rule found')}"
                )
        elif action == "kept_existing":
            if existing_activity is not None:
                log_message(
                    f"Keeping {existing_activity['activity_name']} over {new_activity['activity_name']} - {(rule.explanation if rule else 'No applicable rule found')}"
                )
            else:
                log_message(
                    f"Keeping existing activity over {new_activity['activity_name']} - {(rule.explanation if rule else 'No applicable rule found')}"
                )
        elif action == "kept_both":
            if existing_activity is not None:
                log_message(
                    f"Conflict between {new_activity['activity_name']} and {existing_activity['activity_name']}. Keeping both for user decision."
                )
            else:
                log_message(
                    f"Conflict involving {new_activity['activity_name']}. Keeping both for user decision."
                )
        elif action == "skipped":
            explanation = rule.explanation if rule else "No applicable rule found"
            log_message(f"Skipping {new_activity['activity_name']} - {explanation}")

    # Print the entire explanation trace
    def get_explanation_trace(self) -> List[Dict[str, Any]]:
        """Get the complete explanation trace"""
        return self.explanation_trace


# Main class that coordinates everything. Uses the knowledge base and inference engine to schedule activities optimally,
# taking into account priorities and constraints
class ExpertScheduler:
    """Main expert system for activity scheduling"""

    # Define kb and inference engine
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.inference_engine = InferenceEngine(self.knowledge_base)

    # Apply skip rules to activities of a task, e.g., to mark expired ones as "skipped". Chain:
    # forward_chaining() with activity_status parameter -> applicable_rules with time constraint from activity_status -> apply skip rule
    def set_skipped_activities(self, task: Task, current_time: datetime) -> None:
        """Set activities as skipped based on expert system rules"""
        activities = task["activities"]
        self.inference_engine.forward_chaining(activities, "activity_status")

    # For all activities, resolve temporal conflicts (if they have a time), i.e.:
    # resolve_conflicts -> find_conflicts -> check activities_conflict on time
    #  otherwise do auto_schedule
    # once scheduled in one way or the other, pass them to get_next_activities to get the next ones to perform
    def schedule_activities(
        self, activity_list: List[Activity], current_time: datetime | None
    ) -> tuple[List[Activity], List[Activity]]:
        """Schedule activities using expert system approach"""
        scheduled_activities = []

        for activity in activity_list:
            if activity.get("activity_time"):
                # Fixed time activity - resolve conflicts
                scheduled_activities = self.inference_engine.resolve_conflicts(
                    activity, scheduled_activities
                )
            else:
                # Auto-schedule activity
                scheduled_activities = self._auto_schedule(
                    activity, scheduled_activities
                )

        # Sort by time
        scheduled_activities.sort(
            key=lambda a: self._parse_time(a["activity_time"]) or datetime.max
        )

        # Get next activities
        next_activities = self._get_next_activities(scheduled_activities, current_time)

        return scheduled_activities, next_activities

    # Schedule those without a time, try to find the first available slot of the required duration
    def _auto_schedule(
        self, activity: Activity, activities: List[Activity]
    ) -> List[Activity]:
        """Auto-schedule an activity in the first available slot"""
        last_end_time = START_DAY_TIME_LIMIT

        sorted_activities = sorted(
            activities,
            key=lambda x: self._parse_time(x["activity_time"]) or datetime.max,
        )

        for existing_activity in sorted_activities:
            activity_time = self._parse_time(existing_activity["activity_time"])
            if activity_time:
                available_duration = activity_time - last_end_time
                required_duration = timedelta(minutes=activity["activity_duration"])

                if available_duration >= required_duration:
                    activity["activity_time"] = self._format_time(last_end_time)
                    return activities + [activity]

                last_end_time = max(
                    last_end_time,
                    activity_time
                    + timedelta(minutes=existing_activity["activity_duration"]),
                )

        # Try to fit at the end of the day
        required_duration = timedelta(minutes=activity["activity_duration"])
        if (END_DAY_TIME_LIMIT - last_end_time) >= required_duration:
            activity["activity_time"] = self._format_time(last_end_time)
            return activities + [activity]

        log_message(f"No slot available for {activity['activity_name']}")
        return activities

    # Given the scheduled activities, return the next ones to perform:
    # if the duration falls between current_time - start of day threshold and current_time + end of day threshold, and it's neither done nor skipped,
    # add it.
    # Finally, return them sorted by criticality and priority.
    def _get_next_activities(
        self, activities: List[Activity], current_time: datetime | None
    ) -> List[Activity]:
        """Get next activities to perform"""
        if (
            not current_time
            or current_time < START_DAY_TIME_LIMIT
            or current_time > END_DAY_TIME_LIMIT
        ):
            return []

        upcoming_activities = []

        for activity in activities:
            activity_time = self._parse_time(activity["activity_time"])
            if (
                activity_time
                and current_time - TOLERANCE_START_TIME_ACTIVITY
                <= activity_time
                <= current_time + TOLERANCE_START_TIME_ACTIVITY
                and not activity.get("done")
                and not activity.get("skipped")
            ):
                upcoming_activities.append(activity)

        if not upcoming_activities:
            return []

        # Sort by time and return activities with same critical status and priority
        upcoming_activities.sort(
            key=lambda a: self._parse_time(a["activity_time"]) or datetime.max
        )
        top_activity = upcoming_activities[0]

        return [
            a
            for a in upcoming_activities
            if (
                a["critical"] == top_activity["critical"]
                and a["priority"] == top_activity["priority"]
            )
        ]

    # parse time
    def _parse_time(self, time_str: str | None) -> Optional[datetime]:
        """Parse time string to datetime object"""
        return self.inference_engine._parse_time(time_str)

    def _format_time(self, time_obj: datetime | None) -> str:
        """Format datetime object to string"""
        return time_obj.strftime("%H:%M") if time_obj else "No fixed time"

    def get_explanation_trace(self) -> List[Dict[str, Any]]:
        """Get explanation of all decisions made"""
        return self.inference_engine.get_explanation_trace()


# Wrapper functions to maintain backward compatibility
# functions inside functions to avoid instantiating classes
def set_skipped_activities(task: Task, current_time: datetime) -> None:
    scheduler = ExpertScheduler()
    scheduler.set_skipped_activities(task, current_time)


def schedule_activities(
    activity_list: List[Activity], current_time: datetime | None
) -> tuple[List[Activity], List[Activity]]:
    scheduler = ExpertScheduler()
    return scheduler.schedule_activities(activity_list, current_time)


# Utility functions (maintained for compatibility)
def parse_time(time_str: str | None):
    return (
        datetime.combine(datetime.today(), datetime.strptime(time_str, "%H:%M").time())
        if time_str
        else None
    )


def format_time(time_obj: datetime | None):
    return time_obj.strftime("%H:%M") if time_obj else "No fixed time"
