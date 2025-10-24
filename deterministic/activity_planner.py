from datetime import datetime, timedelta
import math
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum

from utils.interfaces import Activity, Task
from utils.logs import log_message

# Constants
TOLERANCE_START_TIME_ACTIVITY = timedelta(minutes=30)
START_DAY_TIME_LIMIT = datetime.combine(
    datetime.today(), datetime.strptime("07:00", "%H:%M").time()
)
END_DAY_TIME_LIMIT = datetime.combine(
    datetime.today(), datetime.strptime("22:00", "%H:%M").time()
)


class RuleAction(Enum):
    REPLACE = "replace"
    KEEP_EXISTING = "keep_existing"
    KEEP_BOTH = "keep_both"
    SKIP = "skip"


@dataclass
class Rule:
    """Represents a rule in the knowledge base"""

    name: str
    condition: Callable[[Activity, Activity | datetime], bool]
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

    def _initialize_rules(self):
        """Initialize the rule base with scheduling rules"""

        # Critical activity rules
        self.rules.append(
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

    def _initialize_domain_knowledge(self):
        """Initialize domain-specific knowledge"""
        self.domain_knowledge = {
            "day_limits": {"start": START_DAY_TIME_LIMIT, "end": END_DAY_TIME_LIMIT},
            "tolerance": TOLERANCE_START_TIME_ACTIVITY,
            "conflict_tolerance_percentage": 0.1,
            "time_format": "%H:%M",
        }

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


class InferenceEngine:
    """Handles reasoning and rule application"""

    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.working_memory = WorkingMemory()
        self.explanation_trace: List[Dict[str, Any]] = []

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

    def _apply_rule(self, rule: Rule, activities: List[Activity]) -> List[Activity]:
        """Apply a single rule to the activity list"""
        if rule.name == "skip_overdue_activity":
            return self._apply_skip_rule(rule, activities)
        else:
            return self._apply_conflict_rule(rule, activities)

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

    def _apply_conflict_rule(
        self, rule: Rule, activities: List[Activity]
    ) -> List[Activity]:
        """Apply rules for conflict resolution"""
        # This will be called from resolve_conflicts method
        return activities

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

    def _find_conflicts(
        self, new_activity: Activity, activities: List[Activity]
    ) -> List[Activity]:
        """Find activities that conflict with the new one"""
        conflicts = []

        for existing in activities:
            if self._activities_conflict(new_activity, existing):
                conflicts.append(existing)

        return conflicts

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

    def get_explanation_trace(self) -> List[Dict[str, Any]]:
        """Get the complete explanation trace"""
        return self.explanation_trace


class ExpertScheduler:
    """Main expert system for activity scheduling"""

    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.inference_engine = InferenceEngine(self.knowledge_base)

    def set_skipped_activities(self, task: Task, current_time: datetime) -> None:
        """Set activities as skipped based on expert system rules"""
        activities = task["activities"]
        self.inference_engine.forward_chaining(activities, "activity_status")

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
