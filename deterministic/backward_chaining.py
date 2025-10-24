from datetime import datetime, timedelta
import math
from typing import Dict, List, Callable, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import copy

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
    PROVE = "prove"
    DERIVE = "derive"


@dataclass
class Rule:
    """Represents a rule in the knowledge base"""

    name: str
    condition: Callable[[Any, Any], bool]
    action: RuleAction
    explanation: str
    priority: int = 0
    # New fields for backward chaining
    conclusions: List[str] | None = None  # What this rule can prove
    premises: List[str] | None = None  # What this rule needs to prove its conclusions


@dataclass
class Fact:
    """Represents a fact in the working memory"""

    subject: str
    predicate: str
    object: Any
    timestamp: datetime
    confidence: float = 1.0
    source: str = "user"  # user, forward_chain, backward_chain, derived


@dataclass
class Goal:
    """Represents a goal to be proven in backward chaining"""

    predicate: str
    arguments: List[Any]
    confidence_threshold: float = 0.7
    max_depth: int = 5


@dataclass
class ProofNode:
    """Represents a node in the proof tree"""

    goal: Goal
    rule_applied: Optional[Rule] = None
    subgoals: List["ProofNode"] | None = None
    proven: bool = False
    confidence: float = 0.0
    depth: int = 0


class BackwardChaining:
    """
    Implements backward chaining inference for goal-driven reasoning.

    This class allows the system to work backwards from a desired goal
    to find the facts and rules needed to prove that goal.
    """

    def __init__(self, knowledge_base, working_memory):
        self.knowledge_base = knowledge_base
        self.working_memory = working_memory
        self.proof_cache: Dict[str, ProofNode] = {}
        self.inference_trace: List[Dict[str, Any]] = []
        self.max_recursion_depth = 10

    def prove_goal(self, goal: Goal, current_depth: int = 0) -> ProofNode:
        """
        Main entry point for backward chaining.

        Args:
            goal: The goal to prove
            current_depth: Current recursion depth (for cycle detection)

        Returns:
            ProofNode: The proof tree with results
        """
        # Check recursion depth to avoid infinite loops
        if current_depth >= self.max_recursion_depth:
            log_message(f"Maximum recursion depth reached for goal: {goal.predicate}")
            return ProofNode(goal=goal, proven=False, depth=current_depth)

        # Check cache for previously proven goals
        goal_key = self._create_goal_key(goal)
        if goal_key in self.proof_cache:
            log_message(f"Using cached proof for goal: {goal.predicate}")
            return self.proof_cache[goal_key]

        # Create proof node
        proof_node = ProofNode(goal=goal, subgoals=[], depth=current_depth)

        # Log inference attempt
        self._log_inference_attempt(goal, current_depth)

        # Step 1: Check if goal is already proven by existing facts
        if self._check_goal_in_facts(goal):
            proof_node.proven = True
            proof_node.confidence = 1.0
            log_message(f"Goal {goal.predicate} found in existing facts")
            self._cache_proof(goal_key, proof_node)
            return proof_node

        # Step 2: Find applicable rules that can prove this goal
        applicable_rules = self._find_rules_for_goal(goal)

        if not applicable_rules:
            log_message(f"No applicable rules found for goal: {goal.predicate}")
            proof_node.proven = False
            self._cache_proof(goal_key, proof_node)
            return proof_node

        # Step 3: Try to prove goal using each applicable rule
        for rule in applicable_rules:
            rule_proof = self._try_prove_with_rule(goal, rule, current_depth + 1)

            if rule_proof.proven and rule_proof.confidence >= goal.confidence_threshold:
                proof_node.proven = True
                proof_node.confidence = rule_proof.confidence
                proof_node.rule_applied = rule
                proof_node.subgoals = rule_proof.subgoals

                # Add derived fact to working memory
                self._add_derived_fact(goal, rule_proof.confidence)

                log_message(
                    f"Goal {goal.predicate} successfully proven using rule: {rule.name}"
                )
                self._cache_proof(goal_key, proof_node)
                return proof_node

        # Step 4: Goal could not be proven
        proof_node.proven = False
        log_message(f"Failed to prove goal: {goal.predicate}")
        self._cache_proof(goal_key, proof_node)
        return proof_node

    def _check_goal_in_facts(self, goal: Goal) -> bool:
        """
        Check if the goal is already satisfied by existing facts.

        Args:
            goal: The goal to check

        Returns:
            bool: True if goal is satisfied by existing facts
        """
        matching_facts = self.working_memory.get_facts(predicate=goal.predicate)

        for fact in matching_facts:
            if self._fact_matches_goal(fact, goal):
                return True

        return False

    def _fact_matches_goal(self, fact: Fact, goal: Goal) -> bool:
        """
        Check if a specific fact matches the goal arguments.

        Args:
            fact: The fact to check
            goal: The goal to match against

        Returns:
            bool: True if fact matches goal
        """
        # Simple matching - can be extended for more complex unification
        if len(goal.arguments) == 0:
            return True

        # Check if fact object matches goal arguments
        if len(goal.arguments) == 1:
            return fact.object == goal.arguments[0]

        # For more complex matching, implement unification algorithm
        return True

    def _find_rules_for_goal(self, goal: Goal) -> List[Rule]:
        """
        Find all rules that can potentially prove the given goal.

        Args:
            goal: The goal to find rules for

        Returns:
            List[Rule]: Rules that can prove the goal
        """
        applicable_rules = []

        for rule in self.knowledge_base.rules:
            if rule.conclusions and goal.predicate in rule.conclusions:
                applicable_rules.append(rule)

        # Sort by priority (higher priority first)
        applicable_rules.sort(key=lambda r: r.priority, reverse=True)

        return applicable_rules

    def _try_prove_with_rule(
        self, goal: Goal, rule: Rule, current_depth: int
    ) -> ProofNode:
        """
        Try to prove a goal using a specific rule.

        Args:
            goal: The goal to prove
            rule: The rule to use
            current_depth: Current recursion depth

        Returns:
            ProofNode: The proof attempt result
        """
        proof_node = ProofNode(
            goal=goal, rule_applied=rule, subgoals=[], depth=current_depth
        )

        # If rule has no premises, it's a fact-based rule
        if not rule.premises:
            # Check if rule condition is satisfied
            if self._evaluate_rule_condition(rule, goal):
                proof_node.proven = True
                proof_node.confidence = 1.0
                return proof_node
            else:
                proof_node.proven = False
                return proof_node

        # Rule has premises - need to prove all subgoals
        all_subgoals_proven = True
        total_confidence = 1.0

        # Ensure subgoals is a list before appending
        if proof_node.subgoals is None:
            proof_node.subgoals = []

        for premise in rule.premises:
            subgoal = self._create_subgoal_from_premise(premise, goal)
            subgoal_proof = self.prove_goal(subgoal, current_depth)

            proof_node.subgoals.append(subgoal_proof)

            if not subgoal_proof.proven:
                all_subgoals_proven = False
                break
            else:
                # Combine confidence using minimum (conservative approach)
                total_confidence = min(total_confidence, subgoal_proof.confidence)

        if all_subgoals_proven:
            proof_node.proven = True
            proof_node.confidence = total_confidence
        else:
            proof_node.proven = False
            proof_node.confidence = 0.0

        return proof_node

    def _evaluate_rule_condition(self, rule: Rule, goal: Goal) -> bool:
        """
        Evaluate if a rule's condition is satisfied for the given goal.

        Args:
            rule: The rule to evaluate
            goal: The goal context

        Returns:
            bool: True if condition is satisfied
        """
        try:
            # This is a simplified evaluation - in practice, you'd need
            # more sophisticated condition evaluation
            return rule.condition(goal.arguments[0] if goal.arguments else None, None)
        except Exception as e:
            log_message(f"Error evaluating rule condition: {e}")
            return False

    def _create_subgoal_from_premise(self, premise: str, parent_goal: Goal) -> Goal:
        """
        Create a subgoal from a rule premise.

        Args:
            premise: The premise string
            parent_goal: The parent goal for context

        Returns:
            Goal: The created subgoal
        """
        # Simple premise parsing - can be extended for more complex logic
        return Goal(
            predicate=premise,
            arguments=parent_goal.arguments,
            confidence_threshold=parent_goal.confidence_threshold,
            max_depth=parent_goal.max_depth,
        )

    def _add_derived_fact(self, goal: Goal, confidence: float):
        """
        Add a proven goal as a derived fact to working memory.

        Args:
            goal: The proven goal
            confidence: Confidence level of the proof
        """
        fact = Fact(
            subject=goal.arguments[0] if goal.arguments else "system",
            predicate=goal.predicate,
            object=goal.arguments[1] if len(goal.arguments) > 1 else True,
            timestamp=datetime.now(),
            confidence=confidence,
            source="backward_chain",
        )

        self.working_memory.facts.append(fact)

    def _create_goal_key(self, goal: Goal) -> str:
        """
        Create a unique key for goal caching.

        Args:
            goal: The goal to create key for

        Returns:
            str: Unique key for the goal
        """
        args_str = "_".join(str(arg) for arg in goal.arguments)
        return f"{goal.predicate}_{args_str}"

    def _log_inference_attempt(self, goal: Goal, depth: int):
        """
        Log an inference attempt for debugging and explanation.

        Args:
            goal: The goal being attempted
            depth: Current recursion depth
        """
        indent = "  " * depth
        log_message(f"{indent}Attempting to prove goal: {goal.predicate}")

        self.inference_trace.append(
            {
                "timestamp": datetime.now(),
                "action": "prove_goal",
                "goal": goal.predicate,
                "arguments": goal.arguments,
                "depth": depth,
            }
        )

    def _cache_proof(self, goal_key: str, proof_node: ProofNode):
        """
        Cache a proof result for future use.

        Args:
            goal_key: The goal key
            proof_node: The proof result
        """
        self.proof_cache[goal_key] = proof_node

    def get_proof_explanation(self, proof_node: ProofNode) -> Dict[str, Any]:
        """
        Generate a human-readable explanation of the proof.

        Args:
            proof_node: The proof node to explain

        Returns:
            Dict: Explanation structure
        """
        explanation = {
            "goal": proof_node.goal.predicate,
            "proven": proof_node.proven,
            "confidence": proof_node.confidence,
            "rule_used": (
                proof_node.rule_applied.name if proof_node.rule_applied else None
            ),
            "subgoals": [],
        }

        if proof_node.subgoals:
            for subgoal in proof_node.subgoals:
                explanation["subgoals"].append(self.get_proof_explanation(subgoal))

        return explanation

    def clear_cache(self):
        """Clear the proof cache."""
        self.proof_cache.clear()

    def get_inference_trace(self) -> List[Dict[str, Any]]:
        """Get the complete inference trace."""
        return self.inference_trace


class WorkingMemory:
    """Manages current facts and inferences with enhanced backward chaining support"""

    def __init__(self):
        self.facts: List[Fact] = []
        self.inferred_facts: List[Fact] = []
        self.conflict_set: List[Dict[str, Any]] = []

    def add_fact(
        self,
        subject: str,
        predicate: str,
        object: Any,
        confidence: float = 1.0,
        source: str = "user",
    ):
        """Add a fact to working memory with source tracking"""
        fact = Fact(subject, predicate, object, datetime.now(), confidence, source)
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


class EnhancedKnowledgeBase:
    """Enhanced knowledge base with backward chaining support"""

    def __init__(self):
        self.rules: List[Rule] = []
        self.domain_knowledge: Dict[str, Any] = {}
        self._initialize_rules()
        self._initialize_domain_knowledge()

    def _initialize_rules(self):
        """Initialize the rule base with enhanced rules for backward chaining"""

        # Forward chaining rules (existing)
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

        # Backward chaining rules (new)
        self.rules.append(
            Rule(
                name="can_schedule_activity",
                condition=lambda activity, time_slot: self._check_scheduling_feasibility(
                    activity, time_slot
                ),
                action=RuleAction.PROVE,
                explanation="An activity can be scheduled if time slot is available and constraints are met",
                priority=5,
                conclusions=["can_schedule"],
                premises=["time_slot_available", "constraints_satisfied"],
            )
        )

        self.rules.append(
            Rule(
                name="time_slot_available",
                condition=lambda activity, time_slot: self._check_time_availability(
                    activity, time_slot
                ),
                action=RuleAction.PROVE,
                explanation="Time slot is available if no conflicts exist",
                priority=5,
                conclusions=["time_slot_available"],
                premises=["no_conflicts"],
            )
        )

        self.rules.append(
            Rule(
                name="constraints_satisfied",
                condition=lambda activity, context: self._check_constraints(
                    activity, context
                ),
                action=RuleAction.PROVE,
                explanation="Activity constraints are satisfied",
                priority=5,
                conclusions=["constraints_satisfied"],
                premises=[],  # Base fact
            )
        )

        self.rules.append(
            Rule(
                name="medication_timing_valid",
                condition=lambda activity, context: self._check_medication_timing(
                    activity, context
                ),
                action=RuleAction.PROVE,
                explanation="Medication timing follows medical guidelines",
                priority=8,
                conclusions=["medication_timing_valid"],
                premises=["dosage_interval_respected", "food_requirements_met"],
            )
        )

    def _initialize_domain_knowledge(self):
        """Initialize domain-specific knowledge"""
        self.domain_knowledge = {
            "day_limits": {"start": START_DAY_TIME_LIMIT, "end": END_DAY_TIME_LIMIT},
            "tolerance": TOLERANCE_START_TIME_ACTIVITY,
            "conflict_tolerance_percentage": 0.1,
            "time_format": "%H:%M",
            "medical_constraints": {
                "minimum_medication_interval": timedelta(hours=4),
                "meal_medication_gap": timedelta(minutes=30),
            },
        }

    def _check_scheduling_feasibility(self, activity: Any, time_slot: Any) -> bool:
        """Check if an activity can be scheduled at a given time slot"""
        # Simplified implementation - extend based on your needs
        return True

    def _check_time_availability(self, activity: Any, time_slot: Any) -> bool:
        """Check if time slot is available"""
        # Simplified implementation
        return True

    def _check_constraints(self, activity: Any, context: Any) -> bool:
        """Check if activity constraints are satisfied"""
        # Simplified implementation
        return True

    def _check_medication_timing(self, activity: Any, context: Any) -> bool:
        """Check if medication timing is valid"""
        # Simplified implementation
        return True

    def get_applicable_rules(self, context: str) -> List[Rule]:
        """Get rules applicable to a specific context"""
        context_rules = {
            "conflict_resolution": [
                r
                for r in self.rules
                if r.action
                in [RuleAction.REPLACE, RuleAction.KEEP_EXISTING, RuleAction.KEEP_BOTH]
            ],
            "goal_proving": [
                r
                for r in self.rules
                if r.action in [RuleAction.PROVE, RuleAction.DERIVE]
            ],
            "activity_status": [
                r for r in self.rules if r.name in ["skip_overdue_activity"]
            ],
        }
        return context_rules.get(context, self.rules)


class EnhancedInferenceEngine:
    """Enhanced inference engine with both forward and backward chaining"""

    def __init__(self, knowledge_base: EnhancedKnowledgeBase):
        self.kb = knowledge_base
        self.working_memory = WorkingMemory()
        self.backward_chainer = BackwardChaining(knowledge_base, self.working_memory)
        self.explanation_trace: List[Dict[str, Any]] = []

    def prove_goal(
        self, goal_predicate: str, arguments: List[Any] | None = None
    ) -> ProofNode:
        """
        Prove a goal using backward chaining.

        Args:
            goal_predicate: The predicate to prove
            arguments: Arguments for the goal

        Returns:
            ProofNode: The proof result
        """
        goal = Goal(
            predicate=goal_predicate,
            arguments=arguments or [],
            confidence_threshold=0.7,
        )

        return self.backward_chainer.prove_goal(goal)

    def can_schedule_activity(
        self, activity: Activity, time_slot: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if an activity can be scheduled using backward chaining.

        Args:
            activity: The activity to schedule
            time_slot: The proposed time slot

        Returns:
            Tuple: (can_schedule, explanation)
        """
        proof_result = self.prove_goal("can_schedule", [activity, time_slot])

        explanation = self.backward_chainer.get_proof_explanation(proof_result)

        return proof_result.proven, explanation

    def forward_chaining(
        self, activities: List[Activity], context: str | None = None
    ) -> List[Activity]:
        """Apply forward chaining rules (existing functionality)"""
        applicable_rules = self.kb.get_applicable_rules(
            context or "conflict_resolution"
        )
        applicable_rules.sort(key=lambda r: r.priority, reverse=True)

        for rule in applicable_rules:
            if rule.action != RuleAction.PROVE:  # Skip backward chaining rules
                activities = self._apply_rule(rule, activities)

        return activities

    def _apply_rule(self, rule: Rule, activities: List[Activity]) -> List[Activity]:
        """Apply a single rule to the activity list (existing functionality)"""
        # Implementation same as before
        return activities

    def get_complete_explanation(self) -> Dict[str, Any]:
        """Get complete explanation including both forward and backward chaining"""
        return {
            "forward_chaining": self.explanation_trace,
            "backward_chaining": self.backward_chainer.get_inference_trace(),
            "proof_cache": list(self.backward_chainer.proof_cache.keys()),
        }


# Example usage
def demonstrate_backward_chaining():
    """Demonstrate backward chaining functionality"""

    # Create enhanced system
    kb = EnhancedKnowledgeBase()
    inference_engine = EnhancedInferenceEngine(kb)

    # Add some facts
    inference_engine.working_memory.add_fact(
        "patient_anna", "has_condition", "lactose_intolerant"
    )
    inference_engine.working_memory.add_fact(
        "ibuprofen", "medication_type", "anti_inflammatory"
    )

    # Example activity
    activity: Activity = {
        "activity_name": "Take Ibuprofen",
        "activity_time": "08:00",
        "activity_duration": 3,
        "critical": True,
        "priority": 1,
        "without_time": False,
        "done": False,
        "skipped": False,
        "activity_details": "",
        "execution_notes": "",
    }

    # Try to prove if we can schedule this activity
    can_schedule, explanation = inference_engine.can_schedule_activity(
        activity, "08:00"
    )

    print(f"Can schedule activity: {can_schedule}")
    print(f"Explanation: {explanation}")

    return inference_engine


if __name__ == "__main__":
    demo_engine = demonstrate_backward_chaining()
