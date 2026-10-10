from datetime import datetime, timedelta
import math
from typing import Dict, List, Callable, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import copy

#import sys
#print(sys.executable)
#print(sys.path)

from utils.logs import log_message

# Constants
TOLERANCE_START_TIME_ACTIVITY = timedelta(minutes=30)
START_DAY_TIME_LIMIT = datetime.combine(
    datetime.today(), datetime.strptime("07:00", "%H:%M").time()
)
END_DAY_TIME_LIMIT = datetime.combine(
    datetime.today(), datetime.strptime("22:00", "%H:%M").time()
)

# Possible behaviors of a rule (what to do when it triggers)
class RuleAction(Enum):
    REPLACE = "replace"
    KEEP_EXISTING = "keep_existing"
    KEEP_BOTH = "keep_both"
    SKIP = "skip"
    PROVE = "prove"
    DERIVE = "derive"

# Rule of the knowledge base
@dataclass
class Rule:
    """Represents a rule in the knowledge base"""

    name: str
    condition: Callable[[Any, Any], bool]
    action: RuleAction
    explanation: str
    priority: int = 0
    # New fields for backward chaining
    conclusions: List[str] | None = None  # -> what this rule can prove
    premises: List[str] | None = None  # -> what needs to be proven to activate this rule

# A fact ALREADY PRESENT in working memory
@dataclass
class Fact:
    """Represents a fact in the working memory"""

    subject: str
    predicate: str
    object: Any
    timestamp: datetime
    confidence: float = 1.0
    source: str = "user"  # user, forward_chain, backward_chain, derived

# Goal to be proven
@dataclass
class Goal:
    """Represents a goal to be proven in backward chaining"""

    predicate: str
    arguments: List[Any]
    confidence_threshold: float = 0.7
    max_depth: int = 5

# Node of the proof tree. Used to track the proof process of a goal, the applied rules,
# the generated sub-goals, and the final result (proven or not). This way, we can also generate
# detailed explanations of the reasoning process.
@dataclass
class ProofNode:
    """Represents a node in the proof tree"""

    goal: Goal
    rule_applied: Optional[Rule] = None
    subgoals: List["ProofNode"] | None = None
    proven: bool = False
    confidence: float = 0.0
    depth: int = 0

# Implements backward chaining, i.e., the ability to work backwards from a desired goal to find
# the facts and rules needed to prove that goal.
class BackwardChaining:
    """
    Implements backward chaining inference for goal-driven reasoning.

    This class allows the system to work backwards from a desired goal
    to find the facts and rules needed to prove that goal.
    """
    # Initialize knowledge base, working memory, proof cache, and inference trace
    def __init__(self, knowledge_base, working_memory):
        self.knowledge_base = knowledge_base
        self.working_memory = working_memory
        self.proof_cache: Dict[str, ProofNode] = {}
        self.inference_trace: List[Dict[str, Any]] = []
        self.max_recursion_depth = 10

    # Given a goal, try to prove it (main method). Checks cache - creates proof node - checks facts - finds applicable rules - tries each rule - updates cache and working memory - returns result
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

    # Check if the goal is already satisfied in working memory.
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

    # Check if a specific fact satisfies a goal
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

    # Search the knowledge base for all rules that can prove the requested goal. -> e.g., to prove "can_schedule(activity)", find all rules that have "can_schedule" in their conclusions.
    # Sort rules by priority, so that the most relevant ones are tried first.
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

    # Try to prove a goal using a specific rule.
    # Two cases:
    # Case 1 - rule without premises: Check the condition directly.
    # Case 2 - rule with premises: Generate sub-goals.

    # e.g., to prove:
    # can_schedule
    # we need to prove:
    # - time_slot_available
    # - constraints_satisfied

    # So it recursively calls prove_goal, which will call _find_rules_for_goal again
    # to find rules that can prove these sub-goals, AND SO ON UNTIL REACHING EXISTING FACTS
    # IN WORKING MEMORY, OR RULES WITHOUT PREMISES!!!! (BASE CASE OF RECURSION)
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

    # Evaluate the rule's condition, executing the condition code with the necessary parameters (e.g., goal arguments) and returning True or False.
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

    # Transform a rule premise into a new Goal. For example, if the premise is "time_slot_available", create a Goal with predicate "time_slot_available"
    # and arguments taken from the parent goal
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

    # When a goal is proven:
    # automatically adds a new derived fact to working memory.
    # So the system "learns" new information.
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

    # Generate a unique string key for caching. Used to avoid re-proving the same goals.
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

    # Log an inference attempt.
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

    # Save a proof result in the cache
    def _cache_proof(self, goal_key: str, proof_node: ProofNode):
        """
        Cache a proof result for future use.

        Args:
            goal_key: The goal key
            proof_node: The proof result
        """
        self.proof_cache[goal_key] = proof_node

    # Generate a human-readable explanation of the proof tree.
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

    # Clear the proof cache, useful to force the system to re-prove goals (e.g., if the knowledge base has been updated)
    def clear_cache(self):
        """Clear the proof cache."""
        self.proof_cache.clear()

    # Return the complete history of executed reasoning.
    def get_inference_trace(self) -> List[Dict[str, Any]]:
        """Get the complete inference trace."""
        return self.inference_trace

# Manages current facts of the system
class WorkingMemory:
    """Manages current facts and inferences with enhanced backward chaining support"""

    # Initialize lists of facts, inferred facts, and conflicts
    def __init__(self):
        self.facts: List[Fact] = []
        self.inferred_facts: List[Fact] = []
        self.conflict_set: List[Dict[str, Any]] = []

    # Add a new fact to memory, with source tracking (e.g., whether it's an original fact or was inferred by a rule)
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

    # Retrieve facts matching certain criteria
    # (e.g., all facts with predicate "time_slot_available").
    # Useful for backward chaining, when we need to verify if a goal is already satisfied by existing facts.
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

    # Record a conflict for resolution (e.g., if two activities overlap, or if one rule suggests
    # replacing an activity but another rule suggests keeping it)
    def add_conflict(
        self, new_activity: Any, existing_activity: Any, conflict_type: str
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

# Contains the system's knowledge, both as facts and rules. The knowledge base is enriched with rules specific to backward chaining,
# allowing the system to prove complex goals from simpler facts and rules. For example, a rule might state that
# "an activity can be scheduled if the time slot is available and constraints are satisfied". This way, when the system needs to prove
# that an activity can be scheduled, it can work backwards to verify if the time slot is available and if constraints are satisfied,
# proving these sub-goals first and then reaching the final conclusion.
class EnhancedKnowledgeBase:
    """Enhanced knowledge base with backward chaining support"""

    # Initialize rules and domain knowledge
    def __init__(self):
        self.rules: List[Rule] = []
        self.domain_knowledge: Dict[str, Any] = {}
        self._initialize_rules()
        self._initialize_domain_knowledge()

    # Create all rules, both for forward chaining (e.g., conflict resolution) and backward chaining (e.g., proving complex goals).
    # Backward chaining rules also include the definition of conclusions and premises, to guide the proof process.
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

    # Define global knowledge: daily limits, tolerances, medical constraints, etc. This knowledge is used by rules to evaluate whether an activity
    # can be scheduled or whether a goal can be proven.
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

    # Simplified check if an activity can be scheduled in a given time slot, verifying time slot availability,
    # constraint satisfaction, and medication timing validity. -> bypassed, returns true
    def _check_scheduling_feasibility(self, activity: Any, time_slot: Any) -> bool:
        """Check if an activity can be scheduled at a given time slot"""
        # Simplified implementation - extend based on your needs
        return True

    # Check time slot availability, e.g., verifying that no other activities are scheduled in the same slot, or that the slot falls within daily limits.
    # -> bypassed, returns true
    def _check_time_availability(self, activity: Any, time_slot: Any) -> bool:
        """Check if time slot is available"""
        # Simplified implementation
        return True

    # Check constraint satisfaction, e.g., verifying that the activity respects daily limits or medical requirements.
    # -> bypassed, returns true
    def _check_constraints(self, activity: Any, context: Any) -> bool:
        """Check if activity constraints are satisfied"""
        # Simplified implementation
        return True

    # Check validity of medication timing constraints, e.g., verifying that timing respects minimum intervals between doses or meal-related requirements.
    # -> bypassed, returns true
    def _check_medication_timing(self, activity: Any, context: Any) -> bool:
        """Check if medication timing is valid"""
        # Simplified implementation
        return True

    # Return rules applicable to a specific context (e.g., conflict resolution, goal proving, activity status evaluation).
    # This allows better organization of rules and applying only the relevant ones based on context.
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

# Class that combines both forward chaining (for conflict resolution and general rule application)
# and backward chaining (for proving specific and complex goals).
class EnhancedInferenceEngine:
    """Enhanced inference engine with both forward and backward chaining"""

    def __init__(self, knowledge_base: EnhancedKnowledgeBase):
        self.kb = knowledge_base
        self.working_memory = WorkingMemory()
        self.backward_chainer = BackwardChaining(knowledge_base, self.working_memory)
        self.explanation_trace: List[Dict[str, Any]] = []

    # Prove a specific goal using backward chaining. This is the main method called when we want to prove a goal, e.g., "can_schedule(activity)".
    # Acts as a wrapper for the backward chainer's prove_goal
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

    # Check if an activity can be scheduled in a given time slot, using backward chaining to prove the goal "can_schedule(activity, time_slot)".
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

    # Apply forward chaining rules to resolve conflicts and update activity status.
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

    # Should apply a rule.
    # In the future, this function can be extended to implement the specific logic of each rule, e.g.,
    # replacing an activity with another, keeping both, or skipping an activity based on the rule's conditions.
    # -> bypassed, returns activities unchanged
    def _apply_rule(self, rule: Rule, activities: List[Activity]) -> List[Activity]:
        """Apply a single rule to the activity list (existing functionality)"""
        # Implementation same as before
        return activities

    # Return a complete explanation of the inference process, including both the rules
    # applied during forward chaining and the proof process of backward chaining.
    def get_complete_explanation(self) -> Dict[str, Any]:
        """Get complete explanation including both forward and backward chaining"""
        return {
            "forward_chaining": self.explanation_trace,
            "backward_chaining": self.backward_chainer.get_inference_trace(),
            "proof_cache": list(self.backward_chainer.proof_cache.keys()),
        }


# ----------------
# EXAMPLE
# ----------------
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
