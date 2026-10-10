from datetime import datetime
from typing import List, Dict, Any, Optional

from planning.canonical_actions import CANONICAL_ACTIONS
from deterministic.backward_chaining import (
    BackwardChaining,
    Rule,
    RuleAction,
    Goal,
    WorkingMemory,
    ProofNode,
)
from utils.logs import log_message


class _SimpleKnowledgeBase:
    """Minimal knowledge base wrapper holding a list of rules."""

    def __init__(self, rules: List[Rule]):
        self.rules = rules


class GoalDecomposer:
    """
    Decomposes a high-level goal into an ordered chain of canonical actions
    using backward chaining over the canonical action knowledge base.

    Each canonical action is treated as a rule:
      - conclusions = effects  (what the action achieves)
      - premises    = preconditions (what must be true before the action)

    The backward chainer works backwards from the goal predicate, finding
    actions whose effects satisfy the goal, then recursively proving their
    preconditions.  The result is an ordered list of actions (dependencies
    first) that the patient must perform to achieve the goal.
    """

    def __init__(self):
        self.canonical_actions = CANONICAL_ACTIONS

    def _build_rules(self) -> List[Rule]:
        """Build backward chaining rules from CANONICAL_ACTIONS."""
        rules: List[Rule] = []
        for action_name, action_def in self.canonical_actions.items():
            effects = action_def.get("effects", [])
            preconditions = action_def.get("preconditions", [])

            rules.append(
                Rule(
                    name=action_name,
                    condition=lambda action, context: True,
                    action=RuleAction.PROVE,
                    explanation=(
                        f"Action '{action_name}' achieves: "
                        f"{', '.join(effects) if effects else 'nothing'}"
                    ),
                    priority=0,
                    conclusions=effects if effects else None,
                    premises=preconditions if preconditions else None,
                )
            )
        return rules

    def _populate_working_memory(
        self, world_state: Dict[str, bool]
    ) -> WorkingMemory:
        """Populate working memory with facts from the world state dict."""
        wm = WorkingMemory()
        for predicate, value in world_state.items():
            if value:
                wm.add_fact(
                    subject="world",
                    predicate=predicate,
                    object=True,
                    confidence=1.0,
                    source="world_state",
                )
        return wm

    def decompose_goal(
        self,
        goal_predicate: str,
        world_state: Dict[str, bool],
        existing_activities: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Decompose a goal into an ordered chain of canonical actions.

        Args:
            goal_predicate: The goal to achieve (e.g., "rehabilitation_completed")
            world_state: Current world state dict
            existing_activities: Activities already in the patient's schedule.
                Actions whose effects are already achieved by completed
                activities are skipped.

        Returns:
            Ordered list of dicts, each containing:
              - canonical_action: str
              - description: str
              - preconditions: List[str]
              - effects: List[str]
            Dependencies come first (e.g., have_lunch before stretching
            before full_rehabilitation_session).
        """
        rules = self._build_rules()
        kb = _SimpleKnowledgeBase(rules)
        wm = self._populate_working_memory(world_state)

        # If the goal is already satisfied in the world state, no actions needed
        if world_state.get(goal_predicate, False):
            log_message(
                f"Goal '{goal_predicate}' is already satisfied in the world state"
            )
            return []

        chainer = BackwardChaining(kb, wm)
        goal = Goal(
            predicate=goal_predicate,
            arguments=[],
            confidence_threshold=0.7,
        )

        proof = chainer.prove_goal(goal)

        if not proof.proven:
            log_message(
                f"Goal '{goal_predicate}' could not be decomposed: "
                f"no action chain found to achieve it"
            )
            return []

        # Extract ordered action chain from the proof tree (depth-first,
        # dependencies first)
        action_chain: List[Dict[str, Any]] = []
        self._extract_action_chain(proof, action_chain)

        # Filter out actions already completed in existing activities
        if existing_activities:
            completed_effects = set()
            for activity in existing_activities:
                if activity.get("done", False):
                    ca = activity.get("canonical_action")
                    if ca and ca in self.canonical_actions:
                        completed_effects.update(
                            self.canonical_actions[ca].get("effects", [])
                        )
            # Remove actions whose effects are already achieved
            action_chain = [
                a for a in action_chain
                if not all(eff in completed_effects for eff in a["effects"])
            ]

        log_message(
            f"Goal '{goal_predicate}' decomposed into {len(action_chain)} "
            f"actions: {[a['canonical_action'] for a in action_chain]}"
        )

        return action_chain

    def _extract_action_chain(
        self,
        proof_node: ProofNode,
        action_chain: List[Dict[str, Any]],
    ):
        """
        Recursively extract the ordered chain of actions from the proof tree.
        Depth-first traversal ensures dependencies are listed before
        the actions that depend on them.
        """
        # Recurse into subgoals first (dependencies)
        if proof_node.subgoals:
            for subgoal in proof_node.subgoals:
                self._extract_action_chain(subgoal, action_chain)

        # Add the rule's action if it's a canonical action
        if proof_node.rule_applied:
            action_name = proof_node.rule_applied.name
            # Avoid duplicates
            if not any(
                a["canonical_action"] == action_name for a in action_chain
            ):
                action_def = self.canonical_actions.get(action_name, {})
                action_chain.append(
                    {
                        "canonical_action": action_name,
                        "description": action_def.get("description", ""),
                        "preconditions": action_def.get("preconditions", []),
                        "effects": action_def.get("effects", []),
                    }
                )

    def actions_to_activities(
        self,
        action_chain: List[Dict[str, Any]],
        default_duration: int = 30,
        default_priority: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Convert an ordered action chain into activity dicts compatible
        with the GraphState task structure.

        Args:
            action_chain: Output of decompose_goal()
            default_duration: Default duration in minutes
            default_priority: Default priority (lower = higher priority)

        Returns:
            List of Activity dicts ready to be appended to a task
        """
        activities = []
        for action in action_chain:
            activities.append(
                {
                    "activity_name": action["description"],
                    "activity_time": None,
                    "without_time": True,
                    "activity_duration": default_duration,
                    "activity_details": (
                        f"Canonical action: {action['canonical_action']}. "
                        f"Preconditions: {', '.join(action['preconditions'])}. "
                        f"Effects: {', '.join(action['effects'])}."
                    ),
                    "execution_notes": "",
                    "done": False,
                    "skipped": False,
                    "critical": False,
                    "priority": default_priority,
                    "canonical_action": action["canonical_action"],
                }
            )
        return activities
