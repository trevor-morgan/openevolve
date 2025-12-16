"""
Core architectural interfaces for OpenEvolve.

These protocols define the Perception-Reasoning-Action (P-R-A) cycle
and state management contracts to ensure modularity and resilience.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Stateful(Protocol):
    """
    Protocol for components that persist state.
    """

    def save_state(self) -> dict[str, Any]:
        """
        Return the current state as a dictionary.

        The returned dictionary must be JSON-serializable.
        """
        ...

    def load_state(self, state: dict[str, Any]) -> None:
        """
        Restore state from a dictionary.
        """
        ...


class Component(Stateful, Protocol):
    """
    Base protocol for active system components with lifecycle management.
    """

    def start(self) -> None:
        """Initialize and start the component."""
        ...

    def stop(self) -> None:
        """Stop and cleanup the component."""
        ...


class Perceiver(Protocol):
    """
    Perception component: Observes the world state.

    Responsible for gathering context from the database or environment
    to form a coherent observation for the Planner.
    """

    def observe(self, context: Any = None) -> Any:
        """
        Gather information from the environment/database.

        Returns:
            A snapshot or observation object representing the current state.
        """
        ...


class Planner(Stateful, Protocol):
    """
    Reasoning component: Decides the next action.

    Responsible for strategy, logic, and decision making.
    Does not execute side-effects in the environment (pure logic).
    """

    def plan(self, observation: Any) -> Any:
        """
        Decide on an intention/action based on observation.

        Args:
            observation: The state snapshot from the Perceiver.

        Returns:
            A plan or action object describing what to do next.
        """
        ...


class Actor(Protocol):
    """
    Action component: Executes the plan.

    Responsible for realizing the Planner's intention in the world
    (e.g., generating code via LLM, running evaluations).
    """

    async def act(self, plan: Any) -> Any:
        """
        Execute the planned action.

        Args:
            plan: The intention output by the Planner.

        Returns:
            The result of the action (e.g., EvaluationResult).
        """
        ...
