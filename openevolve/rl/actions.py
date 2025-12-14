"""
Action definitions for RL-based adaptive selection

This module defines the action space for the RL policy learner.
"""

from dataclasses import dataclass
from enum import IntEnum


class SelectionAction(IntEnum):
    """Primary selection actions for parent selection

    These actions determine how the next parent program is selected
    from the population for evolution.
    """

    EXPLORATION = 0  # Random sampling for diversity
    EXPLOITATION = 1  # Elite/archive sampling for refinement
    WEIGHTED = 2  # Fitness-proportional sampling
    NOVELTY = 3  # Novelty-seeking (maximize behavioral difference)
    CURIOSITY = 4  # Sample programs with high uncertainty/surprise

    @classmethod
    def from_string(cls, name: str) -> "SelectionAction":
        """Convert string name to action"""
        name_map = {
            "exploration": cls.EXPLORATION,
            "exploitation": cls.EXPLOITATION,
            "weighted": cls.WEIGHTED,
            "novelty": cls.NOVELTY,
            "curiosity": cls.CURIOSITY,
        }
        return name_map.get(name.lower(), cls.WEIGHTED)

    def to_string(self) -> str:
        """Convert action to string name"""
        return self.name.lower()


@dataclass
class ExtendedAction:
    """Extended action space for comprehensive control

    This allows the RL system to control multiple aspects of evolution
    beyond just parent selection.
    """

    selection_mode: SelectionAction
    temperature_modifier: float = 0.0  # -0.3 to +0.3 adjustment
    use_diff_evolution: bool = True  # True = diff, False = full rewrite
    island_target: int | None = None  # Specific island or None for current

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "selection_mode": self.selection_mode.to_string(),
            "temperature_modifier": self.temperature_modifier,
            "use_diff_evolution": self.use_diff_evolution,
            "island_target": self.island_target,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExtendedAction":
        """Create from dictionary"""
        return cls(
            selection_mode=SelectionAction.from_string(data.get("selection_mode", "weighted")),
            temperature_modifier=data.get("temperature_modifier", 0.0),
            use_diff_evolution=data.get("use_diff_evolution", True),
            island_target=data.get("island_target"),
        )


# Number of primary actions (used for policy initialization)
NUM_ACTIONS = len(SelectionAction)

# Action names for logging
ACTION_NAMES = [a.to_string() for a in SelectionAction]
