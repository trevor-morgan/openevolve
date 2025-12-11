"""
Problem Space Evolution for Open-Ended Discovery

This module implements the "Explorer" component - evolving the questions,
not just the answers. Scientific discovery requires exploring new problem
formulations, not just optimizing solutions to fixed problems.
"""

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from openevolve.llm.ensemble import LLMEnsemble

logger = logging.getLogger(__name__)


@dataclass
class ProblemSpace:
    """
    Represents an evolvable problem/research question.

    Unlike static evaluation files, ProblemSpace can mutate to:
    - Add constraints ("sort without comparisons")
    - Change objectives ("optimize for memory, not speed")
    - Expand scope ("handle streaming data")
    - Increase difficulty ("handle adversarial inputs")

    Attributes:
        id: Unique identifier for this problem variant
        parent_id: ID of the problem this was derived from (None for genesis)
        description: Natural language description of the problem
        constraints: List of constraints that solutions must satisfy
        objectives: List of objectives to optimize (can have multiple)
        difficulty_level: Estimated difficulty (increases with mutations)
        generation: How many mutations from the original problem
        evaluation_template: Code template for evaluating solutions
        test_cases: Known test cases for this problem variant
        metadata: Additional problem-specific data
        ontology_id: ID of the ontology (state space) associated with this problem
        ontology_generation: Generation of the ontology when problem was created
        known_variables: List of variable names in the current ontology
    """
    id: str
    description: str
    constraints: List[str] = field(default_factory=list)
    objectives: List[str] = field(default_factory=lambda: ["correctness"])
    difficulty_level: float = 1.0
    generation: int = 0
    parent_id: Optional[str] = None
    evaluation_template: Optional[str] = None
    test_cases: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    # Track which solutions have "solved" this problem
    solved_by: List[str] = field(default_factory=list)

    # Ontology (state space) tracking - for Heisenberg Engine
    ontology_id: Optional[str] = None
    ontology_generation: int = 0
    known_variables: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProblemSpace":
        """Deserialize from dictionary"""
        return cls(**data)

    def to_prompt_context(self) -> str:
        """Format problem for inclusion in LLM prompts"""
        lines = [
            f"## Problem: {self.description}",
            "",
        ]

        if self.constraints:
            lines.append("### Constraints:")
            for c in self.constraints:
                lines.append(f"- {c}")
            lines.append("")

        if self.objectives:
            lines.append("### Objectives (in priority order):")
            for i, obj in enumerate(self.objectives, 1):
                lines.append(f"{i}. {obj}")
            lines.append("")

        # Include ontology information if available
        if self.known_variables:
            lines.append("### Known Variables (State Space):")
            for var in self.known_variables:
                lines.append(f"- {var}")
            lines.append("")
            lines.append(f"Ontology Generation: {self.ontology_generation}")
            lines.append("")

        lines.append(f"Difficulty Level: {self.difficulty_level:.1f}")
        lines.append(f"Problem Generation: {self.generation}")

        return "\n".join(lines)

    def update_for_ontology(
        self,
        ontology_id: str,
        ontology_generation: int,
        variable_names: List[str],
        variable_descriptions: Optional[Dict[str, str]] = None,
    ) -> "ProblemSpace":
        """
        Create a new problem variant updated with ontology information.

        This is called when the Heisenberg Engine discovers new variables -
        the problem description is updated to reflect the expanded state space.

        Args:
            ontology_id: ID of the new ontology
            ontology_generation: Generation number of the ontology
            variable_names: List of variable names in the ontology
            variable_descriptions: Optional dict mapping var names to descriptions

        Returns:
            New ProblemSpace with ontology context added
        """
        import uuid

        # Build description addition for new variables
        new_vars = [v for v in variable_names if v not in self.known_variables]
        desc_addition = ""
        if new_vars:
            var_list = ", ".join(new_vars)
            desc_addition = f"\n\n[ONTOLOGY EXPANDED: New variables discovered: {var_list}. " \
                           f"These may affect solution performance in ways not previously considered.]"

        # Create new problem with ontology info
        new_problem = ProblemSpace(
            id=f"prob_{uuid.uuid4().hex[:8]}",
            parent_id=self.id,
            description=self.description + desc_addition,
            constraints=self.constraints.copy(),
            objectives=self.objectives.copy(),
            difficulty_level=self.difficulty_level,
            generation=self.generation,  # Don't increment - not a problem mutation
            evaluation_template=self.evaluation_template,
            test_cases=self.test_cases.copy(),
            metadata={
                **self.metadata,
                "ontology_expanded": True,
                "previous_ontology_id": self.ontology_id,
                "new_variables": new_vars,
            },
            solved_by=[],  # Reset solutions for new ontology context
            ontology_id=ontology_id,
            ontology_generation=ontology_generation,
            known_variables=variable_names,
        )

        return new_problem

    def is_solved(self, threshold: float = 0.9) -> bool:
        """Check if this problem has been adequately solved"""
        return len(self.solved_by) > 0


@dataclass
class ProblemEvolverConfig:
    """Configuration for problem evolution"""

    # Mutation probabilities
    add_constraint_prob: float = 0.4
    modify_objective_prob: float = 0.3
    increase_difficulty_prob: float = 0.2
    expand_scope_prob: float = 0.1

    # Evolution parameters
    max_constraints: int = 10
    max_difficulty: float = 10.0
    difficulty_increment: float = 0.5

    # LLM settings for mutation
    mutation_temperature: float = 0.9
    mutation_max_tokens: int = 1024


# Default prompt for problem mutation
PROBLEM_MUTATION_SYSTEM = """You are a research director who designs increasingly challenging variants of scientific problems.

Your role is to take a solved problem and create a MORE INTERESTING variant by:
1. Adding constraints that force novel algorithmic approaches
2. Modifying objectives to explore different trade-off spaces
3. Expanding scope to handle more general cases
4. Increasing difficulty to push the boundaries of what's possible

You should create problems that are:
- Scientifically interesting (not just harder for the sake of it)
- Feasible to solve (not impossible)
- Novel (exploring unexplored regions of the problem space)

Output your response as JSON with the following structure:
{
    "description": "New problem description",
    "constraints": ["constraint1", "constraint2", ...],
    "objectives": ["objective1", "objective2", ...],
    "difficulty_delta": 0.5,
    "rationale": "Why this variant is scientifically interesting"
}"""

PROBLEM_MUTATION_USER = """The following problem has been solved:

{problem_context}

Previous solutions achieved these characteristics:
{solution_characteristics}

Generate a MORE CHALLENGING and SCIENTIFICALLY INTERESTING variant of this problem.

Mutation type requested: {mutation_type}

Remember:
- The new problem should explore unexplored regions of the solution space
- It should require genuinely different algorithmic approaches
- It should be feasible but push the boundaries"""


class ProblemEvolver:
    """
    Evolves the problem space to enable open-ended discovery.

    This is the "Explorer" module that prevents the system from
    just optimizing a fixed objective. When a problem is "solved",
    this component mutates it into a harder/more interesting variant.

    Key insight: Scientific discovery comes from asking NEW QUESTIONS,
    not just finding better answers to old questions.
    """

    def __init__(
        self,
        config: ProblemEvolverConfig,
        llm_ensemble: Optional["LLMEnsemble"] = None,
    ):
        self.config = config
        self.llm_ensemble = llm_ensemble

        # Track problem lineage
        self.problem_history: Dict[str, ProblemSpace] = {}
        self.current_problem: Optional[ProblemSpace] = None

        logger.info("Initialized ProblemEvolver")

    def set_genesis_problem(self, problem: ProblemSpace) -> None:
        """Set the initial problem (generation 0)"""
        self.problem_history[problem.id] = problem
        self.current_problem = problem
        logger.info(f"Set genesis problem: {problem.id}")

    def create_genesis_from_evaluator(
        self,
        evaluator_path: str,
        description: str,
    ) -> ProblemSpace:
        """
        Create a genesis problem from an existing evaluator file.

        This bridges OpenEvolve's current evaluator-based approach
        to the evolvable problem space model.
        """
        # Read the evaluator file
        with open(evaluator_path, 'r') as f:
            eval_code = f.read()

        problem = ProblemSpace(
            id=f"genesis_{uuid.uuid4().hex[:8]}",
            description=description,
            evaluation_template=eval_code,
            metadata={
                "evaluator_path": evaluator_path,
                "is_genesis": True,
            }
        )

        self.set_genesis_problem(problem)
        return problem

    def _select_mutation_type(self) -> str:
        """Select which type of mutation to apply"""
        rand = random.random()
        cumulative = 0.0

        mutations = [
            ("add_constraint", self.config.add_constraint_prob),
            ("modify_objective", self.config.modify_objective_prob),
            ("increase_difficulty", self.config.increase_difficulty_prob),
            ("expand_scope", self.config.expand_scope_prob),
        ]

        for mutation_type, prob in mutations:
            cumulative += prob
            if rand < cumulative:
                return mutation_type

        return "add_constraint"  # Default

    async def evolve(
        self,
        parent_problem: ProblemSpace,
        solution_characteristics: Dict[str, Any],
    ) -> ProblemSpace:
        """
        Evolve a solved problem into a more challenging variant.

        Args:
            parent_problem: The problem that was solved
            solution_characteristics: Features of successful solutions
                (e.g., {"complexity": 5, "efficiency": 8, "fitness": 0.95})

        Returns:
            A new, more challenging ProblemSpace
        """
        if not self.llm_ensemble:
            # Fallback to simple mutation without LLM
            return self._simple_evolve(parent_problem)

        mutation_type = self._select_mutation_type()

        # Format the prompt
        user_prompt = PROBLEM_MUTATION_USER.format(
            problem_context=parent_problem.to_prompt_context(),
            solution_characteristics=json.dumps(solution_characteristics, indent=2),
            mutation_type=mutation_type,
        )

        try:
            response = await self.llm_ensemble.generate_with_context(
                system_message=PROBLEM_MUTATION_SYSTEM,
                messages=[{"role": "user", "content": user_prompt}],
            )

            # Parse JSON response
            mutation_data = self._parse_mutation_response(response)

            # Create new problem
            new_problem = ProblemSpace(
                id=f"prob_{uuid.uuid4().hex[:8]}",
                parent_id=parent_problem.id,
                description=mutation_data.get("description", parent_problem.description),
                constraints=parent_problem.constraints + mutation_data.get("constraints", []),
                objectives=mutation_data.get("objectives", parent_problem.objectives),
                difficulty_level=min(
                    parent_problem.difficulty_level + mutation_data.get("difficulty_delta", 0.5),
                    self.config.max_difficulty
                ),
                generation=parent_problem.generation + 1,
                evaluation_template=parent_problem.evaluation_template,
                test_cases=parent_problem.test_cases.copy(),
                metadata={
                    "mutation_type": mutation_type,
                    "rationale": mutation_data.get("rationale", ""),
                    "parent_solutions": solution_characteristics,
                },
            )

            # Ensure constraints don't exceed limit
            if len(new_problem.constraints) > self.config.max_constraints:
                new_problem.constraints = new_problem.constraints[-self.config.max_constraints:]

            self.problem_history[new_problem.id] = new_problem

            logger.info(
                f"Problem evolved: {parent_problem.id} -> {new_problem.id} "
                f"(type: {mutation_type}, difficulty: {new_problem.difficulty_level:.1f})"
            )

            return new_problem

        except Exception as e:
            logger.warning(f"LLM mutation failed: {e}, falling back to simple evolution")
            return self._simple_evolve(parent_problem)

    def _simple_evolve(self, parent_problem: ProblemSpace) -> ProblemSpace:
        """Simple evolution without LLM (fallback)"""
        import random

        new_constraints = parent_problem.constraints.copy()

        # Add a simple constraint
        simple_constraints = [
            "Minimize memory usage",
            "Avoid recursion",
            "No external library imports",
            "Handle edge cases gracefully",
            "Optimize for worst-case complexity",
            "Support incremental/streaming input",
            "Be thread-safe",
            "Handle numerical precision issues",
        ]

        available = [c for c in simple_constraints if c not in new_constraints]
        if available:
            new_constraints.append(random.choice(available))

        new_problem = ProblemSpace(
            id=f"prob_{uuid.uuid4().hex[:8]}",
            parent_id=parent_problem.id,
            description=parent_problem.description,
            constraints=new_constraints,
            objectives=parent_problem.objectives,
            difficulty_level=min(
                parent_problem.difficulty_level + self.config.difficulty_increment,
                self.config.max_difficulty
            ),
            generation=parent_problem.generation + 1,
            evaluation_template=parent_problem.evaluation_template,
            test_cases=parent_problem.test_cases.copy(),
            metadata={"mutation_type": "simple_constraint"},
        )

        self.problem_history[new_problem.id] = new_problem
        return new_problem

    def _parse_mutation_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for mutation data"""
        import re

        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: return empty dict (will use defaults)
        logger.warning("Could not parse mutation response as JSON")
        return {}

    def get_problem_lineage(self, problem_id: str) -> List[ProblemSpace]:
        """Get the full lineage of a problem back to genesis"""
        lineage = []
        current_id = problem_id

        while current_id and current_id in self.problem_history:
            problem = self.problem_history[current_id]
            lineage.append(problem)
            current_id = problem.parent_id

        return list(reversed(lineage))

    def save(self, path: str) -> None:
        """Save problem evolution history"""
        data = {
            "problems": {pid: p.to_dict() for pid, p in self.problem_history.items()},
            "current_problem_id": self.current_problem.id if self.current_problem else None,
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved problem history with {len(self.problem_history)} problems")

    def load(self, path: str) -> None:
        """Load problem evolution history"""
        with open(path, 'r') as f:
            data = json.load(f)

        self.problem_history = {
            pid: ProblemSpace.from_dict(pdata)
            for pid, pdata in data.get("problems", {}).items()
        }

        current_id = data.get("current_problem_id")
        if current_id and current_id in self.problem_history:
            self.current_problem = self.problem_history[current_id]

        logger.info(f"Loaded problem history with {len(self.problem_history)} problems")


# Need this import for _select_mutation_type
import random
