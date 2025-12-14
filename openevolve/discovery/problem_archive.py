"""
Problem Archive for Paired Open‑Ended Co‑Evolution (POET‑style).

The archive keeps multiple active ProblemSpace instances and assigns solver
islands to them. It is intentionally lightweight and heuristic‑driven so it can
run inside OpenEvolve’s existing loop without requiring evaluator rewrites.

Responsibilities:
1. Track per‑problem progress (best fitness, solutions, improvement recency).
2. Decide when a problem should spawn a mutated child problem.
3. Gate admission of new problems by simple novelty/difficulty checks.
4. Assign/reassign islands to active problems.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from openevolve.discovery.problem_space import ProblemSpace

logger = logging.getLogger(__name__)


@dataclass
class ProblemArchiveConfig:
    """Configuration for multi‑problem co‑evolution."""

    enabled: bool = False
    max_active_problems: int = 5
    spawn_after_solutions: int = 5
    novelty_threshold: float = 0.15
    min_difficulty: float = 0.5
    max_difficulty: float = 10.0
    min_islands_per_problem: int = 1


@dataclass
class ProblemRecord:
    """Internal record of a problem’s co‑evolution status."""

    problem: ProblemSpace
    island_ids: list[int] = field(default_factory=list)
    best_fitness: float = 0.0
    solutions: int = 0
    last_improvement_iteration: int = 0
    fitness_history: list[float] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def record_evaluation(
        self,
        fitness: float,
        iteration: int,
        program_id: str,
        solution_threshold: float,
    ) -> None:
        self.fitness_history.append(float(fitness))
        if fitness > self.best_fitness + 1e-9:
            self.best_fitness = float(fitness)
            self.last_improvement_iteration = int(iteration)

        if fitness >= solution_threshold and program_id not in self.problem.solved_by:
            self.problem.solved_by.append(program_id)
            self.solutions = len(self.problem.solved_by)


class ProblemArchive:
    """Archive of active problems and island assignments."""

    def __init__(self, config: ProblemArchiveConfig, num_islands: int):
        self.config = config
        self.num_islands = num_islands
        self.records: dict[str, ProblemRecord] = {}
        self.island_to_problem: dict[int, str] = {}

    def add_problem(self, problem: ProblemSpace) -> None:
        if problem.id in self.records:
            return
        self.records[problem.id] = ProblemRecord(problem=problem)
        logger.info(f"ProblemArchive: added problem {problem.id}")

    def active_problems(self) -> list[ProblemSpace]:
        return [r.problem for r in self.records.values()]

    def get_record(self, problem_id: str) -> ProblemRecord | None:
        return self.records.get(problem_id)

    def get_problem_for_island(self, island_id: int) -> ProblemSpace | None:
        pid = self.island_to_problem.get(island_id)
        if pid and pid in self.records:
            return self.records[pid].problem
        return None

    def assign_island(self, problem_id: str, island_id: int) -> None:
        island_id = island_id % self.num_islands
        if problem_id not in self.records:
            raise KeyError(f"Unknown problem id {problem_id}")

        # Remove from old record if present
        old_pid = self.island_to_problem.get(island_id)
        if old_pid and old_pid in self.records:
            old_rec = self.records[old_pid]
            if island_id in old_rec.island_ids:
                old_rec.island_ids.remove(island_id)

        self.island_to_problem[island_id] = problem_id
        rec = self.records[problem_id]
        if island_id not in rec.island_ids:
            rec.island_ids.append(island_id)

    def initialize_genesis(self, genesis_problem: ProblemSpace) -> None:
        self.add_problem(genesis_problem)
        for island in range(self.num_islands):
            self.assign_island(genesis_problem.id, island)

    def should_spawn_child(self, problem_id: str) -> bool:
        rec = self.records.get(problem_id)
        if not rec:
            return False
        return rec.solutions >= self.config.spawn_after_solutions

    def admit_candidate(self, candidate: ProblemSpace) -> bool:
        """Gate new problem admission by novelty and difficulty."""
        if len(self.records) >= self.config.max_active_problems:
            logger.debug("ProblemArchive: max_active_problems reached, rejecting candidate")
            return False

        if not (
            self.config.min_difficulty <= candidate.difficulty_level <= self.config.max_difficulty
        ):
            logger.debug(
                f"ProblemArchive: candidate difficulty {candidate.difficulty_level:.2f} out of bounds"
            )
            return False

        novelty = self._novelty_score(candidate)
        if novelty < self.config.novelty_threshold:
            logger.debug(f"ProblemArchive: candidate novelty {novelty:.3f} below threshold")
            return False

        return True

    def allocate_island_for_new_problem(self) -> int | None:
        """Pick an island to reassign to a new problem."""
        # Prefer unassigned islands
        for island in range(self.num_islands):
            if island not in self.island_to_problem:
                return island

        # Otherwise pick from a problem that is already solved or has spare islands.
        candidates: list[tuple[float, int, int]] = []
        # score, island_id, solutions
        for island, pid in self.island_to_problem.items():
            rec = self.records.get(pid)
            if not rec:
                continue
            # Keep at least min_islands_per_problem unless solved
            if (
                len(rec.island_ids) <= self.config.min_islands_per_problem
                and rec.solutions < self.config.spawn_after_solutions
            ):
                continue
            # Lower score => more eligible for reassignment
            stagnation = rec.last_improvement_iteration
            score = -(rec.solutions) - 0.01 * stagnation
            candidates.append((score, island, rec.solutions))

        if not candidates:
            return None

        candidates.sort()
        return candidates[0][1]

    def record_program_result(
        self,
        problem_id: str,
        fitness: float,
        iteration: int,
        program_id: str,
        solution_threshold: float,
    ) -> None:
        rec = self.records.get(problem_id)
        if rec is None:
            return
        rec.record_evaluation(fitness, iteration, program_id, solution_threshold)

    def _novelty_score(self, candidate: ProblemSpace) -> float:
        """Compute novelty against existing problems (0..1)."""
        if not self.records:
            return 1.0
        sims = [self._problem_similarity(candidate, r.problem) for r in self.records.values()]
        max_sim = max(sims) if sims else 0.0
        return max(0.0, 1.0 - max_sim)

    def _problem_similarity(self, a: ProblemSpace, b: ProblemSpace) -> float:
        """Cheap similarity using token overlap of description/constraints/objectives."""

        def tokens(p: ProblemSpace) -> set[str]:
            parts = [p.description, *p.constraints, *p.objectives]
            txt = " ".join(parts).lower()
            toks = re.findall(r"[a-z0-9_]+", txt)
            return set(toks)

        ta = tokens(a)
        tb = tokens(b)
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        union = len(ta | tb)
        jacc = inter / max(1, union)
        return float(jacc)
