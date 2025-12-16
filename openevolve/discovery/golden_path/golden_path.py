"""
GoldenPath - The Orchestrator of Ontological Discovery

"I am the Kwisatz Haderach. I am the one who can be many places at once."

The Golden Path is Leto II's prescient vision of humanity's survival - a path
that requires seeing beyond what is currently visible. In OpenEvolve, it is
the framework that discovers hidden variables when evolution hits true walls.

The Golden Path operates autonomously:
1. Prescience watches evolution, detecting true crises (not just slow progress)
2. When an ONTOLOGY_GAP is detected, the path activates
3. Mentat mines programs for hidden patterns
4. SietchFinder proposes hidden variables that might explain success
5. GomJabbar validates these hypotheses rigorously
6. SpiceAgony integrates validated variables into the system

"Without change something sleeps inside us, and seldom awakens.
The sleeper must awaken."
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from .gom_jabbar import GomJabbar, GomJabbarConfig
from .mentat import Mentat, MentatConfig
from .prescience import CrisisType, Prescience, PrescienceConfig, PrescienceReading
from .sietch_finder import SietchFinder, SietchFinderConfig
from .spice_agony import SpiceAgony, SpiceAgonyConfig

logger = logging.getLogger(__name__)


@dataclass
class GoldenPathConfig:
    """Configuration for the Golden Path framework."""

    # Component configs
    prescience: PrescienceConfig = field(default_factory=PrescienceConfig)
    mentat: MentatConfig = field(default_factory=MentatConfig)
    sietch_finder: SietchFinderConfig = field(default_factory=SietchFinderConfig)
    gom_jabbar: GomJabbarConfig = field(default_factory=GomJabbarConfig)
    spice_agony: SpiceAgonyConfig = field(default_factory=SpiceAgonyConfig)

    # Orchestration settings
    min_programs_for_discovery: int = 30
    max_discovery_rounds: int = 5
    cooldown_after_discovery: int = 20  # Iterations to wait after discovery

    # Logging
    log_all_readings: bool = False
    save_discoveries_to_file: bool = True
    discoveries_output_path: str = "golden_path_discoveries.py"


@dataclass
class DiscoveryRound:
    """Record of a discovery round."""

    round_number: int
    iteration: int
    crisis_type: CrisisType
    patterns_found: int
    hypotheses_generated: int
    variables_validated: int
    variables_integrated: int
    discoveries: list[str]  # Names of discovered variables


class GoldenPath:
    """
    The Golden Path - orchestrates autonomous ontological discovery.

    "The concept of progress acts as a protective mechanism to shield us
    from the terrors of the future."

    Usage:
        golden_path = GoldenPath(config, llm_ensemble, domain_context, evaluator_path)

        # In evolution loop:
        golden_path.observe_iteration(iteration, program)

        if golden_path.should_activate():
            await golden_path.run_discovery()

        # Get score adjustment from discoveries
        adjustment = golden_path.get_score_adjustment(code, metrics)
    """

    def __init__(
        self,
        config: GoldenPathConfig | None = None,
        llm_ensemble: Any | None = None,
        domain_context: str = "",
        evaluator_path: str | None = None,
    ):
        self.config = config or GoldenPathConfig()

        # Initialize components
        self.prescience = Prescience(self.config.prescience)
        self.mentat = Mentat(self.config.mentat)
        self.sietch_finder = SietchFinder(
            self.config.sietch_finder,
            llm_ensemble=llm_ensemble,
            domain_context=domain_context,
        )
        self.gom_jabbar = GomJabbar(self.config.gom_jabbar)
        self.spice_agony = SpiceAgony(
            self.config.spice_agony,
            evaluator_path=evaluator_path,
        )

        # Initialize DiscoveryToolkit for external tool orchestration
        self.toolkit = self._init_toolkit(llm_ensemble)

        self.llm_ensemble = llm_ensemble
        self.domain_context = domain_context

        # State
        self.discovery_rounds: list[DiscoveryRound] = []
        self.program_archive: list[dict[str, Any]] = []  # Archive of programs for mining
        self.current_metrics: list[str] = []  # Currently tracked metrics
        self.last_discovery_iteration: int = 0
        self.is_active: bool = False
        self.toolkit_discoveries: list[Any] = []  # Discoveries from external tools

        logger.info("Golden Path initialized - awaiting awakening")
        if self.toolkit:
            available = self.toolkit.get_available_tools()
            logger.info(f"  DiscoveryToolkit: {len(available)} tools available")
            for tool in available:
                logger.info(f"    - {tool.name}: {tool.description[:50]}...")

    def _init_toolkit(self, llm_ensemble: Any | None = None):
        """Initialize the DiscoveryToolkit with available tools."""
        try:
            from .toolkit import create_default_toolkit

            toolkit = create_default_toolkit(llm_ensemble=llm_ensemble)
            return toolkit
        except Exception as e:
            logger.warning(f"Failed to initialize DiscoveryToolkit: {e}")
            return None

    async def _run_toolkit_discovery(self) -> list[Any]:
        """Run external discovery tools via the DiscoveryToolkit."""
        if not self.toolkit:
            return []

        available_tools = self.toolkit.get_available_tools()
        if not available_tools:
            logger.info("  No discovery tools available")
            return []

        logger.info(f"  Running {len(available_tools)} discovery tools...")
        for tool in available_tools:
            logger.info(f"    - {tool.name}")

        # Create tool context
        from .toolkit import ToolContext

        # Find best program for context
        best_program = (
            max(self.program_archive, key=lambda p: p.get("fitness", 0))
            if self.program_archive
            else {}
        )

        context = ToolContext(
            programs=self.program_archive,
            current_metrics=self.current_metrics,
            domain_context=self.domain_context,
            crisis_type=self.prescience.readings_history[-1].crisis_type.value
            if self.prescience.readings_history
            else "unknown",
            crisis_details={
                "last_reading": self.prescience.readings_history[-1].details
                if self.prescience.readings_history
                else {},
            },
            best_fitness=best_program.get("fitness", 0.0),
            best_program_code=best_program.get("code", ""),
            discovery_goal="Find hidden variables that explain fitness variance beyond current metrics",
        )

        # Run tools - either via LLM selection or all available
        try:
            if self.llm_ensemble:
                discoveries = await self.toolkit.select_and_run(context)
            else:
                discoveries = await self.toolkit.run_all_available(context)

            # Log discoveries by tool
            by_tool = {}
            for d in discoveries:
                tool = d.source_tool or "unknown"
                by_tool.setdefault(tool, []).append(d)

            for tool, tool_discoveries in by_tool.items():
                logger.info(f"    {tool}: {len(tool_discoveries)} discoveries")
                for d in tool_discoveries[:2]:  # Show first 2
                    logger.info(f"      - {d.name}: {d.description[:60]}...")

            return discoveries

        except Exception as e:
            logger.error(f"  Toolkit discovery failed: {e}")
            import traceback

            traceback.print_exc()
            return []

    def observe_iteration(
        self,
        iteration: int,
        fitness: float,
        metrics: dict[str, float],
        program_code: str | None = None,
        program_id: str | None = None,
    ) -> PrescienceReading | None:
        """
        Observe an iteration of evolution.

        This feeds data to Prescience and archives programs for future mining.
        """
        # Record in Prescience
        self.prescience.record_iteration(
            iteration=iteration,
            fitness=fitness,
            metrics=metrics,
            program_code=program_code,
            program_id=program_id,
        )

        # Archive program for mining
        if program_code:
            self.program_archive.append(
                {
                    "iteration": iteration,
                    "fitness": fitness,
                    "metrics": metrics.copy(),
                    "code": program_code,
                    "program_id": program_id,
                }
            )

            # Keep archive bounded
            if len(self.program_archive) > 500:
                # Keep best and most recent
                sorted_archive = sorted(
                    self.program_archive, key=lambda p: p["fitness"], reverse=True
                )
                best = sorted_archive[:250]
                recent = self.program_archive[-250:]
                self.program_archive = list({p["program_id"]: p for p in best + recent}.values())

        # Update current metrics list
        self.current_metrics = list(metrics.keys())

        # Take reading if enough data
        reading = self.prescience.take_reading(iteration)

        if self.config.log_all_readings or reading.crisis_type != CrisisType.NONE:
            if reading.crisis_type != CrisisType.NONE:
                logger.info(
                    f"Prescience reading at {iteration}: {reading.crisis_type.value} (conf={reading.confidence:.2f})"
                )

        return reading

    def should_activate(self) -> bool:
        """
        Check if the Golden Path should activate for discovery.

        Activation requires:
        1. Prescience detects ONTOLOGY_GAP
        2. Enough programs in archive
        3. Not in cooldown from previous discovery
        """
        if not self.prescience.readings_history:
            return False

        latest = self.prescience.readings_history[-1]

        # Check for ontology gap
        if latest.crisis_type != CrisisType.ONTOLOGY_GAP:
            return False

        # Check confidence
        if latest.confidence < self.config.prescience.crisis_confidence_threshold:
            return False

        # Check program archive size
        if len(self.program_archive) < self.config.min_programs_for_discovery:
            logger.info(
                f"Golden Path: insufficient programs ({len(self.program_archive)}/{self.config.min_programs_for_discovery})"
            )
            return False

        # Check cooldown
        current_iteration = self.prescience.last_reading_iteration
        if current_iteration - self.last_discovery_iteration < self.config.cooldown_after_discovery:
            return False

        return True

    async def run_discovery(self) -> DiscoveryRound:
        """
        Run a full discovery round.

        This is the main entry point for autonomous ontological discovery.
        """
        round_number = len(self.discovery_rounds) + 1
        current_iteration = self.prescience.last_reading_iteration

        logger.info("=" * 60)
        logger.info(f"GOLDEN PATH ACTIVATED - Discovery Round {round_number}")
        logger.info("=" * 60)

        self.is_active = True
        discoveries = []

        toolkit_discoveries = []

        try:
            # Phase 1: Mentat mines programs for patterns
            logger.info("Phase 1: Mentat analyzing programs...")
            patterns = self.mentat.analyze_programs(self.program_archive)
            logger.info(f"Mentat found {len(patterns)} significant patterns")

            # Phase 2: DiscoveryToolkit runs external tools
            logger.info("Phase 2: DiscoveryToolkit running external discovery tools...")
            toolkit_discoveries = await self._run_toolkit_discovery()
            logger.info(f"DiscoveryToolkit found {len(toolkit_discoveries)} discoveries")

            # Phase 3: SietchFinder proposes hidden variables (from patterns + toolkit)
            logger.info("Phase 3: SietchFinder searching for hidden variables...")
            hypotheses = await self.sietch_finder.find_hidden_variables(
                patterns=patterns,
                programs=self.program_archive,
                current_metrics=self.current_metrics,
            )

            # Add hypotheses from toolkit discoveries that have computation code
            for td in toolkit_discoveries:
                if td.computation_code and td.testable:
                    from .sietch_finder import HiddenVariable

                    hypotheses.append(
                        HiddenVariable(
                            name=td.name,
                            description=td.description,
                            computation_code=td.computation_code,
                            source=f"toolkit:{td.source_tool}",
                            expected_correlation=td.confidence,
                        )
                    )

            logger.info(f"SietchFinder + Toolkit proposed {len(hypotheses)} hidden variables")

            # Phase 4: GomJabbar validates hypotheses
            logger.info("Phase 4: GomJabbar validating hypotheses...")
            validated = []
            for hypothesis in hypotheses:
                result = self.gom_jabbar.validate(
                    variable=hypothesis,
                    programs=self.program_archive,
                    existing_metrics=self.current_metrics,
                )
                if result.passed:
                    validated.append((hypothesis, result))
                    logger.info(f"  ✓ {hypothesis.name} VALIDATED (source: {hypothesis.source})")
                else:
                    logger.info(
                        f"  ✗ {hypothesis.name} failed: {result.failure_reasons[0] if result.failure_reasons else 'unknown'}"
                    )

            logger.info(f"GomJabbar validated {len(validated)}/{len(hypotheses)} variables")

            # Phase 5: SpiceAgony integrates validated variables
            logger.info("Phase 5: SpiceAgony integrating discoveries...")
            for hypothesis, validation_result in validated:
                self.spice_agony.integrate_variable(
                    variable=hypothesis,
                    validation_result=validation_result,
                )
                discoveries.append(hypothesis.name)
                self.current_metrics.append(hypothesis.name)

            if discoveries:
                logger.info(
                    f"SpiceAgony integrated {len(discoveries)} new variables: {discoveries}"
                )

                # Export discoveries if configured
                if self.config.save_discoveries_to_file:
                    self.spice_agony.export_discovered_variables(
                        self.config.discoveries_output_path
                    )

            # Store toolkit discoveries for reference
            self.toolkit_discoveries.extend(toolkit_discoveries)

        except Exception as e:
            logger.error(f"Golden Path discovery failed: {e}")
            import traceback

            traceback.print_exc()

        finally:
            self.is_active = False
            self.last_discovery_iteration = current_iteration

        # Record the round
        round_record = DiscoveryRound(
            round_number=round_number,
            iteration=current_iteration,
            crisis_type=self.prescience.readings_history[-1].crisis_type
            if self.prescience.readings_history
            else CrisisType.NONE,
            patterns_found=len(patterns) if "patterns" in dir() else 0,
            hypotheses_generated=len(hypotheses) if "hypotheses" in dir() else 0,
            variables_validated=len(validated) if "validated" in dir() else 0,
            variables_integrated=len(discoveries),
            discoveries=discoveries,
        )
        self.discovery_rounds.append(round_record)

        logger.info("=" * 60)
        logger.info(f"GOLDEN PATH COMPLETE - Round {round_number}")
        logger.info(f"  Patterns found: {round_record.patterns_found}")
        logger.info(f"  Hypotheses generated: {round_record.hypotheses_generated}")
        logger.info(f"  Variables validated: {round_record.variables_validated}")
        logger.info(f"  Variables integrated: {round_record.variables_integrated}")
        if discoveries:
            logger.info(f"  Discoveries: {', '.join(discoveries)}")
        logger.info("=" * 60)

        return round_record

    def get_score_adjustment(
        self,
        code: str,
        metrics: dict[str, float],
    ) -> float:
        """
        Get score adjustment from discovered variables.

        Call this during evaluation to incorporate discovered variables.
        """
        return self.spice_agony.get_score_adjustment(code, metrics)

    def compute_discovered_metrics(
        self,
        code: str,
        metrics: dict[str, float],
    ) -> dict[str, float]:
        """
        Compute all discovered metrics for a program.

        Returns dict of variable_name -> value for logging/tracking.
        """
        return self.spice_agony.compute_runtime_variables(code, metrics)

    def get_state(self) -> dict[str, Any]:
        """Get current state of the Golden Path for checkpointing."""
        return {
            "discovery_rounds": len(self.discovery_rounds),
            "last_discovery_iteration": self.last_discovery_iteration,
            "active_variables": list(self.spice_agony.active_variables.keys()),
            "ontology_state": self.spice_agony.get_ontology_state(),
            "validation_summary": self.gom_jabbar.get_validation_summary(),
            "prescience_readings": len(self.prescience.readings_history),
        }

    def get_discovered_variables(self) -> list[str]:
        """Get list of all discovered variable names."""
        return list(self.spice_agony.active_variables.keys())

    def force_discovery(self) -> None:
        """
        Force a discovery round regardless of Prescience reading.

        Useful for testing or manual intervention.
        """
        logger.info("Golden Path: Forced activation (bypassing Prescience)")

        # Create a fake ontology gap reading
        from .prescience import PrescienceReading

        fake_reading = PrescienceReading(
            crisis_type=CrisisType.ONTOLOGY_GAP,
            confidence=1.0,
            fitness_gradient=0.0,
            fitness_variance=0.0,
            program_diversity=0.5,
            score_clustering=0.8,
            success_pattern_strength=0.7,
            ontology_coverage=0.5,
            recommended_action="discover_hidden_variables",
            details={"forced": True, "trigger_golden_path": True},
        )
        self.prescience.readings_history.append(fake_reading)

    async def run_discovery_if_needed(self) -> DiscoveryRound | None:
        """
        Check if discovery is needed and run it if so.

        Convenience method for use in evolution loops.
        """
        if self.should_activate():
            return await self.run_discovery()
        return None
