"""
Collaborative Discovery Engine - Multi-Agent Scientific Innovation

A multi-agent system where specialized agents collaborate, debate, and synthesize
NOVEL physics ideas. Unlike literature mining, this generates new concepts from
first principles through adversarial collaboration.

Agent Roles:
1. Theorist - Proposes new physics from first principles
2. Experimentalist - Designs tests and measurements
3. Skeptic - Challenges ideas, finds flaws
4. Synthesizer - Combines insights into working code

The agents debate in rounds, building on each other's ideas while challenging
assumptions. The goal is to discover physics that DOESN'T EXIST in literature.
"""

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openevolve.llm.ensemble import LLMEnsemble

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Specialized agent roles for collaborative discovery"""

    THEORIST = "theorist"
    EXPERIMENTALIST = "experimentalist"
    SKEPTIC = "skeptic"
    SYNTHESIZER = "synthesizer"


@dataclass
class Idea:
    """A physics idea proposed by an agent"""

    id: str
    content: str
    proposer: AgentRole
    physics_variable: str | None = None  # e.g., "thermal_barrier_gradient"
    hypothesis: str | None = None  # e.g., "Steeper gradients reduce loss rate"
    code_sketch: str | None = None  # Python code to compute it
    confidence: float = 0.5
    supporting_arguments: list[str] = field(default_factory=list)
    challenges: list[str] = field(default_factory=list)
    round_proposed: int = 0


@dataclass
class Critique:
    """A critique of an idea from another agent"""

    idea_id: str
    critic: AgentRole
    critique_type: str  # "challenge", "support", "refinement"
    content: str
    suggested_modification: str | None = None


@dataclass
class Synthesis:
    """A synthesis of multiple ideas into testable code"""

    id: str
    source_ideas: list[str]  # idea IDs
    variable_name: str
    description: str
    computation_code: str  # Python code to compute the variable
    validation_test: str  # Python code to validate it correlates with fitness
    consensus_score: float  # 0-1, how much agents agreed


@dataclass
class DebateRound:
    """Record of one debate round"""

    round_number: int
    ideas_proposed: list[Idea]
    critiques: list[Critique]
    eliminated_ideas: list[str]  # idea IDs
    surviving_ideas: list[str]  # idea IDs


@dataclass
class CollaborativeDiscoveryConfig:
    """Configuration for collaborative discovery"""

    max_debate_rounds: int = 5
    min_consensus_for_synthesis: float = 0.6
    max_ideas_per_round: int = 3
    elimination_threshold: float = 0.3  # Ideas below this confidence eliminated
    synthesis_min_ideas: int = 2  # Need at least this many surviving ideas


class DiscoveryAgent:
    """
    A specialized agent that participates in collaborative discovery.

    Each agent has a distinct persona and approach:
    - Theorist: First principles reasoning, mathematical elegance
    - Experimentalist: Empirical grounding, measurement feasibility
    - Skeptic: Find flaws, edge cases, assumptions
    - Synthesizer: Integrate ideas, write code
    """

    def __init__(
        self,
        role: AgentRole,
        llm_ensemble: "LLMEnsemble",
        domain_context: str,
    ):
        self.role = role
        self.llm = llm_ensemble
        self.domain_context = domain_context
        self.persona = self._build_persona()

    def _build_persona(self) -> str:
        """Build the agent's persona prompt"""
        personas = {
            AgentRole.THEORIST: """You are a theoretical physicist who thinks from first principles.
Your approach:
- Start with fundamental physics laws (conservation, symmetry, thermodynamics)
- Propose NEW variables that SHOULD exist based on theory
- Think about what quantities the equations of motion IMPLY but aren't measured
- You prefer elegant, mathematically clean formulations
- You're willing to speculate beyond established physics
- You propose variables that could explain unexplained phenomena

IMPORTANT: Your job is to invent NEW physics concepts, not recite textbook knowledge.
What hidden variables MUST exist to make the physics complete?""",
            AgentRole.EXPERIMENTALIST: """You are an experimental physicist focused on measurement and validation.
Your approach:
- Every variable must be COMPUTABLE from available data
- Design concrete tests to validate hypotheses
- Think about signal-to-noise, measurement error, numerical stability
- Propose practical ways to extract signals from noise
- You ground theoretical ideas in what can actually be measured
- You design clever indirect measurements when direct ones aren't available

IMPORTANT: You make wild theories into testable code. If it can't be computed, it doesn't exist.""",
            AgentRole.SKEPTIC: """You are a critical physicist who finds flaws in reasoning.
Your approach:
- Challenge every assumption - which could be wrong?
- Find edge cases where the idea breaks down
- Question causality - correlation isn't causation
- Look for dimensional analysis errors
- Check conservation law violations
- Ask "what if this is just a proxy for something simpler?"

IMPORTANT: Your job is to STRENGTHEN ideas by attacking them. Weak ideas should fail.
Strong ideas survive your scrutiny and become more robust.""",
            AgentRole.SYNTHESIZER: """You are an integrative physicist who combines insights into working code.
Your approach:
- Find common threads across different ideas
- Write Python code that computes new variables
- Ensure code is numerically stable and efficient
- Create validation tests that check correlations
- Balance theoretical elegance with practical computation
- Produce code that can be integrated into an evaluator

IMPORTANT: You produce WORKING CODE. Every synthesis must include:
1. Variable computation (Python function)
2. Validation test (correlation check)
3. Integration instructions""",
        }
        return personas[self.role]

    async def propose_ideas(
        self,
        crisis_context: dict[str, Any],
        existing_ideas: list[Idea],
        current_round: int,
    ) -> list[Idea]:
        """Propose new physics ideas based on the current crisis"""

        existing_summary = (
            "\n".join(
                [
                    f"- {idea.physics_variable}: {idea.hypothesis} (confidence: {idea.confidence:.2f})"
                    for idea in existing_ideas
                    if idea.physics_variable
                ]
            )
            if existing_ideas
            else "None yet - you're starting fresh."
        )

        prompt = f"""{self.persona}

DOMAIN CONTEXT:
{self.domain_context}

CRISIS SITUATION:
The optimization has plateaued. Current best fitness: {crisis_context.get("best_fitness", "unknown")}
Recent fitness history: {crisis_context.get("fitness_history", [])}
Current metrics being tracked: {crisis_context.get("current_metrics", [])}

EXISTING IDEAS FROM THIS DEBATE:
{existing_summary}

YOUR TASK (Round {current_round}):
Propose 1-3 NEW physics variables that could break through this plateau.

For each idea, provide:
1. VARIABLE NAME: A descriptive snake_case name (e.g., "velocity_space_diffusion_rate")
2. HYPOTHESIS: What you think this variable captures and WHY it matters
3. PHYSICS BASIS: The fundamental physics principle it's based on
4. CODE SKETCH: Python pseudocode to compute it from available data
5. EXPECTED IMPACT: How discovering this could improve optimization

Be CREATIVE and NOVEL. Don't just recite textbook physics. What variables SHOULD exist
but aren't being measured? What hidden degrees of freedom are you missing?

Format each idea as:
### IDEA: <variable_name>
**Hypothesis:** <your hypothesis>
**Physics Basis:** <fundamental principle>
**Code Sketch:**
```python
<pseudocode>
```
**Expected Impact:** <how it helps>
"""

        response = await self.llm.generate(prompt)
        return self._parse_ideas(response, current_round)

    def _parse_ideas(self, response: str, round_num: int) -> list[Idea]:
        """Parse LLM response into structured ideas"""
        ideas = []

        # Split on IDEA markers
        parts = response.split("### IDEA:")
        for part in parts[1:]:  # Skip first empty part
            lines = part.strip().split("\n")
            if not lines:
                continue

            # Extract variable name from first line
            var_name = lines[0].strip()

            # Extract sections
            hypothesis = ""
            code_sketch = ""
            in_code = False

            for line in lines[1:]:
                if line.startswith("**Hypothesis:**"):
                    hypothesis = line.replace("**Hypothesis:**", "").strip()
                elif "```python" in line:
                    in_code = True
                elif "```" in line and in_code:
                    in_code = False
                elif in_code:
                    code_sketch += line + "\n"

            if var_name and hypothesis:
                ideas.append(
                    Idea(
                        id=f"idea_{uuid.uuid4().hex[:8]}",
                        content=part,
                        proposer=self.role,
                        physics_variable=var_name.lower().replace(" ", "_"),
                        hypothesis=hypothesis,
                        code_sketch=code_sketch.strip() if code_sketch else None,
                        round_proposed=round_num,
                    )
                )

        return ideas

    async def critique_ideas(
        self,
        ideas: list[Idea],
        crisis_context: dict[str, Any],
    ) -> list[Critique]:
        """Critique ideas proposed by other agents"""

        if not ideas:
            return []

        ideas_text = "\n\n".join(
            [
                f"### {idea.physics_variable} (by {idea.proposer.value})\n"
                f"Hypothesis: {idea.hypothesis}\n"
                f"Code: {idea.code_sketch or 'Not provided'}"
                for idea in ideas
            ]
        )

        prompt = f"""{self.persona}

DOMAIN CONTEXT:
{self.domain_context}

IDEAS TO CRITIQUE:
{ideas_text}

YOUR TASK:
As the {self.role.value}, critique each idea. You may:
- CHALLENGE: Point out flaws, missing assumptions, or errors
- SUPPORT: Explain why you think this idea is strong
- REFINE: Suggest specific modifications to improve the idea

For each idea, provide your critique in this format:
### CRITIQUE: <variable_name>
**Type:** [CHALLENGE/SUPPORT/REFINE]
**Analysis:** <your critique>
**Suggested Modification:** <if refining, what to change>
**Confidence Adjustment:** [+0.1 to +0.3 for support, -0.1 to -0.3 for challenge, 0 for refine]
"""

        response = await self.llm.generate(prompt)
        return self._parse_critiques(response, ideas)

    def _parse_critiques(self, response: str, ideas: list[Idea]) -> list[Critique]:
        """Parse critique response into structured critiques"""
        critiques = []

        parts = response.split("### CRITIQUE:")
        for part in parts[1:]:
            lines = part.strip().split("\n")
            if not lines:
                continue

            var_name = lines[0].strip().lower().replace(" ", "_")

            # Find matching idea
            matching_idea = None
            for idea in ideas:
                if idea.physics_variable and var_name in idea.physics_variable.lower():
                    matching_idea = idea
                    break

            if not matching_idea:
                continue

            critique_type = "challenge"
            content = ""
            modification = None

            for line in lines[1:]:
                if "**Type:**" in line:
                    type_str = line.replace("**Type:**", "").strip().lower()
                    if "support" in type_str:
                        critique_type = "support"
                    elif "refine" in type_str:
                        critique_type = "refinement"
                elif "**Analysis:**" in line:
                    content = line.replace("**Analysis:**", "").strip()
                elif "**Suggested Modification:**" in line:
                    modification = line.replace("**Suggested Modification:**", "").strip()

            if content:
                critiques.append(
                    Critique(
                        idea_id=matching_idea.id,
                        critic=self.role,
                        critique_type=critique_type,
                        content=content,
                        suggested_modification=modification,
                    )
                )

        return critiques


class CollaborativeDiscovery:
    """
    Orchestrates multi-agent collaborative discovery of new physics.

    The process:
    1. Crisis triggers discovery session
    2. Theorist proposes initial ideas from first principles
    3. All agents debate in rounds:
       - Each agent critiques ideas
       - Low-confidence ideas eliminated
       - Ideas refined based on feedback
    4. Synthesizer produces working code from surviving ideas
    5. Code is validated against fitness data
    """

    def __init__(
        self,
        config: CollaborativeDiscoveryConfig,
        llm_ensemble: "LLMEnsemble",
        domain_context: str,
    ):
        self.config = config
        self.llm = llm_ensemble
        self.domain_context = domain_context

        # Initialize agents
        self.agents = {
            role: DiscoveryAgent(role, llm_ensemble, domain_context) for role in AgentRole
        }

        # State
        self.ideas: dict[str, Idea] = {}
        self.debate_rounds: list[DebateRound] = []
        self.syntheses: list[Synthesis] = []

    async def run_discovery_session(
        self,
        crisis_context: dict[str, Any],
    ) -> list[Synthesis]:
        """
        Run a full collaborative discovery session.

        Args:
            crisis_context: Information about the optimization crisis
                - best_fitness: Current best fitness
                - fitness_history: Recent fitness values
                - current_metrics: What's currently being measured
                - evaluation_artifacts: Any artifacts from evaluation

        Returns:
            List of Synthesis objects with code for new physics variables
        """
        logger.info("Starting collaborative discovery session...")

        # Phase 1: Initial idea generation from Theorist
        logger.info("Phase 1: Theorist generating initial ideas...")
        initial_ideas = await self.agents[AgentRole.THEORIST].propose_ideas(
            crisis_context, [], current_round=1
        )
        for idea in initial_ideas:
            self.ideas[idea.id] = idea

        # Phase 2: Debate rounds
        surviving_ideas = list(self.ideas.values())

        for round_num in range(2, self.config.max_debate_rounds + 1):
            logger.info(f"Phase 2: Debate round {round_num}...")

            round_record = await self._run_debate_round(surviving_ideas, crisis_context, round_num)
            self.debate_rounds.append(round_record)

            # Update surviving ideas
            surviving_ideas = [self.ideas[idea_id] for idea_id in round_record.surviving_ideas]

            if len(surviving_ideas) < self.config.synthesis_min_ideas:
                logger.info("Too few ideas survived - adding more from Experimentalist")
                new_ideas = await self.agents[AgentRole.EXPERIMENTALIST].propose_ideas(
                    crisis_context, surviving_ideas, current_round=round_num
                )
                for idea in new_ideas:
                    self.ideas[idea.id] = idea
                    surviving_ideas.append(idea)

        # Phase 3: Synthesis
        logger.info("Phase 3: Synthesizer combining ideas into code...")
        syntheses = await self._synthesize_ideas(surviving_ideas, crisis_context)
        self.syntheses.extend(syntheses)

        logger.info(f"Discovery session complete. Generated {len(syntheses)} syntheses.")
        return syntheses

    async def _run_debate_round(
        self,
        ideas: list[Idea],
        crisis_context: dict[str, Any],
        round_num: int,
    ) -> DebateRound:
        """Run one round of debate among agents"""

        all_critiques: list[Critique] = []
        new_ideas: list[Idea] = []

        # Each non-theorist agent critiques
        for role in [AgentRole.EXPERIMENTALIST, AgentRole.SKEPTIC]:
            critiques = await self.agents[role].critique_ideas(ideas, crisis_context)
            all_critiques.extend(critiques)

        # Theorist may propose refinements
        refined_ideas = await self.agents[AgentRole.THEORIST].propose_ideas(
            crisis_context, ideas, current_round=round_num
        )
        new_ideas.extend(refined_ideas)

        # Update confidence based on critiques
        for critique in all_critiques:
            if critique.idea_id in self.ideas:
                idea = self.ideas[critique.idea_id]
                if critique.critique_type == "support":
                    idea.confidence = min(1.0, idea.confidence + 0.15)
                    idea.supporting_arguments.append(critique.content)
                elif critique.critique_type == "challenge":
                    idea.confidence = max(0.0, idea.confidence - 0.15)
                    idea.challenges.append(critique.content)
                elif critique.critique_type == "refinement":
                    idea.confidence = min(1.0, idea.confidence + 0.05)

        # Add new ideas
        for idea in new_ideas:
            self.ideas[idea.id] = idea

        # Eliminate low-confidence ideas
        eliminated = []
        surviving = []

        for idea_id, idea in self.ideas.items():
            if idea.confidence < self.config.elimination_threshold:
                eliminated.append(idea_id)
            else:
                surviving.append(idea_id)

        return DebateRound(
            round_number=round_num,
            ideas_proposed=new_ideas,
            critiques=all_critiques,
            eliminated_ideas=eliminated,
            surviving_ideas=surviving,
        )

    async def _synthesize_ideas(
        self,
        ideas: list[Idea],
        crisis_context: dict[str, Any],
    ) -> list[Synthesis]:
        """Have the Synthesizer turn surviving ideas into code"""

        if not ideas:
            return []

        ideas_summary = "\n\n".join(
            [
                f"### {idea.physics_variable}\n"
                f"Hypothesis: {idea.hypothesis}\n"
                f"Confidence: {idea.confidence:.2f}\n"
                f"Supporting: {idea.supporting_arguments}\n"
                f"Challenges: {idea.challenges}\n"
                f"Code Sketch:\n{idea.code_sketch or 'None'}"
                for idea in ideas
            ]
        )

        prompt = f"""{self.agents[AgentRole.SYNTHESIZER].persona}

DOMAIN CONTEXT:
{self.domain_context}

SURVIVING IDEAS FROM DEBATE:
{ideas_summary}

CRISIS CONTEXT:
Best fitness: {crisis_context.get("best_fitness", "unknown")}
Current metrics: {crisis_context.get("current_metrics", [])}

YOUR TASK:
Synthesize these ideas into WORKING Python code. For each new physics variable:

1. Write a function that computes it from available data
2. Write a validation test that checks correlation with fitness
3. Explain how to integrate it into the evaluator

Format each synthesis as:
### SYNTHESIS: <variable_name>
**Description:** <what this variable captures>
**Source Ideas:** <which ideas contributed>

**Computation Code:**
```python
def compute_<variable_name>(metrics: dict, artifacts: dict) -> float:
    \"\"\"
    Compute <variable_name> from evaluation data.

    Args:
        metrics: Dictionary of current metrics (mirror_ratio, well_depth, etc.)
        artifacts: Dictionary of evaluation artifacts (field data, coil config, etc.)

    Returns:
        The computed value of <variable_name>
    \"\"\"
    # Your implementation here
    pass
```

**Validation Test:**
```python
def validate_<variable_name>(history: list[dict]) -> tuple[bool, float]:
    \"\"\"
    Validate that <variable_name> correlates with fitness.

    Args:
        history: List of dicts with 'fitness' and '<variable_name>' keys

    Returns:
        (is_valid, correlation_coefficient)
    \"\"\"
    # Your implementation here
    pass
```

**Integration Notes:** <how to add this to the evaluator>
"""

        response = await self.llm.generate(prompt)
        return self._parse_syntheses(response, ideas)

    def _parse_syntheses(self, response: str, source_ideas: list[Idea]) -> list[Synthesis]:
        """Parse synthesis response into structured syntheses"""
        syntheses = []

        parts = response.split("### SYNTHESIS:")
        for part in parts[1:]:
            lines = part.strip().split("\n")
            if not lines:
                continue

            var_name = lines[0].strip().lower().replace(" ", "_")

            description = ""
            computation_code = ""
            validation_code = ""
            in_computation = False
            in_validation = False

            for line in lines[1:]:
                if "**Description:**" in line:
                    description = line.replace("**Description:**", "").strip()
                elif "**Computation Code:**" in line:
                    in_computation = True
                    in_validation = False
                elif "**Validation Test:**" in line:
                    in_computation = False
                    in_validation = True
                elif "**Integration Notes:**" in line:
                    in_computation = False
                    in_validation = False
                elif "```python" in line or "```" in line:
                    continue
                elif in_computation:
                    computation_code += line + "\n"
                elif in_validation:
                    validation_code += line + "\n"

            if var_name and computation_code:
                # Calculate consensus based on surviving ideas' confidence
                avg_confidence = sum(i.confidence for i in source_ideas) / len(source_ideas)

                syntheses.append(
                    Synthesis(
                        id=f"synth_{uuid.uuid4().hex[:8]}",
                        source_ideas=[i.id for i in source_ideas],
                        variable_name=var_name,
                        description=description,
                        computation_code=computation_code.strip(),
                        validation_test=validation_code.strip(),
                        consensus_score=avg_confidence,
                    )
                )

        return syntheses

    def get_debate_summary(self) -> dict[str, Any]:
        """Get a summary of the debate process"""
        return {
            "total_ideas_proposed": len(self.ideas),
            "debate_rounds": len(self.debate_rounds),
            "ideas_by_proposer": {
                role.value: len([i for i in self.ideas.values() if i.proposer == role])
                for role in AgentRole
            },
            "final_surviving_ideas": [
                {
                    "variable": idea.physics_variable,
                    "confidence": idea.confidence,
                    "proposer": idea.proposer.value,
                }
                for idea in self.ideas.values()
                if idea.confidence >= self.config.elimination_threshold
            ],
            "syntheses": [
                {
                    "variable": s.variable_name,
                    "consensus": s.consensus_score,
                    "has_code": bool(s.computation_code),
                }
                for s in self.syntheses
            ],
        }


# Integration with Heisenberg Engine
async def collaborative_crisis_handler(
    crisis_context: dict[str, Any],
    llm_ensemble: "LLMEnsemble",
    domain_context: str,
    config: CollaborativeDiscoveryConfig | None = None,
) -> list[Synthesis]:
    """
    Handle an epistemic crisis through collaborative discovery.

    This can be called from the DiscoveryEngine when a crisis is detected,
    as an alternative to probe-based discovery.

    Args:
        crisis_context: Information about the crisis
        llm_ensemble: LLM for agent conversations
        domain_context: Description of the physics domain
        config: Optional configuration

    Returns:
        List of syntheses with code for new physics variables
    """
    config = config or CollaborativeDiscoveryConfig()

    discovery = CollaborativeDiscovery(
        config=config,
        llm_ensemble=llm_ensemble,
        domain_context=domain_context,
    )

    syntheses = await discovery.run_discovery_session(crisis_context)

    logger.info(f"Collaborative discovery complete: {discovery.get_debate_summary()}")

    return syntheses
