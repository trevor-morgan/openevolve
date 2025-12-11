"""
Ontology Management for Heisenberg Engine

This module tracks the evolving state space (ontology) during scientific discovery.
An ontology defines what variables the system knows about and can optimize.

Key insight: Scientific breakthroughs often come from discovering NEW variables,
not just optimizing relationships between known ones.

Example:
    - We didn't solve smallpox by optimizing "Bad Air" - we discovered Bacteria
    - We didn't explain orbits with Euclidean geometry - we discovered Space-Time Curvature
"""

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Variable:
    """
    Represents a discovered variable in the state space.

    Variables can come from:
    - User specification (genesis ontology)
    - Probe discovery (Heisenberg Engine)
    - Inference from patterns

    Attributes:
        name: Human-readable variable name (e.g., "cache_locality")
        var_type: Type of variable ("continuous", "categorical", "latent")
        source: How this variable was discovered ("user", "probe", "inferred")
        discovery_method: Which probe/method discovered this (if applicable)
        extraction_code: Python code to extract this variable from raw data
        confidence: How confident we are this is a real variable (0-1)
        description: Human-readable description of what this variable represents
        metadata: Additional variable-specific data
    """

    name: str
    var_type: str = "continuous"  # "continuous", "categorical", "latent"
    source: str = "user"  # "user", "probe", "inferred"
    discovery_method: Optional[str] = None
    extraction_code: Optional[str] = None
    confidence: float = 1.0
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Variable":
        """Deserialize from dictionary"""
        return cls(**data)

    def to_prompt_context(self) -> str:
        """Format variable for inclusion in LLM prompts"""
        lines = [f"- **{self.name}** ({self.var_type})"]
        if self.description:
            lines.append(f"  Description: {self.description}")
        if self.source == "probe":
            lines.append(f"  Discovered via: {self.discovery_method or 'probe'}")
            lines.append(f"  Confidence: {self.confidence:.2f}")
        if self.extraction_code:
            lines.append(f"  Extraction: `{self.extraction_code[:50]}...`")
        return "\n".join(lines)


@dataclass
class Ontology:
    """
    Represents a complete variable/state space definition.

    An ontology is the set of variables that the system knows about.
    Ontologies form a lineage - each expansion creates a new generation
    that inherits from its parent.

    Attributes:
        id: Unique identifier for this ontology version
        generation: How many expansions from the genesis ontology
        parent_id: ID of the parent ontology (None for genesis)
        variables: List of variables in this ontology
        discovered_via: Crisis ID that triggered this expansion (if applicable)
        timestamp: When this ontology was created
        metadata: Additional ontology-specific data
    """

    id: str
    generation: int = 0
    parent_id: Optional[str] = None
    variables: List[Variable] = field(default_factory=list)
    discovered_via: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "id": self.id,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "variables": [v.to_dict() for v in self.variables],
            "discovered_via": self.discovered_via,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Ontology":
        """Deserialize from dictionary"""
        variables = [Variable.from_dict(v) for v in data.get("variables", [])]
        return cls(
            id=data["id"],
            generation=data.get("generation", 0),
            parent_id=data.get("parent_id"),
            variables=variables,
            discovered_via=data.get("discovered_via"),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )

    def to_prompt_context(self) -> str:
        """Format ontology for inclusion in LLM prompts"""
        lines = [
            f"## Current Ontology (Generation {self.generation})",
            "",
            "### Known Variables:",
        ]

        if not self.variables:
            lines.append("- (No variables defined)")
        else:
            for var in self.variables:
                lines.append(var.to_prompt_context())

        if self.generation > 0:
            lines.append("")
            lines.append(f"*This ontology was expanded from generation {self.generation - 1}*")
            if self.discovered_via:
                lines.append(f"*Expansion triggered by crisis: {self.discovered_via}*")

        return "\n".join(lines)

    def get_variable_names(self) -> List[str]:
        """Get list of variable names"""
        return [v.name for v in self.variables]

    def get_variable(self, name: str) -> Optional[Variable]:
        """Get a variable by name"""
        for var in self.variables:
            if var.name == name:
                return var
        return None

    def get_extraction_code(self) -> str:
        """
        Generate combined extraction code for all probe-discovered variables.

        Returns Python code that extracts all discovered variables from artifacts.
        """
        probe_vars = [v for v in self.variables if v.source == "probe" and v.extraction_code]

        if not probe_vars:
            return "# No probe-discovered variables to extract"

        lines = [
            "def extract_ontology_variables(artifacts: dict, metrics: dict) -> dict:",
            '    """Extract discovered variables from evaluation data"""',
            "    extracted = {}",
            "",
        ]

        for var in probe_vars:
            lines.append(f"    # Extract: {var.name}")
            lines.append("    try:")
            # Indent the extraction code
            for code_line in var.extraction_code.split("\n"):
                lines.append(f"        {code_line}")
            lines.append(f'        extracted["{var.name}"] = result')
            lines.append("    except Exception as e:")
            lines.append(f'        extracted["{var.name}"] = None')
            lines.append("")

        lines.append("    return extracted")

        return "\n".join(lines)


class OntologyManager:
    """
    Manages ontology lineage and state space evolution.

    The OntologyManager tracks:
    - The current ontology being used
    - History of all ontology versions
    - Lineage relationships between ontologies

    Usage:
        manager = OntologyManager()
        genesis = manager.create_genesis_ontology(["input_size", "complexity"])

        # Later, when a crisis is detected and new variables discovered:
        new_ontology = manager.expand_ontology(
            new_variables=[Variable(name="cache_locality", ...)],
            discovered_via="crisis_123"
        )
    """

    def __init__(self):
        self.ontology_history: Dict[str, Ontology] = {}
        self.current_ontology: Optional[Ontology] = None

        logger.info("Initialized OntologyManager")

    def create_genesis_ontology(
        self,
        variable_names: List[str] = None,
        variables: List[Variable] = None,
    ) -> Ontology:
        """
        Create the initial (genesis) ontology.

        Args:
            variable_names: Simple list of variable names (creates basic Variables)
            variables: Full Variable objects (takes precedence over variable_names)

        Returns:
            The genesis Ontology (generation 0)
        """
        if variables:
            vars_list = variables
        elif variable_names:
            vars_list = [
                Variable(name=name, source="user", confidence=1.0)
                for name in variable_names
            ]
        else:
            vars_list = []

        genesis = Ontology(
            id=f"ontology_genesis_{uuid.uuid4().hex[:8]}",
            generation=0,
            parent_id=None,
            variables=vars_list,
            metadata={"is_genesis": True},
        )

        self.ontology_history[genesis.id] = genesis
        self.current_ontology = genesis

        logger.info(
            f"Created genesis ontology {genesis.id} with {len(vars_list)} variables: "
            f"{[v.name for v in vars_list]}"
        )

        return genesis

    def expand_ontology(
        self,
        new_variables: List[Variable],
        discovered_via: Optional[str] = None,
    ) -> Ontology:
        """
        Expand the current ontology with newly discovered variables.

        This creates a new ontology generation that includes all variables
        from the parent plus the new discoveries.

        Args:
            new_variables: List of newly discovered Variables
            discovered_via: ID of the crisis that triggered this expansion

        Returns:
            New Ontology with incremented generation
        """
        if self.current_ontology is None:
            # No current ontology - create genesis first
            logger.warning("No current ontology - creating genesis with new variables")
            return self.create_genesis_ontology(variables=new_variables)

        # Inherit all variables from parent
        inherited_vars = [
            Variable.from_dict(v.to_dict())
            for v in self.current_ontology.variables
        ]

        # Add new variables (avoid duplicates by name)
        existing_names = {v.name for v in inherited_vars}
        for new_var in new_variables:
            if new_var.name not in existing_names:
                inherited_vars.append(new_var)
                existing_names.add(new_var.name)
            else:
                logger.warning(
                    f"Variable '{new_var.name}' already exists in ontology - skipping"
                )

        # Create new ontology generation
        new_ontology = Ontology(
            id=f"ontology_gen{self.current_ontology.generation + 1}_{uuid.uuid4().hex[:8]}",
            generation=self.current_ontology.generation + 1,
            parent_id=self.current_ontology.id,
            variables=inherited_vars,
            discovered_via=discovered_via,
            metadata={
                "expanded_from": self.current_ontology.id,
                "new_variables": [v.name for v in new_variables],
            },
        )

        self.ontology_history[new_ontology.id] = new_ontology
        self.current_ontology = new_ontology

        logger.info(
            f"Expanded ontology to generation {new_ontology.generation}: "
            f"added {len(new_variables)} variables "
            f"({[v.name for v in new_variables]}), "
            f"total {len(new_ontology.variables)} variables"
        )

        return new_ontology

    def get_lineage(self, ontology_id: Optional[str] = None) -> List[Ontology]:
        """
        Get the full lineage of an ontology back to genesis.

        Args:
            ontology_id: ID of ontology to get lineage for (default: current)

        Returns:
            List of Ontologies from genesis to specified ontology
        """
        if ontology_id is None:
            if self.current_ontology is None:
                return []
            ontology_id = self.current_ontology.id

        lineage = []
        current_id = ontology_id

        while current_id and current_id in self.ontology_history:
            ontology = self.ontology_history[current_id]
            lineage.append(ontology)
            current_id = ontology.parent_id

        return list(reversed(lineage))

    def get_new_variables_since(self, generation: int) -> List[Variable]:
        """
        Get all variables discovered since a given generation.

        Args:
            generation: The generation to compare from

        Returns:
            List of Variables added after the specified generation
        """
        if self.current_ontology is None:
            return []

        new_vars = []
        for var in self.current_ontology.variables:
            if var.source == "probe":
                # Check if this variable was added after the specified generation
                # by looking at when it was first seen in the lineage
                for onto in self.get_lineage():
                    if onto.generation > generation and var.name in onto.get_variable_names():
                        new_vars.append(var)
                        break

        return new_vars

    def reset(self) -> None:
        """Reset the crisis detector state (called after ontology expansion)"""
        # The OntologyManager doesn't need resetting per se,
        # but this method exists for interface consistency
        pass

    def save(self, path: str) -> None:
        """
        Save ontology history to disk.

        Args:
            path: Path to save the ontology state
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        data = {
            "ontology_history": {
                oid: onto.to_dict()
                for oid, onto in self.ontology_history.items()
            },
            "current_ontology_id": self.current_ontology.id if self.current_ontology else None,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(
            f"Saved ontology history with {len(self.ontology_history)} ontologies to {path}"
        )

    def load(self, path: str) -> None:
        """
        Load ontology history from disk.

        Args:
            path: Path to load the ontology state from
        """
        with open(path, "r") as f:
            data = json.load(f)

        self.ontology_history = {
            oid: Ontology.from_dict(onto_data)
            for oid, onto_data in data.get("ontology_history", {}).items()
        }

        current_id = data.get("current_ontology_id")
        if current_id and current_id in self.ontology_history:
            self.current_ontology = self.ontology_history[current_id]
        else:
            self.current_ontology = None

        logger.info(
            f"Loaded ontology history with {len(self.ontology_history)} ontologies from {path}"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about ontology evolution"""
        if self.current_ontology is None:
            return {
                "current_generation": 0,
                "total_ontologies": 0,
                "total_variables": 0,
                "probe_discovered_variables": 0,
            }

        probe_vars = [v for v in self.current_ontology.variables if v.source == "probe"]

        return {
            "current_generation": self.current_ontology.generation,
            "total_ontologies": len(self.ontology_history),
            "total_variables": len(self.current_ontology.variables),
            "probe_discovered_variables": len(probe_vars),
            "variable_names": self.current_ontology.get_variable_names(),
            "lineage_length": len(self.get_lineage()),
        }
